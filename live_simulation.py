"""
live_simulation.py — Unified live data simulation, scoring, and broadcast.

Streams rows from test_stream.csv through the pre-trained encoder + isolation
forest, writes each scored result to processed_datasets/live_results.jsonl,
and optionally connects WebSocket clients at ws://localhost:8765.

Usage:
    python live_simulation.py [--interval 0.5] [--input <csv_path>] [--port 8765]

Models are loaded ONCE at startup and reused for every row, which eliminates the
per-row disk-read overhead present in the old two-file design.
"""

import argparse
import asyncio
import json
import os
import sys
import signal
import time

import joblib
import numpy as np
import pandas as pd
import websockets

# Paths (loaded from config.py)
import config
BASE_DIR                    = config.BASE_DIR
SCALER_PATH                 = config.LIVE_SCALER_PATH
ENCODER_PATH                = config.LIVE_ENCODER_PATH
IF_PATH                     = config.LIVE_IF_PATH
IF_SCORES_PATH              = config.LIVE_IF_SCORES_PATH
IF_BASELINE_PATH            = config.LIVE_IF_BASELINE_PATH
CALIBRATION_THRESHOLD_PATH  = config.LIVE_CALIBRATION_THRESHOLD_PATH
DEFAULT_INPUT               = config.LIVE_DEFAULT_INPUT
DEFAULT_OUTPUT              = config.LIVE_OUTPUT
PAUSE_FLAG                  = config.LIVE_PAUSE_FLAG

# Fallback percentile thresholds (used only when calibration_thresholds.json is absent)
_FALLBACK_CRITICAL = 95.0
_FALLBACK_HIGH     = 90.0
_FALLBACK_MEDIUM   = 80.0

# ── Global WebSocket client registry ─────────────────────────────────────────
_ws_clients: set = set()
_stop_event: asyncio.Event | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Scorer — loads models once, reuses them for the entire run
# ─────────────────────────────────────────────────────────────────────────────

class LiveScorer:
    def __init__(self) -> None:
        print("[live_simulation] Loading scaler …", flush=True)
        self.scaler = joblib.load(SCALER_PATH)

        print("[live_simulation] Loading encoder …", flush=True)
        from tensorflow.keras.models import load_model  # deferred → faster cold import
        self.encoder = load_model(ENCODER_PATH, compile=False)

        print("[live_simulation] Loading isolation forest …", flush=True)
        # Add repo root to sys.path so the scripts package is importable when
        # this file is launched as a subprocess from a different working directory.
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from scripts.UEBAIsolationForest import UEBAIsolationForest
        self._iforest_cls = None  # kept for symmetry; using loaded instance
        self.iforest = UEBAIsolationForest()
        self.iforest.load(IF_PATH)

        # Load reference score distribution for percentile ranking.
        # Prefer the clean calibration baseline; fall back to full training scores.
        print("[live_simulation] Loading reference score distribution …", flush=True)
        _ref_path = IF_BASELINE_PATH if os.path.exists(IF_BASELINE_PATH) else IF_SCORES_PATH
        self.ref_scores: np.ndarray = np.load(_ref_path)
        _ref_label = "clean calibration baseline" if _ref_path == IF_BASELINE_PATH else "training distribution"
        print(
            f"[live_simulation] Ready. Reference distribution ({_ref_label}): {len(self.ref_scores):,} rows.",
            flush=True,
        )

        # Load calibrated absolute IF thresholds if available; fall back to percentile cutoffs.
        self._if_absolute_thresholds: dict | None = None
        if os.path.exists(CALIBRATION_THRESHOLD_PATH):
            with open(CALIBRATION_THRESHOLD_PATH) as _f:
                _cal = json.load(_f)
            self._if_absolute_thresholds = _cal.get("if")
            print("[live_simulation] Loaded calibrated absolute IF thresholds.", flush=True)
        else:
            print("[live_simulation] calibration_thresholds.json not found — using fallback percentile thresholds.", flush=True)

    def score_row(self, row_df: pd.DataFrame) -> dict:
        """Score one row and return a dict with all required fields."""
        # Preserve metadata before dropping
        meta: dict = {}
        for col in ("user", "pc", "day"):
            if col in row_df.columns:
                meta[col] = str(row_df[col].iloc[0])

        # Keep original CERT time context for downstream sorting/display.
        for ts_col in ("timestamp", "datetime", "date", "day"):
            if ts_col in row_df.columns:
                meta["cert_timestamp"] = str(row_df[ts_col].iloc[0])
                break

        # Feature matrix — drop metadata columns and any unnamed index columns
        # (parquet files created without index_col=0 retain an "Unnamed: 0" column)
        drop_cols = [c for c in row_df.columns
                     if c in ("user", "pc", "day") or str(c).startswith("Unnamed:")]
        feat_df = row_df.drop(columns=drop_cols)

        t0 = time.perf_counter()
        scaled    = self.scaler.transform(feat_df.values.astype("float32"))
        embedding = self.encoder.predict(scaled, verbose=0)
        raw_score = float(self.iforest.anomaly_score(embedding)[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Global percentile (fraction of reference scores strictly below this score)
        percentile = float(np.mean(self.ref_scores < raw_score) * 100)

        # Risk band: use calibrated absolute thresholds when available, else percentile cutoffs
        if self._if_absolute_thresholds is not None:
            if_risk_band = "CRITICAL"
            for label, thresh in self._if_absolute_thresholds.items():
                if raw_score <= thresh:
                    if_risk_band = label
                    break
        else:
            if_risk_band = (
                "CRITICAL" if percentile >= _FALLBACK_CRITICAL else
                "HIGH"     if percentile >= _FALLBACK_HIGH     else
                "MEDIUM"   if percentile >= _FALLBACK_MEDIUM   else
                "LOW"
            )

        payload = {
            **meta,
            "if_anomaly_score":  round(raw_score,  6),
            "if_percentile_rank": round(percentile, 2),
            "if_risk_band":      if_risk_band,
            "_score_ms":         round(elapsed_ms, 1),   # diagnostic; dashboard ignores this
        }
        return payload


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket handler
# ─────────────────────────────────────────────────────────────────────────────

async def _ws_handler(ws) -> None:
    _ws_clients.add(ws)
    print(f"[live_simulation] WS client connected ({len(_ws_clients)} total)", flush=True)
    try:
        await ws.wait_closed()
    finally:
        _ws_clients.discard(ws)
        print(f"[live_simulation] WS client disconnected ({len(_ws_clients)} remaining)", flush=True)


async def _broadcast(payload: dict) -> None:
    if not _ws_clients:
        return
    msg = json.dumps(payload)
    results = await asyncio.gather(
        *[c.send(msg) for c in list(_ws_clients)],
        return_exceptions=True,
    )
    # Remove dead connections
    for client, result in zip(list(_ws_clients), results):
        if isinstance(result, Exception):
            _ws_clients.discard(client)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation producer (async so it can yield control for WS broadcasts)
# ─────────────────────────────────────────────────────────────────────────────

async def _run_simulation(
    scorer: LiveScorer,
    input_path: str,
    output_path: str,
    interval: float,
) -> None:
    assert _stop_event is not None

    if not os.path.exists(input_path):
        print(f"[live_simulation] ERROR: input file not found: {input_path}", flush=True)
        _stop_event.set()
        return

    # Remove any stale pause flag left by a previously interrupted run.
    if os.path.exists(PAUSE_FLAG):
        os.remove(PAUSE_FLAG)

    # Preserve source order: stream rows exactly as they appear in the file.
    if input_path.endswith(".parquet"):
        test_df = pd.read_parquet(input_path)
    else:
        test_df = pd.read_csv(input_path, index_col=0)

    total   = len(test_df)
    print(f"[live_simulation] Streaming {total:,} rows from {os.path.basename(input_path)}", flush=True)

    # Clear / create fresh output file
    with open(output_path, "w", encoding="utf-8") as f:
        pass  # truncate

    for idx in range(total):
        if _stop_event.is_set():
            print("[live_simulation] Stop signal received — exiting.", flush=True)
            break

        row_df = test_df.iloc[[idx]]
        payload = scorer.score_row(row_df)
        payload["event_index"] = idx  # helps dashboard deduplicate rows

        # Append to JSONL output
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        # Broadcast to any connected WS clients
        await _broadcast(payload)

        print(
            f"[live_simulation] [{idx+1:>5}/{total}]  user={payload.get('user','?')}"
            f"  score={payload['if_anomaly_score']:.4f}  pct={payload['if_percentile_rank']:.1f}"
            f"  risk={payload['if_risk_band']}  ({payload['_score_ms']} ms)",
            flush=True,
        )

        # Throttle to simulate real-time arrival
        await asyncio.sleep(interval)

        # Honour pause flag: spin-wait until the dashboard removes it or a stop is signalled.
        if os.path.exists(PAUSE_FLAG):
            print("[live_simulation] Paused — waiting for resume …", flush=True)
            while os.path.exists(PAUSE_FLAG) and not _stop_event.is_set():
                await asyncio.sleep(0.25)
            if not _stop_event.is_set():
                print("[live_simulation] Resumed.", flush=True)

    # Write end-of-stream sentinel so the dashboard can detect natural completion
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"_eos": True}) + "\n")

    print("[live_simulation] Stream complete.", flush=True)
    _stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def _main(args: argparse.Namespace) -> None:
    global _stop_event
    _stop_event = asyncio.Event()

    # Start WebSocket server (non-blocking background task)
    ws_server = await websockets.serve(_ws_handler, "localhost", args.port)
    print(f"[live_simulation] WebSocket listening on ws://localhost:{args.port}", flush=True)

    # Load models (synchronous but happens before any async work starts)
    scorer = LiveScorer()

    # Run simulation; when it finishes, cancel the server
    await _run_simulation(
        scorer,
        input_path=args.input,
        output_path=args.output,
        interval=args.interval,
    )
    ws_server.close()
    await ws_server.wait_closed()


def _handle_signal(signum, frame):
    print(f"\n[live_simulation] Signal {signum} received — shutting down.", flush=True)
    if _stop_event is not None:
        _stop_event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live insider-threat data simulation")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between emitted rows (default: 0.5)")
    parser.add_argument("--input",    type=str,   default=DEFAULT_INPUT,
                        help="Path to the test-stream CSV")
    parser.add_argument("--output",   type=str,   default=DEFAULT_OUTPUT,
                        help="Path for the scored JSONL output file")
    parser.add_argument("--port",     type=int,   default=8765,
                        help="WebSocket port (default: 8765)")
    args = parser.parse_args()

    # Graceful shutdown on SIGTERM / SIGINT
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    asyncio.run(_main(args))
