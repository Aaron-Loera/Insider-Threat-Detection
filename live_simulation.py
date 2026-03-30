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

# ── Paths (all relative to the repo root, i.e. this file's directory) ────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "encoders",         "encoder_model_1", "feature_scaler.pkl")
ENCODER_PATH= os.path.join(BASE_DIR, "encoders",         "encoder_model_1", "encoder_model.keras")
IF_PATH     = os.path.join(BASE_DIR, "isolation_forests","iforest_model_1", "iforest_model.pkl")
# Reference distribution used to compute global percentile ranks
ALERT_TABLE_PARQUET = os.path.join(BASE_DIR, "explainability", "alert_table", "alert_table_2.parquet")
ALERT_TABLE_CSV     = os.path.join(BASE_DIR, "explainability", "alert_table", "alert_table_2.csv")
# Default simulation input
DEFAULT_INPUT  = os.path.join(BASE_DIR, "processed_datasets", "test_stream.csv")
# Live output (one JSON object per line, appended as rows arrive)
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "processed_datasets", "live_results.jsonl")

# Risk-band thresholds (percentile cutoffs, consistent with the dashboard)
HIGH_THRESH   = 95.0
MEDIUM_THRESH = 80.0

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

        # Load historical scores for global percentile computation
        print("[live_simulation] Loading reference score distribution …", flush=True)
        if os.path.exists(ALERT_TABLE_PARQUET):
            ref = pd.read_parquet(ALERT_TABLE_PARQUET, columns=["if_anomaly_score"])
        else:
            ref = pd.read_csv(ALERT_TABLE_CSV, usecols=["if_anomaly_score"])
        self.ref_scores: np.ndarray = ref["if_anomaly_score"].dropna().values
        print(
            f"[live_simulation] Ready. Reference distribution: {len(self.ref_scores):,} rows.",
            flush=True,
        )

    def score_row(self, row_df: pd.DataFrame) -> dict:
        """Score one row and return a dict with all required fields."""
        # Preserve metadata before dropping
        meta: dict = {}
        for col in ("user", "pc", "day"):
            if col in row_df.columns:
                meta[col] = str(row_df[col].iloc[0])

        # Feature matrix — drop metadata columns
        feat_df = row_df.drop(
            columns=[c for c in ("user", "pc", "day") if c in row_df.columns]
        )

        t0 = time.perf_counter()
        scaled    = self.scaler.transform(feat_df.values.astype("float32"))
        embedding = self.encoder.predict(scaled, verbose=0)
        raw_score = float(self.iforest.anomaly_score(embedding)[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Global percentile (fraction of reference scores strictly below this score)
        percentile = float(np.mean(self.ref_scores < raw_score) * 100)

        risk_level = (
            "HIGH"   if percentile >= HIGH_THRESH   else
            "MEDIUM" if percentile >= MEDIUM_THRESH else
            "LOW"
        )

        payload = {
            **meta,
            "anomaly_score":    round(raw_score,  6),
            "percentile_rank":  round(percentile, 2),
            "risk_level":       risk_level,
            "_score_ms":        round(elapsed_ms, 1),   # diagnostic; dashboard ignores this
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

    test_df = pd.read_csv(input_path)
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
            f"  score={payload['anomaly_score']:.4f}  pct={payload['percentile_rank']:.1f}"
            f"  risk={payload['risk_level']}  ({payload['_score_ms']} ms)",
            flush=True,
        )

        # Throttle to simulate real-time arrival
        await asyncio.sleep(interval)

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
