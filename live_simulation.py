"""
live_simulation.py — Unified live data simulation, scoring, and broadcast.

Streams rows from the test-stream dataset through the full ML pipeline:
  scaler → autoencoder (reconstruction error + explanation) → encoder → isolation
  forest (anomaly score + percentile) → AlertObjectBuilder (composite score +
  explanation text)

Each scored row is written to processed_datasets/live_results.jsonl and
broadcast to any connected WebSocket clients at ws://localhost:8765.

The JSONL payload includes every field the dashboard needs:
  ae_percentile_rank, ae_risk_band, if_percentile_rank, if_risk_band,
  composite_score, composite_risk_band, both_signals_high, top_contributors,
  explanation, group_error_*, plus all behavioral passthrough columns for the
  radar chart and heatmap.

Feature columns are determined at startup from the fitted scaler's
feature_names_in_ attribute, so the pipeline adapts automatically when the
dataset gains or loses columns — no hardcoded feature list to maintain.

Usage:
    python live_simulation.py [--interval 0.5] [--input <path>] [--port 8765]
"""

import argparse
import asyncio
import json
import math
import os
import sys
import signal
import time

import joblib
import numpy as np
import pandas as pd
import websockets

import config

BASE_DIR            = config.BASE_DIR
SCALER_PATH         = config.LIVE_SCALER_PATH
ENCODER_PATH        = config.LIVE_ENCODER_PATH
AE_PATH             = config.LIVE_AE_PATH
IF_PATH             = config.LIVE_IF_PATH
IF_SCORES_PATH      = config.LIVE_IF_SCORES_PATH
LIVE_RECON_TABLE    = config.LIVE_RECON_TABLE_PATH
DEFAULT_INPUT       = config.LIVE_DEFAULT_INPUT
DEFAULT_OUTPUT      = config.LIVE_OUTPUT
PAUSE_FLAG          = config.LIVE_PAUSE_FLAG

# Risk-band percentile cutoffs (must match AlertObjectBuilder defaults)
CRITICAL_THRESH = 95.0
HIGH_THRESH     = 90.0
MEDIUM_THRESH   = 80.0

# Hardcoded fallback feature list used only when the scaler pre-dates sklearn 1.0
# and therefore does not carry feature_names_in_.
_FALLBACK_FEATURE_COLS: list[str] = [
    "logon_count", "logoff_count", "off_hours_logon",
    "file_open_count", "file_write_count", "file_copy_count", "file_delete_count",
    "off_hours_files_accessed",
    "usb_insert_count", "usb_remove_count", "off_hours_usb_usage",
    "emails_sent", "external_emails_sent", "attachments_sent", "off_hours_emails",
    "http_total_requests", "http_visit_count", "http_download_count", "http_upload_count",
    "http_jobsite_visits", "http_cloud_storage_visits", "http_suspicious_site_visits",
    "off_hours_http_requests", "http_long_url_count",
    "pc_seen_before", "new_pc_after_stable_history", "pc_prior_use_ratio",
    "primary_pc_activity_ratio", "distinct_pcs_used_prior",
    "pcs_used_count", "non_primary_pc_used_flag", "non_primary_pc_http_requests_flag",
    "non_primary_pc_usb_flag", "non_primary_pc_file_copy_flag",
    "unique_files_accessed", "unique_recipients", "unique_domains_visited",
    "logon_count_zscore", "logoff_count_zscore", "off_hours_logon_zscore",
    "file_open_count_zscore", "file_write_count_zscore", "file_copy_count_zscore",
    "file_delete_count_zscore", "unique_files_accessed_zscore",
    "off_hours_files_accessed_zscore", "usb_insert_count_zscore",
    "usb_remove_count_zscore", "off_hours_usb_usage_zscore",
    "emails_sent_zscore", "external_emails_sent_zscore", "attachments_sent_zscore",
    "off_hours_emails_zscore", "unique_recipients_zscore",
    "http_total_requests_zscore", "http_visit_count_zscore", "http_download_count_zscore",
    "http_upload_count_zscore", "http_jobsite_visits_zscore",
    "http_cloud_storage_visits_zscore", "http_suspicious_site_visits_zscore",
    "off_hours_http_requests_zscore", "http_long_url_count_zscore",
    "unique_domains_visited_zscore",
    "pcs_used_count_zscore", "non_primary_pc_used_flag_zscore",
    "non_primary_pc_http_requests_flag_zscore", "non_primary_pc_usb_flag_zscore",
    "non_primary_pc_file_copy_flag_zscore",
    "logon_count_rolling_delta", "logoff_count_rolling_delta",
    "off_hours_logon_rolling_delta", "file_open_count_rolling_delta",
    "file_write_count_rolling_delta", "file_copy_count_rolling_delta",
    "file_delete_count_rolling_delta", "unique_files_accessed_rolling_delta",
    "off_hours_files_accessed_rolling_delta", "usb_insert_count_rolling_delta",
    "usb_remove_count_rolling_delta", "off_hours_usb_usage_rolling_delta",
    "emails_sent_rolling_delta", "external_emails_sent_rolling_delta",
    "attachments_sent_rolling_delta", "off_hours_emails_rolling_delta",
    "unique_recipients_rolling_delta",
    "http_total_requests_rolling_delta", "http_visit_count_rolling_delta",
    "http_download_count_rolling_delta", "http_upload_count_rolling_delta",
    "http_jobsite_visits_rolling_delta", "http_cloud_storage_visits_rolling_delta",
    "http_suspicious_site_visits_rolling_delta", "off_hours_http_requests_rolling_delta",
    "http_long_url_count_rolling_delta", "unique_domains_visited_rolling_delta",
    "pcs_used_count_rolling_delta", "non_primary_pc_used_flag_rolling_delta",
    "non_primary_pc_http_requests_flag_rolling_delta", "non_primary_pc_usb_flag_rolling_delta",
    "non_primary_pc_file_copy_flag_rolling_delta",
    "off_hours_activity_flag", "usb_file_activity_flag", "external_comm_activity_flag",
    "jobsite_usb_activity_flag", "suspicious_upload_flag", "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]

# Behavioral columns passed through to the JSONL so the Investigation page can
# render the radar chart, heatmap, cross-channel flags, and raw activity table.
_PASSTHROUGH_COLS: list[str] = [
    "logon_count", "logoff_count", "off_hours_logon",
    "file_open_count", "file_write_count", "file_copy_count",
    "file_delete_count", "unique_files_accessed", "off_hours_files_accessed",
    "usb_insert_count", "usb_remove_count", "off_hours_usb_usage",
    "emails_sent", "unique_recipients", "external_emails_sent",
    "attachments_sent", "off_hours_emails",
    "http_total_requests", "http_visit_count", "http_download_count",
    "http_upload_count", "http_jobsite_visits", "http_cloud_storage_visits",
    "http_suspicious_site_visits", "off_hours_http_requests",
    "http_long_url_count", "unique_domains_visited",
    "pcs_used_count", "non_primary_pc_used_flag",
    "non_primary_pc_http_requests_flag", "non_primary_pc_usb_flag",
    "non_primary_pc_file_copy_flag",
    "usb_file_activity_flag", "off_hours_activity_flag",
    "external_comm_activity_flag", "jobsite_usb_activity_flag",
    "suspicious_upload_flag", "cloud_upload_flag", "non_primary_pc_risk_flag",
]

# ── Global WebSocket client registry ─────────────────────────────────────────
_ws_clients: set = set()
_stop_event: asyncio.Event | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Feature-group derivation
# ─────────────────────────────────────────────────────────────────────────────

# Maps substring patterns (checked against the base feature name, without
# _zscore / _rolling_delta suffix) to a channel group name.
_GROUP_RULES: list[tuple[str, str]] = [
    # Logon channel
    ("logon",               "logon"),
    ("logoff",              "logon"),
    # File channel
    ("file",                "file"),
    ("unique_files",        "file"),
    ("off_hours_files",     "file"),
    # Email channel
    ("emails_sent",         "email"),
    ("external_email",      "email"),
    ("attachments",         "email"),
    ("unique_recipients",   "email"),
    ("off_hours_email",     "email"),
    ("external_comm",       "email"),   # external_comm_activity_flag
    # Removable media / device channel
    ("usb",                 "device"),
    ("jobsite_usb",         "device"),  # jobsite_usb_activity_flag
    ("usb_file",            "device"),  # usb_file_activity_flag
    # HTTP channel
    ("http",                "http"),
    ("unique_domains",      "http"),
    ("cloud_upload",        "http"),
    ("suspicious_upload",   "http"),
    ("off_hours_http",      "http"),
    # PC channel
    ("pc_seen",             "pc"),
    ("new_pc",              "pc"),
    ("pc_prior",            "pc"),
    ("primary_pc",          "pc"),
    ("distinct_pcs",        "pc"),
    ("pcs_used",            "pc"),
    ("non_primary_pc",      "pc"),
    # Broad cross-channel flags that don't fit the above
    ("off_hours_activity",  "logon"),
    ("off_hours_usb",       "device"),
]


def _derive_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Build feature_groups dict for ReconstructionErrorExplainer from a list of
    feature column names.

    Each feature is assigned to a channel group based on substring matching of
    its base name (i.e., after stripping _zscore / _rolling_delta suffixes).
    Features that do not match any rule are collected in an "other" group.
    """
    groups: dict[str, list[str]] = {}

    for col in feature_cols:
        # Derive base name by stripping temporal suffixes
        base = col
        for suffix in ("_rolling_delta", "_zscore"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        group = "other"
        for pattern, grp in _GROUP_RULES:
            if pattern in base:
                group = grp
                break

        groups.setdefault(group, []).append(col)

    # Drop the catch-all "other" group if it is empty
    if not groups.get("other"):
        groups.pop("other", None)

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper (handles numpy scalars and tuples from top_contributors)
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        # Convert tuples to lists so top_contributors serialises as [[name, val], ...]
        if isinstance(obj, tuple):
            return super().encode(list(obj))
        return super().encode(obj)

    def iterencode(self, obj, _one_shot=False):
        # Walk the structure and convert tuples to lists before serialisation
        return super().iterencode(self._convert(obj), _one_shot=_one_shot)

    @staticmethod
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _NumpyEncoder._convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_NumpyEncoder._convert(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Scorer — loads all models once, reuses them for the entire run
# ─────────────────────────────────────────────────────────────────────────────

class LiveScorer:
    def __init__(self) -> None:
        # Add repo root to sys.path so scripts package is importable when this
        # file is launched as a subprocess from a different working directory.
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        # ── Scaler ───────────────────────────────────────────────────────────
        print("[live_simulation] Loading scaler …", flush=True)
        self.scaler = joblib.load(SCALER_PATH)

        # Determine expected feature columns from the fitted scaler when possible
        # so the pipeline adapts automatically to dataset changes.
        if hasattr(self.scaler, "feature_names_in_"):
            self._feature_cols: list[str] = list(self.scaler.feature_names_in_)
            print(
                f"[live_simulation] Feature columns from scaler: {len(self._feature_cols)}",
                flush=True,
            )
        else:
            self._feature_cols = _FALLBACK_FEATURE_COLS
            print(
                f"[live_simulation] Scaler has no feature_names_in_; "
                f"using fallback list ({len(self._feature_cols)} cols)",
                flush=True,
            )

        # ── Full autoencoder (for reconstruction error) ──────────────────────
        print("[live_simulation] Loading autoencoder …", flush=True)
        from tensorflow.keras.models import load_model
        self.autoencoder = load_model(AE_PATH, compile=False)

        # ── Encoder (half-network, for isolation-forest embeddings) ──────────
        print("[live_simulation] Loading encoder …", flush=True)
        self.encoder = load_model(ENCODER_PATH, compile=False)

        # ── Isolation forest ─────────────────────────────────────────────────
        print("[live_simulation] Loading isolation forest …", flush=True)
        from scripts.UEBAIsolationForest import UEBAIsolationForest
        self.iforest = UEBAIsolationForest()
        self.iforest.load(IF_PATH)

        # ── Reference score distributions ────────────────────────────────────
        print("[live_simulation] Loading IF reference score distribution …", flush=True)
        self.ref_scores: np.ndarray = np.load(IF_SCORES_PATH)

        print("[live_simulation] Loading AE baseline distribution …", flush=True)
        _recon_df = pd.read_parquet(LIVE_RECON_TABLE)
        ae_baseline: np.ndarray = (
            _recon_df["total_reconstruction_error"].dropna().values
        )

        # ── ReconstructionErrorExplainer ─────────────────────────────────────
        print("[live_simulation] Initialising reconstruction error explainer …", flush=True)
        from scripts.ReconstructionErrorExplainer import ReconstructionErrorExplainer
        feature_groups = _derive_feature_groups(self._feature_cols)
        self.explainer = ReconstructionErrorExplainer(
            feature_names=self._feature_cols,
            feature_groups=feature_groups,
        )

        # ── AlertObjectBuilder (fitted on training distributions) ─────────────
        print("[live_simulation] Initialising alert builder …", flush=True)
        from scripts.AlertObjectBuilder import AlertObjectBuilder
        self.alert_builder = AlertObjectBuilder(top_k=3)
        self.alert_builder.fit_ae_baseline(ae_baseline)
        self.alert_builder.fit_if_baseline(self.ref_scores)

        print(
            f"[live_simulation] Ready. "
            f"IF ref: {len(self.ref_scores):,} rows  "
            f"AE baseline: {len(ae_baseline):,} rows  "
            f"Features: {len(self._feature_cols)}",
            flush=True,
        )

    # ── Feature selection ─────────────────────────────────────────────────────

    def _select_features(self, row_df: pd.DataFrame) -> np.ndarray:
        """Build a (1, n_features) float32 array aligned to self._feature_cols.

        Columns present in row_df are copied; missing columns are filled with 0.
        Extra columns in row_df that are not in _feature_cols are ignored.
        """
        feat_df = pd.DataFrame(0.0, index=[0], columns=self._feature_cols)
        for col in self._feature_cols:
            if col in row_df.columns:
                feat_df[col] = row_df[col].values
        return feat_df.values.astype("float32")

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_row(self, row_df: pd.DataFrame) -> dict:
        """Run the full AE + IF pipeline on one row and return a complete payload."""
        # Preserve metadata columns
        meta: dict = {}
        for col in ("user", "pc", "day"):
            if col in row_df.columns:
                meta[col] = str(row_df[col].iloc[0])

        for ts_col in ("timestamp", "datetime", "date", "day"):
            if ts_col in row_df.columns:
                meta["cert_timestamp"] = str(row_df[ts_col].iloc[0])
                break

        # Build scaled feature matrix
        raw_features = self._select_features(row_df)
        scaled = self.scaler.transform(raw_features)   # shape (1, n_features)

        t0 = time.perf_counter()

        # ── AE reconstruction error + per-feature contributions ───────────────
        # include_feat_err=False omits raw error_* columns (not needed in payload)
        expl_df = self.explainer.explain_to_df(
            scaled,
            self.autoencoder,
            metadata=None,
            include_feat_err=False,
            include_contributions=True,
        )
        total_recon_error = float(expl_df["total_reconstruction_error"].iloc[0])

        # ── IF anomaly score ──────────────────────────────────────────────────
        embedding = self.encoder.predict(scaled, verbose=0)          # (1, latent_dim)
        raw_if_score = float(self.iforest.anomaly_score(embedding)[0])

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Composite alert via AlertObjectBuilder ────────────────────────────
        # Build a pd.Series containing all fields the builder needs.
        alert_input = {
            "user":                      meta.get("user", ""),
            "day":                       meta.get("day", ""),
            "total_reconstruction_error": total_recon_error,
            "if_anomaly_score":          raw_if_score,
        }
        # Contribution columns (needed for top-K extraction)
        for col in expl_df.columns:
            if col.startswith("contribution_"):
                alert_input[col] = float(expl_df[col].iloc[0])
        # Z-scores and rolling deltas (enrich explanation narrative)
        for col in row_df.columns:
            if col.endswith("_zscore") or col.endswith("_rolling_delta"):
                v = row_df[col].iloc[0]
                if hasattr(v, "item"):
                    v = v.item()
                alert_input[col] = v

        alert_dict = self.alert_builder.build_alert_from_row(pd.Series(alert_input))
        # alert_dict keys: user, day, ae_percentile_rank, ae_risk_band,
        #   top_contributors, if_anomaly_score, if_percentile_rank, if_risk_band,
        #   composite_score, composite_risk_band, both_signals_high, explanation

        # ── Group errors (for radar channel breakdown) ────────────────────────
        group_errors = {
            col: float(expl_df[col].iloc[0])
            for col in expl_df.columns
            if col.startswith("group_error_")
        }

        # ── Assemble final payload ────────────────────────────────────────────
        payload: dict = {
            **meta,
            **alert_dict,
            **group_errors,
            "_score_ms": round(elapsed_ms, 1),
        }

        # Behavioral feature passthrough for radar/heatmap/flags charts
        for _col in _PASSTHROUGH_COLS:
            if _col in row_df.columns:
                _v = row_df[_col].iloc[0]
                if hasattr(_v, "item"):
                    _v = _v.item()
                try:
                    if math.isnan(_v):
                        _v = None
                except TypeError:
                    pass
                payload[_col] = _v

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
        print(
            f"[live_simulation] WS client disconnected ({len(_ws_clients)} remaining)",
            flush=True,
        )


async def _broadcast(payload: dict) -> None:
    if not _ws_clients:
        return
    msg = json.dumps(payload, cls=_NumpyEncoder)
    results = await asyncio.gather(
        *[c.send(msg) for c in list(_ws_clients)],
        return_exceptions=True,
    )
    for client, result in zip(list(_ws_clients), results):
        if isinstance(result, Exception):
            _ws_clients.discard(client)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation producer
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

    if input_path.endswith(".parquet"):
        test_df = pd.read_parquet(input_path)
    else:
        test_df = pd.read_csv(input_path, index_col=0)

    total = len(test_df)
    print(
        f"[live_simulation] Streaming {total:,} rows from {os.path.basename(input_path)}",
        flush=True,
    )

    # Truncate / create fresh output file
    with open(output_path, "w", encoding="utf-8") as f:
        pass

    for idx in range(total):
        if _stop_event.is_set():
            print("[live_simulation] Stop signal received — exiting.", flush=True)
            break

        row_df = test_df.iloc[[idx]]
        payload = scorer.score_row(row_df)
        payload["event_index"] = idx

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, cls=_NumpyEncoder) + "\n")

        await _broadcast(payload)

        print(
            f"[live_simulation] [{idx+1:>5}/{total}]  user={payload.get('user','?')}"
            f"  ae={payload.get('ae_percentile_rank', 0):.1f}pct"
            f"  if={payload.get('if_percentile_rank', 0):.1f}pct"
            f"  composite={payload.get('composite_score', 0):.1f}"
            f"  risk={payload.get('composite_risk_band','?')}"
            f"  ({payload['_score_ms']} ms)",
            flush=True,
        )

        await asyncio.sleep(interval)

        # Honour pause flag: spin-wait until the dashboard removes it or stop is signalled.
        if os.path.exists(PAUSE_FLAG):
            print("[live_simulation] Paused — waiting for resume …", flush=True)
            while os.path.exists(PAUSE_FLAG) and not _stop_event.is_set():
                await asyncio.sleep(0.25)
            if not _stop_event.is_set():
                print("[live_simulation] Resumed.", flush=True)

    # End-of-stream sentinel so the dashboard can detect natural completion
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

    ws_server = await websockets.serve(_ws_handler, "localhost", args.port)
    print(f"[live_simulation] WebSocket listening on ws://localhost:{args.port}", flush=True)

    scorer = LiveScorer()

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
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Seconds between emitted rows (default: 0.5)",
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="Path to the test-stream CSV or Parquet",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Path for the scored JSONL output file",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="WebSocket port (default: 8765)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    asyncio.run(_main(args))
