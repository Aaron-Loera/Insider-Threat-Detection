"""
live_replay.py — Cloud-compatible live-feed simulation using pre-scored data.

Replays the test-stream portion of the pre-merged dataset without requiring
tensorflow, joblib, or any ML models.  Produces identical JSONL output to
live_simulation.py so the dashboard sees no difference.

Usage:
    python live_replay.py [--interval 0.5] [--output <path>] [--port 8765]

On Streamlit Cloud: launched as a subprocess by the dashboard when the user
clicks ▶ START LIVE SIMULATION.  Requires only packages already installed:
    huggingface_hub, pandas, pyarrow, (stdlib: os, json, time, argparse)
"""

import argparse
import json
import os
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

DEFAULT_OUTPUT = config.LIVE_OUTPUT
PAUSE_FLAG     = config.LIVE_PAUSE_FLAG
HF_REPO        = "DSKittens/ueba-dashboard-dat"
HF_FILENAME    = "merged_dataset_5.parquet"

# Local copy of the merged parquet (built by scripts/build_merged_parquet.py)
_LOCAL_MERGED = os.path.join(
    BASE_DIR, "explainability", "alert_table", "merged_dataset_5.parquet"
)

# Columns loaded from the merged parquet.  Behavioral feature columns are
# included so the Investigation page's radar chart and heatmap populate
# correctly for live records — without them every feature renders as NaN.
_COLS = [
    "user", "day",
    "if_anomaly_score", "if_percentile_rank", "if_risk_band",
    # Auth
    "logon_count", "logoff_count", "off_hours_logon",
    # File
    "file_open_count", "file_write_count", "file_copy_count",
    "file_delete_count", "unique_files_accessed", "off_hours_files_accessed",
    # Removable media
    "usb_insert_count", "usb_remove_count", "off_hours_usb_usage",
    # Email
    "emails_sent", "unique_recipients", "external_emails_sent",
    "attachments_sent", "off_hours_emails",
    # HTTP
    "http_total_requests", "http_visit_count", "http_download_count", "http_upload_count",
    "http_jobsite_visits", "http_cloud_storage_visits", "http_suspicious_site_visits",
    "off_hours_http_requests", "http_long_url_count", "unique_domains_visited",
    # PC / endpoint
    "pcs_used_count", "non_primary_pc_used_flag", "non_primary_pc_http_requests_flag",
    "non_primary_pc_usb_flag", "non_primary_pc_file_copy_flag",
    # Cross-channel flags
    "usb_file_activity_flag", "off_hours_activity_flag", "external_comm_activity_flag",
    "jobsite_usb_activity_flag", "suspicious_upload_flag", "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_parquet_safe(path: str) -> "pd.DataFrame":
    """Read only the _COLS that actually exist in the parquet.

    The HF-hosted version may have been built before all behavioural columns
    were added.  Requesting a column that doesn't exist raises a KeyError, so
    we filter _COLS against the file schema first.
    """
    import pandas as pd
    import pyarrow.parquet as _pq

    schema_names = set(_pq.read_schema(path).names)
    use_cols = [c for c in _COLS if c in schema_names]
    missing = set(_COLS) - schema_names
    if missing:
        print(f"[live_replay] WARN: {len(missing)} cols not in parquet (will be NaN): "
              f"{sorted(missing)[:6]}{'…' if len(missing) > 6 else ''}", flush=True)
    return pd.read_parquet(path, columns=use_cols)


def _load_records() -> list[dict]:
    """Load the test-stream records (last 10 % of dates) from local or HF parquet."""
    import pandas as pd

    print("[live_replay] Loading pre-scored dataset …", flush=True)

    if os.path.exists(_LOCAL_MERGED):
        df = _read_parquet_safe(_LOCAL_MERGED)
        print("[live_replay] Loaded from local merged parquet.", flush=True)
    else:
        # Try HF hub cache first (dashboard's load_data() will already have downloaded it)
        try:
            from huggingface_hub import hf_hub_download
            import streamlit as _st
            _token = _st.secrets.get("huggingface", {}).get("token", None)
        except Exception:
            _token = os.environ.get("HF_TOKEN", None)

        try:
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(
                repo_id=HF_REPO,
                filename=HF_FILENAME,
                repo_type="dataset",
                token=_token,
            )
            df = _read_parquet_safe(cached)
            print(f"[live_replay] Loaded from HF cache: {cached}", flush=True)
        except Exception as e:
            print(f"[live_replay] ERROR: could not load dataset — {e}", flush=True)
            sys.exit(1)

    df["day"] = pd.to_datetime(df["day"], errors="coerce")

    # Test-stream = last 10 % of unique days (matching the chronological split used at train time)
    unique_days = sorted(df["day"].dropna().unique())
    cutoff_idx  = int(len(unique_days) * 0.90)
    test_days   = set(unique_days[cutoff_idx:])
    df = df[df["day"].isin(test_days)].sort_values("day").reset_index(drop=True)

    # The merged parquet only contains training data, so the "test" days above
    # are actually within the training date range and are already present in the
    # dashboard's user_data.  The Investigation fragment deduplicates by date,
    # which means every replayed record would be silently dropped.
    #
    # Fix: shift all replay dates to start the day after the last training day.
    # This makes live records appear as genuinely new dates the dashboard has
    # never seen, so the merge logic appends them correctly.
    last_train_day = df["day"].max()
    min_replay_day = df["day"].min()
    _date_offset = last_train_day - min_replay_day + pd.Timedelta(days=1)
    df["day"] = df["day"] + _date_offset

    # Behavioural feature columns present in the parquet (subset of _COLS)
    _FEAT_COLS = [c for c in _COLS if c not in ("user", "day", "if_anomaly_score", "if_percentile_rank", "if_risk_band")]

    records = []
    for idx, row in df.iterrows():
        rec = {
            "user":               str(row["user"]),
            "day":                str(row["day"].date()),
            "cert_timestamp":     str(row["day"].date()),
            "if_anomaly_score":   round(float(row["if_anomaly_score"]), 6),
            "if_percentile_rank": round(float(row["if_percentile_rank"]), 2),
            "if_risk_band":       str(row["if_risk_band"]),
            "_score_ms":          0.0,
            "event_index":        int(idx),
        }
        # Pass through all behavioural feature columns so the Investigation
        # page's radar chart, heatmap, and cross-channel flags render correctly.
        import math as _math
        for col in _FEAT_COLS:
            if col in row.index:
                val = row[col]
                if isinstance(val, float) and _math.isnan(val):
                    rec[col] = None
                else:
                    rec[col] = val.item() if hasattr(val, "item") else val
        records.append(rec)

    print(f"[live_replay] {len(records):,} records to replay "
          f"({len(test_days)} test days).", flush=True)
    return records


def _is_paused() -> bool:
    return os.path.exists(PAUSE_FLAG)


# ── Main replay loop ──────────────────────────────────────────────────────────

def run(interval: float, output: str) -> None:
    records = _load_records()

    # Clear any previous output
    if os.path.exists(output):
        os.remove(output)

    print(f"[live_replay] Writing to {output} at {interval}s per record …", flush=True)

    with open(output, "a", encoding="utf-8") as fh:
        for rec in records:
            # Honour pause flag
            while _is_paused():
                time.sleep(0.25)

            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            # Progress every 100 records
            if rec["event_index"] % 100 == 0:
                print(
                    f"[live_replay] {rec['event_index']:,} / {len(records):,} "
                    f"— {rec['user']} {rec['day']} [{rec['if_risk_band']}]",
                    flush=True,
                )

            time.sleep(interval)

        # End-of-stream sentinel so the dashboard knows the feed is complete
        fh.write(json.dumps({"_eos": True}) + "\n")
        fh.flush()

    print("[live_replay] Stream complete.", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay pre-scored live-feed data.")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between records (default: 0.5)")
    parser.add_argument("--output",   type=str,   default=DEFAULT_OUTPUT,
                        help="Path to JSONL output file (default: live_results.jsonl)")
    # Accept --port for interface compatibility with live_simulation.py (ignored here)
    parser.add_argument("--port", type=int, default=8765, help="(ignored)")
    args = parser.parse_args()

    run(interval=args.interval, output=args.output)
