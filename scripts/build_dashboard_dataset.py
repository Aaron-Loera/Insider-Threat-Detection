"""Build the slim *dashboard* serving parquet for Streamlit Cloud (v6).

Why this exists
---------------
The dashboard only renders ~56 of the 414 columns in ``ueba_dataset_6b.parquet``.
Loading the full 414-column matrix (432 MB) plus the alert table and merging them
at runtime peaked at several GB and was OOM-killed on Streamlit Community Cloud's
1 GB tier. This script pre-computes a single, pre-merged, column-projected file
that the dashboard loads directly.

It is the v6 successor to ``build_merged_parquet.py`` (which did the same for v5).

What it does
------------
1. Reads ONLY the dashboard-relevant columns from v6b + alert_table_6
   (engineered model features — z-scores, rolling sums/deltas, peer z-scores,
   sub-day intensity, late-night counters — are intentionally excluded; the
   dashboard never reads their values, only parses their names from the
   per-user ``top_contributors`` text served separately).
2. Downcasts float64->float32 / int64->int32.
3. Dictionary-encodes low-cardinality string columns (user, profile fields,
   risk bands) to category — large RAM + disk win, no data loss.
4. Writes ``processed_datasets/ueba_dataset_6/ueba_dataset_6_dashboard.parquet``
   (zstd). Result: ~25 MB on disk, ~277 MB resident, ~491 MB peak on load.

Usage
-----
    python scripts/build_dashboard_dataset.py            # version from config
    python scripts/build_dashboard_dataset.py --version 6

After building, upload it with ``scripts/upload_dashboard_dataset.py`` and redeploy.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Dashboard serving schema ────────────────────────────────────────────────
# Keep this list in sync with what dashboard/app.py actually renders. Anything
# not listed here is dropped from the serving layer.
KEYS = ["user", "day"]

# Base behavioral channels (raw daily counts) — RAW_FEATURES in app.py
BASE_CHANNELS = [
    "logon_count", "logoff_count", "off_hours_logon",
    "file_open_count", "file_write_count", "file_copy_count",
    "file_delete_count", "unique_files_accessed", "off_hours_files_accessed",
    "usb_insert_count", "usb_remove_count", "off_hours_usb_usage",
    "emails_sent", "unique_recipients", "external_emails_sent",
    "attachments_sent", "off_hours_emails",
    "http_total_requests", "http_visit_count", "http_download_count", "http_upload_count",
    "http_jobsite_visits", "http_cloud_storage_visits", "http_suspicious_site_visits",
    "off_hours_http_requests", "http_long_url_count", "unique_domains_visited",
    "pcs_used_count", "non_primary_pc_used_flag", "non_primary_pc_http_requests_flag",
    "non_primary_pc_usb_flag", "non_primary_pc_file_copy_flag",
]

# Cross-channel risk flags — CROSS_FLAGS in app.py
CROSS_FLAGS = [
    "usb_file_activity_flag", "off_hours_activity_flag", "external_comm_activity_flag",
    "jobsite_usb_activity_flag", "suspicious_upload_flag", "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]

# User-profile enrichment (per-user constants; cheap as category)
PROFILE = [
    "employee_name", "department", "role", "supervisor",
    "functional_unit", "is_active", "role_sensitivity",
]

# Cold-start gate the dashboard re-enforces at load time
GATE = ["baseline_complete"]

# Scores / risk bands — these live in the alert table
SCORE_COLS = [
    "if_anomaly_score", "if_percentile_rank", "if_risk_band",
    "ae_percentile_rank", "ae_risk_band",
    "composite_score", "composite_risk_band",
]

# Columns we want from each source. v6b carries behavioral + profile + gate;
# the alert table contributes the scores/bands. Keys come from both.
V6B_WANT = KEYS + BASE_CHANNELS + CROSS_FLAGS + PROFILE + GATE
ALERT_WANT = KEYS + SCORE_COLS


def _read_cols(path: str, wanted: list[str]) -> pd.DataFrame:
    present = set(pq.read_schema(path).names)
    use = [c for c in wanted if c in present]
    missing = [c for c in wanted if c not in present]
    if missing:
        print(f"  [WARN] not found in {os.path.basename(path)}, skipped: {missing}")
    df = pq.read_table(path, columns=use).to_pandas()
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype("int32")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="dataset version (default: config.MODEL_VERSION)")
    args = ap.parse_args()

    try:
        import config
        mv = args.version or config.MODEL_VERSION
    except ImportError:
        mv = args.version or "6"

    v6b = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{mv}", f"ueba_dataset_{mv}b.parquet")
    alert = os.path.join(BASE_DIR, "explainability", "alert_table", f"alert_table_{mv}", f"alert_table_{mv}.parquet")
    out = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{mv}", f"ueba_dataset_{mv}_dashboard.parquet")

    for p in (v6b, alert):
        if not os.path.exists(p):
            sys.exit(f"[ERROR] required input not found: {p}")

    print(f"Reading v6b columns from {v6b} …")
    ueba = _downcast(_read_cols(v6b, V6B_WANT))
    print(f"  ueba: {ueba.shape}")

    print(f"Reading alert/score columns from {alert} …")
    al = _downcast(_read_cols(alert, ALERT_WANT))
    print(f"  alert: {al.shape}")

    print("Merging (left join on user, day) …")
    alert_only = ["user", "day"] + [c for c in al.columns if c not in ueba.columns]
    merged = ueba.merge(al[alert_only], on=["user", "day"], how="left")
    del ueba, al
    gc.collect()
    _downcast(merged)

    # Dictionary-encode low-cardinality strings
    for c in merged.columns:
        if merged[c].dtype == "object" and merged[c].nunique(dropna=True) < len(merged) * 0.5:
            merged[c] = merged[c].astype("category")

    print(f"  merged: {merged.shape}, {merged.memory_usage(deep=True).sum()/1e6:.0f} MB resident")

    tmp = out + ".tmp"
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, out)
    print(f"Wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, {len(merged.columns)} cols)")
    print("Next: python scripts/upload_dashboard_dataset.py --version", mv)


if __name__ == "__main__":
    main()
