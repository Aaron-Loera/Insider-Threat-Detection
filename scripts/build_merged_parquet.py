"""Build a single pre-merged parquet for Streamlit Cloud deployment.

Merges the analyst risk-score table with the UEBA behavioral features into one
file, excluding the heavyweight string columns (explanation, top_contributors)
that live in the separate details/{user}.parquet files.

Output: explainability/alert_table/merged_dataset_5.parquet

Peak RAM on cloud when loading this single file is ~250 MB, compared to
~700-800 MB when the old approach loaded two files and merged them at runtime.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ──────────────────────────────────────────────────────────────────────
ANALYST_PARQUET = os.path.join(
    BASE_DIR, "explainability", "alert_table", "alert_table_5.parquet"
)
UEBA_PARQUET = os.path.join(
    BASE_DIR, "processed_datasets", "ueba_dataset_5", "ueba_dataset_5_train.parquet"
)
OUT_PATH = os.path.join(
    BASE_DIR, "explainability", "alert_table", "merged_dataset_5.parquet"
)

# ── Columns ────────────────────────────────────────────────────────────────────
ANALYST_COLS = [
    "user", "day",
    "if_anomaly_score", "ae_percentile_rank", "ae_risk_band",
    "if_percentile_rank", "if_risk_band",
]

UEBA_COLS = [
    "user", "pc", "day",
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
    "usb_file_activity_flag", "off_hours_activity_flag", "external_comm_activity_flag",
    "jobsite_usb_activity_flag", "suspicious_upload_flag", "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]


def load_parquet_cols(path, wanted_cols):
    schema = pq.read_schema(path)
    present = set(schema.names)
    use = [c for c in wanted_cols if c in present]
    missing = set(wanted_cols) - present
    if missing:
        print(f"  [WARN] columns not found and will be skipped: {sorted(missing)}")
    return pd.read_parquet(path, columns=use)


def downcast(df):
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    return df


def main():
    print("Loading analyst table …")
    analyst = load_parquet_cols(ANALYST_PARQUET, ANALYST_COLS)
    downcast(analyst)
    analyst["day"] = pd.to_datetime(analyst["day"], errors="coerce")
    print(f"  analyst: {analyst.shape}, {analyst.memory_usage(deep=True).sum()/1e6:.1f} MB")

    print("Loading UEBA dataset …")
    ueba = load_parquet_cols(UEBA_PARQUET, UEBA_COLS)
    downcast(ueba)
    ueba["day"] = pd.to_datetime(ueba["day"], errors="coerce")
    print(f"  ueba: {ueba.shape}, {ueba.memory_usage(deep=True).sum()/1e6:.1f} MB")

    print("Merging …")
    analyst_extra = [c for c in ANALYST_COLS if c not in ("user", "day")]
    merged = ueba.merge(
        analyst[["user", "day"] + analyst_extra],
        on=["user", "day"],
        how="left",
    )
    import gc; gc.collect()
    print(f"  merged: {merged.shape}, {merged.memory_usage(deep=True).sum()/1e6:.1f} MB")

    print(f"Writing → {OUT_PATH}")
    merged.to_parquet(
        OUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
        row_group_size=1000,  # same granularity as alert_table_5 for predicate pushdown
    )
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Done. File size: {size_mb:.1f} MB")
    print(f"Columns ({len(merged.columns)}): {list(merged.columns)}")


if __name__ == "__main__":
    main()
