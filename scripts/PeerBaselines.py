"""
PeerBaselines.py — Department-level rolling peer baseline computation.

Computes 30-day rolling averages for all raw behavioral features, grouped by
department. Output is wide-format parquet used by the dashboard Investigation
tab to compare a user against their departmental peer group.
"""

import os
import numpy as np
import pandas as pd

# Raw behavioral features (mirrors RAW_FEATURES in dashboard/app.py)
BASELINE_FEATURES = [
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
    "pcs_used_count",
]

_DEFAULT_OUTPUT_PATH = os.path.join(
    "processed_datasets", "peer_baselines.parquet"
)


def compute_peer_baselines(
    ueba_df: pd.DataFrame,
    window_days: int = 30,
    min_periods: int = 1,
    output_path: str = _DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Computes rolling department-level peer baselines for all behavioral features.

    For each (department, day), the daily mean across all users in that department
    is computed first, then a rolling window is applied over that daily mean.
    This prevents single-user outliers from distorting the peer average.

    Departments with fewer than 2 users still produce valid output; the rolling
    average equals the single user's own value for those days, which the dashboard
    interprets correctly (peer trace ≈ user trace = no additional signal).

    Schema: [department, day, <feature_1>, ..., <feature_N>]  — wide format.
    Each cell is the 30-day rolling average for that feature in that department.

    Args:
        ueba_df: Layer B (user, day) dataset with a ``department`` column.
        window_days: Rolling look-back window in calendar days.
        min_periods: Minimum observations required to produce a non-NaN value.
        output_path: Destination parquet path.

    Returns:
        pd.DataFrame: Wide-format peer baseline table.
    """
    df = ueba_df.copy()
    df["day"] = pd.to_datetime(df["day"])

    feat_cols = [f for f in BASELINE_FEATURES if f in df.columns]
    if not feat_cols:
        raise ValueError(
            "No BASELINE_FEATURES found in ueba_df. "
            "Ensure Layer B has been built before calling compute_peer_baselines()."
        )

    # Step 1 — collapse to (department, day) mean across users
    dept_daily = (
        df.groupby(["department", "day"])[feat_cols]
        .mean()
        .reset_index()
        .sort_values(["department", "day"])
    )

    # Step 2 — rolling window per department (time-indexed)
    result_parts: list[pd.DataFrame] = []
    for dept, grp in dept_daily.groupby("department", sort=False):
        grp = grp.set_index("day").sort_index()
        rolled = (
            grp[feat_cols]
            .rolling(f"{window_days}D", min_periods=min_periods)
            .mean()
            .reset_index()
        )
        rolled["department"] = dept
        result_parts.append(rolled)

    if not result_parts:
        return pd.DataFrame(columns=["department", "day"] + feat_cols)

    baselines = pd.concat(result_parts, ignore_index=True)

    # Downcast to float32
    for col in feat_cols:
        baselines[col] = baselines[col].astype("float32")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    baselines.to_parquet(output_path, index=False)

    n_depts = baselines["department"].nunique()
    n_days = baselines["day"].nunique()
    print(
        f"Peer baselines saved — {n_depts} departments × {n_days} days "
        f"({len(baselines):,} rows) → {output_path}"
    )

    # Coverage check
    if "department" in ueba_df.columns:
        covered = ueba_df["department"].isin(baselines["department"])
        pct = covered.mean() * 100
        print(f"Coverage: {pct:.1f}% of user-day records have a peer baseline.")

    return baselines