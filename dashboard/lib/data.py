"""Cached data loaders and feature-group derivation for the dashboard.

Every loader is a Streamlit cache primitive, so app.py and the lib aggregation
helpers can call them freely and get the same in-process objects back. load_data()
loads the slim serving parquet (config.DASHBOARD_PARQUET, honouring UEBA_BASE_DIR)
and pre-computes the per-user risk summary; get_feature_groups() returns the
RAW_FEATURES / CROSS_FLAGS / CHANNELS lists filtered to the columns actually
present in the loaded frame.

Paths come from config (the repo-root shim re-exporting ueba.config) and the
disposition DB API from db — imported the same way app.py imports them.
"""

import os

import pandas as pd
import streamlit as st
from db import upsert_disposition

from config import (
    ANALYST_TABLE_PARQUET,
    PEER_BASELINES_PATH,
    UEBA_A_CSV,
    UEBA_A_PARQUET,
)

ALERT_STATUS_OPTIONS = ["NEW", "INVESTIGATING", "RESOLVED", "DISMISSED"]


def _on_status_change(user, day, key):
    upsert_disposition(user, day, st.session_state[key])


# ── Feature-group definitions (canonical order; filtered to present columns) ──
_RAW_FEATURES = [
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
]

_CROSS_FLAGS = [
    "usb_file_activity_flag", "off_hours_activity_flag", "external_comm_activity_flag",
    "jobsite_usb_activity_flag", "suspicious_upload_flag", "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]

# Channel groupings for radar / breakdown charts.
_CHANNELS = {
    "Authentication": ["logon_count", "logoff_count", "off_hours_logon"],
    "File Access":    ["file_open_count", "file_write_count", "file_copy_count",
                       "file_delete_count", "unique_files_accessed", "off_hours_files_accessed"],
    "Removable Media": ["usb_insert_count", "usb_remove_count", "off_hours_usb_usage"],
    "Email":          ["emails_sent", "unique_recipients", "external_emails_sent",
                       "attachments_sent", "off_hours_emails"],
    "HTTP Activity":  ["http_total_requests", "http_visit_count", "http_download_count",
                       "http_upload_count", "http_jobsite_visits", "http_cloud_storage_visits",
                       "http_suspicious_site_visits", "off_hours_http_requests",
                       "http_long_url_count", "unique_domains_visited"],
    "PC Activity":    ["pcs_used_count", "non_primary_pc_used_flag",
                       "non_primary_pc_http_requests_flag", "non_primary_pc_usb_flag",
                       "non_primary_pc_file_copy_flag"],
}


@st.cache_data(show_spinner=False)
def _load_user_detail_df(user: str) -> "pd.DataFrame":
    """Download ALL detail rows for one user and cache the result.

    Local:  reads from the main analyst parquet with a user-level DNF filter
            (1268 sorted row groups → only the user's ~1270 rows are read).
    Cloud:  downloads a tiny per-user parquet from HF details/ folder (~46 KB).
            The full 193 MB analyst parquet is NEVER loaded on cloud — each
            user file is independently tiny, making the download cost negligible.
    """
    _DETAIL_COLS = ["user", "day", "top_contributors", "explanation"]
    safe = str(user).replace("/", "_").replace("\\", "_")

    try:
        if os.path.exists(ANALYST_TABLE_PARQUET):
            import pyarrow.parquet as pq
            tbl = pq.read_table(
                ANALYST_TABLE_PARQUET,
                columns=_DETAIL_COLS,
                filters=[[("user", "=", user)]],
            )
            return tbl.to_pandas()
        else:
            # Per-user file: ~46 KB download regardless of total dataset size
            from ueba.serving.hf_io import get_dataset_file
            path = get_dataset_file(f"details/{safe}.parquet")
            return pd.read_parquet(path)
    except Exception as _e:
        import logging as _logging
        _logging.getLogger(__name__).warning("_load_user_detail_df(%s) failed: %s", user, _e)
        return pd.DataFrame(columns=_DETAIL_COLS)


@st.cache_data(show_spinner=False)
def fetch_alert_detail(user: str, day_str: str) -> dict:
    """Return top_contributors and explanation for one (user, day) record.

    Delegates the expensive I/O to _load_user_detail_df (cached per-user),
    then filters to the requested day. On cloud this costs ~46 KB per unique
    user, cached after the first lookup — compared to 193 MB per call before.
    """
    try:
        df = _load_user_detail_df(user)
        if df.empty:
            return {}
        day_ts = pd.Timestamp(day_str)
        if "day" in df.columns:
            df["day"] = pd.to_datetime(df["day"], errors="coerce")
            match = df[df["day"] == day_ts]
        else:
            return {}
        if match.empty:
            return {}
        r = match.iloc[0]
        return {
            "top_contributors": r.get("top_contributors", None),
            "explanation": r.get("explanation", "") or "",
        }
    except Exception as _e:
        import logging as _logging
        _logging.getLogger(__name__).warning("fetch_alert_detail(%s, %s) failed: %s", user, day_str, _e)
        return {}


def get_alert_detail(user, day, key):
    """Fetch explanation or top_contributors for a single (user, day) record."""
    day_str = str(day.date()) if hasattr(day, "date") else str(day)
    return fetch_alert_detail(str(user), day_str).get(key, None)


@st.cache_resource(show_spinner=False)
def load_ueba_a():
    """Load UEBA Table A — (user, pc, day) granular rows used for PC-level drill-down."""
    if os.path.exists(UEBA_A_PARQUET):
        df = pd.read_parquet(UEBA_A_PARQUET)
    elif os.path.exists(UEBA_A_CSV):
        df = pd.read_csv(UEBA_A_CSV)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
    else:
        return None
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_peer_baselines():
    """Load peer baseline parquet. Returns None if file not yet generated."""
    if not os.path.exists(PEER_BASELINES_PATH):
        return None
    df = pd.read_parquet(PEER_BASELINES_PATH)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return df


@st.cache_resource(show_spinner="Loading dataset...")
def load_data():
    """Merge alert_table_6 and ueba_dataset_6b, then pre-compute user_risk.

    Uses @st.cache_resource so the DataFrame is stored by reference — no pickle
    serialisation overhead on repeated Streamlit reruns.
    """
    import gc
    import logging as _logging

    import pyarrow.parquet as _pq

    from config import MODEL_VERSION
    _log = _logging.getLogger("ueba.load_data")
    _log.warning("[load_data] started")

    _MV = MODEL_VERSION

    # ── Serving layer ────────────────────────────────────────────────────────
    # Load ONE pre-merged "dashboard" parquet that contains only the columns the
    # UI actually renders (~76 cols), dictionary-encoded for low-cardinality
    # strings and stored as float32. This replaces the previous runtime path of
    # downloading the full 414-column v6b (432 MB) plus the alert table and
    # merging them in-process, which peaked at several GB and was OOM-killed on
    # Streamlit Community Cloud's 1 GB tier. The slim file peaks at ~0.65 GB.
    # Rebuild it with scripts/build_dashboard_dataset.py whenever v6b / the alert
    # table changes, then re-upload to the HF dataset repo.
    #
    # The local path comes from config.DASHBOARD_PARQUET (not __file__) so it
    # honours UEBA_BASE_DIR — identical resolution under normal operation, but it
    # lets the AppTest smoke harness point the app at a synthetic dataset tree.
    import config as _config
    _DASH_LOCAL_PATH = _config.DASHBOARD_PARQUET
    from ueba.serving.hf_io import get_dataset_file
    if not os.path.exists(_DASH_LOCAL_PATH):
        _log.warning("[load_data] downloading dashboard parquet from HuggingFace…")
    _DASH_PATH = get_dataset_file(
        f"ueba_dataset_{_MV}_dashboard.parquet",
        version=_MV,
        local_path=_DASH_LOCAL_PATH,
    )

    def _downcast(df):
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype("int32")
        return df

    # Memory-frugal Arrow→pandas conversion: split_blocks + self_destruct frees
    # each Arrow column as soon as it is converted, roughly halving peak RSS.
    _log.warning("[load_data] loading slim dashboard dataset")
    _tbl = _pq.read_table(_DASH_PATH)
    merged = _tbl.to_pandas(split_blocks=True, self_destruct=True)
    del _tbl
    gc.collect()
    merged["day"] = pd.to_datetime(merged["day"], errors="coerce")
    _downcast(merged)  # no-op if file already float32; protective if rebuilt otherwise
    for _col in ("user", "ae_risk_band", "if_risk_band"):
        if _col in merged.columns:
            merged[_col] = merged[_col].astype("category")
    gc.collect()
    _log.warning(f"[load_data] loaded: {merged.shape}, {merged.memory_usage(deep=True).sum()/1e6:.1f} MB")

    # Cast risk bands to ordered categorical
    _risk_cat = pd.CategoricalDtype(categories=["LOW", "MEDIUM", "HIGH", "CRITICAL"], ordered=True)
    for _col in ("ae_risk_band", "if_risk_band"):
        if _col in merged.columns:
            merged[_col] = merged[_col].astype(_risk_cat)

    # Defense in depth: enforce the v6 baseline_complete gate at read time.
    # Upstream Alert_Object_Builder already filters baseline_complete=False rows before
    # risk banding, but re-applying here protects against alert_table regenerations that
    # skipped the filter (cold-start users would otherwise surface as false CRITICAL).
    if "baseline_complete" in merged.columns:
        _ungated = ~merged["baseline_complete"].fillna(False).astype(bool)
        for _band_col in ("ae_risk_band", "if_risk_band", "composite_risk_band"):
            if _band_col in merged.columns:
                _crit_mask = _ungated & (merged[_band_col] == "CRITICAL")
                if _crit_mask.any():
                    merged.loc[_crit_mask, _band_col] = "HIGH"
                    _log.warning(
                        f"[load_data] baseline_complete gate demoted "
                        f"{int(_crit_mask.sum())} CRITICAL→HIGH in {_band_col}"
                    )

    # ── Pre-compute per-user risk summary ──
    user_risk = (
        merged.groupby("user", observed=True)
        .agg(
            max_score=("if_anomaly_score", "max"),
            mean_score=("if_anomaly_score", "mean"),
            max_percentile=("ae_percentile_rank", "max"),
            alert_days=("day", "nunique"),
            critical_count=("ae_risk_band", lambda x: (x == "CRITICAL").sum()),
            high_count=("ae_risk_band", lambda x: (x == "HIGH").sum()),
            medium_count=("ae_risk_band", lambda x: (x == "MEDIUM").sum()),
        )
        .reset_index()
        .sort_values("max_percentile", ascending=False)
    )

    _all_users = sorted(merged["user"].unique().tolist())
    _ds_min = merged["day"].min().date()
    _ds_max = merged["day"].max().date()

    _log.warning("[load_data] complete")
    return merged, user_risk, _all_users, _ds_min, _ds_max


_USER_PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_profiles.parquet")


@st.cache_data(show_spinner=False)
def load_user_profiles() -> pd.DataFrame:
    """Load pre-built user profile table (department, role, supervisor, role_sensitivity)."""
    if os.path.exists(_USER_PROFILES_PATH):
        return pd.read_parquet(_USER_PROFILES_PATH)
    return pd.DataFrame(columns=["user", "department", "role", "supervisor", "role_sensitivity"])


@st.cache_resource(show_spinner=False)
def get_feature_groups() -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Return (RAW_FEATURES, CROSS_FLAGS, CHANNELS) filtered to loaded columns.

    Replaces the module-level derivation that filtered the canonical lists against
    merged_df.columns. Cached as a resource (computed once per process) and calls
    load_data() internally for the column set — the same cached frame.
    """
    merged, *_ = load_data()
    cols = set(merged.columns)
    raw_features = [f for f in _RAW_FEATURES if f in cols]
    cross_flags = [f for f in _CROSS_FLAGS if f in cols]
    channels = {k: [f for f in v if f in cols] for k, v in _CHANNELS.items()}
    channels = {k: v for k, v in channels.items() if v}
    return raw_features, cross_flags, channels
