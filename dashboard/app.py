import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import subprocess
import sys
import time
import ast as _ast

# ──────────────────────────────────────────────────────────────
# Page Config & Custom CSS
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UEBA — Insider Threat Detection",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #d4d4d4;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container { padding-top: 0rem; padding-bottom: 1rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #1a1a1a;
    }
    [data-testid="stSidebar"] * { border-radius: 0 !important; }

    /* On desktop: always force sidebar visible regardless of Streamlit's stored state */
    @media (min-width: 769px) {
        [data-testid="stSidebar"] {
            transform: none !important;
            margin-left: 0 !important;
            visibility: visible !important;
            width: 280px !important;
            min-width: 280px !important;
        }
        /* Collapse button stays hidden on desktop */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"] {
            display: none !important;
        }
    }

    /* On mobile: both buttons visible for toggle */
    @media (max-width: 768px) {
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"] {
            display: none !important; /* hidden — our custom hamburger handles this */
        }
    }
    /* Zero out Streamlit's injected top padding on sidebar wrappers */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    /* Logo branding block: negative margin pulls it past any remaining offset */
    .sidebar-branding {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 14px;
        border-bottom: 1px solid #1a1a1a;
        margin-top: -4.5rem !important;
        padding: 14px 16px;
        gap: 10px;
    }

    [data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        display: flex; flex-direction: column; gap: 0;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        padding: 12px 16px !important;
        margin: 0 !important;
        border-left: 3px solid transparent;
        color: #666 !important;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        color: #ccc !important;
        background: #111 !important;
        border-left-color: #333 !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        color: #fff !important;
        border-left-color: #e84545 !important;
        background: #111 !important;
    }

    /* ── Sidebar section labels ── */
    .sidebar-section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 16px 16px 6px 16px;
        margin: 0;
    }

    /* ── Page Title ── */
    .page-header-block {
        border-bottom: 1px solid #1a1a1a;
        padding-bottom: 14px;
        margin-bottom: 24px;
    }
    .page-title {
        font-family: 'Inter', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.1;
    }
    .page-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: #555;
        margin: 3px 0 0 0 !important;
        padding: 0 !important;
    }

    /* ── Project title badge (top-right, scrolls with page) ── */
    .project-title-badge {
        display: block;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 8px;
    }

    /* ── Critical Alert Notice Banner ── */
    @keyframes alertGlow {
        0%   {
            box-shadow:
                0 0  4px #e8454500,
                0 0 12px #e8454500,
                0 0 30px #e8454500,
                inset 0 0  0px #e8454500;
            border-color: #e8454588;
            background: #0d0000;
        }
        50%  {
            box-shadow:
                0 0  8px #e84545cc,
                0 0 24px #e8454599,
                0 0 55px #e8454544,
                inset 0 0 18px #e8454518;
            border-color: #e84545ff;
            background: #180000;
        }
        100% {
            box-shadow:
                0 0  4px #e8454500,
                0 0 12px #e8454500,
                0 0 30px #e8454500,
                inset 0 0  0px #e8454500;
            border-color: #e8454588;
            background: #0d0000;
        }
    }
    @keyframes alertTitleGlow {
        0%   { text-shadow: 0 0 0px #e8454500; color: #e84545cc; }
        50%  { text-shadow: 0 0 10px #e84545cc, 0 0 22px #e8454566; color: #ff6b6b; }
        100% { text-shadow: 0 0 0px #e8454500; color: #e84545cc; }
    }
    @keyframes alertIconPulse {
        0%   { text-shadow: none; opacity: 0.6; }
        50%  { text-shadow: 0 0 8px #e84545ff, 0 0 16px #e8454599; opacity: 1; }
        100% { text-shadow: none; opacity: 0.6; }
    }
    .alert-notice-banner {
        background: #0d0000;
        border: 1px solid #e8454588;
        border-left: 4px solid #e84545;
        border-radius: 0;
        padding: 16px 20px;
        margin: 12px 0 16px 0;
        animation: alertGlow 2.2s ease-in-out infinite;
        transition: background 0.1s;
    }
    .alert-notice-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 10px;
    }
    .alert-notice-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #e84545cc;
        display: flex;
        align-items: center;
        gap: 8px;
        animation: alertTitleGlow 2.2s ease-in-out infinite;
    }
    .alert-notice-title::before {
        content: '▲';
        font-size: 11px;
        animation: alertIconPulse 1.1s ease-in-out infinite;
    }
    .alert-notice-count {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #999;
        letter-spacing: 1px;
    }
    .alert-notice-rows {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 4px;
    }
    .alert-notice-row-item {
        background: #160000;
        border: 1px solid #3a0000;
        padding: 6px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #cccccc;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .alert-notice-row-item .u-id  { color: #ffffff; font-weight: 600; }
    .alert-notice-row-item .u-pct { color: #e84545; }
    .alert-notice-row-item .u-days { color: #888; font-size: 11px; }

    /* ── KPI Cards ── */
    .kpi-scroll-wrapper {
        position: relative;
        display: flex;
        align-items: center;
        gap: 0;
        margin-bottom: 16px;
    }
    .kpi-scroll-arrow {
        flex-shrink: 0;
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        color: #888;
        font-size: 16px;
        width: 32px;
        height: 100%;
        min-height: 90px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
        transition: color 0.15s, border-color 0.15s;
    }
    .kpi-scroll-arrow:hover { color: #fff; border-color: #444; }
    .kpi-scroll-row {
        display: flex;
        flex-direction: row;
        gap: 12px;
        overflow-x: auto;
        padding-bottom: 4px;
        scroll-behavior: smooth;
        scrollbar-width: none;
    }
    .kpi-scroll-row::-webkit-scrollbar { display: none; }
    .kpi-card {
        background: #0a0a0a;
        border-radius: 0;
        padding: 20px 24px;
        border-left: 3px solid;
        border-top: 1px solid #1a1a1a;
        min-width: 180px;
        flex-shrink: 0;
        border-right: 1px solid #1a1a1a;
        border-bottom: 1px solid #1a1a1a;
        min-height: 120px;
        box-sizing: border-box;
    }
    .kpi-card h3 {
        margin: 0; font-size: 12px; color: #666666; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1.5px;
        font-family: 'JetBrains Mono', monospace;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .kpi-card h1 {
        margin: 8px 0 0 0; font-size: 28px; font-weight: 600;
        font-family: 'JetBrains Mono', monospace; letter-spacing: -0.5px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        font-size: clamp(16px, 2.2vw, 28px);
    }
    .kpi-card p {
        margin: 6px 0 0 0; font-size: 12px; color: #555555;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase; letter-spacing: 0.5px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #999999;
        margin: 32px 0 8px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #1a1a1a;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Risk badges ── */
    .risk-high   { color: #e84545; font-weight: 600; }
    .risk-medium { color: #d4a017; font-weight: 600; }
    .risk-low    { color: #3a86a8; font-weight: 600; }

    /* ── Metric tweaks ── */
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: #666666 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
    }

    /* ── Inputs & Widgets ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        border-radius: 0 !important;
        border-color: #1a1a1a !important;
        background-color: #0a0a0a !important;
    }
    .stSlider > div > div > div { border-radius: 0 !important; }
    button[kind="primary"], .stDownloadButton > button {
        border-radius: 0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        font-size: 11px;
    }

    /* ── DataFrame ── */
    .stDataFrame { border-radius: 0 !important; }
    [data-testid="stDataFrame"] > div { border-radius: 0 !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0 !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stHeader"] {
        background-color: #000000 !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    /* ── Remove ALL border-radius globally ── */
    div[data-testid], .element-container,
    /* buttons */
    button, .stButton > button, [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"], [data-testid="baseButton-minimal"],
    /* inputs & textareas */
    input, textarea, select,
    [data-testid="stTextInput"] > div, [data-testid="stTextInput"] input,
    [data-testid="stDateInput"] > div, [data-testid="stDateInput"] input,
    [data-testid="stNumberInput"] > div, [data-testid="stNumberInput"] input,
    /* multiselect */
    [data-testid="stMultiSelect"] > div,
    [data-testid="stMultiSelect"] span[data-baseweb="tag"],
    [data-baseweb="tag"], [data-baseweb="input"], [data-baseweb="select"],
    [data-baseweb="popover"], [data-baseweb="menu"],
    /* modal / dialog */
    [data-testid="stModal"], [data-testid="stModal"] > div,
    [data-testid="stModalContent"], [role="dialog"],
    [role="dialog"] > div, [role="dialog"] section,
    /* misc */
    .stAlert, [data-testid="stNotificationContentSuccess"],
    [data-testid="stNotificationContentError"],
    [data-testid="stExpander"], [data-testid="stExpanderDetails"],
    [data-baseweb="card"], [data-baseweb="notification"] {
        border-radius: 0 !important;
    }

    /* ── Responsive: Large screens (>1400px) ── */
    @media (min-width: 1401px) {
        .kpi-card h1 { font-size: 28px; }
        .kpi-card h3 { font-size: 12px; }
        .kpi-card p  { font-size: 12px; }
    }

    /* ── Responsive: Medium screens (1024–1400px) ── */
    @media (max-width: 1400px) {
        .kpi-card { padding: 16px 18px; min-height: 110px; }
        .kpi-card h1 { font-size: 22px; }
        .kpi-card h3 { font-size: 11px; letter-spacing: 1px; }
        .kpi-card p  { font-size: 11px; }
        .section-header { font-size: 14px; letter-spacing: 1.5px; }
    }

    /* ── Responsive: Small screens (768–1023px) ── */
    @media (max-width: 1023px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .kpi-card { padding: 14px 14px; min-height: 100px; }
        .kpi-card h1 { font-size: 20px; }
        .kpi-card h3 { font-size: 10px; letter-spacing: 0.8px; }
        .kpi-card p  { font-size: 10px; }
        .section-header { font-size: 13px; margin: 20px 0 10px 0; }
        [data-testid="stMetricValue"] { font-size: 22px; }
        [data-testid="stMetricLabel"] { font-size: 11px !important; }
    }

    /* ── Responsive: Extra small (<768px) ── */
    @media (max-width: 767px) {
        .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
        .kpi-card { padding: 12px 12px; min-height: 90px; border-left-width: 2px; }
        .kpi-card h1 { font-size: 18px; }
        .kpi-card h3 { font-size: 9px; }
        .kpi-card p  { font-size: 9px; }
        .section-header { font-size: 12px; letter-spacing: 1px; }
        [data-testid="stMetricValue"] { font-size: 20px; }
        [data-testid="stMetricLabel"] { font-size: 10px !important; }
    }

    /* ── Info popover button ── */
    [data-testid="stPopover"] button,
    [data-testid="stBasePopoverButton"] {
        background-color: transparent !important;
        border: none !important;
        color: #444 !important;
        font-size: 14px !important;
        padding: 0 4px !important;
        min-height: unset !important;
        line-height: 1 !important;
        white-space: nowrap !important;
    }
    [data-testid="stPopover"] button:hover,
    [data-testid="stBasePopoverButton"]:hover {
        color: #4a9eff !important;
        background-color: transparent !important;
        border: none !important;
    }

    /* ── Investigate list buttons ── */
    [data-testid="stButton"] button {
        background-color: #0e0e0e !important;
        border: 1px solid #2a2a2a !important;
        color: #888 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.5px !important;
        padding: 4px 8px !important;
        white-space: nowrap !important;
        transition: border-color 0.15s, color 0.15s !important;
    }
    [data-testid="stButton"] button:hover {
        border-color: #4a9eff !important;
        color: #4a9eff !important;
        background-color: #0e1a2e !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────

# Resolve paths relative to this script so any team member can run it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Local path overrides ──────────────────────────────────────────────────────
# Contributors can create paths.local.py at the project root to point to their
# own data locations without touching this file.  See paths.local.example.py.
import importlib.util as _ilu
_local = None
_local_path = os.path.join(BASE_DIR, "paths.local.py")
if os.path.exists(_local_path):
    _spec = _ilu.spec_from_file_location("paths_local", _local_path)
    _local = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_local)

def _local_path_or(attr: str, default: str) -> str:
    """Return override from paths.local.py if set, otherwise the default."""
    val = getattr(_local, attr, None) if _local is not None else None
    return val if val else default

# Prefer Parquet (5-10x faster I/O); fall back to CSV
_analyst_override = _local_path_or("ANALYST_TABLE", "")
if _analyst_override:
    ANALYST_TABLE_PARQUET = _analyst_override if _analyst_override.endswith(".parquet") else ""
    ANALYST_TABLE_CSV     = _analyst_override if _analyst_override.endswith(".csv")     else _analyst_override
else:
    ANALYST_TABLE_PARQUET = os.path.join(BASE_DIR, "explainability", "alert_table", "alert_table_4.parquet")
    ANALYST_TABLE_CSV     = os.path.join(BASE_DIR, "explainability", "alert_table", "alert_table_4.csv")

_ueba_override = _local_path_or("UEBA_DATASET", "")
if _ueba_override:
    UEBA_PARQUET = _ueba_override if _ueba_override.endswith(".parquet") else ""
    UEBA_CSV     = _ueba_override if _ueba_override.endswith(".csv")     else _ueba_override
else:
    UEBA_PARQUET = os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset_4", "ueba_dataset_4_train.parquet")
    UEBA_CSV     = os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset_4", "ueba_dataset_4_train.csv")
LIVE_OUTPUT     = os.path.join(BASE_DIR, "processed_datasets", "live_results.jsonl")
LIVE_PAUSE_FLAG = os.path.join(BASE_DIR, "processed_datasets", "live_pause.flag")
LIVE_SIM_SCRIPT = os.path.join(BASE_DIR, "live_simulation.py")

# Only load columns the dashboard actually uses
UEBA_COLS = [
    "user", "pc", "day",
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


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    """Load analyst table + UEBA dataset, merge, pre-compute user_risk."""
    import gc

    # ── Load analyst table ──
    if os.path.exists(ANALYST_TABLE_PARQUET):
        analyst = pd.read_parquet(ANALYST_TABLE_PARQUET)
    else:
        analyst = pd.read_csv(ANALYST_TABLE_CSV)

    # ── Load UEBA dataset ──
    if os.path.exists(UEBA_PARQUET):
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(UEBA_PARQUET).names
        use = [c for c in UEBA_COLS if c in schema_cols]
        ueba = pd.read_parquet(UEBA_PARQUET, columns=use)
    else:
        all_cols = pd.read_csv(UEBA_CSV, nrows=0).columns.tolist()
        use_idx = [i for i, c in enumerate(all_cols) if c in UEBA_COLS]
        ueba = pd.read_csv(UEBA_CSV, usecols=use_idx)

    # Downcast numeric columns to reduce memory
    for df in [analyst, ueba]:
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    # Normalize day column
    for df in [analyst, ueba]:
        if "day" in df.columns:
            df["day"] = pd.to_datetime(df["day"], errors="coerce")

    # Merge — include alert-context columns from the alert table
    analyst_cols = [
        "user", "day", "if_anomaly_score", "ae_percentile_rank", "ae_risk_band",
        "top_contributors", "if_percentile_rank", "if_risk_band", "explanation",
    ]
    analyst_cols = [c for c in analyst_cols if c in analyst.columns]
    merged = ueba.merge(analyst[analyst_cols], on=["user", "day"], how="left")

    # Pre-compute per-user risk summary (expensive groupby, do once)
    user_risk = (
        merged.groupby("user")
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

    # Cast risk band to ordered categorical — makes .isin() and groupby ~3x faster
    _risk_cat = pd.CategoricalDtype(
        categories=["LOW", "MEDIUM", "HIGH", "CRITICAL"], ordered=True
    )
    for _col in ("ae_risk_band", "if_risk_band"):
        if _col in merged.columns:
            merged[_col] = merged[_col].astype(_risk_cat)

    # Pre-group by user so the Investigation tab can do O(1) lookups instead
    # of scanning the full 1.5M-row frame on every user switch
    user_data_dict: dict[str, pd.DataFrame] = {
        u: grp.reset_index(drop=True)
        for u, grp in merged.groupby("user", observed=True)
    }

    del ueba, analyst
    gc.collect()

    return merged, user_risk, user_data_dict


try:
    merged_df, user_risk, user_data_dict = load_data()
    DATA_LOADED = True
except Exception:
    DATA_LOADED = False


# ──────────────────────────────────────────────────────────────
# If data hasn't been generated yet, show instructions
# ──────────────────────────────────────────────────────────────

if not DATA_LOADED:
    st.title("INSIDER THREAT DETECTION")
    st.error(
        "**Data files not found.** Please run the preprocessing and model notebooks first:\n\n"
        "1. `CERT_Preprocessing.ipynb` → generates `processed_datasets/ueba_dataset_4/ueba_dataset_4_train.csv`\n"
        "2. `Autoencoder.ipynb` → trains the encoder model\n"
        "3. `Isolation_Forest.ipynb` → generates anomaly scores\n"
        "4. `Alert_Object_Builder.ipynb` → generates `explainability/alert_table/alert_table_4.csv`"
    )
    st.stop()


# ──────────────────────────────────────────────────────────────
# Pre-compute derived data used across tabs
# ──────────────────────────────────────────────────────────────

# Base behavioral feature columns (raw counts)
RAW_FEATURES = [
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
RAW_FEATURES = [f for f in RAW_FEATURES if f in merged_df.columns]

CROSS_FLAGS = [
    "usb_file_activity_flag", "off_hours_activity_flag", "external_comm_activity_flag",
    "jobsite_usb_activity_flag", "suspicious_upload_flag", "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]
CROSS_FLAGS = [f for f in CROSS_FLAGS if f in merged_df.columns]

# Channel groupings for radar / breakdown charts
CHANNELS = {
    "Authentication": ["logon_count", "logoff_count", "off_hours_logon"],
    "File Access":    ["file_open_count", "file_write_count", "file_copy_count",
                       "file_delete_count", "unique_files_accessed", "off_hours_files_accessed"],
    "Removable Media":["usb_insert_count", "usb_remove_count", "off_hours_usb_usage"],
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
# Filter to features actually present; drop channels with no data
CHANNELS = {k: [f for f in v if f in merged_df.columns] for k, v in CHANNELS.items()}
# Drop empty channels (dataset may not have all features)
CHANNELS = {k: v for k, v in CHANNELS.items() if v}

# Explicit channel→color mapping so each channel keeps its color regardless of which are present in filtered data
CHANNEL_COLOR_MAP = {
    "Authentication":  "#ffffff",
    "File Access":     "#e84545",
    "Removable Media": "#d4a017",
    "Email":           "#3a86a8",
    "HTTP Activity":   "#7a7a7a",
    "PC Activity":     "#6b4a2d",
}

# user_risk is now pre-computed inside load_data() and cached


# ──────────────────────────────────────────────────────────────
# Plotly theme defaults
# ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#0a0a0a",
    font=dict(family="Inter, sans-serif", color="#999999", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#1a1a1a"),
    yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#1a1a1a"),
)

RISK_TIERS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
RISK_COLORS = {"CRITICAL": "#ff1744", "HIGH": "#e84545", "MEDIUM": "#d4a017", "LOW": "#3a86a8"}

# ──────────────────────────────────────────────────────────────
# Alert context helpers
# ──────────────────────────────────────────────────────────────

import ast as _ast

_FEATURE_LABELS: dict[str, str] = {
    # ── Authentication ──
    "off_hours_logon":                          "after-hours logon activity",
    "off_hours_logon_count":                    "after-hours logon activity",
    "logon_count":                              "elevated logon frequency",
    # ── File activity ──
    "file_open_count":                          "high file open activity",
    "file_write_count":                         "file write activity",
    "file_copy_count":                          "file copy activity",
    "file_delete_count":                        "file deletion activity",
    "unique_files_accessed":                    "access to many unique files",
    "off_hours_files_accessed":                 "after-hours file access",
    # ── USB ──
    "usb_insert_count":                         "USB device insertion",
    "usb_remove_count":                         "USB device removals",
    "off_hours_usb_usage":                      "after-hours USB usage",
    "usb_file_activity_flag":                   "USB-related file activity",
    "jobsite_usb_activity_flag":                "job-site browsing combined with USB activity",
    # ── Email ──
    "emails_sent":                              "high email volume",
    "unique_recipients":                        "many unique email recipients",
    "external_emails":                          "external email activity",
    "external_email_count":                     "external email activity",
    "attachements_sent":                        "email attachment activity",
    "attachments_sent":                         "email attachment activity",
    "off_hours_emails":                         "after-hours email activity",
    # ── Cross-channel flags ──
    "off_hours_activity_flag":                  "off-hours behavioral anomalies",
    "external_comm_activity_flag":              "external communication anomalies",
    # ── HTTP / Web activity (raw counts) ──
    "http_total_requests":                      "total web requests",
    "http_visit_count":                         "web page visits",
    "http_download_count":                      "web downloads",
    "http_upload_count":                        "web uploads",
    "http_jobsite_visits":                      "job-site website visits",
    "http_cloud_storage_visits":                "cloud storage website visits",
    "http_suspicious_site_visits":              "suspicious website visits",
    "off_hours_http_requests":                  "after-hours HTTP activity",
    "http_long_url_count":                      "long-URL HTTP activity",
    "unique_domains_visited":                   "unique websites visited",
    # ── HTTP cross-channel flags ──
    "jobsite_usb_activity_flag":                "job-site browsing combined with USB activity",
    "suspicious_upload_flag":                   "suspicious web upload activity",
    "cloud_upload_flag":                        "cloud storage upload activity",
    # ── Known z-score variants ──
    "file_delete_count_zscore":                 "unusually high file deletion activity",
    "file_write_count_zscore":                  "unusually high file write activity",
    "file_copy_count_zscore":                   "unusually high file copy activity",
    "file_open_count_zscore":                   "unusually high file open activity",
    "unique_files_accessed_zscore":             "unusually high unique file access",
    "off_hours_files_accessed_zscore":          "unusually high after-hours file access",
    "off_hours_logon_zscore":                   "unusually high after-hours logon activity",
    "http_long_url_count_zscore":               "unusually high long-URL HTTP activity",
    "off_hours_http_requests_zscore":           "unusually high after-hours HTTP activity",
    "attachments_sent_zscore":                  "unusually high attachment-sending activity",
    "attachements_sent_zscore":                 "unusually high attachment-sending activity",
    "external_emails_sent_zscore":              "unusually high external email activity",
    "external_email_count_zscore":              "unusually high external email activity",
    "emails_sent_zscore":                       "unusually high email volume",
    "unique_recipients_zscore":                 "unusually high number of unique email recipients",
    "usb_insert_count_zscore":                  "unusually high USB insertion activity",
    "http_total_requests_zscore":               "unusually high web request volume",
    "http_visit_count_zscore":                  "unusually high web browsing activity",
    "http_download_count_zscore":               "unusually high web download activity",
    "http_upload_count_zscore":                 "unusually high web upload activity",
    "http_jobsite_visits_zscore":               "unusually high job-site browsing activity",
    "http_cloud_storage_visits_zscore":         "unusually high cloud storage activity",
    "http_suspicious_site_visits_zscore":       "unusually high suspicious site visits",
    "unique_domains_visited_zscore":            "unusually high number of unique websites visited",
    # ── Known rolling-delta variants ──
    "file_delete_count_rolling_delta":          "sudden spike in file deletions",
    "file_write_count_rolling_delta":           "sudden spike in file write activity",
    "file_copy_count_rolling_delta":            "sudden spike in file copy activity",
    "file_open_count_rolling_delta":            "sudden spike in file open activity",
    "unique_files_accessed_rolling_delta":      "sudden increase in unique file access",
    "off_hours_files_accessed_rolling_delta":   "sudden increase in after-hours file access",
    "off_hours_logon_rolling_delta":            "sudden increase in after-hours logons",
    "emails_sent_rolling_delta":                "sudden spike in email volume",
    "external_emails_rolling_delta":            "sudden spike in external email activity",
    "attachments_sent_rolling_delta":           "sudden spike in attachment-sending",
    "attachements_sent_rolling_delta":          "sudden spike in attachment-sending",
    "usb_insert_count_rolling_delta":           "sudden spike in USB device activity",
    "http_requests_rolling_delta":              "sudden spike in HTTP requests",
    "off_hours_http_requests_rolling_delta":    "sudden increase in after-hours HTTP activity",
    "http_total_requests_rolling_delta":        "sudden spike in web requests",
    "http_visit_count_rolling_delta":           "sudden spike in web browsing",
    "http_download_count_rolling_delta":        "sudden spike in web downloads",
    "http_upload_count_rolling_delta":          "sudden spike in web uploads",
    "http_jobsite_visits_rolling_delta":        "sudden increase in job-site browsing",
    "http_cloud_storage_visits_rolling_delta":  "sudden increase in cloud storage website visits",
    "http_suspicious_site_visits_rolling_delta": "sudden increase in suspicious site visits",
    "http_long_url_count_rolling_delta":        "sudden spike in long-URL HTTP activity",
    "unique_domains_visited_rolling_delta":     "sudden increase in unique websites visited",
}

# Base-name phrases used by the pattern-matching fallback in prettify_feature_name()
_BASE_LABELS: dict[str, str] = {
    "file_delete_count":        "file deletions",
    "file_write_count":         "file write activity",
    "file_copy_count":          "file copy activity",
    "file_open_count":          "file open activity",
    "unique_files_accessed":    "unique file access",
    "off_hours_files_accessed": "after-hours file access",
    "off_hours_logon":          "after-hours logons",
    "off_hours_logon_count":    "after-hours logons",
    "logon_count":              "logon frequency",
    "logoff_count":             "logoff activity",
    "external_emails":          "external email activity",
    "external_email_count":     "external email activity",
    "external_emails_sent":     "external email activity",
    "http_long_url":            "long-URL HTTP activity",
    "off_hours_http":           "after-hours HTTP activity",
    "attachments_sent":         "attachment-sending activity",
    "attachements_sent":        "attachment-sending activity",
    "emails_sent":              "email volume",
    "unique_recipients":        "unique email recipients",
    "off_hours_emails":         "after-hours email activity",
    "usb_insert_count":         "USB device activity",
    "usb_remove_count":         "USB device removals",
    "off_hours_usb_usage":      "after-hours USB usage",
    "http_requests":            "HTTP requests",
    "http_long_url":            "long-URL HTTP activity",
    "off_hours_http":           "after-hours HTTP activity",
    "http_total_requests":          "web requests",
    "http_visit_count":             "web page visits",
    "http_download_count":          "web downloads",
    "http_upload_count":            "web uploads",
    "http_jobsite_visits":          "job-site browsing",
    "http_cloud_storage_visits":    "cloud storage website visits",
    "http_suspicious_site_visits":  "suspicious site visits",
    "unique_domains_visited":       "unique websites visited",
}


def _humanize_base(base: str) -> str:
    """Map a bare feature base name to a readable phrase for pattern-generated sentences."""
    return _BASE_LABELS.get(base, base.replace("_", " "))


def parse_top_contributors(raw) -> list[str]:
    """Return a list of feature name strings from top_contributors, however it is stored."""
    if raw is None:
        return []
    if isinstance(raw, float):
        return []  # NaN from a left-join miss
    if isinstance(raw, list):
        names = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                names.append(str(item[0]))
            elif isinstance(item, str):
                names.append(item)
        return names
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = _ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parse_top_contributors(parsed)
        except Exception:
            pass
    return []


def prettify_feature_name(name: str) -> str:
    """Convert a raw feature name to an analyst-readable investigation phrase."""
    cleaned = name.strip().replace("contribution_", "")
    # 1. Exact dictionary hit — covers flags, known features, and common variants
    if cleaned in _FEATURE_LABELS:
        return _FEATURE_LABELS[cleaned]
    # 2. Pattern: z-score suffix → "unusually high <base>"
    if cleaned.endswith("_zscore"):
        base = cleaned[: -len("_zscore")]
        return f"unusually high {_humanize_base(base)}"
    # 3. Pattern: rolling-delta suffix → "sudden spike/increase in <base>"
    if cleaned.endswith("_rolling_delta"):
        base = cleaned[: -len("_rolling_delta")]
        human = _humanize_base(base)
        if base.endswith(("_count", "_sent")):
            return f"sudden spike in {human}"
        return f"sudden increase in {human}"
    # 4. Last resort: replace underscores with spaces
    return cleaned.replace("_", " ")


def build_alert_summary(top_contributors_raw) -> str:
    """
    Return a short, analyst-friendly sentence describing the top contributing
    behaviors for an alert. Shows up to 3 contributors; appends 'and additional
    signals' when the full list contains more than 3 unique contributors.
    """
    features = parse_top_contributors(top_contributors_raw)
    if not features:
        return "No contributor detail available for this alert."

    labels: list[str] = []
    seen: set[str] = set()
    for f in features:
        lbl = prettify_feature_name(f)
        if lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)

    has_more = len(labels) > 3
    display = labels[:3]

    if len(display) == 1:
        return f"This alert is primarily driven by {display[0]}."

    if len(display) == 2:
        return f"This alert is mainly driven by {display[0]} and {display[1]}."

    # Exactly 3 shown
    body = f"{display[0]}, {display[1]}, and {display[2]}"
    suffix = ", and additional signals" if has_more else ""
    return f"This alert is mainly driven by {body}{suffix}."


# ──────────────────────────────────────────────────────────────
# Sidebar — Global Filters
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Navigation pages
# ──────────────────────────────────────────────────────────────

NAV_PAGES = [
    "Alerts",
    "Overview",
    "Investigation",
    "Channels",
]

# Live simulation state is shared across all pages.
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "live_proc" not in st.session_state:
    st.session_state.live_proc = None  # subprocess.Popen or None
if "live_paused" not in st.session_state:
    st.session_state.live_paused = False

# Consume any programmatic navigation request NOW — before any widget is
# instantiated — so we can write session_state.nav_page freely.
if st.session_state.get("_nav_request"):
    st.session_state.nav_page = st.session_state.pop("_nav_request")

with st.sidebar:
    # ── DSK Team Logo ──
    st.markdown(
        "<div class='sidebar-branding'>"
        "<svg width='44' height='44' viewBox='0 0 100 100' fill='none' xmlns='http://www.w3.org/2000/svg' style='flex-shrink:0;'>"
        "<path d='M25 85 L25 40 L15 15 L30 30 L50 25 L70 30 L85 15 L75 40 L75 85 Z' "
        "fill='#e84545' opacity='0.9'/>"
        "<circle cx='38' cy='50' r='5' fill='#000'/>"
        "<circle cx='62' cy='50' r='5' fill='#000'/>"
        "<path d='M45 60 Q50 65 55 60' stroke='#000' stroke-width='2' fill='none'/>"
        "<line x1='20' y1='55' x2='38' y2='52' stroke='#000' stroke-width='1.5'/>"
        "<line x1='20' y1='60' x2='38' y2='58' stroke='#000' stroke-width='1.5'/>"
        "<line x1='62' y1='52' x2='80' y2='55' stroke='#000' stroke-width='1.5'/>"
        "<line x1='62' y1='58' x2='80' y2='60' stroke='#000' stroke-width='1.5'/>"
        "</svg>"
        "<div style='line-height:1;'>"
        "<div style='font-family:JetBrains Mono,monospace; font-size:18px; letter-spacing:4px; "
        "color:#ffffff; font-weight:700;'>DSK</div>"
        "<div style='font-family:JetBrains Mono,monospace; font-size:9px; letter-spacing:1.5px; "
        "color:#555555; text-transform:uppercase; margin-top:3px;'>Data Structure Kittens</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Navigation ──
    st.markdown("<p class='sidebar-section-label'>Navigation</p>", unsafe_allow_html=True)
    active_page = st.radio(
        "Nav",
        NAV_PAGES,
        index=0,
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown("<div style='border-top:1px solid #1a1a1a; margin:8px 0 0 0;'></div>", unsafe_allow_html=True)

    # ── Global live status (visible on every page) ──
    _live_rows_received = 0
    _stream_done = False
    _live_session_active = bool(st.session_state.live_mode or st.session_state.live_paused)
    if _live_session_active and os.path.exists(LIVE_OUTPUT):
        with open(LIVE_OUTPUT, "r", encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _obj = json.loads(_line)
                except json.JSONDecodeError:
                    continue
                if _obj.get("_eos"):
                    _stream_done = True
                else:
                    _live_rows_received += 1

    _proc = st.session_state.live_proc
    _proc_running = _proc is not None and _proc.poll() is None
    if not _live_session_active:
        _status_color = "#666"
        _status_label = "IDLE"
    elif st.session_state.live_paused:
        _status_color = "#f5a623"
        _status_label = "PAUSED"
    elif _proc_running:
        _status_color = "#3c9"
        _status_label = "RUNNING"
    else:
        _status_color = "#e84545"
        _status_label = "COMPLETE" if _stream_done else "STOPPED"

    st.markdown(
        f"<div style='margin:12px 0 6px 0;'>"
        f"<span style='font-family:JetBrains Mono,monospace; font-size:11px; "
        f"color:{_status_color}; letter-spacing:1.5px;'>● {_status_label}</span>"
        f"<span style='font-family:JetBrains Mono,monospace; font-size:10px; "
        f"color:#555; margin-left:10px;'>{_live_rows_received:,} rows received</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='border-top:1px solid #1a1a1a; margin:16px 0;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace; font-size:9px; color:#333; "
        "text-transform:uppercase; letter-spacing:1.5px; line-height:1.8;'>"
        "DSK &mdash; Data Structure Kittens<br>Senior Design Project &middot; 2026</div>",
        unsafe_allow_html=True,
    )

# Users list (used across pages for search)
all_users = sorted(merged_df["user"].unique())

# Cap data points sent to Plotly — browser rendering is the main bottleneck
MAX_PLOT_POINTS = 50_000

# ──────────────────────────────────────────────────────────────
# Global filter state (persists across page navigations)
# ──────────────────────────────────────────────────────────────
_DS_MIN = merged_df["day"].min().date()
_DS_MAX = merged_df["day"].max().date()
if "flt_date_start" not in st.session_state:
    st.session_state.flt_date_start = _DS_MIN
if "flt_date_end" not in st.session_state:
    st.session_state.flt_date_end = _DS_MAX
if "flt_risk" not in st.session_state:
    st.session_state.flt_risk = list(RISK_TIERS)


@st.dialog("Filters")
def show_filters():
    st.markdown("**Date Range**")
    dr = st.date_input(
        "Date Range",
        value=(st.session_state.flt_date_start, st.session_state.flt_date_end),
        min_value=_DS_MIN,
        max_value=_DS_MAX,
        label_visibility="collapsed",
        key="dlg_date",
    )
    st.markdown("**Risk Levels**")
    rl = st.multiselect(
        "Risk Levels",
        RISK_TIERS,
        default=st.session_state.flt_risk,
        label_visibility="collapsed",
        key="dlg_risk",
    )
    st.markdown("")
    apply_col, reset_col = st.columns(2)
    with apply_col:
        if st.button("Apply", use_container_width=True, type="primary"):
            if isinstance(dr, tuple) and len(dr) == 2:
                st.session_state.flt_date_start = dr[0]
                st.session_state.flt_date_end = dr[1]
            st.session_state.flt_risk = rl if rl else list(RISK_TIERS)
            st.rerun()
    with reset_col:
        if st.button("Reset", use_container_width=True):
            st.session_state.flt_date_start = _DS_MIN
            st.session_state.flt_date_end = _DS_MAX
            st.session_state.flt_risk = list(RISK_TIERS)
            st.rerun()


@st.cache_data(show_spinner=False)
def _cached_filtered_df(date_start, date_end, risk_levels: tuple) -> pd.DataFrame:
    """Return a cached slice of merged_df. Only recomputes when filters change."""
    mask = (
        merged_df["ae_risk_band"].isin(risk_levels)
        & (merged_df["day"].dt.date >= date_start)
        & (merged_df["day"].dt.date <= date_end)
    )
    return merged_df[mask].copy()


def _get_filtered_df() -> pd.DataFrame:
    """Return merged_df sliced by current session_state filter values."""
    return _cached_filtered_df(
        st.session_state.flt_date_start,
        st.session_state.flt_date_end,
        tuple(sorted(st.session_state.flt_risk)),
    )


@st.cache_data(show_spinner=False)
def _pop_channel_avgs() -> dict[str, float]:
    """Pre-compute population channel averages from the full dataset (run once)."""
    result: dict[str, float] = {}
    for channel, feats in CHANNELS.items():
        valid = [f for f in feats if f in merged_df.columns]
        if valid:
            result[channel] = float(merged_df[valid].mean().sum())
    return result


@st.cache_data(show_spinner=False)
def _channel_time_series(date_start, date_end, risk_levels: tuple) -> pd.DataFrame:
    """Cached channel-volume-by-day aggregation for the Channels tab."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    parts = []
    for channel, feats in CHANNELS.items():
        valid = [f for f in feats if f in fdf.columns]
        if valid:
            daily = fdf.groupby(fdf["day"].dt.date)[valid].sum().sum(axis=1).reset_index()
            daily.columns = ["Date", "Volume"]
            daily["Channel"] = channel
            parts.append(daily)
    return pd.concat(parts) if parts else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _corr_matrix(date_start, date_end, risk_levels: tuple, corr_feats: tuple) -> pd.DataFrame:
    """Cached Pearson correlation matrix for the Channels tab."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    sample = fdf if len(fdf) <= MAX_PLOT_POINTS else fdf.sample(MAX_PLOT_POINTS, random_state=42)
    return sample[list(corr_feats)].corr()


_SECTION_INFO = {
    "Risk Distribution": (
        "**Risk Distribution**\n\n"
        "A donut chart showing what proportion of activity records fall into each risk tier:\n\n"
        "- **HIGH** — anomaly score in the top percentile; warrants immediate review\n"
        "- **MEDIUM** — elevated but not critical; worth monitoring\n"
        "- **LOW** — behavior consistent with typical baseline activity\n\n"
        "Risk level is assigned per record based on the anomaly score percentile "
        "produced by the Isolation Forest model."
    ),
    "Alert Trend Over Time": (
        "**Alert Trend Over Time**\n\n"
        "An area chart showing how many alerts were generated each day, stacked by risk level.\n\n"
        "Use this to identify **spikes** or **sustained elevations** in suspicious activity "
        "within the selected date range. A sudden spike often correlates with a specific "
        "incident (e.g. large file exfiltration, off-hours access)."
    ),
    "Top 10 Riskiest Users": (
        "**Top 10 Riskiest Users**\n\n"
        "A ranked list of the users with the highest overall risk score.\n\n"
        "Each entry shows:\n"
        "- **Percentile** — how the user ranks relative to all users (100 = most anomalous)\n"
        "- **High-risk days** — number of days flagged HIGH by the model\n"
        "- **Badge** — CRITICAL (≥80th pct), HIGH (≥60th), or ELEVATED (below 60th)\n\n"
        "Click **Investigate →** to jump directly to that user's full behavioral profile."
    ),
    "Anomaly Score Distribution": (
        "**Anomaly Score Distribution**\n\n"
        "A histogram of raw anomaly scores across all records, coloured by risk level.\n\n"
        "- Scores **near 0** indicate behavior very close to the normal baseline\n"
        "- Scores **toward 1** represent increasingly anomalous activity\n\n"
        "The Isolation Forest model assigns each record a score based on how easily it "
        "can be isolated from the rest of the dataset. Outliers require fewer splits and "
        "therefore receive higher scores."
    ),
    "Cross-Channel Risk Flags (Global)": (
        "**Cross-Channel Risk Flags**\n\n"
        "Counts of records where suspicious activity co-occurred across multiple data channels:\n\n"
        "- **USB + File Write** — removable device use paired with large file write activity\n"
        "- **Off-Hours Activity** — logins or actions recorded outside normal working hours\n"
        "- **External Communication** — significant outbound traffic to external endpoints\n"
        "- **Job Site + USB** — job-site browsing paired with USB insertion on the same day\n"
        "- **Suspicious Upload** — HTTP upload to a suspicious or uncategorized domain\n"
        "- **Cloud Upload** — HTTP upload to a known cloud storage service\n"
        "- **Non-Primary PC** — sensitive activity (file copy, USB, HTTP) from an atypical endpoint\n\n"
        "These compound flags are stronger indicators of insider threat than any single "
        "channel signal alone."
    ),
    "Anomaly Score Timeline": (
        "**Anomaly Score Timeline**\n\n"
        "A day-by-day line chart of the selected user's anomaly score.\n\n"
        "- A **persistent elevation** suggests consistently unusual behavior\n"
        "- A **sudden spike** may point to a discrete incident on that date\n\n"
        "Use the date filter to narrow the window and correlate peaks with raw activity records below."
    ),
    "Behavioral Profile (Avg Activity)": (
        "**Behavioral Profile**\n\n"
        "A radar chart comparing the selected user's **average feature values** (solid line) "
        "against the **global population average** (dashed line).\n\n"
        "Each axis represents one of six behavioral channels: Authentication, File Access, "
        "Removable Media, Email, HTTP Activity, and PC Activity. "
        "Axes where the user extends significantly beyond the population average indicate "
        "dimensions of behavior worth investigating."
    ),
    "Daily Feature Activity": (
        "**Daily Feature Activity**\n\n"
        "A heatmap of the user's raw feature values over time.\n\n"
        "- Each **row** is one behavioral feature\n"
        "- Each **column** is one day\n"
        "- **Darker cells** = higher-than-usual activity on that day and feature\n\n"
        "This lets you pinpoint exactly which features drove an anomaly spike on a given date."
    ),
    "Cross-Channel Risk Indicators": (
        "**Cross-Channel Risk Indicators**\n\n"
        "A summary of whether this user triggered any multi-channel co-occurrence flags:\n\n"
        "- **USB + File Write** — device use coincided with large file write events\n"
        "- **Off-Hours Activity** — actions occurred outside normal business hours\n"
        "- **External Communication** — outbound connections to external hosts were detected\n"
        "- **Job Site + USB** — job-site browsing paired with USB insertion on the same day\n"
        "- **Suspicious Upload** — HTTP upload to a suspicious or uncategorized domain\n"
        "- **Cloud Upload** — HTTP upload to a known cloud storage service\n"
        "- **Non-Primary PC** — sensitive activity from an atypical endpoint\n\n"
        "Combinations of multiple flags substantially increase the likelihood of an insider threat."
    ),
    "Raw Activity Records": (
        "**Raw Activity Records**\n\n"
        "A full table of every aggregated daily record for the selected user within the "
        "current filter window.\n\n"
        "Each row represents one day and includes all behavioral features (email, file, "
        "HTTP, logon, device activity), the computed anomaly score, and the assigned risk level. "
        "Use this to audit exactly what the model saw on any particular date."
    ),
    "Channel Activity Volume": (
        "**Channel Activity Volume**\n\n"
        "A line chart showing the total number of events recorded per day across each data channel "
        "(Authentication, File Access, Removable Media, Email, HTTP Activity, PC Activity) "
        "within the selected filters.\n\n"
        "Channels with disproportionately high volumes relative to peers can indicate "
        "a data exfiltration path that warrants deeper investigation."
    ),
    "Channel Volume Share": (
        "**Channel Volume Share**\n\n"
        "A donut chart showing each channel's **percentage share** of all recorded events "
        "across Authentication, File Access, Removable Media, Email, HTTP Activity, and PC Activity.\n\n"
        "This gives a quick sense of which channels dominate activity organizationally. "
        "A sudden shift in these proportions between time periods may indicate an attack campaign."
    ),
    "Feature Distributions by Risk Level": (
        "**Feature Distributions by Risk Level**\n\n"
        "Box plots for each numeric feature, grouped by risk level (HIGH / MEDIUM / LOW).\n\n"
        "The box shows the interquartile range (25th–75th percentile); the line inside is the median. "
        "Features where the HIGH box sits far above LOW are the **strongest predictors** "
        "of anomalous behavioral in this dataset."
    ),
    "Feature Correlation Matrix": (
        "**Feature Correlation Matrix**\n\n"
        "A heatmap of Pearson correlations between all numeric features.\n\n"
        "- **+1 (dark red)** — features rise and fall together\n"
        "- **−1 (dark blue)** — features move in opposite directions\n"
        "- **~0** — no linear relationship\n\n"
        "Highly correlated features may be redundant for modelling, while unexpected "
        "correlations can reveal undocumented behavioral patterns."
    ),
}


def section_header(title: str, key: str) -> None:
    """Render a section header with an optional ⓘ info popover."""
    info = _SECTION_INFO.get(title, "")
    if info:
        _hdr_left, _hdr_right = st.columns([9, 1], vertical_alignment="bottom")
        with _hdr_left:
            st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
        with _hdr_right:
            with st.popover("ⓘ", use_container_width=True):
                st.markdown(info, unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def _filter_bar(key: str):
    """Render filter button + active-filter summary. Opens modal on click."""
    _fb_left, _fb_right = st.columns([7, 1])
    with _fb_right:
        if st.button("Filter", key=key, use_container_width=True):
            show_filters()
    active_risks = ", ".join(st.session_state.flt_risk) if len(st.session_state.flt_risk) < 3 else "All risk levels"
    _fb_left.caption(
        f"Date: {st.session_state.flt_date_start} to {st.session_state.flt_date_end}   |   "
        f"Risk: {active_risks}"
    )

st.markdown("<div class='project-title-badge'>Insider Threat Detection</div>", unsafe_allow_html=True)

# ── Mobile sidebar toggle (injected once into parent document) ──
components.html(
    """
    <script>
    (function(){
        var p = window.parent.document;
        if (p.getElementById('mob-sidebar-btn')) return;
        var btn = p.createElement('button');
        btn.id = 'mob-sidebar-btn';
        btn.innerHTML = '&#9776;';
        btn.style.cssText = [
            'position:fixed','top:12px','left:12px','z-index:99999',
            'background:#0a0a0a','border:1px solid #1a1a1a','color:#aaa',
            'width:38px','height:38px','font-size:18px','cursor:pointer',
            'display:none','align-items:center','justify-content:center',
            'padding:0'
        ].join(';');
        p.body.appendChild(btn);
        function show(){
            btn.style.display = window.parent.innerWidth <= 768 ? 'flex' : 'none';
        }
        window.parent.addEventListener('resize', show);
        show();
        btn.addEventListener('click', function(){
            var sidebar = p.querySelector('[data-testid="stSidebar"]');
            var collapse = p.querySelector('[data-testid="stSidebarCollapseButton"]');
            var expand   = p.querySelector('[data-testid="stExpandSidebarButton"]');
            if (sidebar) {
                var rect = sidebar.getBoundingClientRect();
                if (rect.width > 50) {
                    if (collapse) collapse.click();
                } else {
                    if (expand) expand.click();
                }
            } else if (expand) { expand.click(); }
        });
    })();
    </script>
    """,
    height=0,
)
# ══════════════════════════════════════════════════════════════

if active_page == "Overview":
    st.markdown(
        "<div class='page-header-block'>"
        "<h1 class='page-title'>Overview</h1>"
        "<p class='page-subtitle'>High-level summary of monitored users, risk levels, anomaly scores, and cross-channel threat indicators.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _filter_bar("ov_flt")
    filtered_df = _get_filtered_df()

    total_users = filtered_df["user"].nunique()
    total_records = len(filtered_df)
    critical_risk_users = filtered_df[filtered_df["ae_risk_band"] == "CRITICAL"]["user"].nunique()
    high_risk_users = filtered_df[filtered_df["ae_risk_band"] == "HIGH"]["user"].nunique()
    medium_risk_users = filtered_df[filtered_df["ae_risk_band"] == "MEDIUM"]["user"].nunique()
    avg_anomaly = filtered_df["if_anomaly_score"].mean()
    detection_rate = (filtered_df["ae_risk_band"].isin(["CRITICAL", "HIGH", "MEDIUM"]).sum() / max(len(filtered_df), 1)) * 100

    st.markdown(
        "<div class='kpi-scroll-wrapper'>"
        "<div class='kpi-scroll-arrow' id='kpi-arrow-left'>&#8592;</div>"
        f"<div class='kpi-scroll-row' id='kpi-row'>"
        f"<div class='kpi-card' style='border-color:#ffffff'><h3>Users Monitored</h3><h1 style='color:#ffffff'>{total_users:,}</h1><p>Active in period</p></div>"
        f"<div class='kpi-card' style='border-color:#ff1744'><h3>Critical Risk</h3><h1 style='color:#ff1744'>{critical_risk_users}</h1><p>&ge; 95th percentile</p></div>"
        f"<div class='kpi-card' style='border-color:#e84545'><h3>High Risk</h3><h1 style='color:#e84545'>{high_risk_users}</h1><p>&ge; 90th percentile</p></div>"
        f"<div class='kpi-card' style='border-color:#d4a017'><h3>Medium Risk</h3><h1 style='color:#d4a017'>{medium_risk_users}</h1><p>&ge; 80th percentile</p></div>"
        f"<div class='kpi-card' style='border-color:#666666'><h3>Total Records</h3><h1 style='color:#cccccc'>{total_records:,}</h1><p>User-day observations</p></div>"
        f"<div class='kpi-card' style='border-color:#666666'><h3>Avg Anomaly Score</h3><h1 style='color:#cccccc'>{avg_anomaly:.4f}</h1><p>Across all records</p></div>"
        f"<div class='kpi-card' style='border-color:#666666'><h3>Detection Rate</h3><h1 style='color:#cccccc'>{detection_rate:.1f}%</h1><p>Medium + High + Critical alerts</p></div>"
        "</div>"
        "<div class='kpi-scroll-arrow' id='kpi-arrow-right'>&#8594;</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    components.html(
        "<script>"
        "(function(){"
        "  var p = window.parent.document;"
        "  function wire(){"
        "    var row = p.getElementById('kpi-row');"
        "    var lft = p.getElementById('kpi-arrow-left');"
        "    var rgt = p.getElementById('kpi-arrow-right');"
        "    if(!row||!lft||!rgt){setTimeout(wire,100);return;}"
        "    lft.addEventListener('click',function(){row.scrollBy({left:-200,behavior:'smooth'});});"
        "    rgt.addEventListener('click',function(){row.scrollBy({left:200,behavior:'smooth'});});"
        "  }"
        "  wire();"
        "})();"
        "</script>",
        height=0,
    )

    st.markdown("")

    # ── Row 2: Risk Distribution + Alerts Over Time ──
    col_left, col_right = st.columns([1, 2])

    with col_left:
        section_header("Risk Distribution", "sh_risk_dist")
        risk_counts = filtered_df["ae_risk_band"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_donut = px.pie(
            risk_counts, values="Count", names="Risk Level",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            hole=0.6,
        )
        fig_donut.update_layout(**PLOTLY_LAYOUT, showlegend=True, height=340,
                                legend=dict(font=dict(size=10, family="JetBrains Mono")))
        fig_donut.update_traces(textinfo="label+percent", textfont_size=11,
                                textfont_family="JetBrains Mono")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        section_header("Alert Trend Over Time", "sh_alert_trend")
        daily_alerts = (
            filtered_df.groupby([filtered_df["day"].dt.date, "ae_risk_band"])
            .size()
            .reset_index(name="count")
        )
        daily_alerts.columns = ["Date", "Risk Level", "Count"]
        fig_trend = px.area(
            daily_alerts, x="Date", y="Count", color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="", yaxis_title="Alert Count")
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Row 3: Top Risky Users + Score Distribution ──
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        section_header("Top 10 Riskiest Users", "sh_top_users")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:13px;color:#555;margin:0 0 12px 0;'>"
            "Click a user to open their investigation profile.</p>",
            unsafe_allow_html=True,
        )
        top_users = user_risk.head(10).copy()
        for rank, row in enumerate(top_users.itertuples(), start=1):
            uid = row.user
            score = row.max_percentile
            days = row.critical_count + row.high_count
            # Color badge based on score
            if score >= 95:
                badge_color = "#ff1744"
                badge_label = "CRITICAL"
            elif score >= 90:
                badge_color = "#e84545"
                badge_label = "HIGH"
            elif score >= 80:
                badge_color = "#d4a017"
                badge_label = "MEDIUM"
            else:
                badge_color = "#3a86a8"
                badge_label = "LOW"

            col_rank, col_info, col_btn = st.columns([1, 5, 3])
            with col_rank:
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:13px;"
                    f"color:#444;font-weight:600;padding-top:4px;text-align:center;'>#{rank}</div>",
                    unsafe_allow_html=True,
                )
            with col_info:
                st.markdown(
                    f"<div style='padding:2px 0 1px 0;'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:13px;"
                    f"color:#e0e0e0;font-weight:600;'>{uid}</span>"
                    f"&nbsp;&nbsp;<span style='background:{badge_color}22;color:{badge_color};font-size:10px;"
                    f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:1px 5px;"
                    f"border:1px solid {badge_color}55;'>{badge_label}</span>"
                    f"<br><span style='font-family:Inter,sans-serif;font-size:11px;"
                    f"color:#555;line-height:1.5;'>"
                    f"Percentile {score:.1f} &middot; {days} high-risk day{'s' if days != 1 else ''}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("Investigate →", key=f"inv_btn_{uid}", use_container_width=True):
                    st.session_state["inv_user_select"] = uid
                    st.session_state["_nav_request"] = "Investigation"
                    st.rerun()
            st.markdown("<div style='border-bottom:1px solid #111;margin:0;'></div>", unsafe_allow_html=True)

    with col_right2:
        section_header("Anomaly Score Distribution", "sh_score_dist")
        # Sample for histogram to avoid sending 2M+ points to Plotly
        hist_df = filtered_df if len(filtered_df) <= MAX_PLOT_POINTS else filtered_df.sample(MAX_PLOT_POINTS, random_state=42)
        fig_hist = px.histogram(
            hist_df, x="if_anomaly_score", nbins=80,
            color="ae_risk_band", color_discrete_map=RISK_COLORS,
            labels={"if_anomaly_score": "Anomaly Score", "ae_risk_band": "Risk Level"},
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=440, barmode="overlay")
        fig_hist.update_traces(opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 4: Cross-Channel Risk Flags Summary ──
    if CROSS_FLAGS:
        section_header("Cross-Channel Risk Flags (Global)", "sh_cross_flags")
        _flag_info = [
            ("usb_file_activity_flag",      "USB + File Write",      "#e84545"),
            ("off_hours_activity_flag",     "Off-Hours Activity",    "#d4a017"),
            ("external_comm_activity_flag", "External Communication","#3a86a8"),
            ("jobsite_usb_activity_flag",   "Job Site + USB",        "#9b59b6"),
            ("suspicious_upload_flag",      "Suspicious Upload",     "#e67e22"),
            ("cloud_upload_flag",           "Cloud Upload",          "#00b4d8"),
            ("non_primary_pc_risk_flag",    "Non-Primary PC",        "#7f8c8d"),
        ]
        _cards_html = ""
        for flag, label, color in _flag_info:
            if flag in CROSS_FLAGS and flag in filtered_df.columns:
                count = int(filtered_df[flag].sum())
                pct = (count / max(len(filtered_df), 1)) * 100
                _cards_html += (
                    f"<div style='flex:1;background:#0a0a0a;border:1px solid #1a1a1a;"
                    f"border-left:3px solid {color};padding:14px 18px;min-width:160px;'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#555;"
                    f"text-transform:uppercase;letter-spacing:1.5px;'>{label}</div>"
                    f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:8px;'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:24px;"
                    f"color:{color};font-weight:600;'>{count:,}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
                    f"color:#444;'>{pct:.1f}%</span>"
                    f"</div></div>"
                )
        st.markdown(
            f"<div style='display:flex;gap:12px;margin:4px 0 16px 0;flex-wrap:wrap;'>{_cards_html}</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# PAGE: Investigation
# ══════════════════════════════════════════════════════════════

if active_page == "Investigation":
    st.markdown(
        "<div class='page-header-block'>"
        "<h1 class='page-title'>Investigation</h1>"
        "<p class='page-subtitle'>Deep-dive into a single user&#39;s behavioral timeline, radar profile, and raw activity records.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _filter_bar("inv_flt")
    filtered_df = _get_filtered_df()

    # Build user list ordered by risk (highest first)
    _risk_sorted = user_risk["user"].tolist()
    _remaining = [u for u in all_users if u not in set(_risk_sorted)]
    _all_users_sorted = _risk_sorted + _remaining

    selected_user = st.selectbox(
        "Search User ID",
        _all_users_sorted,
        index=None,
        placeholder="Type to search, e.g. acm2278",
        key="inv_user_select",
    )

    if selected_user is None:
        st.info("Select a user above to begin investigation. Users are sorted by risk (highest first).")
        st.stop()

    # O(1) lookup from pre-grouped dict — avoids scanning 1.5 M rows per user switch
    _u_rows = user_data_dict.get(selected_user, pd.DataFrame())
    if not _u_rows.empty:
        _u_mask = (
            _u_rows["ae_risk_band"].isin(st.session_state.flt_risk)
            & (_u_rows["day"].dt.date >= st.session_state.flt_date_start)
            & (_u_rows["day"].dt.date <= st.session_state.flt_date_end)
        )
        user_data = _u_rows[_u_mask].sort_values("day")
    else:
        user_data = pd.DataFrame()

    if user_data.empty:
        st.warning("No data for this user in the current filter range.")
        st.stop()

    # ── User KPI Row ──
    u1, u2, u3, u4, u5, u6 = st.columns(6)
    u_max_score = user_data["if_anomaly_score"].max()
    u_max_pctl = user_data["ae_percentile_rank"].max()
    u_crit_days = (user_data["ae_risk_band"] == "CRITICAL").sum()
    u_high_days = (user_data["ae_risk_band"] == "HIGH").sum()
    u_med_days = (user_data["ae_risk_band"] == "MEDIUM").sum()
    u_total_days = len(user_data)

    # Determine overall user risk label
    if u_max_pctl >= 95:
        u_risk_label, u_risk_color = "CRITICAL", "#ff1744"
    elif u_max_pctl >= 90:
        u_risk_label, u_risk_color = "HIGH", "#ff6b6b"
    elif u_max_pctl >= 80:
        u_risk_label, u_risk_color = "MEDIUM", "#feca57"
    else:
        u_risk_label, u_risk_color = "LOW", "#48dbfb"

    u1.metric("Overall Risk", u_risk_label)
    u2.metric("Peak Percentile", f"{u_max_pctl:.1f}")
    u3.metric("Critical-Risk Days", u_crit_days)
    u4.metric("High-Risk Days", u_high_days)
    u5.metric("Medium-Risk Days", u_med_days)
    u6.metric("Days Observed", u_total_days)

    # ── Anomaly Timeline ──
    section_header("Anomaly Score Timeline", "sh_score_timeline")
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=user_data["day"], y=user_data["if_anomaly_score"],
        mode="lines+markers", name="Anomaly Score",
        line=dict(color="#ffffff", width=1.5),
        marker=dict(size=4, color="#ffffff"),
    ))
    # Color markers by risk level
    for risk, color in RISK_COLORS.items():
        subset = user_data[user_data["ae_risk_band"] == risk]
        if not subset.empty:
            fig_timeline.add_trace(go.Scatter(
                x=subset["day"], y=subset["if_anomaly_score"],
                mode="markers", name=risk,
                marker=dict(size=8, color=color, symbol="square", line=dict(width=1, color="#000")),

            ))
    fig_timeline.update_layout(**PLOTLY_LAYOUT, height=320, xaxis_title="Date", yaxis_title="Anomaly Score")
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Behavioral Radar Chart + Activity Heatmap ──
    col_radar, col_heat = st.columns(2)

    with col_radar:
        section_header("Behavioral Profile (Avg Activity)", "sh_beh_profile")
        # Compute average of each channel for this user vs population
        radar_categories = []
        user_vals = []
        pop_vals = []
        for channel, feats in CHANNELS.items():
            valid_feats = [f for f in feats if f in user_data.columns]
            if valid_feats:
                radar_categories.append(channel)
                user_vals.append(user_data[valid_feats].mean().sum())
                pop_vals.append(_pop_channel_avgs().get(channel, 0.0))

        if radar_categories:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=user_vals, theta=radar_categories, fill="toself",
                name=selected_user, line=dict(color="#e84545", width=2),
                fillcolor="rgba(232,69,69,0.15)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=pop_vals, theta=radar_categories, fill="toself",
                name="Population Avg", line=dict(color="#3a86a8", width=1), opacity=0.6,
                fillcolor="rgba(58,134,168,0.1)",
            ))
            fig_radar.update_layout(**PLOTLY_LAYOUT, height=380,
                                    polar=dict(bgcolor="#0a0a0a",
                                               radialaxis=dict(visible=True, color="#333333"),
                                               angularaxis=dict(color="#444444")),
                                    showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)

    with col_heat:
        section_header("Daily Feature Activity", "sh_daily_feat")
        # Show heatmap of raw feature values over time for this user
        heat_feats = [f for f in RAW_FEATURES if f in user_data.columns]
        if heat_feats and len(user_data) > 1:
            heat_data = user_data.set_index("day")[heat_feats].T
            heat_data.columns = [d.strftime("%m/%d") if hasattr(d, "strftime") else str(d) for d in heat_data.columns]
            # Limit columns for readability
            if heat_data.shape[1] > 30:
                heat_data = heat_data.iloc[:, -30:]
            fig_heat = px.imshow(
                heat_data.values, x=heat_data.columns, y=heat_data.index,
                color_continuous_scale=[[0, "#0a0a0a"], [0.3, "#1a1a1a"], [0.6, "#d4a017"], [1, "#e84545"]],
                aspect="auto",
                labels=dict(x="Date", y="Feature", color="Value"),
            )
            fig_heat.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Not enough data points for heatmap.")

    # ── Cross-Channel Flags for This User ──
    if CROSS_FLAGS:
        section_header("Cross-Channel Risk Indicators", "sh_cross_ind")
        _flag_detail = [
            ("usb_file_activity_flag",      "USB + File Write",   "USB + file write on same day",           "#e84545"),
            ("off_hours_activity_flag",     "Off-Hours Activity", "Outside 9 AM \u2013 5 PM",               "#d4a017"),
            ("external_comm_activity_flag", "External Comms",     "Emails to external domains",             "#3a86a8"),
            ("jobsite_usb_activity_flag",   "Job Site + USB",     "Job-site browsing + USB on same day",    "#9b59b6"),
            ("suspicious_upload_flag",      "Suspicious Upload",  "HTTP upload to suspicious domain",       "#e67e22"),
            ("cloud_upload_flag",           "Cloud Upload",       "HTTP upload to cloud storage",           "#00b4d8"),
            ("non_primary_pc_risk_flag",    "Non-Primary PC",     "Sensitive activity from atypical endpoint", "#7f8c8d"),
        ]
        _total = len(user_data)
        _ucards = ""
        for flag, label, desc, color in _flag_detail:
            if flag in user_data.columns:
                triggered = int(user_data[flag].sum())
                pct = (triggered / max(_total, 1)) * 100
                _ucards += (
                    f"<div style='flex:1;background:#0a0a0a;border:1px solid #1a1a1a;"
                    f"border-left:3px solid {color};padding:14px 18px;min-width:160px;'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#555;"
                    f"text-transform:uppercase;letter-spacing:1.5px;'>{label}</div>"
                    f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:8px;'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:22px;"
                    f"color:{color};font-weight:600;'>{triggered}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
                    f"color:#444;'>/ {_total} days &middot; {pct:.0f}%</span>"
                    f"</div>"
                    f"<div style='font-family:Inter,sans-serif;font-size:11px;color:#333;"
                    f"margin-top:6px;'>{desc}</div>"
                    f"</div>"
                )
        st.markdown(
            f"<div style='display:flex;gap:12px;margin:4px 0 16px 0;flex-wrap:wrap;'>{_ucards}</div>",
            unsafe_allow_html=True,
        )

    # ── Raw Activity Table ──
    section_header("Raw Activity Records", "sh_raw_records")
    display_cols = ["day", "ae_risk_band", "if_anomaly_score", "ae_percentile_rank"] + RAW_FEATURES + CROSS_FLAGS
    display_cols = [c for c in display_cols if c in user_data.columns]
    st.dataframe(
        user_data[display_cols].sort_values("day", ascending=False),
        use_container_width=True, height=350,
    )


# ══════════════════════════════════════════════════════════════
# PAGE: Alerts
# ══════════════════════════════════════════════════════════════

if active_page == "Alerts":
    st.markdown(
        "<div class='page-header-block'>"
        "<h1 class='page-title'>Alerts</h1>"
        "<p class='page-subtitle'>Sortable, filterable list of anomaly detection alerts with behavioral context.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Live-simulation control row ────────────────────────────
    ctrl_start, ctrl_pause, ctrl_right = st.columns([3, 2, 7])
    with ctrl_start:
        if not st.session_state.live_mode:
            if st.button("▶ START LIVE SIMULATION", key="start_live", use_container_width=True):
                # Clear any previous output and stale pause flag
                if os.path.exists(LIVE_OUTPUT):
                    os.remove(LIVE_OUTPUT)
                if os.path.exists(LIVE_PAUSE_FLAG):
                    os.remove(LIVE_PAUSE_FLAG)
                # Launch the unified simulation script as a subprocess
                proc = subprocess.Popen(
                    [sys.executable, LIVE_SIM_SCRIPT, "--interval", "0.5"],
                    cwd=BASE_DIR,
                )
                st.session_state.live_proc = proc
                st.session_state.live_mode = True
                st.session_state.live_paused = False
                st.rerun()
        else:
            if st.button("⏹ STOP LIVE SIMULATION", key="stop_live", use_container_width=True):
                proc = st.session_state.live_proc
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                # Remove pause flag so the process isn't blocked on next start
                if os.path.exists(LIVE_PAUSE_FLAG):
                    os.remove(LIVE_PAUSE_FLAG)
                st.session_state.live_proc = None
                st.session_state.live_mode = False
                st.session_state.live_paused = False
                st.rerun()
    with ctrl_pause:
        if st.session_state.live_mode:
            if not st.session_state.live_paused:
                if st.button("⏸ PAUSE", key="pause_live", use_container_width=True):
                    with open(LIVE_PAUSE_FLAG, "w", encoding="utf-8") as _pf:
                        pass  # existence of the file signals pause
                    st.session_state.live_paused = True
                    st.rerun()
            else:
                if st.button("▶ RESUME", key="resume_live", use_container_width=True):
                    if os.path.exists(LIVE_PAUSE_FLAG):
                        os.remove(LIVE_PAUSE_FLAG)
                    st.session_state.live_paused = False
                    st.rerun()

    # ── LIVE mode ─────────────────────────────────────────────
    if st.session_state.live_mode:
        # Check whether the subprocess is still running
        proc = st.session_state.live_proc
        proc_running = proc is not None and proc.poll() is None
        stream_done  = False

        # Read all scored rows emitted so far
        live_rows = []
        if os.path.exists(LIVE_OUTPUT):
            with open(LIVE_OUTPUT, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("_eos"):
                        stream_done = True
                    else:
                        live_rows.append(obj)

        # Status strip
        if st.session_state.live_paused:
            _status_color = "#f5a623"
            _status_label = "PAUSED"
        elif proc_running:
            _status_color = "#3c9"
            _status_label = "RUNNING"
        else:
            _status_color = "#e84545"
            _status_label = "COMPLETE" if stream_done else "STOPPED"
        with ctrl_right:
            st.markdown(
                f"<span style='font-family:JetBrains Mono,monospace; font-size:11px; "
                f"color:{_status_color}; letter-spacing:1.5px;'>● {_status_label}</span>"
                f"<span style='font-family:JetBrains Mono,monospace; font-size:10px; "
                f"color:#555; margin-left:16px;'>{len(live_rows):,} rows received</span>",
                unsafe_allow_html=True,
            )

        if live_rows:
            live_df = pd.DataFrame(live_rows)
            # Keep only the columns users care about; drop duplicate/diagnostic fields.
            live_df = live_df.drop(columns=[c for c in ("day", "_score_ms") if c in live_df.columns])

            # Most-recent rows first using original CERT timestamp when available.
            if "cert_timestamp" in live_df.columns:
                live_df["_sort_ts"] = pd.to_datetime(live_df["cert_timestamp"], errors="coerce")
                _sort_cols = ["_sort_ts"]
                _sort_dirs = [False]
                if "event_index" in live_df.columns:
                    # Tie-break equal timestamps by latest arrival first.
                    _sort_cols.append("event_index")
                    _sort_dirs.append(False)
                elif "if_percentile_rank" in live_df.columns:
                    _sort_cols.append("if_percentile_rank")
                    _sort_dirs.append(False)
                live_df = live_df.sort_values(by=_sort_cols, ascending=_sort_dirs, kind="stable")
                live_df = live_df.drop(columns=[c for c in ("_sort_ts", "event_index") if c in live_df.columns])
            else:
                # Fallback for older payloads that do not include source timestamps.
                live_df = live_df.iloc[::-1]
                live_df = live_df.drop(columns=[c for c in ("event_index",) if c in live_df.columns])

            # Cap at 500 to keep the table snappy
            live_df = live_df.head(500).reset_index(drop=True)

            # Column config for the live table
            _live_col_cfg = {
                "cert_timestamp": st.column_config.TextColumn("CERT Timestamp"),
                "risk_level":     st.column_config.TextColumn("Risk Level"),
                "anomaly_score":  st.column_config.NumberColumn("Anomaly Score", format="%.6f"),
                "ae_percentile_rank":st.column_config.ProgressColumn("Percentile", min_value=0, max_value=100, format="%.1f"),
            }
            st.dataframe(live_df, use_container_width=True, height=500, column_config=_live_col_cfg)

            st.download_button(
                "EXPORT LIVE ALERTS",
                data=live_df.to_csv(index=False).encode("utf-8"),
                file_name="live_alerts.csv",
                mime="text/csv",
            )

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

            # ── Live charts (based solely on current live alerts table data) ──
            live_chart_df = live_df.copy()

            col_live_left, col_live_right = st.columns(2)

            with col_live_left:
                section_header("Risk Distribution", "sh_live_risk_dist")
                if "if_risk_band" in live_chart_df.columns and not live_chart_df.empty:
                    _risk_counts = (
                        live_chart_df["if_risk_band"]
                        .fillna("LOW")
                        .value_counts()
                        .rename_axis("Risk Level")
                        .reset_index(name="Count")
                    )
                    fig_live_donut = px.pie(
                        _risk_counts,
                        values="Count",
                        names="Risk Level",
                        color="Risk Level",
                        color_discrete_map=RISK_COLORS,
                        hole=0.6,
                    )
                    fig_live_donut.update_layout(
                        **PLOTLY_LAYOUT,
                        showlegend=True,
                        height=340,
                        legend=dict(font=dict(size=10, family="JetBrains Mono")),
                    )
                    fig_live_donut.update_traces(
                        textinfo="label+percent",
                        textfont_size=11,
                        textfont_family="JetBrains Mono",
                    )
                    st.plotly_chart(fig_live_donut, use_container_width=True)
                else:
                    st.info("Risk band data is not available in live alerts yet.")

            with col_live_right:
                section_header("Anomaly Score Distribution", "sh_live_score_dist")
                if "if_anomaly_score" in live_chart_df.columns and "if_risk_band" in live_chart_df.columns:
                    fig_live_hist = px.histogram(
                        live_chart_df,
                        x="if_anomaly_score",
                        nbins=50,
                        color="if_risk_band",
                        color_discrete_map=RISK_COLORS,
                        labels={"if_anomaly_score": "Anomaly Score", "if_risk_band": "Risk Level"},
                    )
                    fig_live_hist.update_layout(**PLOTLY_LAYOUT, height=340, barmode="overlay")
                    fig_live_hist.update_traces(opacity=0.75)
                    st.plotly_chart(fig_live_hist, use_container_width=True)
                else:
                    st.info("Anomaly-score or risk-band fields are not available in live alerts yet.")
        else:
            st.info("Waiting for first scored row… (models are loading)")

        # Auto-refresh while the process is running and not paused
        if proc_running and not st.session_state.live_paused:
            time.sleep(1)
            st.rerun()
        elif stream_done and st.session_state.live_mode:
            # Natural end-of-stream: auto-stop
            st.session_state.live_proc = None
            st.session_state.live_mode = False
            st.success("Simulation complete — all test rows processed.")

    # ── STATIC mode ───────────────────────────────────────────
    else:
        _filter_bar("al_flt")
        filtered_df = _get_filtered_df()

        # ── Critical Alert Notice ──────────────────────────────────
        _high_df = filtered_df[filtered_df["ae_risk_band"].isin(["CRITICAL", "HIGH"])]
        _high_user_count = _high_df["user"].nunique()
        _high_record_count = len(_high_df)

        if _high_user_count > 0:
            _top_alert_users = (
                _high_df.groupby("user", observed=True)
                .agg(
                    peak_pct=("ae_percentile_rank", "max") if "ae_percentile_rank" in _high_df.columns else ("if_anomaly_score", "max"),
                    high_days=("ae_risk_band", "count"),
                )
                .sort_values("peak_pct", ascending=False)
                .head(5)
                .reset_index()
            )

            _user_pills = ""
            for _, _row in _top_alert_users.iterrows():
                _user_pills += (
                    f"<div class='alert-notice-row-item'>"
                    f"<span class='u-id'>{_row['user']}</span>"
                    f"<span class='u-pct'>P{_row['peak_pct']:.0f}</span>"
                    f"<span class='u-days'>{int(_row['high_days'])} critical/high-risk day{'s' if _row['high_days'] != 1 else ''}</span>"
                    f"</div>"
                )

            _notice_html = (
                "<div class='alert-notice-banner'>"
                "<div class='alert-notice-header'>"
                f"<span class='alert-notice-title'>Active Alerts Requiring Immediate Attention</span>"
                f"<span class='alert-notice-count'>{_high_user_count:,} critical/high-risk user{'s' if _high_user_count != 1 else ''} &nbsp;&middot;&nbsp; {_high_record_count:,} flagged record{'s' if _high_record_count != 1 else ''}</span>"
                "</div>"
                f"<div class='alert-notice-rows'>{_user_pills}</div>"
                "</div>"
            )
            st.markdown(_notice_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

        # ── Top 10 Riskiest Users in Alerts ──
        section_header("Top 10 Riskiest Users", "sh_top_users")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:12px;color:#555;margin:0 0 12px 0;'>"
            "Click a user to open their investigation profile.</p>",
            unsafe_allow_html=True,
        )
        top_users = (
            filtered_df.groupby("user", observed=True)
            .agg(
                max_percentile=("ae_percentile_rank", "max"),
                critical_count=("ae_risk_band", lambda x: (x == "CRITICAL").sum()),
                high_count=("ae_risk_band", lambda x: (x == "HIGH").sum()),
            )
            .reset_index()
            .sort_values("max_percentile", ascending=False)
            .head(10)
        )

        if top_users.empty:
            st.info("No users available in the current filter range.")
        else:
            for rank, row in enumerate(top_users.itertuples(), start=1):
                uid = row.user
                score = row.max_percentile
                days = int(row.critical_count + row.high_count)
                if score >= 95:
                    badge_color = "#ff1744"
                    badge_label = "CRITICAL"
                elif score >= 90:
                    badge_color = "#e84545"
                    badge_label = "HIGH"
                elif score >= 80:
                    badge_color = "#d4a017"
                    badge_label = "MEDIUM"
                else:
                    badge_color = "#3a86a8"
                    badge_label = "LOW"

                col_rank, col_info, col_btn = st.columns([1, 5, 3])
                with col_rank:
                    st.markdown(
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        f"color:#444;font-weight:600;padding-top:4px;text-align:center;'>#{rank}</div>",
                        unsafe_allow_html=True,
                    )
                with col_info:
                    st.markdown(
                        f"<div style='padding:2px 0 1px 0;'>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        f"color:#e0e0e0;font-weight:600;'>{uid}</span>"
                        f"&nbsp;&nbsp;<span style='background:{badge_color}22;color:{badge_color};font-size:9px;"
                        f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:1px 5px;"
                        f"border:1px solid {badge_color}55;'>{badge_label}</span>"
                        f"<br><span style='font-family:Inter,sans-serif;font-size:10px;"
                        f"color:#555;line-height:1.5;'>"
                        f"Percentile {score:.1f} &middot; {days} high-risk day{'s' if days != 1 else ''}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("Investigate →", key=f"al_top_inv_{uid}", use_container_width=True):
                        st.session_state["inv_user_select"] = uid
                        st.session_state["_nav_request"] = "Investigation"
                        st.rerun()
                st.markdown("<div style='border-bottom:1px solid #111;margin:0;'></div>", unsafe_allow_html=True)

        # Alert severity filter within this tab
        alert_cols = st.columns([2, 2, 2, 6])
        with alert_cols[0]:
            alert_risk = st.multiselect("Severity", RISK_TIERS, default=["CRITICAL", "HIGH", "MEDIUM"], key="alert_sev")
        with alert_cols[1]:
            min_pctl = st.slider("Min Percentile", 0.0, 100.0, 0.0, key="min_pctl")
        with alert_cols[2]:
            max_results = st.number_input("Max Rows", min_value=10, max_value=10000, value=500, step=50, key="max_rows")

        alert_data = filtered_df[
            (filtered_df["ae_risk_band"].isin(alert_risk)) &
            (filtered_df["ae_percentile_rank"] >= min_pctl)
        ].sort_values("ae_percentile_rank", ascending=False).head(int(max_results))

        # Cap card rendering to keep the UI responsive
        CARD_LIMIT = 100
        total_alerts = len(alert_data)
        card_data = alert_data.head(CARD_LIMIT)

        if total_alerts == 0:
            st.info("No alerts match the current filters.")
        else:
            if total_alerts > CARD_LIMIT:
                st.caption(
                    f"Displaying top {CARD_LIMIT} of {total_alerts:,} matching alerts. "
                    "Use the export button below for the complete set."
                )

            # ── Column header row ──
            st.markdown(
                "<div style='display:grid;grid-template-columns:72px 1fr 108px 90px 130px;"
                "gap:8px;padding:6px 4px;border-bottom:1px solid #1a1a1a;margin-bottom:2px;'>"
                "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                "text-transform:uppercase;letter-spacing:1.5px;'>Risk</span>"
                "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                "text-transform:uppercase;letter-spacing:1.5px;'>User / Investigation hint</span>"
                "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                "text-transform:uppercase;letter-spacing:1.5px;'>Day</span>"
                "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                "text-transform:uppercase;letter-spacing:1.5px;'>Percentile</span>"
                "<span></span>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Per-alert card rows ──
            for i, row in enumerate(card_data.itertuples()):
                risk    = getattr(row, "ae_risk_band",    "LOW")
                user    = getattr(row, "user",           "—")
                day_val = getattr(row, "day",            None)
                day_str = day_val.strftime("%Y-%m-%d") if hasattr(day_val, "strftime") else str(day_val)
                pctl    = getattr(row, "ae_percentile_rank", 0.0)
                top_raw = getattr(row, "top_contributors", None)
                summary = build_alert_summary(top_raw)

                risk_color = RISK_COLORS.get(risk, "#666666")

                c_risk, c_info, c_day, c_pctl, c_btn = st.columns([1, 5, 2, 1, 2])

                with c_risk:
                    st.markdown(
                        f"<div style='padding-top:5px;'>"
                        f"<span style='background:{risk_color}22;color:{risk_color};font-size:10px;"
                        f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:2px 6px;"
                        f"border:1px solid {risk_color}55;display:inline-block;'>{risk}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with c_info:
                    st.markdown(
                        f"<div style='padding:2px 0 4px 0;'>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:13px;"
                        f"color:#e0e0e0;font-weight:600;'>{user}</span>"
                        f"<br><span style='font-family:Inter,sans-serif;font-size:12px;"
                        f"color:#666;line-height:1.5;'>{summary}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with c_day:
                    st.markdown(
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        f"color:#888;padding-top:5px;'>{day_str}</div>",
                        unsafe_allow_html=True,
                    )

                with c_pctl:
                    st.markdown(
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        f"color:{risk_color};padding-top:5px;'>P{pctl:.1f}</div>",
                        unsafe_allow_html=True,
                    )

                with c_btn:
                    if st.button("Investigate →", key=f"al_inv_{i}", use_container_width=True):
                        st.session_state["inv_user_select"] = user
                        st.session_state["_nav_request"] = "Investigation"
                        st.rerun()

                st.markdown(
                    "<div style='border-bottom:1px solid #0d0d0d;margin:2px 0;'></div>",
                    unsafe_allow_html=True,
                )

        # ── Export (columns + top_contributors if present) ──
        alert_display_cols = ["user", "day", "ae_risk_band", "if_anomaly_score", "ae_percentile_rank"]
        alert_display_cols += [c for c in CROSS_FLAGS if c in alert_data.columns]
        if "top_contributors" in alert_data.columns:
            alert_display_cols.append("top_contributors")

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "EXPORT ALERTS",
            data=alert_data[alert_display_cols].to_csv(index=False).encode("utf-8"),
            file_name="insider_threat_alerts.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════
# PAGE: Channels
# ══════════════════════════════════════════════════════════════

if active_page == "Channels":
    st.markdown(
        "<div class='page-header-block'>"
        "<h1 class='page-title'>Channels</h1>"
        "<p class='page-subtitle'>Compare behavioral feature distributions across activity channels for the filtered population.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _filter_bar("ch_flt")
    filtered_df = _get_filtered_df()

    # ── Channel volume over time ──
    ch1, ch2 = st.columns(2)

    with ch1:
        section_header("Channel Activity Volume", "sh_chan_vol")
        channel_ts_df = _channel_time_series(
            st.session_state.flt_date_start,
            st.session_state.flt_date_end,
            tuple(sorted(st.session_state.flt_risk)),
        )
        if not channel_ts_df.empty:
            fig_ch_ts = px.line(channel_ts_df, x="Date", y="Volume", color="Channel",
                                color_discrete_map=CHANNEL_COLOR_MAP)
            fig_ch_ts.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_ch_ts, use_container_width=True)

    with ch2:
        section_header("Channel Volume Share", "sh_chan_share")
        ch_totals = {}
        for channel, feats in CHANNELS.items():
            valid = [f for f in feats if f in filtered_df.columns]
            if valid:
                ch_totals[channel] = filtered_df[valid].sum().sum()
        if ch_totals:
            ch_df = pd.DataFrame(list(ch_totals.items()), columns=["Channel", "Total Events"])
            fig_ch_pie = px.pie(ch_df, values="Total Events", names="Channel",
                                color="Channel", color_discrete_map=CHANNEL_COLOR_MAP,
                                hole=0.6)
            fig_ch_pie.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_ch_pie, use_container_width=True)

    # ── Feature-level box plots ──
    section_header("Feature Distributions by Risk Level", "sh_feat_dist")
    selected_feature = st.selectbox("Select Feature", RAW_FEATURES, key="feat_box")
    if selected_feature in filtered_df.columns:
        # Downsample for box plot — browser can't handle 2M+ points
        box_df = filtered_df if len(filtered_df) <= MAX_PLOT_POINTS else filtered_df.sample(MAX_PLOT_POINTS, random_state=42)
        fig_box = px.box(
            box_df, x="ae_risk_band", y=selected_feature,
            color="ae_risk_band", color_discrete_map=RISK_COLORS,
            category_orders={"ae_risk_band": RISK_TIERS},
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_title="Risk Level", yaxis_title=selected_feature)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation heatmap ──
    section_header("Feature Correlation Matrix", "sh_feat_corr")
    corr_feats = [f for f in RAW_FEATURES if f in merged_df.columns]
    if len(corr_feats) >= 2:
        corr_matrix = _corr_matrix(
            st.session_state.flt_date_start,
            st.session_state.flt_date_end,
            tuple(sorted(st.session_state.flt_risk)),
            tuple(corr_feats),
        )
        fig_corr = px.imshow(
            corr_matrix, x=corr_feats, y=corr_feats,
            color_continuous_scale=[[0, "#3a86a8"], [0.5, "#0a0a0a"], [1, "#e84545"]],
            zmin=-1, zmax=1,
            aspect="auto",
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


# Keep live row counter and status fresh while browsing non-Alerts pages.
if active_page != "Alerts" and st.session_state.live_mode:
    _proc = st.session_state.live_proc
    _proc_running = _proc is not None and _proc.poll() is None
    if _proc_running and not st.session_state.live_paused:
        time.sleep(1)
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Footer — Data & Feature Gaps Note
# ──────────────────────────────────────────────────────────────

st.markdown("---")
