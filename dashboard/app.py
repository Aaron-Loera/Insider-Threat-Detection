import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

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
        font-size: 9px;
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
        font-size: 13px;
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
        font-size: 12px;
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
        font-size: 11px;
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
        font-size: 11px;
        color: #cccccc;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .alert-notice-row-item .u-id  { color: #ffffff; font-weight: 600; }
    .alert-notice-row-item .u-pct { color: #e84545; }
    .alert-notice-row-item .u-days { color: #888; font-size: 10px; }

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
        margin: 0; font-size: 11px; color: #666666; font-weight: 500;
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
        margin: 6px 0 0 0; font-size: 11px; color: #555555;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase; letter-spacing: 0.5px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #999999;
        margin: 32px 0 16px 0;
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
        font-size: 24px;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 11px !important;
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
        .kpi-card h3 { font-size: 11px; }
        .kpi-card p  { font-size: 11px; }
    }

    /* ── Responsive: Medium screens (1024–1400px) ── */
    @media (max-width: 1400px) {
        .kpi-card { padding: 16px 18px; min-height: 110px; }
        .kpi-card h1 { font-size: 22px; }
        .kpi-card h3 { font-size: 10px; letter-spacing: 1px; }
        .kpi-card p  { font-size: 10px; }
        .section-header { font-size: 12px; letter-spacing: 1.5px; }
    }

    /* ── Responsive: Small screens (768–1023px) ── */
    @media (max-width: 1023px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .kpi-card { padding: 14px 14px; min-height: 100px; }
        .kpi-card h1 { font-size: 20px; }
        .kpi-card h3 { font-size: 9px; letter-spacing: 0.8px; }
        .kpi-card p  { font-size: 9px; }
        .section-header { font-size: 11px; margin: 20px 0 10px 0; }
        [data-testid="stMetricValue"] { font-size: 20px; }
        [data-testid="stMetricLabel"] { font-size: 10px !important; }
    }

    /* ── Responsive: Extra small (<768px) ── */
    @media (max-width: 767px) {
        .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
        .kpi-card { padding: 12px 12px; min-height: 90px; border-left-width: 2px; }
        .kpi-card h1 { font-size: 18px; }
        .kpi-card h3 { font-size: 8px; }
        .kpi-card p  { font-size: 8px; }
        .section-header { font-size: 10px; letter-spacing: 1px; }
        [data-testid="stMetricValue"] { font-size: 18px; }
        [data-testid="stMetricLabel"] { font-size: 9px !important; }
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

# Prefer Parquet (5-10x faster I/O); fall back to CSV
ANALYST_TABLE_PARQUET = os.path.join(BASE_DIR, "static_dashboards", "table_1.parquet")
ANALYST_TABLE_CSV = os.path.join(BASE_DIR, "static_dashboards", "table_1.csv")
UEBA_PARQUET = os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset.parquet")
UEBA_CSV = os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset.csv")

# Only load columns the dashboard actually uses
UEBA_COLS = [
    "user", "pc", "day",
    "logon_count", "logoff_count", "off_hours_logon",
    "file_open_count", "file_write_count", "file_copy_count",
    "file_delete_count", "unique_files_accessed", "off_hours_files_accessed",
    "usb_insert_count", "usb_remove_count", "off_hours_usb_usage",
    "emails_sent", "unique_recipients", "external_emails",
    "attachements_sent", "off_hours_emails",
    "usb_file_activity_flag", "off_hours_activity_flag", "external_comm_activity_flag",
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

    # Merge
    analyst_cols = ["user", "day", "anomaly_scores", "percentile_rank", "risk_levels"]
    analyst_cols = [c for c in analyst_cols if c in analyst.columns]
    merged = ueba.merge(analyst[analyst_cols], on=["user", "day"], how="inner")

    # Pre-compute per-user risk summary (expensive groupby, do once)
    user_risk = (
        merged.groupby("user")
        .agg(
            max_score=("anomaly_scores", "max"),
            mean_score=("anomaly_scores", "mean"),
            max_percentile=("percentile_rank", "max"),
            alert_days=("day", "nunique"),
            high_count=("risk_levels", lambda x: (x == "HIGH").sum()),
            medium_count=("risk_levels", lambda x: (x == "MEDIUM").sum()),
        )
        .reset_index()
        .sort_values("max_percentile", ascending=False)
    )

    del ueba, analyst
    gc.collect()

    return merged, user_risk


try:
    merged_df, user_risk = load_data()
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
        "1. `CERT_Preprocessing.ipynb` → generates `processed_datasets/ueba_dataset.csv`\n"
        "2. `Autoencoder.ipynb` → trains the encoder model\n"
        "3. `Isolation_Forest.ipynb` → generates anomaly scores\n"
        "4. `Static_Dashboard.ipynb` → generates `static_dashboards/table_1.csv`"
    )
    st.stop()


# ──────────────────────────────────────────────────────────────
# Pre-compute derived data used across tabs
# ──────────────────────────────────────────────────────────────

# Base behavioral feature columns (raw counts)
RAW_FEATURES = [
    "logon_count", "logoff_count", "off_hours_logon",
    "file_open_count", "file_write_count", "file_copy_count",
    "file_delete_count", "unique_files_accessed", "off_hours_files_accessed",
    "usb_insert_count", "usb_remove_count", "off_hours_usb_usage",
    "emails_sent", "unique_recipients", "external_emails",
    "attachements_sent", "off_hours_emails",
]
RAW_FEATURES = [f for f in RAW_FEATURES if f in merged_df.columns]

CROSS_FLAGS = [
    "usb_file_activity_flag", "off_hours_activity_flag", "external_comm_activity_flag",
]
CROSS_FLAGS = [f for f in CROSS_FLAGS if f in merged_df.columns]

# Channel groupings for radar / breakdown charts
CHANNELS = {
    "Authentication": ["logon_count", "logoff_count", "off_hours_logon"],
    "File Access":    ["file_open_count", "file_write_count", "file_copy_count",
                       "file_delete_count", "unique_files_accessed", "off_hours_files_accessed"],
    "Removable Media":["usb_insert_count", "usb_remove_count", "off_hours_usb_usage"],
    "Email":          ["emails_sent", "unique_recipients", "external_emails",
                       "attachements_sent", "off_hours_emails"],
}
# Filter to features actually present
CHANNELS = {k: [f for f in v if f in merged_df.columns] for k, v in CHANNELS.items()}

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

RISK_COLORS = {"HIGH": "#e84545", "MEDIUM": "#d4a017", "LOW": "#3a86a8"}


# ──────────────────────────────────────────────────────────────
# Sidebar — Global Filters
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Navigation pages
# ──────────────────────────────────────────────────────────────

NAV_PAGES = [
    "Overview",
    "Investigation",
    "Alerts",
    "Channels",
]

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
    st.session_state.flt_risk = ["HIGH", "MEDIUM", "LOW"]


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
        ["HIGH", "MEDIUM", "LOW"],
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
            st.session_state.flt_risk = rl if rl else ["HIGH", "MEDIUM", "LOW"]
            st.rerun()
    with reset_col:
        if st.button("Reset", use_container_width=True):
            st.session_state.flt_date_start = _DS_MIN
            st.session_state.flt_date_end = _DS_MAX
            st.session_state.flt_risk = ["HIGH", "MEDIUM", "LOW"]
            st.rerun()


def _get_filtered_df():
    """Return merged_df sliced by current session_state filter values."""
    mask = merged_df["risk_levels"].isin(st.session_state.flt_risk)
    mask &= merged_df["day"].dt.date >= st.session_state.flt_date_start
    mask &= merged_df["day"].dt.date <= st.session_state.flt_date_end
    return merged_df[mask]


_SECTION_INFO = {
    "Risk Distribution": (
        "**Risk Distribution**\n\n"
        "A donut chart showing what proportion of activity records fall into each risk tier:\n\n"
        "- **HIGH** — anomaly score in the top percentile; warrants immediate review\n"
        "- **MEDIUM** — elevated but not critical; worth monitoring\n"
        "- **LOW** — behaviour consistent with typical baseline activity\n\n"
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
        "Click **Investigate →** to jump directly to that user's full behavioural profile."
    ),
    "Anomaly Score Distribution": (
        "**Anomaly Score Distribution**\n\n"
        "A histogram of raw anomaly scores across all records, coloured by risk level.\n\n"
        "- Scores **near 0** indicate behaviour very close to the normal baseline\n"
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
        "- **External Communication** — significant outbound traffic to external endpoints\n\n"
        "These compound flags are stronger indicators of insider threat than any single "
        "channel signal alone."
    ),
    "Anomaly Score Timeline": (
        "**Anomaly Score Timeline**\n\n"
        "A day-by-day line chart of the selected user's anomaly score.\n\n"
        "- A **persistent elevation** suggests consistently unusual behaviour\n"
        "- A **sudden spike** may point to a discrete incident on that date\n\n"
        "Use the date filter to narrow the window and correlate peaks with raw activity records below."
    ),
    "Behavioral Profile (Avg Activity)": (
        "**Behavioral Profile**\n\n"
        "A radar chart comparing the selected user's **average feature values** (solid line) "
        "against the **global population average** (dashed line).\n\n"
        "Each axis represents a behavioural feature (e.g. email volume, logon count, file writes). "
        "Axes where the user extends significantly beyond the population average indicate "
        "dimensions of behaviour worth investigating."
    ),
    "Daily Feature Activity": (
        "**Daily Feature Activity**\n\n"
        "A heatmap of the user's raw feature values over time.\n\n"
        "- Each **row** is one behavioural feature\n"
        "- Each **column** is one day\n"
        "- **Darker cells** = higher-than-usual activity on that day and feature\n\n"
        "This lets you pinpoint exactly which features drove an anomaly spike on a given date."
    ),
    "Cross-Channel Risk Indicators": (
        "**Cross-Channel Risk Indicators**\n\n"
        "A summary of whether this user triggered any multi-channel co-occurrence flags:\n\n"
        "- **USB + File Write** — device use coincided with large file write events\n"
        "- **Off-Hours Activity** — actions occurred outside normal business hours\n"
        "- **External Communication** — outbound connections to external hosts were detected\n\n"
        "Combinations of multiple flags substantially increase the likelihood of an insider threat."
    ),
    "Raw Activity Records": (
        "**Raw Activity Records**\n\n"
        "A full table of every aggregated daily record for the selected user within the "
        "current filter window.\n\n"
        "Each row represents one day and includes all behavioural features (email, file, "
        "HTTP, logon, device activity), the computed anomaly score, and the assigned risk level. "
        "Use this to audit exactly what the model saw on any particular date."
    ),
    "Channel Activity Volume": (
        "**Channel Activity Volume**\n\n"
        "A bar chart showing the total number of events recorded across each data channel "
        "(email, file, HTTP, logon, device) within the selected filters.\n\n"
        "Channels with disproportionately high volumes relative to peers can indicate "
        "a data exfiltration path that warrants deeper investigation."
    ),
    "Channel Volume Share": (
        "**Channel Volume Share**\n\n"
        "A pie chart showing each channel's **percentage share** of all recorded events.\n\n"
        "This gives a quick sense of which channels dominate activity organisationally. "
        "A sudden shift in these proportions between time periods may indicate an attack campaign."
    ),
    "Feature Distributions by Risk Level": (
        "**Feature Distributions by Risk Level**\n\n"
        "Box plots for each numeric feature, grouped by risk level (HIGH / MEDIUM / LOW).\n\n"
        "The box shows the interquartile range (25th–75th percentile); the line inside is the median. "
        "Features where the HIGH box sits far above LOW are the **strongest predictors** "
        "of anomalous behaviour in this dataset."
    ),
    "Feature Correlation Matrix": (
        "**Feature Correlation Matrix**\n\n"
        "A heatmap of Pearson correlations between all numeric features.\n\n"
        "- **+1 (dark red)** — features rise and fall together\n"
        "- **−1 (dark blue)** — features move in opposite directions\n"
        "- **~0** — no linear relationship\n\n"
        "Highly correlated features may be redundant for modelling, while unexpected "
        "correlations can reveal undocumented behavioural patterns."
    ),
}


def section_header(title: str, key: str) -> None:
    """Render a section header with a right-aligned ⓘ info popover."""
    c_title, c_info = st.columns([12, 1])
    c_title.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
    with c_info:
        st.markdown("<div style='padding-top:28px;'>", unsafe_allow_html=True)
        with st.popover("ⓘ", use_container_width=True):
            body = _SECTION_INFO.get(title, "No description available.")
            st.markdown(body)
        st.markdown("</div>", unsafe_allow_html=True)


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

    # ── Critical Alert Notice ──────────────────────────────────
    _high_df = filtered_df[filtered_df["risk_levels"] == "HIGH"]
    _high_user_count = _high_df["user"].nunique()
    _high_record_count = len(_high_df)

    if _high_user_count > 0:
        # Top 5 users by peak percentile (or anomaly score fallback)
        _top_alert_users = (
            _high_df.groupby("user", observed=True)
            .agg(
                peak_pct=("percentile_rank", "max") if "percentile_rank" in _high_df.columns else ("anomaly_scores", "max"),
                high_days=("risk_levels", "count"),
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
                f"<span class='u-days'>{int(_row['high_days'])} high-risk day{'s' if _row['high_days'] != 1 else ''}</span>"
                f"</div>"
            )

        _notice_html = (
            "<div class='alert-notice-banner'>"
            "<div class='alert-notice-header'>"
            f"<span class='alert-notice-title'>Active Alerts Requiring Immediate Attention</span>"
            f"<span class='alert-notice-count'>{_high_user_count:,} high-risk user{'s' if _high_user_count != 1 else ''} &nbsp;&middot;&nbsp; {_high_record_count:,} flagged record{'s' if _high_record_count != 1 else ''}</span>"
            "</div>"
            f"<div class='alert-notice-rows'>{_user_pills}</div>"
            "</div>"
        )
        st.markdown(_notice_html, unsafe_allow_html=True)

        if st.button("View All Alerts \u2192", key="ov_goto_alerts", type="primary"):
            st.session_state._nav_request = "Alerts"
            st.rerun()

        st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    total_users = filtered_df["user"].nunique()
    total_records = len(filtered_df)
    high_risk_users = filtered_df[filtered_df["risk_levels"] == "HIGH"]["user"].nunique()
    medium_risk_users = filtered_df[filtered_df["risk_levels"] == "MEDIUM"]["user"].nunique()
    avg_anomaly = filtered_df["anomaly_scores"].mean()
    detection_rate = (filtered_df["risk_levels"].isin(["HIGH", "MEDIUM"]).sum() / max(len(filtered_df), 1)) * 100

    st.markdown(
        "<div class='kpi-scroll-wrapper'>"
        "<div class='kpi-scroll-arrow' id='kpi-arrow-left'>&#8592;</div>"
        f"<div class='kpi-scroll-row' id='kpi-row'>"
        f"<div class='kpi-card' style='border-color:#ffffff'><h3>Users Monitored</h3><h1 style='color:#ffffff'>{total_users:,}</h1><p>Active in period</p></div>"
        f"<div class='kpi-card' style='border-color:#e84545'><h3>High Risk</h3><h1 style='color:#e84545'>{high_risk_users}</h1><p>&ge; 95th percentile</p></div>"
        f"<div class='kpi-card' style='border-color:#d4a017'><h3>Medium Risk</h3><h1 style='color:#d4a017'>{medium_risk_users}</h1><p>&ge; 80th percentile</p></div>"
        f"<div class='kpi-card' style='border-color:#666666'><h3>Total Records</h3><h1 style='color:#cccccc'>{total_records:,}</h1><p>User-day observations</p></div>"
        f"<div class='kpi-card' style='border-color:#666666'><h3>Avg Anomaly Score</h3><h1 style='color:#cccccc'>{avg_anomaly:.4f}</h1><p>Across all records</p></div>"
        f"<div class='kpi-card' style='border-color:#666666'><h3>Detection Rate</h3><h1 style='color:#cccccc'>{detection_rate:.1f}%</h1><p>Medium + High alerts</p></div>"
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
        risk_counts = filtered_df["risk_levels"].value_counts().reset_index()
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
            filtered_df.groupby([filtered_df["day"].dt.date, "risk_levels"])
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
            "<p style='font-family:Inter,sans-serif;font-size:12px;color:#555;margin:0 0 12px 0;'>"
            "Click a user to open their investigation profile.</p>",
            unsafe_allow_html=True,
        )
        top_users = user_risk.head(10).copy()
        for rank, row in enumerate(top_users.itertuples(), start=1):
            uid = row.user
            score = row.max_percentile
            days = row.high_count
            # Color badge based on score
            if score >= 80:
                badge_color = "#e84545"
                badge_label = "CRITICAL"
            elif score >= 60:
                badge_color = "#d4a017"
                badge_label = "HIGH"
            else:
                badge_color = "#4a9eff"
                badge_label = "ELEVATED"

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
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#e0e0e0;font-weight:600;'>{uid}</span>"
                    f"&nbsp;&nbsp;<span style='background:{badge_color}22;color:{badge_color};font-size:9px;"
                    f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:1px 5px;"
                    f"border:1px solid {badge_color}55;'>{badge_label}</span>"
                    f"<br><span style='font-family:Inter,sans-serif;font-size:10px;color:#555;'>"
                    f"Percentile {score:.1f} &middot; {days} high-risk day{'s' if days != 1 else ''}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("Investigate →", key=f"inv_btn_{uid}", use_container_width=True):
                    st.session_state["inv_user_search"] = uid
                    st.session_state["_nav_request"] = "Investigation"
                    st.rerun()
            st.markdown("<div style='border-bottom:1px solid #111;margin:0;'></div>", unsafe_allow_html=True)

    with col_right2:
        section_header("Anomaly Score Distribution", "sh_score_dist")
        # Sample for histogram to avoid sending 2M+ points to Plotly
        hist_df = filtered_df if len(filtered_df) <= MAX_PLOT_POINTS else filtered_df.sample(MAX_PLOT_POINTS, random_state=42)
        fig_hist = px.histogram(
            hist_df, x="anomaly_scores", nbins=80,
            color="risk_levels", color_discrete_map=RISK_COLORS,
            labels={"anomaly_scores": "Anomaly Score", "risk_levels": "Risk Level"},
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=440, barmode="overlay")
        fig_hist.update_traces(opacity=0.75)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 4: Cross-Channel Risk Flags Summary ──
    if CROSS_FLAGS:
        section_header("Cross-Channel Risk Flags (Global)", "sh_cross_flags")
        flag_labels = {
            "usb_file_activity_flag": "USB + File Write",
            "off_hours_activity_flag": "Off-Hours Activity",
            "external_comm_activity_flag": "External Communication",
        }
        flag_data = []
        for flag in CROSS_FLAGS:
            count = int(filtered_df[flag].sum()) if flag in filtered_df.columns else 0
            flag_data.append({"Flag": flag_labels.get(flag, flag), "Triggered": count})
        flag_df = pd.DataFrame(flag_data)

        fc1, fc2, fc3 = st.columns(3)
        for i, (col, row) in enumerate(zip([fc1, fc2, fc3], flag_data)):
            with col:
                pct = (row["Triggered"] / max(len(filtered_df), 1)) * 100
                col.metric(row["Flag"], f'{row["Triggered"]:,}', f"{pct:.1f}% of records")


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

    inv_col1, inv_col2 = st.columns([2, 4])
    with inv_col1:
        user_input = st.text_input(
            "Search User ID",
            value="",
            placeholder="Type a user ID, e.g. acm2278",
            key="inv_user_search",
        )
    # Find matching users as the admin types
    if user_input:
        query = user_input.lower().strip()
        matches = [u for u in all_users if query in u.lower()]
    else:
        # Default: show top 20 riskiest users
        matches = user_risk["user"].head(20).tolist()

    with inv_col2:
        if matches:
            # Pre-select exact match if found, otherwise first match
            exact = [u for u in matches if u.lower() == (user_input or "").lower().strip()]
            default_idx = matches.index(exact[0]) if exact else 0
            selected_user = st.selectbox(
                "Select from matches",
                matches,
                index=default_idx,
                key="inv_user_select",
            )
        else:
            st.warning(f"No user found matching '{user_input}'")
            st.stop()

    user_data = filtered_df[filtered_df["user"] == selected_user].sort_values("day")

    if user_data.empty:
        st.warning("No data for this user in the current filter range.")
        st.stop()

    # ── User KPI Row ──
    u1, u2, u3, u4, u5 = st.columns(5)
    u_max_score = user_data["anomaly_scores"].max()
    u_max_pctl = user_data["percentile_rank"].max()
    u_high_days = (user_data["risk_levels"] == "HIGH").sum()
    u_med_days = (user_data["risk_levels"] == "MEDIUM").sum()
    u_total_days = len(user_data)

    # Determine overall user risk label
    if u_max_pctl >= 95:
        u_risk_label, u_risk_color = "HIGH", "#ff6b6b"
    elif u_max_pctl >= 80:
        u_risk_label, u_risk_color = "MEDIUM", "#feca57"
    else:
        u_risk_label, u_risk_color = "LOW", "#48dbfb"

    u1.metric("Overall Risk", u_risk_label)
    u2.metric("Peak Percentile", f"{u_max_pctl:.1f}")
    u3.metric("High-Risk Days", u_high_days)
    u4.metric("Medium-Risk Days", u_med_days)
    u5.metric("Days Observed", u_total_days)

    # ── Anomaly Timeline ──
    section_header("Anomaly Score Timeline", "sh_score_timeline")
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=user_data["day"], y=user_data["anomaly_scores"],
        mode="lines+markers", name="Anomaly Score",
        line=dict(color="#ffffff", width=1.5),
        marker=dict(size=4, color="#ffffff"),
    ))
    # Color markers by risk level
    for risk, color in RISK_COLORS.items():
        subset = user_data[user_data["risk_levels"] == risk]
        if not subset.empty:
            fig_timeline.add_trace(go.Scatter(
                x=subset["day"], y=subset["anomaly_scores"],
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
                pop_vals.append(filtered_df[valid_feats].mean().sum())

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
        fc1, fc2, fc3 = st.columns(3)
        flag_details = [
            ("usb_file_activity_flag",        "USB + FILE WRITE",    "USB inserted AND files written on same day"),
            ("off_hours_activity_flag",        "OFF-HOURS ACTIVITY",  "Activity detected outside 9 AM - 5 PM"),
            ("external_comm_activity_flag",    "EXTERNAL COMMS",      "Emails sent to external domains"),
        ]
        for col_w, (flag, label, desc) in zip([fc1, fc2, fc3], flag_details):
            if flag in user_data.columns:
                triggered = int(user_data[flag].sum())
                total = len(user_data)
                col_w.metric(label, f"{triggered} / {total} days")
                col_w.caption(desc)

    # ── Raw Activity Table ──
    section_header("Raw Activity Records", "sh_raw_records")
    display_cols = ["day", "risk_levels", "anomaly_scores", "percentile_rank"] + RAW_FEATURES + CROSS_FLAGS
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
        "<p class='page-subtitle'>Sortable, filterable table of all anomaly detection alerts. Click column headers to sort.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _filter_bar("al_flt")
    filtered_df = _get_filtered_df()

    # Alert severity filter within this tab
    alert_cols = st.columns([2, 2, 2, 6])
    with alert_cols[0]:
        alert_risk = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"], key="alert_sev")
    with alert_cols[1]:
        min_pctl = st.slider("Min Percentile", 0.0, 100.0, 0.0, key="min_pctl")
    with alert_cols[2]:
        max_results = st.number_input("Max Rows", min_value=10, max_value=10000, value=500, step=50, key="max_rows")

    alert_data = filtered_df[
        (filtered_df["risk_levels"].isin(alert_risk)) &
        (filtered_df["percentile_rank"] >= min_pctl)
    ].sort_values("percentile_rank", ascending=False).head(max_results)

    alert_display_cols = ["user", "day", "risk_levels", "anomaly_scores", "percentile_rank"]
    # Add cross-channel flags if present
    alert_display_cols += [c for c in CROSS_FLAGS if c in alert_data.columns]

    st.dataframe(
        alert_data[alert_display_cols],
        use_container_width=True,
        height=500,
        column_config={
            "risk_levels": st.column_config.TextColumn("Risk Level"),
            "anomaly_scores": st.column_config.NumberColumn("Anomaly Score", format="%.6f"),
            "percentile_rank": st.column_config.ProgressColumn("Percentile", min_value=0, max_value=100, format="%.1f"),
        },
    )

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
        channel_ts = []
        for channel, feats in CHANNELS.items():
            valid = [f for f in feats if f in filtered_df.columns]
            if valid:
                daily = filtered_df.groupby(filtered_df["day"].dt.date)[valid].sum().sum(axis=1).reset_index()
                daily.columns = ["Date", "Volume"]
                daily["Channel"] = channel
                channel_ts.append(daily)
        if channel_ts:
            channel_ts_df = pd.concat(channel_ts)
            fig_ch_ts = px.line(channel_ts_df, x="Date", y="Volume", color="Channel",
                                color_discrete_sequence=["#ffffff", "#e84545", "#d4a017", "#3a86a8"])
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
                                color_discrete_sequence=["#ffffff", "#e84545", "#d4a017", "#3a86a8"],
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
            box_df, x="risk_levels", y=selected_feature,
            color="risk_levels", color_discrete_map=RISK_COLORS,
            category_orders={"risk_levels": ["HIGH", "MEDIUM", "LOW"]},
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_title="Risk Level", yaxis_title=selected_feature)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation heatmap ──
    section_header("Feature Correlation Matrix", "sh_feat_corr")
    corr_feats = [f for f in RAW_FEATURES if f in filtered_df.columns]
    if len(corr_feats) >= 2:
        # Correlation converges fast — 50k rows is more than enough
        corr_sample = filtered_df if len(filtered_df) <= MAX_PLOT_POINTS else filtered_df.sample(MAX_PLOT_POINTS, random_state=42)
        corr_matrix = corr_sample[corr_feats].corr()
        fig_corr = px.imshow(
            corr_matrix, x=corr_feats, y=corr_feats,
            color_continuous_scale=[[0, "#3a86a8"], [0.5, "#0a0a0a"], [1, "#e84545"]],
            zmin=-1, zmax=1,
            aspect="auto",
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Footer — Data & Feature Gaps Note
# ──────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("DATA GAPS & RECOMMENDED ENHANCEMENTS"):
    st.markdown("""
    The following features appear on industry UEBA dashboards but are **not yet available** in our current CERT dataset.  
    If we want to bring these in, the preprocessing pipeline would need to be updated:

    | Feature | Why It Matters | Status |
    |---------|---------------|--------|
    | **User Department / Role** | Contextualize risk — a DBA accessing databases is normal, a marketing user is not | Not in dataset |
    | **Incident Response Status** | Track alert lifecycle (New → Investigating → Resolved) | No workflow layer |
    | **IP / Geo-location** | Detect impossible travel or unusual network source | Not in CERT r6.2 |
    | **Data Volume Transferred** | Flag large exfiltrations (MB/GB moved) | Not in current features |
    | **Application Usage** | Which apps a user opened (especially sensitive ones) | Not in CERT r6.2 |
    | **Peer Group Comparison** | Compare user to their department/role peers | No department info |
    | **Historical Baseline Trends** | 30/60/90 day rolling baselines with drift detection | Partially available via rolling deltas |
    | **Alert Acknowledge / Dismiss** | Analyst feedback loop to reduce false positives | No persistence layer |
    
    *Bring these up with the team to decide which are feasible for our scope.*
    """)
