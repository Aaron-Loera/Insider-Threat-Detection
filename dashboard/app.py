import streamlit as st
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
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #1a1a1a;
    }
    [data-testid="stSidebar"] * { border-radius: 0 !important; }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #0a0a0a;
        border-radius: 0;
        padding: 20px 24px;
        border-left: 3px solid;
        border-top: 1px solid #1a1a1a;
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

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 1px solid #1a1a1a; background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 28px;
        border-radius: 0 !important;
        font-weight: 500;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'JetBrains Mono', monospace;
        color: #666666;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ffffff !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #cccccc;
        background: #0a0a0a !important;
    }

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

    /* ── Hide Streamlit branding (keep sidebar toggle visible) ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] { display: none !important; }

    /* Header: keep visible for sidebar toggle */
    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }

    /* ── Sidebar toggle buttons (Streamlit 1.54+) ── */
    /* Expand button (visible when sidebar is collapsed) */
    [data-testid="stExpandSidebarButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        background-color: #0a0a0a !important;
        border: 1px solid #333 !important;
        border-radius: 0 !important;
        color: #ffffff !important;
    }
    [data-testid="stExpandSidebarButton"]:hover {
        background-color: #1a1a1a !important;
    }
    [data-testid="stExpandSidebarButton"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }
    /* Collapse button (inside sidebar) */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }

    /* ── Remove all border-radius globally ── */
    div[data-testid] { border-radius: 0 !important; }
    .element-container { border-radius: 0 !important; }

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
        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
            font-size: 11px;
            letter-spacing: 1px;
        }
    }

    /* ── Responsive: Small screens (768–1023px) ── */
    @media (max-width: 1023px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .kpi-card { padding: 14px 14px; min-height: 100px; }
        .kpi-card h1 { font-size: 20px; }
        .kpi-card h3 { font-size: 9px; letter-spacing: 0.8px; }
        .kpi-card p  { font-size: 9px; }
        .section-header { font-size: 11px; margin: 20px 0 10px 0; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 10px;
            letter-spacing: 0.8px;
        }
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
        .stTabs [data-baseweb="tab"] {
            padding: 8px 8px;
            font-size: 9px;
            letter-spacing: 0.5px;
        }
        [data-testid="stMetricValue"] { font-size: 18px; }
        [data-testid="stMetricLabel"] { font-size: 9px !important; }
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

with st.sidebar:
    # ── DSK Team Logo ──
    st.markdown(
        "<div style='padding:4px 0 16px 0; border-bottom:1px solid #1a1a1a; margin-bottom:20px; text-align:center;'>"
        # Cat silhouette icon (SVG)
        "<div style='margin-bottom:10px;'>"
        "<svg width='80' height='80' viewBox='0 0 100 100' fill='none' xmlns='http://www.w3.org/2000/svg'>"
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
        "</div>"
        # Team name
        "<div style='font-family:JetBrains Mono,monospace; font-size:18px; letter-spacing:6px; "
        "color:#ffffff; font-weight:700;'>DSK</div>"
        "<div style='font-family:JetBrains Mono,monospace; font-size:10px; letter-spacing:2px; "
        "color:#555555; text-transform:uppercase; margin-top:4px;'>Data Structure Kittens</div>"
        "<div style='margin-top:12px; font-family:JetBrains Mono,monospace; font-size:10px; "
        "letter-spacing:1.5px; color:#666; text-transform:uppercase;'>UEBA Threat Detection</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<p style='font-family:JetBrains Mono,monospace; font-size:10px; color:#555; "
                "text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;'>Date Range</p>",
                unsafe_allow_html=True)
    min_date = merged_df["day"].min().date()
    max_date = merged_df["day"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
    )

    st.markdown("<p style='font-family:JetBrains Mono,monospace; font-size:10px; color:#555; "
                "text-transform:uppercase; letter-spacing:2px; margin:16px 0 4px 0;'>Risk Levels</p>",
                unsafe_allow_html=True)
    risk_filter = st.multiselect(
        "Risk Levels",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='border-top:1px solid #1a1a1a; margin:20px 0;'></div>", unsafe_allow_html=True)

    st.markdown("<p style='font-family:JetBrains Mono,monospace; font-size:10px; color:#555; "
                "text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;'>Search User</p>",
                unsafe_allow_html=True)
    all_users = sorted(merged_df["user"].unique())
    user_search = st.text_input("Search User ID", placeholder="e.g. acm2278", label_visibility="collapsed")

    st.markdown("<div style='border-top:1px solid #1a1a1a; margin:20px 0;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace; font-size:9px; color:#333; "
        "text-transform:uppercase; letter-spacing:1.5px; line-height:1.8;'>"
        "DSK &mdash; Data Structure Kittens<br>Senior Design Project &middot; 2026</div>",
        unsafe_allow_html=True,
    )

# Apply global filters
mask = merged_df["risk_levels"].isin(risk_filter)

if isinstance(date_range, tuple) and len(date_range) == 2:
    mask &= (merged_df["day"].dt.date >= date_range[0]) & (merged_df["day"].dt.date <= date_range[1])

filtered_df = merged_df[mask]


# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────

st.markdown(
    "<div style='margin-bottom:4px;'>"
    "<h1 style='margin:0; font-family:Inter,sans-serif; font-weight:700; font-size:28px; "
    "color:#ffffff; letter-spacing:-0.5px;'>INSIDER THREAT DETECTION</h1>"
    "<p style='color:#444444; margin:4px 0 0 0; font-family:JetBrains Mono,monospace; "
    "font-size:11px; text-transform:uppercase; letter-spacing:2px;'>"
    "User &amp; Entity Behavior Analytics</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Cap data points sent to Plotly — browser rendering is the main bottleneck
MAX_PLOT_POINTS = 50_000


# ──────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────

tab_overview, tab_investigate, tab_alerts, tab_channels = st.tabs([
    "OVERVIEW", "INVESTIGATION", "ALERTS", "CHANNELS"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════

with tab_overview:

    # ── KPI Row ──
    total_users = filtered_df["user"].nunique()
    total_records = len(filtered_df)
    high_risk_users = filtered_df[filtered_df["risk_levels"] == "HIGH"]["user"].nunique()
    medium_risk_users = filtered_df[filtered_df["risk_levels"] == "MEDIUM"]["user"].nunique()
    avg_anomaly = filtered_df["anomaly_scores"].mean()
    detection_rate = (filtered_df["risk_levels"].isin(["HIGH", "MEDIUM"]).sum() / max(len(filtered_df), 1)) * 100

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(
            f"""<div class='kpi-card' style='border-color:#ffffff'>
            <h3>Users Monitored</h3><h1 style='color:#ffffff'>{total_users:,}</h1>
            <p>Active in period</p></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"""<div class='kpi-card' style='border-color:#e84545'>
            <h3>High Risk</h3><h1 style='color:#e84545'>{high_risk_users}</h1>
            <p>&ge; 95th percentile</p></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(
            f"""<div class='kpi-card' style='border-color:#d4a017'>
            <h3>Medium Risk</h3><h1 style='color:#d4a017'>{medium_risk_users}</h1>
            <p>&ge; 80th percentile</p></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(
            f"""<div class='kpi-card' style='border-color:#666666'>
            <h3>Total Records</h3><h1 style='color:#cccccc'>{total_records:,}</h1>
            <p>User-day observations</p></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(
            f"""<div class='kpi-card' style='border-color:#666666'>
            <h3>Avg Anomaly Score</h3><h1 style='color:#cccccc'>{avg_anomaly:.4f}</h1>
            <p>Across all records</p></div>""", unsafe_allow_html=True)
    with k6:
        st.markdown(
            f"""<div class='kpi-card' style='border-color:#666666'>
            <h3>Detection Rate</h3><h1 style='color:#cccccc'>{detection_rate:.1f}%</h1>
            <p>Medium + High alerts</p></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 2: Risk Distribution + Alerts Over Time ──
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("<div class='section-header'>Risk Distribution</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Alert Trend Over Time</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Top 15 Riskiest Users</div>", unsafe_allow_html=True)
        top_users = user_risk.head(15).copy()
        fig_top = px.bar(
            top_users, x="max_percentile", y="user", orientation="h",
            color="high_count", color_continuous_scale=[[0, "#1a1a1a"], [0.5, "#d4a017"], [1, "#e84545"]],
            labels={"max_percentile": "Max Percentile Rank", "user": "User", "high_count": "High-Risk Days"},
        )
        fig_top.update_layout(**PLOTLY_LAYOUT, height=440)
        fig_top.update_yaxes(autorange="reversed", gridcolor="#1a1a1a")
        st.plotly_chart(fig_top, use_container_width=True)

    with col_right2:
        st.markdown("<div class='section-header'>Anomaly Score Distribution</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Cross-Channel Risk Flags (Global)</div>", unsafe_allow_html=True)
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
# TAB 2 — User Investigation
# ══════════════════════════════════════════════════════════════

with tab_investigate:
    st.markdown("<div class='section-header'>User Investigation</div>", unsafe_allow_html=True)

    # User search — admin can type any user ID directly
    inv_col1, inv_col2 = st.columns([2, 4])
    with inv_col1:
        user_input = st.text_input(
            "Search User ID",
            value=user_search.strip() if user_search else "",
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
    st.markdown("<div class='section-header'>Anomaly Score Timeline</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Behavioral Profile (Avg Activity)</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Daily Feature Activity</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Cross-Channel Risk Indicators</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section-header'>Raw Activity Records</div>", unsafe_allow_html=True)
    display_cols = ["day", "risk_levels", "anomaly_scores", "percentile_rank"] + RAW_FEATURES + CROSS_FLAGS
    display_cols = [c for c in display_cols if c in user_data.columns]
    st.dataframe(
        user_data[display_cols].sort_values("day", ascending=False),
        use_container_width=True, height=350,
    )


# ══════════════════════════════════════════════════════════════
# TAB 3 — Alert Feed
# ══════════════════════════════════════════════════════════════

with tab_alerts:
    st.markdown("<div class='section-header'>Alert Feed</div>", unsafe_allow_html=True)
    st.caption("Sortable, filterable table of all anomaly detection alerts. Click column headers to sort.")

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
# TAB 4 — Channel Breakdown
# ══════════════════════════════════════════════════════════════

with tab_channels:
    st.markdown("<div class='section-header'>Activity Channel Breakdown</div>", unsafe_allow_html=True)
    st.caption("Compare behavioral feature distributions across activity channels for the filtered population.")

    # ── Channel volume over time ──
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("<div class='section-header'>Channel Activity Volume</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>Channel Volume Share</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section-header'>Feature Distributions by Risk Level</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section-header'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
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
