import os
import sys

# Ensure the dashboard directory is importable so `from db import …` resolves the
# same way whether the app is launched by `streamlit run dashboard/app.py` (which
# puts this dir on sys.path[0]) or headlessly via streamlit.testing.v1.AppTest
# (which does not). Idempotent: a no-op once the path is present.
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

sys.stderr.write("[APP] module-level execution started\n")
sys.stderr.flush()
import html as _html_mod
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from db import get_all_dispositions, init_db
from lib.auth import require_auth
from lib.data import ALERT_STATUS_OPTIONS, _on_status_change
from lib.theme import inject_base_css
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Page Config & Custom CSS
# ──────────────────────────────────────────────────────────────

icon_path = Path(__file__).parent / "assets" / "dsk_kitten.png"
page_icon = Image.open(icon_path) if icon_path.exists() else "■"

st.set_page_config(
    page_title="InsiderGuard AI — Insider Threat Detection",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_base_css()


# ──────────────────────────────────────────────────────────────
# Authentication gate (logout → cookie restore → guest → dev bypass → login)
# ──────────────────────────────────────────────────────────────

require_auth()


init_db()

# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
import datetime as _dt
import logging as _auth_log

_auth_log.getLogger("ueba.startup").warning(
    f"[STARTUP] authenticated — reached data-load section at {_dt.datetime.now(_dt.timezone.utc).isoformat()}"
)

# Resolve the project root so config.py (at the root) is importable.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# All path configuration is centralized in config.py.  Per-contributor
# overrides live in paths.local.py (gitignored). See paths.local.example.py.
# Only the live-simulation paths are still referenced inline here; the dataset /
# model paths moved into lib.data along with the loaders.
import logging as _startup_log

# Cached loaders live in lib.data; import them so the module-level bootstrap below
# (which populates the page-facing globals) and the page bodies see the same names.
from lib.data import (
    get_alert_detail as _get_alert_detail,
)
from lib.data import (
    get_feature_groups,
    load_data,
    load_peer_baselines,
    load_ueba_a,
    load_user_profiles,
)

from config import (
    LIVE_OUTPUT,
    LIVE_PAUSE_FLAG,
    LIVE_SIM_SCRIPT,
)

_slog = _startup_log.getLogger("ueba.startup")
try:
    merged_df, user_risk, all_users, _DS_MIN, _DS_MAX = load_data()
    _slog.warning("[STARTUP] load_data() complete")
    ueba_a_df = load_ueba_a()
    peer_baselines_df = load_peer_baselines()
    user_profiles_df = load_user_profiles()
    _slog.warning("[STARTUP] load_ueba_a() complete — DATA_LOADED=True")
    DATA_LOADED = True
except Exception as _load_err:
    import traceback as _tb
    ueba_a_df = None
    peer_baselines_df = None
    user_profiles_df = pd.DataFrame(columns=["user", "department", "role", "supervisor", "role_sensitivity", "employee_name"])
    DATA_LOADED = False
    _LOAD_ERROR = _tb.format_exc()
    _slog.warning(f"[STARTUP] load_data FAILED: {_LOAD_ERROR}")
else:
    _LOAD_ERROR = None


# ──────────────────────────────────────────────────────────────
# If data hasn't been generated yet, show instructions
# ──────────────────────────────────────────────────────────────

if not DATA_LOADED:
    st.title("INSIDER THREAT DETECTION")
    st.error("**Failed to load data.** See details below.")
    if _LOAD_ERROR:
        st.code(_LOAD_ERROR, language="text")
    st.stop()


# ──────────────────────────────────────────────────────────────
# Pre-compute derived data used across tabs
# ──────────────────────────────────────────────────────────────

# Feature groups (raw counts, cross-channel flags, channel map) are derived from
# the loaded columns by lib.data.get_feature_groups(). The page bodies reference
# the RAW_FEATURES / CROSS_FLAGS / CHANNELS names as before.
RAW_FEATURES, CROSS_FLAGS, CHANNELS = get_feature_groups()

# user_risk is now pre-computed inside load_data() and cached

# Theme constants (palette, Plotly defaults, plot caps) and the analyst-facing
# label/summary helpers live in lib.* — imported here so the rest of app.py uses
# the names unchanged. assign_band_from_percentile still comes from ueba.risk.
from lib.labels import (
    build_alert_summary,
    parse_top_contributors,
    parse_top_contributors_with_values,
    prettify_feature_name,
)

# Live-simulation output readers (the polling half of live mode) live in lib.live;
# the simulation subprocess control stays inline below until Phase 7.4.
from lib.live import (
    _cached_live_file_stats,
    _cached_live_rows,
    _get_live_user_data,
)
from lib.theme import (
    CHANNEL_COLOR_MAP,
    PLOTLY_LAYOUT,
    RISK_COLORS,
    RISK_TIERS,
)

from ueba.risk import assign_band_from_percentile

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
if "live_page" not in st.session_state:
    st.session_state.live_page = 0

# Consume any programmatic navigation request NOW — before any widget is
# instantiated — so we can write session_state.nav_page freely.
if st.session_state.get("_nav_request"):
    st.session_state.nav_page = st.session_state.pop("_nav_request")

with st.sidebar:
    # ── InsiderGuard AI Logo ──
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
        "<div style='line-height:1; white-space:nowrap;'>"
        "<div style='font-family:JetBrains Mono,monospace; font-size:13px; letter-spacing:1.5px; "
        "color:#ffffff; font-weight:700;'>InsiderGuard AI</div>"
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
    if _live_session_active:
        _live_rows_received, _stream_done = _cached_live_file_stats()

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

    # ── Signed-in user + logout (fixed to bottom of sidebar) ──
    _signed_in_email = st.session_state.get("auth_user_email", "")
    _email_html = (
        f"<p class='so-email'>{_signed_in_email}</p>"
        if _signed_in_email else ""
    )
    st.markdown(
        f"<div class='signout-panel'>"
        f"{_email_html}"
        f"<a class='so-btn' href='?logout=true' target='_self'>SIGN OUT</a>"
        f"</div>",
        unsafe_allow_html=True,
    )

# all_users, _DS_MIN, _DS_MAX are returned from load_data() — no per-rerun scan needed
# MAX_PLOT_POINTS (the Plotly downsample cap) is imported from lib.theme above.

# ──────────────────────────────────────────────────────────────
# Global filter state (persists across page navigations)
# ──────────────────────────────────────────────────────────────
# Live records can extend past the static dataset maximum once a simulation has
# run; _cached_live_max_date (lib.live) tracks that effective ceiling. The filter
# dialog + flt_* session-state defaults live in lib.filters.
from lib.filters import _filter_bar, _get_filtered_df, _ov_args, init_filter_state, show_filters
from lib.live import _cached_live_max_date

_DS_MIN = merged_df["day"].min().date()
_DS_MAX = merged_df["day"].max().date()
_ds_live_max = _cached_live_max_date(_DS_MAX)
init_filter_state(_DS_MIN, _DS_MAX, _ds_live_max)


# Cached aggregations over the filtered frame live in lib.aggregations; page bodies
# call these names directly. _get_filtered_df / _ov_args / _filter_bar are imported
# from lib.filters above.
from lib.aggregations import (
    _al_top_users,
    _ch_box_sample,
    _ch_totals,
    _channel_time_series,
    _corr_matrix,
    _ov_daily_alerts,
    _ov_flag_counts,
    _ov_histogram_sample,
    _ov_kpis,
    _ov_risk_counts,
    _peer_channel_avgs,
    _pop_channel_avgs,
)

filtered_df = _get_filtered_df()

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


# _filter_bar is imported from lib.filters above.


def _render_card_carousel(cards: list[str], state_key: str, visible_count: int = 4) -> None:
    """Render KPI cards with native previous/next controls."""
    if not cards:
        return

    visible_count = max(1, min(visible_count, len(cards)))
    max_start = max(len(cards) - visible_count, 0)
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    start = min(max(int(st.session_state[state_key]), 0), max_start)

    left_col, card_col, right_col = st.columns([0.45, 8, 0.45], vertical_alignment="center")
    with left_col:
        prev_clicked = st.button("←", key=f"{state_key}_prev", use_container_width=True, disabled=(start == 0))
    with right_col:
        next_clicked = st.button("→", key=f"{state_key}_next", use_container_width=True, disabled=(start == max_start))

    if prev_clicked:
        start = max(0, start - visible_count)
    if next_clicked:
        start = min(max_start, start + visible_count)
    st.session_state[state_key] = start

    with card_col:
        visible_cards = cards[start : start + visible_count]
        cols = st.columns(visible_count)
        for col, card_html in zip(cols, visible_cards):
            col.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

st.markdown("<div class='project-title-badge'>Insider Threat Detection</div>", unsafe_allow_html=True)

# ── Mobile sidebar toggle (injected once into parent document) ──
st.html(
    """
    <script>
    (function(){
        var p = document;
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
            btn.style.display = window.innerWidth <= 768 ? 'flex' : 'none';
        }
        window.addEventListener('resize', show);
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
    """
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

    _kpis = _ov_kpis(*_ov_args())
    total_users        = _kpis["total_users"]
    total_records      = _kpis["total_records"]
    critical_risk_users = _kpis["critical_users"]
    high_risk_users    = _kpis["high_users"]
    medium_risk_users  = _kpis["medium_users"]
    avg_anomaly        = _kpis["avg_anomaly"]
    detection_rate     = _kpis["detection_rate"]

    _overview_kpi_cards = [
        f"<div class='kpi-card' style='border-color:#ffffff'><h3>Users Monitored</h3><h1 style='color:#ffffff'>{total_users:,}</h1><p>Active in period</p></div>",
        f"<div class='kpi-card' style='border-color:{RISK_COLORS['CRITICAL']}'><h3>Critical Risk</h3><h1 style='color:{RISK_COLORS['CRITICAL']}'>{critical_risk_users}</h1><p>&ge; 95th percentile</p></div>",
        f"<div class='kpi-card' style='border-color:#e84545'><h3>High Risk</h3><h1 style='color:#e84545'>{high_risk_users}</h1><p>&ge; 90th percentile</p></div>",
        f"<div class='kpi-card' style='border-color:#d4a017'><h3>Medium Risk</h3><h1 style='color:#d4a017'>{medium_risk_users}</h1><p>&ge; 80th percentile</p></div>",
        f"<div class='kpi-card' style='border-color:#666666'><h3>Total Records</h3><h1 style='color:#cccccc'>{total_records:,}</h1><p>User-day observations</p></div>",
        f"<div class='kpi-card' style='border-color:#666666'><h3>Avg Anomaly Score</h3><h1 style='color:#cccccc'>{avg_anomaly:.4f}</h1><p>Across all records</p></div>",
        f"<div class='kpi-card' style='border-color:#666666'><h3>Detection Rate</h3><h1 style='color:#cccccc'>{detection_rate:.1f}%</h1><p>Medium + High + Critical alerts</p></div>",
    ]
    _render_card_carousel(
        _overview_kpi_cards,
        "overview_kpi_card_start",
        visible_count=4,
    )

    st.markdown("")

    # ── Disposition Breakdown Row ──
    section_header("Alert Dispositions", "sh_alert_disp")
    _all_disps = {(r["user"], r["day"]): r["status"] for r in get_all_dispositions()}
    _day_strs = filtered_df["day"].apply(
        lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d).split("T")[0].split(" ")[0]
    )
    _statuses = [_all_disps.get((u, d), "NEW") for u, d in zip(filtered_df["user"], _day_strs)]
    _status_counts = pd.Series(_statuses).value_counts()
    _disp_total     = len(filtered_df)
    _disp_new       = int(_status_counts.get("NEW", 0))
    _disp_invest    = int(_status_counts.get("INVESTIGATING", 0))
    _disp_resolved  = int(_status_counts.get("RESOLVED", 0))
    _disp_dismissed = int(_status_counts.get("DISMISSED", 0))

    _dc1, _dc2, _dc3, _dc4, _dc5 = st.columns(5)
    _dc1.markdown(f"<div class='kpi-card' style='border-color:#666666'><h3>Total Alerts</h3><h1 style='color:#cccccc'>{_disp_total:,}</h1><p>User-day records</p></div>", unsafe_allow_html=True)
    _dc2.markdown(f"<div class='kpi-card' style='border-color:{RISK_COLORS['CRITICAL']}'><h3>New</h3><h1 style='color:{RISK_COLORS['CRITICAL']}'>{_disp_new:,}</h1><p>Awaiting triage</p></div>", unsafe_allow_html=True)
    _dc3.markdown(f"<div class='kpi-card' style='border-color:#d4a017'><h3>Investigating</h3><h1 style='color:#d4a017'>{_disp_invest:,}</h1><p>In progress</p></div>", unsafe_allow_html=True)
    _dc4.markdown(f"<div class='kpi-card' style='border-color:#22c55e'><h3>Resolved</h3><h1 style='color:#22c55e'>{_disp_resolved:,}</h1><p>Closed — confirmed</p></div>", unsafe_allow_html=True)
    _dc5.markdown(f"<div class='kpi-card' style='border-color:#555555'><h3>Dismissed</h3><h1 style='color:#888888'>{_disp_dismissed:,}</h1><p>Closed — false positive</p></div>", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 2: Risk Distribution + Alerts Over Time ──
    col_left, col_right = st.columns([1, 2])

    with col_left:
        section_header("Risk Distribution", "sh_risk_dist")
        risk_counts = _ov_risk_counts(*_ov_args())
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
        daily_alerts = _ov_daily_alerts(*_ov_args())
        # Stack order: LOW at bottom, CRITICAL at top — reflects true data proportions
        _trend_stack_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        fig_trend = px.bar(
            daily_alerts, x="Date", y="Count", color="Risk Level",
            color_discrete_map=RISK_COLORS,
            category_orders={"Risk Level": _trend_stack_order},
            barmode="stack",
        )
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="", yaxis_title="Alert Count",
                                bargap=0, bargroupgap=0)
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
            # Color badge based on percentile → shared band assignment / palette.
            badge_label = assign_band_from_percentile(score)
            badge_color = RISK_COLORS[badge_label]

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
        hist_df = _ov_histogram_sample(*_ov_args())
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
        _flag_counts_cache = _ov_flag_counts(*_ov_args(), flags=tuple(CROSS_FLAGS))
        _flag_cards = []
        for flag, label, color in _flag_info:
            if flag in CROSS_FLAGS and flag in _flag_counts_cache:
                count, pct = _flag_counts_cache[flag]
                _flag_cards.append(
                    f"<div class='kpi-card' style='border-color:{color}'>"
                    f"<h3>{label}</h3>"
                    f"<h1 style='color:{color}'>{count:,}</h1>"
                    f"<p>{pct:.1f}% of records</p>"
                    f"</div>"
                )
        _render_card_carousel(
            _flag_cards,
            "overview_cross_flag_card_start",
            visible_count=4,
        )


# ══════════════════════════════════════════════════════════════
# Investigation — live-updating fragment
# ══════════════════════════════════════════════════════════════
# All dynamic per-user content lives inside this fragment so that Streamlit
# can rerun it every 2 seconds independently of the main script.  This avoids:
#   • time.sleep() blocking the UI thread
#   • st.stop() cutting off the refresh loop when user_data is temporarily empty
#   • full-page reruns on every tick (only the fragment DOM subtree updates)

@st.fragment(run_every="2s")
def _render_investigation_content() -> None:
    _user: str | None = st.session_state.get("inv_user_select")
    if _user is None:
        return

    _inv_merged, _, _, _, _ = load_data()
    _inv_ueba_a = load_ueba_a()

    # Historical rows — date filter intentionally omitted so that live records
    # (which fall outside the historical date range) are never silently dropped.
    _u_rows = _inv_merged.loc[_inv_merged["user"] == _user].reset_index(drop=True)
    if not _u_rows.empty:
        _u_mask = _u_rows["ae_risk_band"].isin(st.session_state.flt_risk)
        user_data = _u_rows[_u_mask].sort_values("day")
    else:
        user_data = pd.DataFrame()

    # ── Merge live data ─────────────────────────────────────────────────────
    # _inv_live_mode tracks whether the dashboard launched the simulation (for
    # the status banner).  Live data is ALWAYS merged when LIVE_OUTPUT has
    # content — this covers both dashboard-launched and CLI-launched simulations.
    _inv_live_mode = bool(st.session_state.live_mode or st.session_state.live_paused)
    _inv_live_count = 0
    if os.path.exists(LIVE_OUTPUT) and os.path.getsize(LIVE_OUTPUT) > 0:
        _live_u = _get_live_user_data(_user)
        if not _live_u.empty:
            # Apply risk-band filter only — live records are expected to fall
            # outside the historical date range, so the date filter is skipped
            # for live rows to prevent them from being silently dropped.
            _lm = _live_u["ae_risk_band"].isin(st.session_state.flt_risk)
            _live_u = _live_u[_lm].copy()
            if not _live_u.empty:
                # Prefer historical rows for days already in user_data; live rows extend the timeline
                _existing_days = (
                    set(user_data["day"].dt.date) if not user_data.empty else set()
                )
                _new_live = _live_u[~_live_u["day"].dt.date.isin(_existing_days)].copy()
                if not _new_live.empty:
                    user_data = (
                        pd.concat([user_data, _new_live], ignore_index=True)
                        .sort_values("day")
                        .reset_index(drop=True)
                    )
                _inv_live_count = len(_live_u)

    if user_data.empty:
        if _inv_live_mode or (os.path.exists(LIVE_OUTPUT) and os.path.getsize(LIVE_OUTPUT) > 0):
            st.info("No data for this user yet — waiting for live records to arrive.")
        else:
            st.warning("No data for this user in the current filter range.")
        # Return (not st.stop()) so the fragment keeps auto-running and picks
        # up the first live record as soon as it arrives.
        return

    # ── Live investigation status banner ────────────────────────────────────
    if _inv_live_mode or _inv_live_count > 0:
        if not _inv_live_mode:
            # Live data present but simulation wasn't started from this dashboard session
            _inv_live_status = "ACTIVE"
            _inv_live_color = "#3a86a8"
            _inv_live_dot = "●"
        elif st.session_state.live_mode and not st.session_state.live_paused:
            _inv_live_status = "LIVE"
            _inv_live_color = "#e84545"
            _inv_live_dot = "●"
        else:
            _inv_live_status = "PAUSED"
            _inv_live_color = "#d4a017"
            _inv_live_dot = "⏸"
        _inv_live_msg = (
            f"{_inv_live_dot} {_inv_live_status} &mdash; "
            f"{_inv_live_count} live record{'s' if _inv_live_count != 1 else ''} merged into view"
            if _inv_live_count > 0
            else f"{_inv_live_dot} {_inv_live_status} &mdash; awaiting live records for this user"
        )
        st.markdown(
            f"<div style='background:{_inv_live_color}11;border:1px solid {_inv_live_color}33;"
            f"border-left:3px solid {_inv_live_color};padding:6px 14px;margin:0 0 14px 0;"
            f"display:flex;align-items:center;gap:8px;'>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:{_inv_live_color};letter-spacing:1px;'>{_inv_live_msg}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── User Profile Section ──
    _PROFILE_COLS = ["user", "employee_name", "department", "role", "supervisor", "role_sensitivity"]
    _avail = [c for c in _PROFILE_COLS if c in _inv_merged.columns]
    if len(_avail) > 1:
        user_profiles_df = _inv_merged[_avail].drop_duplicates("user").reset_index(drop=True)
    else:
        user_profiles_df = pd.DataFrame(columns=_PROFILE_COLS)
    _prof_row = user_profiles_df[user_profiles_df["user"] == _user]
    _prof = _prof_row.iloc[0] if not _prof_row.empty else None

    _name      = _prof["employee_name"]     if _prof is not None else _user
    _dept      = _prof["department"]        if _prof is not None else "—"
    _role      = _prof["role"]              if _prof is not None else "—"
    _sup_id    = _prof["supervisor"]        if _prof is not None else None
    _rs        = float(_prof["role_sensitivity"]) if _prof is not None else None

    # Resolve supervisor display name (Name · user_id)
    if _sup_id:
        _sup_prof = user_profiles_df[user_profiles_df["user"] == _sup_id]
        _sup_name = _sup_prof.iloc[0]["employee_name"] if not _sup_prof.empty else _sup_id
        _sup_display = f"{_sup_name} &middot; <span style='color:#555;font-size:11px;'>{_sup_id}</span>"
    else:
        _sup_display = "—"

    _all_u_rows = _u_rows
    _total_alerts = len(_all_u_rows)
    _first_alert  = _all_u_rows["day"].min().strftime("%Y-%m-%d") if not _all_u_rows.empty else "—"
    _last_alert   = _all_u_rows["day"].max().strftime("%Y-%m-%d") if not _all_u_rows.empty else "—"

    if _rs is not None:
        if _rs >= 0.85:
            _rs_color, _rs_label = RISK_COLORS["CRITICAL"], f"{_rs:.2f} · Critical"
        elif _rs >= 0.70:
            _rs_color, _rs_label = RISK_COLORS["HIGH"], f"{_rs:.2f} · High"
        elif _rs >= 0.50:
            _rs_color, _rs_label = RISK_COLORS["MEDIUM"], f"{_rs:.2f} · Medium"
        else:
            _rs_color, _rs_label = RISK_COLORS["LOW"], f"{_rs:.2f} · Low"
    else:
        _rs_color, _rs_label = "#555", "—"

    def _prof_field(label: str, value: str, value_color: str = "#e0e0e0") -> str:
        return (
            f"<div style='display:flex;flex-direction:column;gap:2px;'>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:9px;"
            f"text-transform:uppercase;letter-spacing:1.2px;color:#555;'>{label}</span>"
            f"<span style='font-family:Inter,sans-serif;font-size:13px;font-weight:500;"
            f"color:{value_color};'>{value}</span>"
            f"</div>"
        )

    st.markdown(
        "<div style='background:#0a0a0a;border:1px solid #1c1c1c;padding:16px 20px;margin:0 0 18px 0;'>"
        # Name header row
        f"<div style='margin-bottom:14px;border-bottom:1px solid #1c1c1c;padding-bottom:10px;"
        f"display:flex;align-items:baseline;gap:12px;'>"
        f"<span style='font-family:Inter,sans-serif;font-size:16px;font-weight:600;color:#e0e0e0;'>{_name}</span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;color:#555;'>{_user}</span>"
        f"</div>"
        # Fields grid
        "<div style='display:grid;grid-template-columns:repeat(6,1fr);gap:16px;'>"
        + _prof_field("Department", _dept)
        + _prof_field("Role", _role)
        + _prof_field("Supervisor", _sup_display)
        + _prof_field("Role Sensitivity", _rs_label, _rs_color)
        + _prof_field("Total Alerts", f"{_total_alerts:,}")
        + _prof_field("First / Last Alert", f"{_first_alert}&nbsp;&nbsp;→&nbsp;&nbsp;{_last_alert}")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Alert History ──
    _ah_disps = {(r["user"], r["day"]): r["status"] for r in get_all_dispositions()}

    _FEAT_CHANNEL: dict[str, str] = {}
    for _ch, _ch_feats in CHANNELS.items():
        for _f in _ch_feats:
            _FEAT_CHANNEL[_f] = _ch

    _DISP_COLORS = {
        "NEW":           "#3a86a8",
        "INVESTIGATING": "#d4a017",
        "RESOLVED":      "#2ec27e",
        "DISMISSED":     "#555555",
    }

    def _threat_cats(tc_raw) -> list[str]:
        feats = parse_top_contributors(tc_raw)
        seen: set[str] = set()
        cats: list[str] = []
        for f in feats:
            base = f.replace("_zscore", "").replace("_rolling_delta", "")
            ch = _FEAT_CHANNEL.get(f) or _FEAT_CHANNEL.get(base)
            if ch is None and (f in CROSS_FLAGS or base in CROSS_FLAGS):
                ch = "Cross-Channel"
            if ch and ch not in seen:
                seen.add(ch)
                cats.append(ch)
        return cats

    _ah_rows_df = _all_u_rows.sort_values("day", ascending=False).reset_index(drop=True)
    section_header("Alert History", "sh_alert_history")

    _AH_PAGE_SIZE = 10
    _ah_total = len(_ah_rows_df)
    _ah_total_pages = max(1, (_ah_total + _AH_PAGE_SIZE - 1) // _AH_PAGE_SIZE)

    _ah_page_key = f"ah_page_{_user}"
    if _ah_page_key not in st.session_state or st.session_state.get("_ah_last_user") != _user:
        st.session_state[_ah_page_key] = 0
    st.session_state["_ah_last_user"] = _user

    _ah_page = st.session_state[_ah_page_key]
    _ah_slice = _ah_rows_df.iloc[_ah_page * _AH_PAGE_SIZE : (_ah_page + 1) * _AH_PAGE_SIZE]

    def _cat_badge(cat: str) -> str:
        c = CHANNEL_COLOR_MAP.get(cat, "#bb44f0")
        return (
            f"<span style='background:{c}22;color:{c};font-size:9px;"
            f"font-family:JetBrains Mono,monospace;letter-spacing:0.8px;"
            f"padding:1px 5px;border:1px solid {c}55;margin-right:3px;"
            f"white-space:nowrap;display:inline-block;'>{cat}</span>"
        )

    _ah_tbody_parts: list[str] = []
    for _, _ahr in _ah_slice.iterrows():
        _ahr_day = _ahr["day"]
        _ahr_day_str = _ahr_day.strftime("%Y-%m-%d") if hasattr(_ahr_day, "strftime") else str(_ahr_day)
        _ahr_risk = str(_ahr.get("ae_risk_band", "")).upper()
        _ahr_pctl = float(_ahr.get("ae_percentile_rank", 0.0))
        _ahr_rc = RISK_COLORS.get(_ahr_risk, "#666666")
        _ahr_cats = _threat_cats(_ahr.get("top_contributors"))
        _ahr_disp = _ah_disps.get((_ahr["user"], _ahr_day_str), "NEW")
        _ahr_dc = _DISP_COLORS.get(_ahr_disp, "#555555")

        _ahr_cat_html = (
            "".join(_cat_badge(c) for c in _ahr_cats)
            if _ahr_cats
            else "<span style='color:#444;font-family:JetBrains Mono,monospace;font-size:11px;'>—</span>"
        )

        _ah_tbody_parts.append(
            "<tr>"
            f"<td style='font-family:JetBrains Mono,monospace;font-size:11px;color:#aaa;"
            f"padding:8px 16px 8px 0;border-bottom:1px solid #0f0f0f;white-space:nowrap;"
            f"vertical-align:middle;'>{_ahr_day_str}</td>"
            f"<td style='padding:8px 16px 8px 0;border-bottom:1px solid #0f0f0f;"
            f"white-space:nowrap;vertical-align:middle;'>"
            f"<span style='background:{_ahr_rc}22;color:{_ahr_rc};font-size:9px;"
            f"font-family:JetBrains Mono,monospace;letter-spacing:1px;"
            f"padding:2px 6px;border:1px solid {_ahr_rc}55;'>{_ahr_risk}</span></td>"
            f"<td style='font-family:JetBrains Mono,monospace;font-size:11px;color:{_ahr_rc};"
            f"padding:8px 16px 8px 0;border-bottom:1px solid #0f0f0f;white-space:nowrap;"
            f"vertical-align:middle;'>P{_ahr_pctl:.1f}</td>"
            f"<td style='padding:8px 16px 8px 0;border-bottom:1px solid #0f0f0f;"
            f"min-width:200px;vertical-align:middle;'>{_ahr_cat_html}</td>"
            f"<td style='padding:8px 0 8px 0;border-bottom:1px solid #0f0f0f;"
            f"white-space:nowrap;vertical-align:middle;'>"
            f"<span style='background:{_ahr_dc}22;color:{_ahr_dc};font-size:9px;"
            f"font-family:JetBrains Mono,monospace;letter-spacing:1px;"
            f"padding:2px 6px;border:1px solid {_ahr_dc}55;'>{_ahr_disp}</span></td>"
            "</tr>"
        )

    _ah_th = (
        "font-family:JetBrains Mono,monospace;font-size:10px;font-weight:600;"
        "color:#555;text-transform:uppercase;letter-spacing:1.2px;"
        "padding-bottom:10px;padding-right:16px;border-bottom:1px solid #1a1a1a;"
    )
    st.markdown(
        "<div style='background:#0a0a0a;border:1px solid #1c1c1c;padding:14px 18px 10px 18px;"
        "margin:0 0 4px 0;overflow-x:auto;-webkit-overflow-scrolling:touch;'>"
        "<table style='width:100%;border-collapse:collapse;min-width:600px;'>"
        "<thead><tr>"
        f"<th style='{_ah_th}'>Day</th>"
        f"<th style='{_ah_th}'>Risk</th>"
        f"<th style='{_ah_th}'>AE Pctl</th>"
        f"<th style='{_ah_th}'>Threat Categories</th>"
        f"<th style='{_ah_th.replace('padding-right:16px;', 'padding-right:0;')}'>Disposition</th>"
        "</tr></thead>"
        f"<tbody>{''.join(_ah_tbody_parts)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )

    _ah_pg_left, _ah_pg_mid, _ah_pg_right = st.columns([1, 4, 1])
    _ah_start = _ah_page * _AH_PAGE_SIZE + 1
    _ah_end = min(_ah_start + _AH_PAGE_SIZE - 1, _ah_total)
    with _ah_pg_left:
        if st.button("← Previous", key="ah_prev", disabled=(_ah_page == 0), use_container_width=True):
            st.session_state[_ah_page_key] -= 1
            st.rerun()
    with _ah_pg_mid:
        st.markdown(
            f"<div style='text-align:center;font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:#555;padding-top:6px;'>{_ah_start}–{_ah_end} of {_ah_total}</div>",
            unsafe_allow_html=True,
        )
    with _ah_pg_right:
        if st.button("Next →", key="ah_next", disabled=(_ah_page >= _ah_total_pages - 1), use_container_width=True):
            st.session_state[_ah_page_key] += 1
            st.rerun()

    # ── User KPI Row ──
    u1, u2, u3, u4, u5, u6 = st.columns(6)
    u_max_pctl = user_data["ae_percentile_rank"].max()
    u_crit_days = (user_data["ae_risk_band"] == "CRITICAL").sum()
    u_high_days = (user_data["ae_risk_band"] == "HIGH").sum()
    u_med_days = (user_data["ae_risk_band"] == "MEDIUM").sum()
    u_total_days = len(user_data)

    # Determine overall user risk label (shared band assignment)
    u_risk_label = assign_band_from_percentile(u_max_pctl)

    u1.metric("Overall Risk", u_risk_label)
    u2.metric("Peak Percentile", f"{u_max_pctl:.1f}")
    u3.metric("Critical-Risk Days", u_crit_days)
    u4.metric("High-Risk Days", u_high_days)
    u5.metric("Medium-Risk Days", u_med_days)
    u6.metric("Days Observed", u_total_days)

    # ── Alert Context Summary (shown when navigating from Alerts tab) ──
    _alert_ctx = st.session_state.get("inv_alert_context")
    if _alert_ctx and _alert_ctx.get("user") == _user:
        _ctx_risk = _alert_ctx.get("risk", "")
        _ctx_risk_color = RISK_COLORS.get(_ctx_risk, "#666666")
        _ctx_day = _alert_ctx.get("day", "")
        _ctx_pctl = _alert_ctx.get("percentile", 0.0)
        _ctx_summary = _alert_ctx.get("summary") or "Summary unavailable for this alert."
        st.markdown(
            f"<div style='background:#0a0a0a;border:1px solid #1a1a1a;"
            f"border-left:3px solid {_ctx_risk_color};padding:14px 18px;margin:0 0 20px 0;'>"
            "<div class='inv-card-label'>Alert Summary</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#888;margin-bottom:6px;'>"
            f"<span style='background:{_ctx_risk_color}22;color:{_ctx_risk_color};font-size:9px;"
            f"letter-spacing:1px;padding:2px 6px;border:1px solid {_ctx_risk_color}55;"
            f"margin-right:10px;'>{_ctx_risk}</span>"
            f"Day: {_ctx_day}&nbsp;&nbsp;&middot;&nbsp;&nbsp;Percentile: P{_ctx_pctl:.1f}</div>"
            f"<div style='font-family:Inter,sans-serif;font-size:12px;color:#bbb;line-height:1.6;'>"
            f"{_ctx_summary}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Raw Alert Record ──
        _ctx_day_ts = pd.to_datetime(_ctx_day, errors="coerce")
        _raw_row = _inv_merged[
            (_inv_merged["user"] == _user) &
            (_inv_merged["day"] == _ctx_day_ts)
        ]

        _ALERT_RECORD_FIELDS = [
            ("user",               "User"),
            ("day",                "Day"),
            ("risk_levels",        "AE Risk Band"),
            ("ae_percentile_rank",    "AE Percentile"),
            ("anomaly_scores",     "IF Score"),
            ("if_percentile_rank", "IF Percentile"),
            ("if_risk_band",       "IF Risk Band"),
        ]

        def _fmt_alert_val(col, val):
            if col == "day" and hasattr(val, "strftime"):
                return val.strftime("%Y-%m-%d")
            if isinstance(val, str):
                return val
            try:
                fv = float(val)
                return str(int(fv)) if fv == int(fv) else f"{fv:.2f}"
            except (TypeError, ValueError, OverflowError):
                return str(val)

        _BADGE_COLS = {"risk_levels", "if_risk_band"}
        _kv_parts = []
        if not _raw_row.empty:
            _rec0 = _raw_row.iloc[0]
            for _c, _l in _ALERT_RECORD_FIELDS:
                if _c not in _raw_row.columns:
                    continue
                _v = _rec0[_c]
                if not isinstance(_v, str) and pd.isnull(_v):
                    continue
                _val_str = _fmt_alert_val(_c, _v)
                if _c in _BADGE_COLS:
                    _bc = RISK_COLORS.get(str(_val_str).upper(), "#666666")
                    _val_html = (
                        f"<span style='background:{_bc}22;color:{_bc};font-size:10px;"
                        f"font-family:JetBrains Mono,monospace;letter-spacing:1px;"
                        f"padding:2px 8px;border:1px solid {_bc}55;'>{_val_str}</span>"
                    )
                elif _c == "explanation":
                    _val_html = (
                        f"<span style='font-family:Inter,sans-serif;font-size:12px;"
                        f"color:#bbb;line-height:1.6;'>{_val_str}</span>"
                    )
                else:
                    _val_html = (
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        f"color:#ccc;'>{_val_str}</span>"
                    )
                _kv_parts.append(
                    f"<span class='inv-field-label'>{_l}</span>"
                    f"<span style='padding:5px 0 3px 0;align-self:center;'>{_val_html}</span>"
                )

        _kv_grid = "".join(_kv_parts) if _kv_parts else (
            "<span style='font-family:Inter,sans-serif;font-size:12px;color:#666;"
            "grid-column:1/-1;'>Raw alert record unavailable.</span>"
        )
        st.markdown(
            f"<div style='background:#0a0a0a;border:1px solid #1a1a1a;"
            f"border-left:3px solid {_ctx_risk_color};padding:14px 18px;margin:0 0 12px 0;'>"
            "<div class='inv-card-label'>Raw Alert Record</div>"
            f"<div style='display:grid;grid-template-columns:140px 1fr;gap:2px 16px;'>{_kv_grid}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Top Reconstruction Error Contributors ──
        _tc_raw = _get_alert_detail(_user, _ctx_day_ts, "top_contributors")
        _tc_pairs = parse_top_contributors_with_values(_tc_raw)

        if _tc_pairs:
            _tc_table_rows = []
            _seen_feats: set = set()
            for _feat, _contrib_val in _tc_pairs:
                # Deduplicate on the original feature name only — not on the
                # resolved base column. Two contributors like
                # file_copy_count_zscore and file_copy_count_rolling_delta
                # are distinct contributors even if they share a base feature.
                if _feat in _seen_feats:
                    continue
                _seen_feats.add(_feat)
                try:
                    _rv_str = f"{float(_contrib_val) * 100:.1f}%" if _contrib_val is not None else "—"
                except (TypeError, ValueError):
                    _rv_str = "—"
                _tc_table_rows.append((_feat, prettify_feature_name(_feat), _rv_str))

            if _tc_table_rows:
                _tc_tbody_html = "".join(
                    f"<tr>"
                    f"<td class='inv-feat-name'>{_f}</td>"
                    f"<td style='font-family:Inter,sans-serif;font-size:12px;color:#999;"
                    f"padding:8px 16px 8px 0;text-align:left;border-bottom:1px solid #111;"
                    f"vertical-align:middle;'>{_l}</td>"
                    f"<td style='font-family:JetBrains Mono,monospace;font-size:12px;color:#e0e0e0;"
                    f"padding:8px 0;border-bottom:1px solid #111;vertical-align:middle;"
                    f"font-weight:600;text-align:right;white-space:nowrap;'>{_r}</td>"
                    f"</tr>"
                    for _f, _l, _r in _tc_table_rows
                )
                st.markdown(
                    f"<div style='background:#0a0a0a;border:1px solid #1a1a1a;"
                    f"border-left:3px solid {_ctx_risk_color};padding:14px 18px;margin:0 0 20px 0;'>"
                    "<div class='inv-card-label'>Top Reconstruction Error Contributors</div>"
                    "<table style='width:100%;border-collapse:collapse;'>"
                    "<thead><tr>"
                    "<th class='inv-th' style='width:44%;text-align:left;'>Feature</th>"
                    "<th class='inv-th' style='width:38%;text-align:left;padding-left:0;'>Description</th>"
                    "<th class='inv-th' style='width:18%;text-align:right;padding-right:0;white-space:nowrap;'>Error Contribution</th>"
                    "</tr></thead>"
                    f"<tbody>{_tc_tbody_html}</tbody>"
                    "</table></div>",
                    unsafe_allow_html=True,
                )

        # ── PC-Level Drill-Down (UEBA Table A) ──
        if _inv_ueba_a is not None:
            _drill = _inv_ueba_a[
                (_inv_ueba_a["user"] == _user) &
                (_inv_ueba_a["day"] == _ctx_day_ts)
            ].copy()
            _drop_cols = [c for c in ("user", "day") if c in _drill.columns]
            if _drop_cols:
                _drill = _drill.drop(columns=_drop_cols)
            if "pc" in _drill.columns:
                _drill = _drill.sort_values("pc").reset_index(drop=True)
            else:
                _drill = _drill.reset_index(drop=True)

            if _drill.empty:
                st.markdown(
                    f"<div style='background:#0a0a0a;border:1px solid #1a1a1a;"
                    f"border-left:3px solid {_ctx_risk_color};padding:14px 18px;margin:0 0 20px 0;'>"
                    "<div class='inv-card-label'>PC-Level Drill-Down &mdash; UEBA Table A</div>"
                    "<div style='font-family:Inter,sans-serif;font-size:11px;color:#555;'>"
                    "No PC-level UEBA Table A rows found for this user/day.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                _drill_display = _drill.drop(columns=["pc"], errors="ignore")
                _drill_cols = list(_drill_display.columns)
                _drill_last = _drill_cols[-1] if _drill_cols else None
                _drill_th_html = "".join(
                    f"<th style='font-family:JetBrains Mono,monospace;font-size:11px;font-weight:600;"
                    f"color:#aaaaaa;text-align:left;text-transform:uppercase;letter-spacing:1.2px;"
                    f"white-space:nowrap;padding-top:0;padding-bottom:12px;"
                    f"padding-right:{'0' if col == _drill_last else '20px'};"
                    f"padding-left:{'0' if col == _drill_cols[0] else '20px'};"
                    f"border-bottom:1px solid #1a1a1a;"
                    f"border-right:{'none' if col == _drill_last else '1px solid #2e2e2e'};"
                    f"min-width:130px;'>{col.replace('_', ' ')}</th>"
                    for col in _drill_cols
                )
                _drill_tbody_html = "".join(
                    "<tr>" + "".join(
                        f"<td style='font-family:JetBrains Mono,monospace;font-size:12px;color:#cccccc;"
                        f"padding-top:10px;padding-bottom:10px;"
                        f"padding-right:{'0' if col == _drill_last else '20px'};"
                        f"padding-left:{'0' if col == _drill_cols[0] else '20px'};"
                        f"border-bottom:1px solid #111;vertical-align:middle;white-space:nowrap;"
                        f"border-right:{'none' if col == _drill_last else '1px solid #2e2e2e'};"
                        f"min-width:130px;'>{_fmt_alert_val(col, row[col])}</td>"
                        for col in _drill_cols
                    ) + "</tr>"
                    for _, row in _drill_display.iterrows()
                )
                st.markdown(
                    f"<div style='background:#0a0a0a;border:1px solid #1a1a1a;"
                    f"border-left:3px solid {_ctx_risk_color};padding:14px 18px 18px 18px;margin:0 0 20px 0;'>"
                    "<div class='inv-card-label'>PC-Level Drill-Down &mdash; UEBA Table A</div>"
                    "<div style='overflow-x:auto;-webkit-overflow-scrolling:touch;'>"
                    "<table style='border-collapse:collapse;width:max-content;min-width:100%;'>"
                    f"<thead><tr>{_drill_th_html}</tr></thead>"
                    f"<tbody>{_drill_tbody_html}</tbody>"
                    "</table></div></div>",
                    unsafe_allow_html=True,
                )

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

        # Compute per-channel averages: user vs dept peer group vs global population
        radar_categories = []
        user_vals = []
        peer_vals = []
        pop_vals = []

        user_dept = (
            user_data["department"].iloc[0]
            if "department" in user_data.columns and len(user_data) > 0
            else None
        )

        day_min = user_data["day"].min() if "day" in user_data.columns else None
        day_max = user_data["day"].max() if "day" in user_data.columns else None

        peer_avgs = _peer_channel_avgs(
            user_dept,
            day_min=day_min,
            day_max=day_max,
        ) if user_dept else {}

        for channel, feats in CHANNELS.items():
            valid_feats = [f for f in feats if f in user_data.columns]
            if valid_feats:
                radar_categories.append(channel)
                user_vals.append(user_data[valid_feats].mean().sum())
                peer_vals.append(peer_avgs.get(channel, 0.0))
                pop_vals.append(_pop_channel_avgs().get(channel, 0.0))

        if radar_categories:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=user_vals, theta=radar_categories, fill="toself",
                name=_user, line=dict(color="#e84545", width=2),
                fillcolor="rgba(232,69,69,0.15)",
        ))

        if any(v > 0 for v in peer_vals):
            fig_radar.add_trace(go.Scatterpolar(
                r=peer_vals,
                theta=radar_categories,
                fill="toself",
                name=f"Dept Avg ({user_dept})",
                line=dict(color="#d4a017", width=2),
                fillcolor="rgba(212,160,23,0.12)",
            ))

        fig_radar.add_trace(go.Scatterpolar(
            r=pop_vals,
            theta=radar_categories,
            fill="toself",
            name="Population Avg",
            line=dict(color="#3a86a8", width=1),
            opacity=0.6,
            fillcolor="rgba(58,134,168,0.1)",
        ))

        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            polar=dict(
                bgcolor="#0a0a0a",
                radialaxis=dict(visible=True, color="#333333"),
                angularaxis=dict(color="#444444"),
            ),
            showlegend=True,
        )
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
    # Auto-refresh is handled by the fragment's run_every="2s" — no time.sleep needed.


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
    else:
        _render_investigation_content()


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
    # Detect Streamlit Cloud: the repo is mounted at /mount/src/ on the cloud runner.
    # On Streamlit Cloud use live_replay.py (no tensorflow/joblib needed).
    # Locally use live_simulation.py (full ML scoring pipeline).
    _ON_CLOUD = os.path.exists("/mount/src")
    _LIVE_SCRIPT = (
        os.path.join(BASE_DIR, "live_replay.py")
        if _ON_CLOUD
        else LIVE_SIM_SCRIPT
    )
    ctrl_start, ctrl_pause = st.columns([3, 2])
    with ctrl_start:
        if not st.session_state.live_mode:
            if st.button("▶ START LIVE SIMULATION", key="start_live", use_container_width=True):
                # Clear any previous output and stale pause flag
                if os.path.exists(LIVE_OUTPUT):
                    os.remove(LIVE_OUTPUT)
                if os.path.exists(LIVE_PAUSE_FLAG):
                    os.remove(LIVE_PAUSE_FLAG)
                _cached_live_max_date.clear()
                _cached_live_file_stats.clear()
                _cached_live_rows.clear()
                _get_live_user_data.clear()
                st.session_state.live_page = 0
                # On cloud: live_replay.py (pre-scored data, no ML deps)
                # Locally: live_simulation.py (full encoder + isolation forest)
                proc = subprocess.Popen(
                    [sys.executable, _LIVE_SCRIPT, "--interval", "0.5"],
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
                st.session_state.live_page = 0
                st.rerun()
    with ctrl_pause:
            if not st.session_state.live_paused:
                if st.button("⏸   PAUSE", key="pause_live", use_container_width=True):
                    with open(LIVE_PAUSE_FLAG, "w", encoding="utf-8") as _pf:
                        pass  # existence of the file signals pause
                    st.session_state.live_paused = True
                    st.rerun()
            else:
                if st.button("▶   RESUME", key="resume_live", use_container_width=True):
                    if os.path.exists(LIVE_PAUSE_FLAG):
                        os.remove(LIVE_PAUSE_FLAG)
                    st.session_state.live_paused = False
                    st.rerun()
    st.html(
        "<script>"
        "(function(){"
        "  function styleSimBtns(){"
        "    var btns=document.querySelectorAll('[data-testid=\"stButton\"] button');"
        "    for(var i=0;i<btns.length;i++){"
        "      var txt=(btns[i].innerText||btns[i].textContent||'').trim();"
        "      if(txt.indexOf('LIVE SIMULATION')!==-1||txt==='⏸   PAUSE'||txt==='▶   RESUME'){"
        "        btns[i].style.setProperty('padding','16px 24px','important');"
        "        btns[i].style.setProperty('font-size','13px','important');"
        "        btns[i].style.setProperty('letter-spacing','4px','important');"
        "        btns[i].style.setProperty('color','#cccccc','important');"
        "        btns[i].style.setProperty('border-color','#2a2a2a','important');"
        "      }"
        "    }"
        "  }"
        "  setTimeout(styleSimBtns,50);"
        "  setTimeout(styleSimBtns,300);"
        "})();"
        "</script>"
    )
    st.markdown(
        "<hr style='border:none;border-top:1px solid #111;margin:16px 0 0 0;'>",
        unsafe_allow_html=True,
    )

    # ── LIVE mode ─────────────────────────────────────────────
    if st.session_state.live_mode:

        # Check whether the subprocess is still running
        proc = st.session_state.live_proc
        proc_running = proc is not None and proc.poll() is None
        stream_done  = False

        # Read all scored rows emitted so far (cached to avoid re-reading 500 MB+ per rerun)
        live_rows, stream_done = _cached_live_rows()

        # Detect crash: process has exited (non-None poll) before writing any rows
        if not live_rows and not proc_running and proc is not None:
            _exit_code = proc.poll()
            if _exit_code is not None and _exit_code != 0:
                st.error(
                    f"Live simulation process exited with code {_exit_code} before producing "
                    "any output. Check the terminal for error details (e.g. missing model files "
                    "or a feature-count mismatch). Click **⏹ STOP LIVE SIMULATION** to reset."
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

            # Keep all rows to allow infinite scrolling expansion
            live_df = live_df.reset_index(drop=True)

            # Normalize live payload fields so this table matches the static Alerts layout.
            # NOTE: column names must NOT start with "_" — itertuples() silently
            # drops underscore-prefixed attributes from namedtuples, which caused
            # every row to fall back to the default "LOW" / NaN.
            if "ae_risk_band" in live_df.columns:
                live_df["ui_risk_band"] = live_df["ae_risk_band"].astype(str).str.upper()
            elif "if_risk_band" in live_df.columns:
                live_df["ui_risk_band"] = live_df["if_risk_band"].astype(str).str.upper()
            elif "risk_level" in live_df.columns:
                live_df["ui_risk_band"] = live_df["risk_level"].astype(str).str.upper()
            else:
                live_df["ui_risk_band"] = "LOW"

            if "ae_percentile_rank" in live_df.columns:
                live_df["ui_percentile"] = pd.to_numeric(live_df["ae_percentile_rank"], errors="coerce")
            elif "if_percentile_rank" in live_df.columns:
                live_df["ui_percentile"] = pd.to_numeric(live_df["if_percentile_rank"], errors="coerce")
            else:
                live_df["ui_percentile"] = np.nan

            if "if_anomaly_score" in live_df.columns:
                live_df["ui_anomaly_score"] = pd.to_numeric(live_df["if_anomaly_score"], errors="coerce")
            elif "anomaly_score" in live_df.columns:
                live_df["ui_anomaly_score"] = pd.to_numeric(live_df["anomaly_score"], errors="coerce")
            else:
                live_df["ui_anomaly_score"] = np.nan

            # Normalize composite_score: in live mode, use IF percentile as composite (w1=0, w2=1)
            # since autoencoder isn't part of live pipeline
            if "if_percentile_rank" in live_df.columns:
                live_df["ui_composite_score"] = pd.to_numeric(live_df["if_percentile_rank"], errors="coerce")
            elif "composite_score" in live_df.columns:
                live_df["ui_composite_score"] = pd.to_numeric(live_df["composite_score"], errors="coerce")
            else:
                live_df["ui_composite_score"] = np.nan

            # Assign risk bands from numeric composite score (shared assignment;
            # NaN scores fall back to LOW).
            def assign_live_risk_band(score):
                if pd.isna(score):
                    return "LOW"
                return assign_band_from_percentile(float(score))

            live_df["ui_risk_band"] = live_df["ui_composite_score"].apply(assign_live_risk_band)
            _live_risk_counts = {
                tier: int((live_df["ui_risk_band"] == tier).sum()) for tier in RISK_TIERS
            }

            section_header("Filter by Risk Level", "sh_live_risk")
            _live_sev_cols = st.columns([1, 1, 1, 1, 4])
            _live_tier_checked = {}
            for _idx, _tier in enumerate(RISK_TIERS):
                _color = RISK_COLORS[_tier]
                _count = _live_risk_counts[_tier]
                with _live_sev_cols[_idx]:
                    _live_tier_checked[_tier] = st.checkbox(
                        _tier,
                        value=True,
                        key=f"live_alert_sev_{_tier}",
                    )
                    st.markdown(
                        f"<span style='background:{_color}22;color:{_color};font-size:9px;"
                        f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:1px 6px;"
                        f"border:1px solid {_color}55;display:inline-block;margin-top:-6px;'>"
                        f"{_count:,} alert{'s' if _count != 1 else ''}</span>",
                        unsafe_allow_html=True,
                    )

            live_alert_risk = [t for t, checked in _live_tier_checked.items() if checked]
            live_alert_data = live_df[live_df["ui_risk_band"].isin(live_alert_risk)].copy()

            # Keep the most-recent-first order from live_df (simulates real-time arrival).
            live_alert_data = live_alert_data.reset_index(drop=True)

            live_total_alerts = len(live_alert_data)

            if live_total_alerts == 0:
                st.info("No live alerts match the current risk-level filter.")
            else:
                st.caption(f"Showing {live_total_alerts:,} matching alert{'s' if live_total_alerts != 1 else ''}.")

                # Build the entire table as a single HTML block inside a scrollable container.
                # Height is sized for ~10 visible rows (~56px each) plus header.
                _ROW_HEIGHT = 56
                _SCROLL_HEIGHT = _ROW_HEIGHT * 10 + 40  # 10 rows + header

                _html_parts = [
                    "<div style='max-height:{h}px;overflow-y:auto;border:1px solid #1a1a1a;"
                    "border-radius:4px;'>".format(h=_SCROLL_HEIGHT),
                    # ── sticky header ──
                    "<div style='display:grid;grid-template-columns:72px 1fr 100px 90px;"
                    "gap:8px;padding:6px 8px;border-bottom:1px solid #1a1a1a;"
                    "position:sticky;top:0;background:#0a0a0a;z-index:1;'>",
                    "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                    "text-transform:uppercase;letter-spacing:1.5px;'>Risk</span>",
                    "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                    "text-transform:uppercase;letter-spacing:1.5px;'>User / Investigation hint</span>",
                    "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                    "text-transform:uppercase;letter-spacing:1.5px;'>Day</span>",
                    "<span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                    "text-transform:uppercase;letter-spacing:1.5px;'>Score</span>",
                    "</div>",
                ]

                for _r in live_alert_data.itertuples():
                    _risk = str(getattr(_r, "ui_risk_band", "LOW")).upper()
                    if _risk not in RISK_COLORS:
                        _risk = "LOW"
                    _rc = RISK_COLORS.get(_risk, "#666666")

                    _user = getattr(_r, "user", "--")
                    if isinstance(_user, float) and pd.isna(_user):
                        _user = "--"

                    _cert_ts = getattr(_r, "cert_timestamp", None)
                    if _cert_ts:
                        _ts_p = pd.to_datetime(_cert_ts, errors="coerce")
                        _day_s = _ts_p.strftime("%Y-%m-%d") if pd.notna(_ts_p) else str(_cert_ts)
                    else:
                        _day_s = "--"

                    _cs = getattr(_r, "ui_composite_score", np.nan)
                    _sc_d = f"{float(_cs):.1f}" if pd.notna(_cs) else "--"

                    _top_raw = _get_alert_detail(str(_user), _ts_p, "top_contributors")
                    _summary = build_alert_summary(_top_raw)
                    if _summary == "No contributor detail available for this alert.":
                        _expl = _get_alert_detail(str(_user), _ts_p, "explanation") or ""
                        if isinstance(_expl, str) and _expl.strip():
                            _summary = _expl.strip()
                    # Escape HTML in user-derived strings
                    _user_safe = _html_mod.escape(str(_user))
                    _summary_safe = _html_mod.escape(str(_summary))
                    _day_safe = _html_mod.escape(str(_day_s))

                    _html_parts.append(
                        f"<div style='display:grid;grid-template-columns:72px 1fr 100px 90px;"
                        f"gap:8px;padding:8px 8px;border-bottom:1px solid #0d0d0d;align-items:start;'>"
                        f"<div style='padding-top:2px;'>"
                        f"<span style='background:{_rc}22;color:{_rc};font-size:10px;"
                        f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:2px 6px;"
                        f"border:1px solid {_rc}55;display:inline-block;'>{_risk}</span></div>"
                        f"<div>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:13px;"
                        f"color:#e0e0e0;font-weight:600;'>{_user_safe}</span>"
                        f"<br><span style='font-family:Inter,sans-serif;font-size:12px;"
                        f"color:#666;line-height:1.5;'>{_summary_safe}</span></div>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                        f"color:#888;padding-top:4px;white-space:nowrap;'>{_day_safe}</div>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                        f"color:#999;padding-top:4px;text-align:center;white-space:nowrap;'>{_sc_d}</div>"
                        f"</div>"
                    )

                _html_parts.append("</div>")  # close scrollable container
                st.markdown("".join(_html_parts), unsafe_allow_html=True)

                # ── Investigate action (outside the scrollable table) ──
                _live_users = live_alert_data["user"].dropna().unique().tolist()
                if _live_users:
                    _inv_cols = st.columns([3, 1])
                    with _inv_cols[0]:
                        _sel_user = st.selectbox(
                            "Select a user to investigate",
                            options=_live_users,
                            key="live_inv_select",
                            label_visibility="collapsed",
                            placeholder="Select a user to investigate…",
                        )
                    with _inv_cols[1]:
                        if st.button("Investigate →", key="live_inv_btn", use_container_width=True):
                            st.session_state["inv_user_select"] = _sel_user
                            st.session_state["_nav_request"] = "Investigation"
                            st.rerun()

            if live_total_alerts > 0:
                st.download_button(
                    "EXPORT LIVE ALERTS",
                    data=live_alert_data.to_csv(index=False).encode("utf-8"),
                    file_name="live_alerts.csv",
                    mime="text/csv",
                )

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

            # ── Live charts (based solely on current live alerts table data) ──
            live_chart_df = live_df.copy()

            col_live_left, col_live_right = st.columns(2)

            with col_live_left:
                section_header("Risk Distribution", "sh_live_risk_dist")
                if "ui_risk_band" in live_chart_df.columns and not live_chart_df.empty:
                    _risk_counts = (
                        live_chart_df["ui_risk_band"]
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
                if "ui_anomaly_score" in live_chart_df.columns and "ui_risk_band" in live_chart_df.columns:
                    fig_live_hist = px.histogram(
                        live_chart_df,
                        x="ui_anomaly_score",
                        nbins=50,
                        color="ui_risk_band",
                        color_discrete_map=RISK_COLORS,
                        labels={"ui_anomaly_score": "Anomaly Score", "ui_risk_band": "Risk Level"},
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
            st.session_state.live_paused = False
            st.success("Simulation complete — all test rows processed.")

    # ── STATIC mode ───────────────────────────────────────────
    else:
        # Reset pagination and clear live state when returning to static mode
        if st.session_state.get("live_page") != 0:
            st.session_state.live_page = 0

        # ── Top 10 Riskiest Users in Alerts ──
        section_header("Top 10 Riskiest Users", "sh_top_users")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:12px;color:#555;margin:0 0 12px 0;'>"
            "Click a user to open their investigation profile.</p>",
            unsafe_allow_html=True,
        )
        top_users = _al_top_users(*_ov_args())

        if top_users.empty:
            st.info("No users available in the current filter range.")
        else:
            for rank, row in enumerate(top_users.itertuples(), start=1):
                uid = row.user
                score = row.max_percentile
                days = int(row.critical_count + row.high_count)
                badge_label = assign_band_from_percentile(score)
                badge_color = RISK_COLORS[badge_label]

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

        # ── Date/risk summary (no filter button here) ──
        _active_risks = ", ".join(st.session_state.flt_risk) if len(st.session_state.flt_risk) < 4 else "All risk levels"
        st.caption(
            f"Date: {st.session_state.flt_date_start} to {st.session_state.flt_date_end}   |   Risk: {_active_risks}"
        )

        # ── Alert Filters (unified) ──
        # Pull all alert filter state from session (set via Filters modal)
        _disp_filter        = st.session_state.get("flt_disp_filter", "Show New Only")
        show_suppressed_alerts = st.session_state.get("flt_view_suppressed", False)
        min_pctl            = st.session_state.get("flt_min_pctl", 0.0)
        sort_choice         = st.session_state.get("flt_sort_choice", "Highest score first")
        max_results         = st.session_state.get("flt_max_rows", 500)

        # Centralized risk-band mapping — extend here when new bands are added
        _RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

        # Build triage disposition lookup from SQLite: {(user, day_str): status}
        _triage_all = {(r["user"], r["day"]): r["status"] for r in get_all_dispositions()}

        # Map radio label → disposition status value
        _DISP_FILTER_MAP = {
            "Show New Only":      "NEW",
            "Show Investigating": "INVESTIGATING",
            "Show Resolved":      "RESOLVED",
            "Show Dismissed":     "DISMISSED",
        }

        # Suppressed Alerts mode: show only suppressed alerts (top 10)
        if show_suppressed_alerts:
            alert_data = (
                filtered_df[
                    filtered_df["status"].astype(str).str.upper() == "SUPPRESSED"
                ].copy()
                if "status" in filtered_df.columns
                else pd.DataFrame()
            )
            if "ae_percentile_rank" in alert_data.columns:
                alert_data = alert_data[alert_data["ae_percentile_rank"] >= min_pctl]
                alert_data = alert_data.sort_values("ae_percentile_rank", ascending=False)
            elif "if_percentile_rank" in alert_data.columns:
                alert_data = alert_data[alert_data["if_percentile_rank"] >= min_pctl]
                alert_data = alert_data.sort_values("if_percentile_rank", ascending=False)
        else:
            # Normal mode: use the shared risk-level filter and exclude suppressed alerts
            alert_data = filtered_df[
                (filtered_df["ae_percentile_rank"] >= min_pctl) &
                (
                    ~(filtered_df["status"].astype(str).str.upper() == "SUPPRESSED")
                    if "status" in filtered_df.columns
                    else True
                )
            ].copy()

            # Apply triage disposition filter (SQLite-backed, row-level lookup)
            if _disp_filter in _DISP_FILTER_MAP:
                _target = _DISP_FILTER_MAP[_disp_filter]
                _day_strs = alert_data["day"].apply(
                    lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime")
                    else str(d).split("T")[0].split(" ")[0]
                )
                _row_statuses = pd.Series(
                    [_triage_all.get((u, d), "NEW") for u, d in zip(alert_data["user"], _day_strs)],
                    index=alert_data.index,
                )
                alert_data = alert_data[_row_statuses == _target]

        # Sort based on explicit outcome label (skip in suppressed mode, already sorted)
        if not show_suppressed_alerts:
            if sort_choice == "Highest score first":
                if "ae_percentile_rank" in alert_data.columns:
                    alert_data = alert_data.sort_values("ae_percentile_rank", ascending=False)
            elif sort_choice == "Lowest score first":
                if "ae_percentile_rank" in alert_data.columns:
                    alert_data = alert_data.sort_values("ae_percentile_rank", ascending=True)
            elif sort_choice in ("Highest severity first", "Lowest severity first"):
                _asc = sort_choice == "Lowest severity first"
                alert_data["_risk_sort_key"] = alert_data["ae_risk_band"].astype(str).map(_RISK_ORDER).fillna(-1)
                if "ae_percentile_rank" in alert_data.columns:
                    alert_data = alert_data.sort_values(
                        ["_risk_sort_key", "ae_percentile_rank"],
                        ascending=[_asc, False],
                    ).drop(columns=["_risk_sort_key"])
                else:
                    alert_data = alert_data.sort_values(
                        ["_risk_sort_key"],
                        ascending=[_asc],
                    ).drop(columns=["_risk_sort_key"])
            elif sort_choice == "Most recent first":
                alert_data["day"] = pd.to_datetime(alert_data["day"], errors="coerce")
                alert_data = alert_data.sort_values("day", ascending=False)
            elif sort_choice == "Oldest first":
                alert_data["day"] = pd.to_datetime(alert_data["day"], errors="coerce")
                alert_data = alert_data.sort_values("day", ascending=True)
            elif sort_choice == "User A–Z":
                alert_data = alert_data.sort_values("user", ascending=True)
            else:  # User Z–A
                alert_data = alert_data.sort_values("user", ascending=False)
            alert_data = alert_data.head(int(max_results))

                    #########################

        # Cap card rendering to keep the UI responsive
        CARD_LIMIT = 10
        total_alerts = len(alert_data)
        _total_pages = max(1, (total_alerts + CARD_LIMIT - 1) // CARD_LIMIT)
        if "alert_feed_page" not in st.session_state:
            st.session_state["alert_feed_page"] = 0
        _page = min(st.session_state["alert_feed_page"], _total_pages - 1)
        st.session_state["alert_feed_page"] = _page
        card_data = alert_data.iloc[_page * CARD_LIMIT : (_page + 1) * CARD_LIMIT]

        if total_alerts == 0:
            _af_hdr, _af_btn = st.columns([9, 1], vertical_alignment="bottom")
            with _af_hdr:
                st.markdown("<div class='section-header'>Alert Feed</div>", unsafe_allow_html=True)
            with _af_btn:
                if st.button("Filters", key="al_flt_empty", use_container_width=True):
                    show_filters()
            if show_suppressed_alerts:
                st.info("No suppressed alerts found in the current filter range.")
            else:
                st.info("No alerts match the current filters.")
        else:
            # ── Alert Feed header with Filters button ──
            _af_hdr, _af_btn = st.columns([9, 1], vertical_alignment="bottom")
            with _af_hdr:
                st.markdown("<div class='section-header'>Alert Feed</div>", unsafe_allow_html=True)
            with _af_btn:
                if st.button("Filters", key="al_flt", use_container_width=True):
                    show_filters()
            if total_alerts > CARD_LIMIT:
                st.caption(
                    f"Page {_page + 1} of {_total_pages} &nbsp;·&nbsp; {total_alerts:,} matching alerts."
                )

            # ── Column header row ──
            _HDR = (
                "font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                "text-transform:uppercase;letter-spacing:1.5px;"
            )
            if show_suppressed_alerts:
                _h_risk, _h_info, _h_rule, _h_day, _h_pctl, _h_status, _h_btn = st.columns([1, 4, 3, 2, 1, 2, 2])
                _h_risk.markdown(f"<span style='{_HDR}'>Risk</span>", unsafe_allow_html=True)
                _h_info.markdown(f"<span style='{_HDR}'>User / Reason</span>", unsafe_allow_html=True)
                _h_rule.markdown(f"<span style='{_HDR}'>Suppression Rule</span>", unsafe_allow_html=True)
                _h_day.markdown(f"<span style='{_HDR}'>Day</span>", unsafe_allow_html=True)
                _h_pctl.markdown(f"<span style='{_HDR}'>Pctl</span>", unsafe_allow_html=True)
                _h_status.markdown(f"<span style='{_HDR}'>Status</span>", unsafe_allow_html=True)
            else:
                _h_risk, _h_info, _h_day, _h_pctl, _h_status, _h_btn = st.columns([1, 5, 2, 1, 2, 2])
                _h_risk.markdown(f"<span style='{_HDR}'>Risk</span>", unsafe_allow_html=True)
                _h_info.markdown(f"<span style='{_HDR}'>User / Investigation hint</span>", unsafe_allow_html=True)
                _h_day.markdown(f"<span style='{_HDR}'>Day</span>", unsafe_allow_html=True)
                _h_pctl.markdown(f"<span style='{_HDR}'>Percentile</span>", unsafe_allow_html=True)
                _h_status.markdown(f"<span style='{_HDR}'>Status</span>", unsafe_allow_html=True)
            st.markdown(
                "<div style='border-bottom:1px solid #1a1a1a;margin:0 0 2px 0;'></div>",
                unsafe_allow_html=True,
            )

            # ── Per-alert card rows ──
            _disp_lookup = {(r["user"], r["day"]): r["status"] for r in get_all_dispositions()}
            for i, row in enumerate(card_data.itertuples()):
                risk    = getattr(row, "composite_risk_band", "MEDIUM") if show_suppressed_alerts else getattr(row, "ae_risk_band", "LOW")
                user    = getattr(row, "user",           "—")
                day_val = getattr(row, "day",            None)
                day_str = (day_val.strftime("%Y-%m-%d") if hasattr(day_val, "strftime")
                           else str(day_val).split("T")[0].split(" ")[0])
                pctl    = getattr(row, "ae_percentile_rank", 0.0)
                top_raw = _get_alert_detail(getattr(row, "user", ""), getattr(row, "day", None), "top_contributors")
                summary = build_alert_summary(top_raw)
                status  = str(getattr(row, "status", "")).upper()
                supp_rule = getattr(row, "suppression_rule", None) or "—"

                risk_color = RISK_COLORS.get(risk, "#666666")
                _disp_key = f"disp_{user}_{day_str}"
                _cur_status = _disp_lookup.get((user, day_str), "NEW")
                if _disp_key not in st.session_state:
                    st.session_state[_disp_key] = _cur_status

                if show_suppressed_alerts:
                    c_risk, c_info, c_rule, c_day, c_pctl, c_status, c_btn = st.columns([1, 4, 3, 2, 1, 2, 2])
                else:
                    c_risk, c_info, c_day, c_pctl, c_status, c_btn = st.columns([1, 5, 2, 1, 2, 2])

                with c_risk:
                    st.markdown(
                        f"<div style='padding-top:5px;'>"
                        f"<span style='background:{risk_color}22;color:{risk_color};font-size:10px;"
                        f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:2px 6px;"
                        f"border:1px solid {risk_color}55;display:inline-block;'>{risk}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if status == "SUPPRESSED" and show_suppressed_alerts:
                        st.markdown(
                            "<div style='padding-top:2px;'>"
                            "<span style='background:#9a8b7722;color:#9a8b77;font-size:9px;"
                            "font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:2px 6px;"
                            "border:1px solid #9a8b7755;display:inline-block;'>SUPPRESSED</span>"
                            "</div>",
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

                if show_suppressed_alerts:
                    with c_rule:
                        st.markdown(
                            f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                            f"color:#d4a017;padding-top:5px;word-break:break-all;'>"
                            f"{_html_mod.escape(str(supp_rule))}</div>",
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

                with c_status:
                    st.selectbox(
                        "Status",
                        options=ALERT_STATUS_OPTIONS,
                        key=_disp_key,
                        on_change=_on_status_change,
                        args=(user, day_str, _disp_key),
                        label_visibility="collapsed",
                    )

                with c_btn:
                    if st.button("Investigate →", key=f"al_inv_{i}_{show_suppressed_alerts}", use_container_width=True):
                        st.session_state["inv_user_select"] = user
                        st.session_state["inv_alert_context"] = {
                            "user": user,
                            "day": day_str,
                            "risk": risk,
                            "percentile": pctl,
                            "summary": summary,
                        }
                        st.session_state["_nav_request"] = "Investigation"
                        st.rerun()

                st.markdown(
                    "<div style='border-bottom:1px solid #0d0d0d;margin:2px 0;'></div>",
                    unsafe_allow_html=True,
                )

            # ── Pagination controls ──
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            _p_prev_col, _p_info_col, _p_next_col = st.columns([2, 3, 2])
            with _p_prev_col:
                if st.button("← Previous", key="al_prev", use_container_width=True, disabled=(_page == 0)):
                    st.session_state["alert_feed_page"] = _page - 1
                    st.rerun()
            with _p_info_col:
                st.markdown(
                    f"<div style='text-align:center;font-family:JetBrains Mono,monospace;"
                    f"font-size:11px;color:#555;padding-top:6px;'>"
                    f"Page {_page + 1} / {_total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with _p_next_col:
                if st.button("Next →", key="al_next", use_container_width=True, disabled=(_page >= _total_pages - 1)):
                    st.session_state["alert_feed_page"] = _page + 1
                    st.rerun()

        # ── Export (columns + top_contributors if present) ──
        alert_display_cols = ["user", "day", "ae_risk_band", "if_anomaly_score", "ae_percentile_rank"]
        alert_display_cols = [c for c in alert_display_cols if c in alert_data.columns]
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
        ch_totals = _ch_totals(*_ov_args())
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
    box_df = _ch_box_sample(*_ov_args(), selected_feature)
    if not box_df.empty:
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
# Investigation is excluded: its @st.fragment(run_every="2s") handles its own
# refresh cycle independently.  A full st.rerun() here would reset the fragment
# timer every second, preventing it from ever firing on its own schedule.
if active_page not in ("Alerts", "Investigation") and st.session_state.live_mode:
    _proc = st.session_state.live_proc
    _proc_running = _proc is not None and _proc.poll() is None
    if _proc_running and not st.session_state.live_paused:
        time.sleep(1)
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Footer — Data & Feature Gaps Note
# ──────────────────────────────────────────────────────────────

st.markdown("---")
