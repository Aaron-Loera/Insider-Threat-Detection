import sys
sys.stderr.write("[APP] module-level execution started\n"); sys.stderr.flush()
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
import time
import ast as _ast
import hmac
import hashlib
import html as _html_mod

try:
    import pyrebase
    _PYREBASE_AVAILABLE = True
except ImportError:
    _PYREBASE_AVAILABLE = False

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
    /* Give sidebar content enough bottom padding so nothing hides behind the fixed panel */
    [data-testid="stSidebarUserContent"] {
        padding-bottom: 120px !important;
    }
    /* Fixed sign-out panel at sidebar bottom */
    .signout-panel {
        position: fixed;
        bottom: 0;
        left: 0;
        width: var(--sidebar-width, 280px);
        background: #0a0a0a;
        border-top: 1px solid #1a1a1a;
        padding: 14px 16px;
        z-index: 999;
    }
    .signout-panel .so-email {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
        word-break: break-all;
        margin: 0 0 8px 0;
    }
    .signout-panel .so-btn {
        display: block;
        width: 100%;
        background: #0e0e0e;
        border: 1px solid #2a2a2a;
        color: #888;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-decoration: none;
        text-align: center;
        padding: 12px 0;
        transition: border-color 0.15s, color 0.15s;
    }
    .signout-panel .so-btn:hover {
        border-color: #e84545;
        color: #e84545;
        background: #1a0000;
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
            
    /* ── Investigation card labels (Alert Summary, Raw Alert Record, etc.) ── */
    .inv-card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        font-weight: 900;
        color: #bbbbbb;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 14px;
    }

    /* ── Investigation field labels (User, Day, AE Risk Band …) ── */
    .inv-field-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 600;
        color: #999999;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 6px 0 4px 0;
        align-self: center;
    }

    /* ── Investigation table: column headers ── */
    .inv-th {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 600;
        color: #999999;
        text-align: left;
        padding: 0 16px 10px 0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        border-bottom: 1px solid #1a1a1a;
    }

    /* ── Investigation table: feature-name cells ── */
    .inv-feat-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #888888;
        padding: 8px 16px 8px 0;
        border-bottom: 1px solid #111;
        vertical-align: middle;
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
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div * {
        cursor: pointer !important;
    }
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
# Firebase Authentication
# ──────────────────────────────────────────────────────────────

def _make_auth_token(email: str) -> str:
    secret = st.secrets["firebase"]["apiKey"].encode()
    return hmac.new(secret, email.lower().encode(), hashlib.sha256).hexdigest()

def _set_auth_cookie(email: str):
    token = _make_auth_token(email)
    components.html(
        f"""<script>
        document.cookie = "auth_email={email}; path=/; max-age=86400; SameSite=Strict";
        document.cookie = "auth_token={token}; path=/; max-age=86400; SameSite=Strict";
        </script>""",
        height=0,
    )

def _clear_auth_cookie():
    components.html(
        """<script>
        document.cookie = "auth_email=; path=/; max-age=0";
        document.cookie = "auth_token=; path=/; max-age=0";
        </script>""",
        height=0,
    )


def _get_firebase_auth():
    """Return a Pyrebase Auth object initialised from st.secrets."""
    if not _PYREBASE_AVAILABLE:
        st.error(
            "**Missing dependency:** `pyrebase4` is required for authentication. "
            "Run `pip install pyrebase4` then restart the app."
        )
        st.stop()
    firebase_cfg = st.secrets.get("firebase", None)
    if firebase_cfg is None:
        st.error(
            "**Firebase config missing.** Add a `[firebase]` section to "
            "`.streamlit/secrets.toml`. See the dashboard README for setup instructions."
        )
        st.stop()
    app = pyrebase.initialize_app(dict(firebase_cfg))
    return app.auth()


def _render_login():
    """Render a full-page login form. Authenticates via Firebase Email/Password."""
    st.markdown("""
    <style>
    /* ── Hide sidebar entirely on the login page ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] { display: none !important; }

    /* ── Full-bleed black background ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background-color: #000000 !important;
    }
    /* Vertically + horizontally centre the login card */
    [data-testid="stMain"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 100vh !important;
    }
    .block-container {
        padding: 24px 40px 36px 40px !important;
        max-width: 460px !important;
        width: 100% !important;
        margin: 0 auto !important;
        flex: none !important;
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-top: 2px solid #e84545;
    }

    /* ── Input labels ── */
    .block-container [data-testid="stTextInput"] label p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #555 !important;
    }

    /* ── Input fields ── */
    .block-container [data-testid="stTextInput"] input {
        background-color: #000000 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0 !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        padding: 10px 14px !important;
        caret-color: #e84545 !important;
        transition: border-color 0.15s ease !important;
    }
    .block-container [data-testid="stTextInput"] input:focus {
        border-color: #e84545 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .block-container [data-testid="stTextInput"] input::placeholder {
        color: #333 !important;
    }
    /* Hide "Press Enter to apply" helper on login inputs */
    .block-container [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* ── Sign-in button ── */
    div[data-testid="stColumn"]:nth-child(2) [data-testid="stButton"] button {
        background-color: #e84545 !important;
        border: none !important;
        border-radius: 0 !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        padding: 12px 0 !important;
        width: 100% !important;
        transition: background-color 0.15s ease !important;
    }
    .block-container [data-testid="stButton"] button:hover {
        background-color: #c73333 !important;
    }

    /* ── Error alert ── */
    .block-container [data-testid="stNotificationContentError"],
    .block-container .stAlert {
        background-color: #1a0000 !important;
        border: 1px solid #e84545 !important;
        border-radius: 0 !important;
        color: #ee8888 !important;
        font-size: 12px !important;
    }

    /* ── Logo heading ── */
    .login-logo-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 22px;
    }
    .login-logo-dsk {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 6px;
        color: #ffffff;
        display: block;
        line-height: 1;
    }
    .login-logo-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 8px;
        letter-spacing: 2px;
        color: #444;
        text-transform: uppercase;
        margin-top: 5px;
        display: block;
    }
    .login-divider {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 0 0 22px 0;
    }
    .login-heading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #444;
        margin: 0 0 20px 0;
    }
    .login-footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 8px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #2a2a2a;
        text-align: center;
        margin-top: 24px;
    }

    /* ── Responsive: phones / small screens ── */
    @media (max-width: 640px) {
        .block-container {
            max-width: 100% !important;
            padding: 24px 18px 22px 18px !important;
        }
        .login-logo-row svg { width: 36px; height: 36px; }
        .login-logo-dsk { font-size: 22px; letter-spacing: 3px; }
        .login-logo-sub { font-size: 7px; letter-spacing: 1.5px; }
        .login-heading  { font-size: 8px; letter-spacing: 2px; margin-bottom: 14px; }
        .login-divider  { margin: 16px 0 18px 0; }
        .login-footer   { font-size: 7px; margin-top: 18px; }
        .block-container [data-testid="stTextInput"] input {
            font-size: 12px !important;
            padding: 9px 12px !important;
        }
        .block-container [data-testid="stButton"] button {
            font-size: 10px !important;
            padding: 10px 0 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='login-logo-row'>"
        "<svg width='48' height='48' viewBox='0 0 100 100' fill='none' xmlns='http://www.w3.org/2000/svg' style='flex-shrink:0;'>"
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
        "<div>"
        "<span class='login-logo-dsk'>DSK</span>"
        "<span class='login-logo-sub'>Data Structure Kittens</span>"
        "</div>"
        "</div>"
        "<hr class='login-divider'>"
        "<p class='login-heading'>Analyst Portal &mdash; Sign In</p>",
        unsafe_allow_html=True,
    )

    email = st.text_input(
        "Email address",
        placeholder="analyst@organisation.com",
        key="login_email",
    )
    password = st.text_input(
        "Password",
        type="password",
        placeholder="••••••••",
        key="login_password",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.session_state.get("_login_error"):
        st.error(st.session_state["_login_error"])

    if st.button("SIGN IN", use_container_width=True, type="primary"):
        if not email or not password:
            st.session_state["_login_error"] = "Email and password are required."
            st.rerun()
        else:
            try:
                auth = _get_firebase_auth()
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state["authenticated"] = True
                st.session_state["auth_user_email"] = user.get("email", email)
                st.session_state["auth_id_token"] = user.get("idToken", "")
                st.session_state["_set_cookie"] = True
                st.session_state.pop("_login_error", None)
                st.rerun()
            except Exception as exc:
                msg = str(exc)
                if any(k in msg for k in ("INVALID_PASSWORD", "EMAIL_NOT_FOUND", "INVALID_LOGIN_CREDENTIALS", "INVALID_EMAIL")):
                    st.session_state["_login_error"] = "Invalid email or password."
                elif "TOO_MANY_ATTEMPTS_TRY_LATER" in msg:
                    st.session_state["_login_error"] = "Too many failed attempts. Try again later."
                elif "USER_DISABLED" in msg:
                    st.session_state["_login_error"] = "This account has been disabled."
                elif "OPERATION_NOT_ALLOWED" in msg:
                    st.session_state["_login_error"] = "Email/password sign-in is not enabled in Firebase."
                elif "EMAIL_NOT_VERIFIED" in msg:
                    st.session_state["_login_error"] = "Please verify your email address before signing in."
                else:
                    st.session_state["_login_error"] = f"Sign-in failed: {msg}"
                st.rerun()

    st.markdown(
        "<p class='login-footer'>UEBA Insider Threat Detection &mdash; Senior Design Project &middot; 2026</p>",
        unsafe_allow_html=True,
    )


# ── Handle logout via query param (set by the fixed sidebar button) ──
_is_logout = st.query_params.get("logout") == "true"
if _is_logout:
    st.session_state.clear()
    st.query_params.clear()
    _clear_auth_cookie()

# ── Restore session from auth cookie (survives page refresh) ──
if not _is_logout and not st.session_state.get("authenticated", False):
    _cookies = st.context.cookies
    _c_email = _cookies.get("auth_email", "")
    _c_token = _cookies.get("auth_token", "")
    if _c_email and _c_token and hmac.compare_digest(_c_token, _make_auth_token(_c_email)):
        st.session_state["authenticated"] = True
        st.session_state["auth_user_email"] = _c_email

# ── Auth gate — show login and stop until the user has signed in ──
if not st.session_state.get("authenticated", False):
    _render_login()
    st.stop()

# ── Set auth cookie after first successful login ──
if st.session_state.pop("_set_cookie", False):
    _set_auth_cookie(st.session_state.get("auth_user_email", ""))


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
import datetime as _dt
import logging as _auth_log
_auth_log.getLogger("ueba.startup").warning(
    f"[STARTUP] authenticated — reached data-load section at {_dt.datetime.utcnow().isoformat()}"
)

# Resolve the project root so config.py (at the root) is importable.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# All path configuration is centralized in config.py.  Per-contributor
# overrides live in paths.local.py (gitignored). See paths.local.example.py.
from config import (
    ANALYST_TABLE_PARQUET, ANALYST_TABLE_CSV,
    UEBA_PARQUET, UEBA_CSV,
    UEBA_A_PARQUET, UEBA_A_CSV,
    LIVE_OUTPUT, LIVE_PAUSE_FLAG, LIVE_SIM_SCRIPT,
)

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


@st.cache_data(show_spinner=False)
def _load_user_detail_df(user: str) -> "pd.DataFrame":
    """Download ALL detail rows for one user and cache the result.

    Local:  reads from the main analyst parquet with a user-level DNF filter
            (1268 sorted row groups → only the user's ~1270 rows are read).
    Cloud:  downloads a tiny per-user parquet from HF details/ folder (~46 KB).
            The full 193 MB analyst parquet is NEVER loaded on cloud — each
            user file is independently tiny, making the download cost negligible.
    """
    _HF_REPO = "DSKittens/ueba-dashboard-dat"
    _hf_token = st.secrets.get("huggingface", {}).get("token", None)
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
            url = f"hf://datasets/{_HF_REPO}/details/{safe}.parquet"
            return pd.read_parquet(url, storage_options={"token": _hf_token})
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


@st.cache_resource(show_spinner="Loading dataset...")
def load_data():
    """Load pre-merged parquet and pre-compute user_risk.

    Uses @st.cache_resource (not cache_data) so the DataFrame is stored by
    reference — no pickle serialisation overhead. cache_data would pickle the
    250 MB DataFrame to disk on first call, temporarily doubling peak RAM and
    pushing the container past the 1 GB Streamlit Cloud free-tier limit.

    On cloud: downloads merged_dataset_5.parquet (62 MB) via hf_hub_download,
    which streams to the HF local disk cache before reading — avoids holding
    the compressed bytes AND the decompressed Arrow table in memory at the same
    time (unlike the hf:// fsspec path which buffers in-process).
    """
    import gc
    import logging as _logging
    _log = _logging.getLogger("ueba.load_data")
    _log.warning("[load_data] started")

    _HF_REPO  = "DSKittens/ueba-dashboard-dat"
    _hf_token = st.secrets.get("huggingface", {}).get("token", None)

    # Local path for the pre-merged parquet (built by scripts/build_merged_parquet.py)
    _MERGED_LOCAL = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "explainability", "alert_table", "merged_dataset_5.parquet",
    )

    def _downcast(df):
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype("int32")
        return df

    # ── Load single pre-merged file ──
    _log.warning("[load_data] loading merged dataset")
    if os.path.exists(_MERGED_LOCAL):
        import pyarrow.parquet as _pq
        _tbl = _pq.read_table(_MERGED_LOCAL)
        merged = _tbl.to_pandas()
        del _tbl
        _log.warning("[load_data] loaded from local merged parquet")
    else:
        # Cloud: hf_hub_download caches the file to disk before reading.
        # This avoids buffering the compressed bytes in-process (unlike hf://).
        from huggingface_hub import hf_hub_download as _hf_dl
        import pyarrow.parquet as _pq
        _cached = _hf_dl(
            repo_id=_HF_REPO,
            filename="merged_dataset_5.parquet",
            repo_type="dataset",
            token=_hf_token,
        )
        _log.warning(f"[load_data] file cached at {_cached}")
        _tbl = _pq.read_table(_cached)
        _log.warning(f"[load_data] arrow table: {_tbl.nbytes/1e6:.0f} MB")
        merged = _tbl.to_pandas()
        del _tbl
        _log.warning("[load_data] converted to pandas")

    gc.collect()
    _downcast(merged)
    merged["day"] = pd.to_datetime(merged["day"], errors="coerce")
    for _col in ("user", "ae_risk_band", "if_risk_band"):
        if _col in merged.columns:
            merged[_col] = merged[_col].astype("category")
    gc.collect()
    _log.warning(f"[load_data] merged: {merged.shape}, {merged.memory_usage(deep=True).sum()/1e6:.1f} MB")

    # Cast risk bands to ordered categorical
    _risk_cat = pd.CategoricalDtype(categories=["LOW", "MEDIUM", "HIGH", "CRITICAL"], ordered=True)
    for _col in ("ae_risk_band", "if_risk_band"):
        if _col in merged.columns:
            merged[_col] = merged[_col].astype(_risk_cat)

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


import logging as _startup_log
_slog = _startup_log.getLogger("ueba.startup")
try:
    merged_df, user_risk, all_users, _DS_MIN, _DS_MAX = load_data()
    _slog.warning("[STARTUP] load_data() complete")
    ueba_a_df = load_ueba_a()
    _slog.warning("[STARTUP] load_ueba_a() complete — DATA_LOADED=True")
    DATA_LOADED = True
except Exception as _load_err:
    import traceback as _tb
    ueba_a_df = None
    DATA_LOADED = False
    _LOAD_ERROR = _tb.format_exc()
    _slog.warning(f"[STARTUP] load_data FAILED: {_LOAD_ERROR}")
else:
    _LOAD_ERROR = None

def _get_alert_detail(user, day, key):
    """Fetch explanation or top_contributors for a single (user, day) record."""
    day_str = str(day.date()) if hasattr(day, "date") else str(day)
    return fetch_alert_detail(str(user), day_str).get(key, None)


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
    "Authentication":  "#00b4d8",
    "File Access":     "#e84545",
    "Removable Media": "#d4a017",
    "Email":           "#3a86a8",
    "HTTP Activity":   "#9b59b6",
    "PC Activity":     "#e67e22",
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
RISK_COLORS = {"CRITICAL": "#bb44f0", "HIGH": "#e84545", "MEDIUM": "#d4a017", "LOW": "#3a86a8"}

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


def parse_top_contributors_with_values(raw) -> list[tuple]:
    """Return list of (feature_name, contribution_value) tuples from top_contributors."""
    if raw is None:
        return []
    if isinstance(raw, float):
        return []  # NaN from a left-join miss
    if isinstance(raw, list):
        pairs = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), item[1]))
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                pairs.append((str(item[0]), None))
            elif isinstance(item, str):
                pairs.append((item, None))
        return pairs
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = _ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parse_top_contributors_with_values(parsed)
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


@st.cache_data(ttl=5, show_spinner=False)
def _get_live_user_data(user: str) -> pd.DataFrame:
    """Load live-scored rows for *user* from LIVE_OUTPUT, normalized to match user_data columns.

    Delegates to _cached_live_rows to avoid a redundant full-file scan; the shared
    2-second cache means all live-data consumers pay at most one read per TTL window.
    """
    rows, _ = _cached_live_rows()
    if not rows:
        return pd.DataFrame()
    user_rows = [r for r in rows if r.get("user") == user]
    if not user_rows:
        return pd.DataFrame()
    df = pd.DataFrame(user_rows)
    if "cert_timestamp" not in df.columns:
        return pd.DataFrame()
    df["day"] = pd.to_datetime(df["cert_timestamp"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["day"])
    if df.empty:
        return pd.DataFrame()
    for _col in ("if_anomaly_score", "if_percentile_rank"):
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
    # Keep one row per day: worst-of-day (highest percentile rank)
    df = (
        df.sort_values("if_percentile_rank", ascending=False)
        .drop_duplicates(subset=["day"])
        .copy()
    )
    # Map to column names the Investigation page expects
    df["ae_risk_band"] = (
        df["if_risk_band"].astype(str).str.upper()
        if "if_risk_band" in df.columns
        else "LOW"
    )
    df["ae_percentile_rank"] = df.get("if_percentile_rank", np.nan)
    df["_live_row"] = True
    return df.reset_index(drop=True)


@st.cache_data(ttl=2, show_spinner=False)
def _cached_live_file_stats():
    """Return (row_count, stream_done) for the live output file.

    Delegates to _cached_live_rows so at most one file scan occurs per TTL window
    regardless of how many callers need live-file metadata.
    """
    rows, stream_done = _cached_live_rows()
    return len(rows), stream_done


@st.cache_data(ttl=2, show_spinner=False)
def _cached_live_rows():
    """Read all scored rows from the live output file; return (rows_list, stream_done).

    Cached for 2 seconds to avoid re-reading a potentially 500 MB+ file on every
    Streamlit rerun.  Called by the Alerts page live mode on every auto-refresh.
    Clear via _cached_live_rows.clear() when starting a new simulation.
    """
    if not os.path.exists(LIVE_OUTPUT):
        return [], False
    rows: list[dict] = []
    stream_done = False
    try:
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
                    rows.append(obj)
    except Exception:
        pass
    return rows, stream_done


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

# Cap data points sent to Plotly — browser rendering is the main bottleneck
MAX_PLOT_POINTS = 50_000

# ──────────────────────────────────────────────────────────────
# Global filter state (persists across page navigations)
# ──────────────────────────────────────────────────────────────
_DS_MIN = merged_df["day"].min().date()
_DS_MAX = merged_df["day"].max().date()
# When live simulation has run, live records extend beyond _DS_MAX. Track the
# effective maximum so the date slider and filter can reach those dates.
@st.cache_data(ttl=60, show_spinner=False)
def _cached_live_max_date():
    """Scan the live output file at most once per minute to find the latest date in
    live records.  Delegates to _cached_live_rows to avoid a separate full-file scan.
    Clear via _cached_live_max_date.clear() when starting a new simulation.
    """
    try:
        rows, _ = _cached_live_rows()
        live_max = _DS_MAX
        for _o in rows:
            _ts = _o.get("cert_timestamp")
            if not _ts:
                continue
            _d = pd.to_datetime(_ts, errors="coerce")
            if pd.notna(_d) and _d.date() > live_max:
                live_max = _d.date()
        return live_max
    except Exception:
        return _DS_MAX

_ds_live_max = _cached_live_max_date()
if "flt_date_start" not in st.session_state:
    st.session_state.flt_date_start = _DS_MIN
if "flt_date_end" not in st.session_state:
    st.session_state.flt_date_end = _DS_MAX
if "flt_risk" not in st.session_state:
    st.session_state.flt_risk = list(RISK_TIERS)
if "flt_alert_sev" not in st.session_state:
    st.session_state.flt_alert_sev = ["CRITICAL", "HIGH"]
if "flt_min_pctl" not in st.session_state:
    st.session_state.flt_min_pctl = 0.0
if "flt_max_rows" not in st.session_state:
    st.session_state.flt_max_rows = 500
if "flt_sort_choice" not in st.session_state:
    st.session_state.flt_sort_choice = "Highest score first"


@st.dialog("Filters")
def show_filters():
    st.markdown("**Date Range**")
    dr = st.date_input(
        "Date Range",
        value=(st.session_state.flt_date_start, st.session_state.flt_date_end),
        min_value=_DS_MIN,
        max_value=_ds_live_max,
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

    # ── Alerts-specific controls (only shown on the Alerts page) ──
    _on_alerts = (active_page == "Alerts" and not st.session_state.get("live_mode", False))
    if _on_alerts:
        st.markdown("---")
        st.markdown("**Alert Severity**")
        _sev_options = list(RISK_TIERS)
        dlg_alert_sev = st.multiselect(
            "Alert Severity",
            _sev_options,
            default=st.session_state.flt_alert_sev,
            label_visibility="collapsed",
            key="dlg_alert_sev",
        )
        st.markdown("**Sort Alerts By**")
        _sort_opts = [
            "Highest score first",
            "Lowest score first",
            "Highest severity first",
            "Lowest severity first",
            "Most recent first",
            "Oldest first",
            "User A\u2013Z",
            "User Z\u2013A",
        ]
        dlg_sort = st.selectbox(
            "Sort Alerts By",
            _sort_opts,
            index=_sort_opts.index(st.session_state.flt_sort_choice)
            if st.session_state.flt_sort_choice in _sort_opts else 0,
            label_visibility="collapsed",
            key="dlg_sort",
        )
        _pctl_col, _rows_col = st.columns(2)
        with _pctl_col:
            st.markdown("**Min Percentile**")
            dlg_min_pctl = st.slider(
                "Min Percentile",
                0.0, 100.0,
                st.session_state.flt_min_pctl,
                label_visibility="collapsed",
                key="dlg_min_pctl",
            )
        with _rows_col:
            st.markdown("**Max Rows**")
            dlg_max_rows = st.number_input(
                "Max Rows",
                min_value=10, max_value=10000,
                value=st.session_state.flt_max_rows,
                step=50,
                label_visibility="collapsed",
                key="dlg_max_rows",
            )

    st.markdown("")
    apply_col, reset_col = st.columns(2)
    with apply_col:
        if st.button("Apply", use_container_width=True, type="primary"):
            if isinstance(dr, tuple) and len(dr) == 2:
                st.session_state.flt_date_start = dr[0]
                st.session_state.flt_date_end = dr[1]
            st.session_state.flt_risk = rl if rl else list(RISK_TIERS)
            if _on_alerts:
                st.session_state.flt_alert_sev = dlg_alert_sev if dlg_alert_sev else ["CRITICAL", "HIGH"]
                st.session_state.flt_sort_choice = dlg_sort
                st.session_state.flt_min_pctl = float(dlg_min_pctl)
                st.session_state.flt_max_rows = int(dlg_max_rows)
            st.rerun()
    with reset_col:
        if st.button("Reset", use_container_width=True):
            st.session_state.flt_date_start = _DS_MIN
            st.session_state.flt_date_end = _DS_MAX
            st.session_state.flt_risk = list(RISK_TIERS)
            if _on_alerts:
                st.session_state.flt_alert_sev = ["CRITICAL", "HIGH"]
                st.session_state.flt_sort_choice = "Highest score first"
                st.session_state.flt_min_pctl = 0.0
                st.session_state.flt_max_rows = 500
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


# ── Cached Overview aggregations ──────────────────────────────────────────────
# These run on 1.2M rows so must be memoised; they recompute only when filters
# change, not on every Streamlit rerun.

@st.cache_data(show_spinner=False)
def _ov_kpis(date_start, date_end, risk_levels: tuple) -> dict:
    """All 7 Overview KPI values in one pass over the filtered frame."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    risk_str = fdf["ae_risk_band"].astype(str)
    n = max(len(fdf), 1)
    return {
        "total_users":    int(fdf["user"].nunique()),
        "total_records":  len(fdf),
        "critical_users": int(fdf.loc[risk_str == "CRITICAL", "user"].nunique()),
        "high_users":     int(fdf.loc[risk_str == "HIGH",     "user"].nunique()),
        "medium_users":   int(fdf.loc[risk_str == "MEDIUM",   "user"].nunique()),
        "avg_anomaly":    float(fdf["if_anomaly_score"].mean()),
        "detection_rate": float(risk_str.isin(["CRITICAL", "HIGH", "MEDIUM"]).sum() / n * 100),
    }


@st.cache_data(show_spinner=False)
def _ov_risk_counts(date_start, date_end, risk_levels: tuple) -> pd.DataFrame:
    """Risk-band value counts for the donut chart."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    rc = fdf["ae_risk_band"].astype(str).value_counts().reset_index()
    rc.columns = ["Risk Level", "Count"]
    return rc


@st.cache_data(show_spinner=False)
def _ov_daily_alerts(date_start, date_end, risk_levels: tuple) -> pd.DataFrame:
    """Daily alert counts by risk band for the trend bar chart."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    daily = (
        fdf.groupby([fdf["day"].dt.date, fdf["ae_risk_band"].astype(str)])
        .size()
        .reset_index(name="Count")
    )
    daily.columns = ["Date", "Risk Level", "Count"]
    return daily


@st.cache_data(show_spinner=False)
def _ov_histogram_sample(date_start, date_end, risk_levels: tuple, n: int = 50_000) -> pd.DataFrame:
    """Sampled (score, risk_band) pairs for the anomaly score histogram."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    sub = fdf[["if_anomaly_score", "ae_risk_band"]].copy()
    sub["ae_risk_band"] = sub["ae_risk_band"].astype(str)
    return sub if len(sub) <= n else sub.sample(n, random_state=42)


@st.cache_data(show_spinner=False)
def _ov_flag_counts(date_start, date_end, risk_levels: tuple, flags: tuple) -> dict[str, tuple[int, float]]:
    """Cross-channel flag sums: {flag: (count, pct)}."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    n = max(len(fdf), 1)
    return {
        f: (int(fdf[f].sum()), float(fdf[f].sum() / n * 100))
        for f in flags if f in fdf.columns
    }


def _ov_args() -> tuple:
    """Shorthand for the three filter keys used by all cached functions."""
    return (
        st.session_state.flt_date_start,
        st.session_state.flt_date_end,
        tuple(sorted(st.session_state.flt_risk)),
    )


# ── Cached Alerts aggregations ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _al_top_users(date_start, date_end, risk_levels: tuple) -> pd.DataFrame:
    """Top 10 users by max percentile for the Alerts tab."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    return (
        fdf.groupby("user", observed=True)
        .agg(
            max_percentile=("ae_percentile_rank", "max"),
            critical_count=("ae_risk_band", lambda x: (x == "CRITICAL").sum()),
            high_count=("ae_risk_band", lambda x: (x == "HIGH").sum()),
        )
        .reset_index()
        .sort_values("max_percentile", ascending=False)
        .head(10)
    )


@st.cache_data(show_spinner=False)
def _al_alert_data(
    date_start, date_end, risk_levels: tuple,
    alert_risk: tuple, min_pctl: float, max_results: int, sort_choice: str,
) -> pd.DataFrame:
    """Filtered + sorted alert feed for the Alerts tab."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    _RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    data = fdf[
        (fdf["ae_risk_band"].isin(alert_risk)) &
        (fdf["ae_percentile_rank"] >= min_pctl)
    ].copy()
    if sort_choice == "Highest score first":
        data = data.sort_values("ae_percentile_rank", ascending=False)
    elif sort_choice == "Lowest score first":
        data = data.sort_values("ae_percentile_rank", ascending=True)
    elif sort_choice in ("Highest severity first", "Lowest severity first"):
        _asc = sort_choice == "Lowest severity first"
        data["_risk_sort_key"] = data["ae_risk_band"].astype(str).map(_RISK_ORDER).fillna(-1)
        data = data.sort_values(
            ["_risk_sort_key", "ae_percentile_rank"], ascending=[_asc, False]
        ).drop(columns=["_risk_sort_key"])
    elif sort_choice == "Most recent first":
        data = data.sort_values("day", ascending=False)
    elif sort_choice == "Oldest first":
        data = data.sort_values("day", ascending=True)
    elif sort_choice == "User A–Z":
        data = data.sort_values("user", ascending=True)
    else:
        data = data.sort_values("user", ascending=False)
    return data.head(int(max_results))


# ── Cached Channels aggregations ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _ch_totals(date_start, date_end, risk_levels: tuple) -> dict[str, float]:
    """Channel volume totals for the Channels tab donut chart."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    result: dict[str, float] = {}
    for channel, feats in CHANNELS.items():
        valid = [f for f in feats if f in fdf.columns]
        if valid:
            result[channel] = float(fdf[valid].sum().sum())
    return result


@st.cache_data(show_spinner=False)
def _ch_box_sample(date_start, date_end, risk_levels: tuple, feature: str) -> pd.DataFrame:
    """Down-sampled (ae_risk_band, feature) frame for the Channels box plot."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    if feature not in fdf.columns:
        return pd.DataFrame()
    subset = fdf[["ae_risk_band", feature]]
    return subset if len(subset) <= MAX_PLOT_POINTS else subset.sample(MAX_PLOT_POINTS, random_state=42)


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

    _kpis = _ov_kpis(*_ov_args())
    total_users        = _kpis["total_users"]
    total_records      = _kpis["total_records"]
    critical_risk_users = _kpis["critical_users"]
    high_risk_users    = _kpis["high_users"]
    medium_risk_users  = _kpis["medium_users"]
    avg_anomaly        = _kpis["avg_anomaly"]
    detection_rate     = _kpis["detection_rate"]

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
            # Color badge based on score
            if score >= 95:
                badge_color = "#bb44f0"
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
        _cards_inner = ""
        for flag, label, color in _flag_info:
            if flag in CROSS_FLAGS and flag in _flag_counts_cache:
                count, pct = _flag_counts_cache[flag]
                _cards_inner += (
                    f"<div class='kpi-card' style='border-color:{color}'>"
                    f"<h3>{label}</h3>"
                    f"<h1 style='color:{color}'>{count:,}</h1>"
                    f"<p>{pct:.1f}% of records</p>"
                    f"</div>"
                )
        st.markdown(
            "<div class='kpi-scroll-wrapper'>"
            "<div class='kpi-scroll-arrow' id='cf-arrow-left'>&#8592;</div>"
            f"<div class='kpi-scroll-row' id='cf-row'>{_cards_inner}</div>"
            "<div class='kpi-scroll-arrow' id='cf-arrow-right'>&#8594;</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        components.html(
            "<script>"
            "(function(){"
            "  var p = window.parent.document;"
            "  function wire(){"
            "    var row = p.getElementById('cf-row');"
            "    var lft = p.getElementById('cf-arrow-left');"
            "    var rgt = p.getElementById('cf-arrow-right');"
            "    if(!row||!lft||!rgt){setTimeout(wire,100);return;}"
            "    lft.addEventListener('click',function(){row.scrollBy({left:-200,behavior:'smooth'});});"
            "    rgt.addEventListener('click',function(){row.scrollBy({left:200,behavior:'smooth'});});"
            "  }"
            "  wire();"
            "})();"
            "</script>",
            height=0,
        )
        st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)


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
        st.stop()

    _u_rows = merged_df.loc[merged_df["user"] == selected_user].reset_index(drop=True)
    if not _u_rows.empty:
        _u_mask = (
            _u_rows["ae_risk_band"].isin(st.session_state.flt_risk)
            & (_u_rows["day"].dt.date >= st.session_state.flt_date_start)
            & (_u_rows["day"].dt.date <= st.session_state.flt_date_end)
        )
        user_data = _u_rows[_u_mask].sort_values("day")
    else:
        user_data = pd.DataFrame()

    # ── Merge live data when simulation is active ───────────────────────────
    _inv_live_active = bool(st.session_state.live_mode or st.session_state.live_paused)
    _inv_live_count = 0
    if _inv_live_active and os.path.exists(LIVE_OUTPUT):
        _live_u = _get_live_user_data(selected_user)
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
        if _inv_live_active:
            st.info("No data for this user yet — waiting for live records to arrive.")
        else:
            st.warning("No data for this user in the current filter range.")
        st.stop()

    # ── Live investigation status banner ────────────────────────────────────
    if _inv_live_active:
        _inv_live_status = (
            "LIVE" if st.session_state.live_mode and not st.session_state.live_paused else "PAUSED"
        )
        _inv_live_color = "#e84545" if _inv_live_status == "LIVE" else "#d4a017"
        _inv_live_dot = "●" if _inv_live_status == "LIVE" else "⏸"
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

        # ── Alert Context Summary (shown when navigating from Alerts tab) ──
    _alert_ctx = st.session_state.get("inv_alert_context")
    if _alert_ctx and _alert_ctx.get("user") == selected_user:
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
        _raw_row = merged_df[
            (merged_df["user"] == selected_user) &
            (merged_df["day"] == _ctx_day_ts)
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
        _tc_raw = _get_alert_detail(selected_user, _ctx_day_ts, "top_contributors")
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
        if ueba_a_df is not None:
            _drill = ueba_a_df[
                (ueba_a_df["user"] == selected_user) &
                (ueba_a_df["day"] == _ctx_day_ts)
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

    # ── Auto-refresh while live simulation is running ───────────────────────
    if st.session_state.live_mode and not st.session_state.live_paused:
        time.sleep(1)
        st.rerun()


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
    components.html(
        "<script>"
        "(function(){"
        "  function styleSimBtns(){"
        "    var btns=window.parent.document.querySelectorAll('[data-testid=\"stButton\"] button');"
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
        "</script>",
        height=0,
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

            # Assign risk bands from numeric composite score
            def assign_live_risk_band(score):
                if pd.isna(score):
                    return "LOW"
                score = float(score)
                if score >= 95:
                    return "CRITICAL"
                elif score >= 90:
                    return "HIGH"
                elif score >= 80:
                    return "MEDIUM"
                return "LOW"

            live_df["ui_risk_band"] = live_df["ui_composite_score"].apply(assign_live_risk_band)
            _live_risk_counts = {
                tier: int((live_df["ui_risk_band"] == tier).sum()) for tier in RISK_TIERS
            }

            section_header("Filter by severity", "sh_live_sev")
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
                st.info("No live alerts match the current severity filter.")
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
                if score >= 95:
                    badge_color = "#bb44f0"
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

        # ── Date/risk summary (no filter button here) ──
        _active_risks = ", ".join(st.session_state.flt_risk) if len(st.session_state.flt_risk) < 4 else "All risk levels"
        st.caption(
            f"Date: {st.session_state.flt_date_start} to {st.session_state.flt_date_end}   |   Risk: {_active_risks}"
        )

        # Read alert controls from session state (set via filter modal)
        alert_risk  = st.session_state.flt_alert_sev or ["CRITICAL", "HIGH"]
        min_pctl    = st.session_state.flt_min_pctl
        max_results = st.session_state.flt_max_rows
        sort_choice = st.session_state.flt_sort_choice

        # Active alert filter summary pill
        _sev_summary = " ".join(
            f"<span style='background:{RISK_COLORS.get(t,'#888')}22;color:{RISK_COLORS.get(t,'#888')};"
            f"font-size:9px;font-family:JetBrains Mono,monospace;letter-spacing:1px;"
            f"padding:1px 7px;border:1px solid {RISK_COLORS.get(t,'#888')}55;"
            f"display:inline-block;margin-right:4px;'>{t}</span>"
            for t in RISK_TIERS if t in alert_risk
        )
        st.markdown(
            f"<p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#444;margin:4px 0 16px 0;'>"
            f"{_sev_summary}&nbsp; &middot;&nbsp; Min P{min_pctl:.0f}"
            f"&nbsp; &middot;&nbsp; {sort_choice}&nbsp; &middot;&nbsp; Max {max_results:,} rows</p>",
            unsafe_allow_html=True,
        )

        alert_data = _al_alert_data(
            *_ov_args(), tuple(alert_risk), min_pctl, max_results, sort_choice
        )

        # Cap card rendering to keep the UI responsive
        CARD_LIMIT = 10
        total_alerts = len(alert_data)
        card_data = alert_data.head(CARD_LIMIT)

        if total_alerts == 0:
            st.info("No alerts match the current filters.")
        else:
            # ── Alert Feed header with Filter button inline ──
            _af_left, _af_right = st.columns([9, 1], vertical_alignment="bottom")
            with _af_left:
                st.markdown("<div class='section-header'>Alert Feed</div>", unsafe_allow_html=True)
            with _af_right:
                if st.button("Filter", key="al_flt", use_container_width=True):
                    show_filters()
            if total_alerts > CARD_LIMIT:
                st.caption(
                    f"Displaying top {CARD_LIMIT} of {total_alerts:,} matching alerts. "
                    "Use the export button below for the complete set."
                )

            # ── Column header row ──
            _HDR = (
                "font-family:JetBrains Mono,monospace;font-size:9px;color:#444;"
                "text-transform:uppercase;letter-spacing:1.5px;"
            )
            _h_risk, _h_info, _h_day, _h_pctl, _h_btn = st.columns([1, 5, 2, 1, 2])
            _h_risk.markdown(f"<span style='{_HDR}'>Risk</span>", unsafe_allow_html=True)
            _h_info.markdown(f"<span style='{_HDR}'>User / Investigation hint</span>", unsafe_allow_html=True)
            _h_day.markdown(f"<span style='{_HDR}'>Day</span>", unsafe_allow_html=True)
            _h_pctl.markdown(f"<span style='{_HDR}'>Percentile</span>", unsafe_allow_html=True)
            st.markdown(
                "<div style='border-bottom:1px solid #1a1a1a;margin:0 0 2px 0;'></div>",
                unsafe_allow_html=True,
            )

            # ── Per-alert card rows ──
            for i, row in enumerate(card_data.itertuples()):
                risk    = getattr(row, "ae_risk_band",    "LOW")
                user    = getattr(row, "user",           "—")
                day_val = getattr(row, "day",            None)
                day_str = day_val.strftime("%Y-%m-%d") if hasattr(day_val, "strftime") else str(day_val)
                pctl    = getattr(row, "ae_percentile_rank", 0.0)
                top_raw = _get_alert_detail(getattr(row, "user", ""), getattr(row, "day", None), "top_contributors")
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
