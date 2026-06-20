import os
import sys

# Make both the dashboard dir and the project root importable BEFORE any local
# import. The dashboard dir lets `from db import …` / `from lib.… import …`
# resolve; the project root lets the `config` shim resolve. This must run first:
# lib.data (imported below) does `from config import …` at its module top, so the
# root has to be on sys.path before that import — `streamlit run` only adds the
# entrypoint's dir, and AppTest adds neither. Idempotent.
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DASHBOARD_DIR)
for _p in (_DASHBOARD_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.stderr.write("[APP] module-level execution started\n")
sys.stderr.flush()

import time

import streamlit as st
from db import init_db
from lib import live
from lib.auth import require_auth
from lib.data import get_feature_groups, load_data, load_peer_baselines, load_ueba_a, load_user_profiles
from lib.filters import init_filter_state
from lib.live import _cached_live_max_date
from lib.pages import alerts, channels, investigation, overview
from lib.theme import inject_base_css
from lib.ui import render_chrome, render_sidebar
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Page config → base CSS → auth gate → DB init
# ──────────────────────────────────────────────────────────────
icon_path = os.path.join(_DASHBOARD_DIR, "assets", "dsk_kitten.png")
page_icon = Image.open(icon_path) if os.path.exists(icon_path) else "■"

st.set_page_config(
    page_title="InsiderGuard AI — Insider Threat Detection",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_base_css()
require_auth()
init_db()

# ──────────────────────────────────────────────────────────────
# Data bootstrap — load everything once, gate the app on success
# ──────────────────────────────────────────────────────────────
import logging as _startup_log

_slog = _startup_log.getLogger("ueba.startup")
try:
    merged_df, user_risk, all_users, _DS_MIN, _DS_MAX = load_data()
    _slog.warning("[STARTUP] load_data() complete")
    # Warm the secondary caches so the first page render doesn't pay for them and a
    # missing optional input surfaces here (gated) rather than mid-page.
    load_ueba_a()
    load_peer_baselines()
    load_user_profiles()
    get_feature_groups()
    _slog.warning("[STARTUP] secondary loads complete — DATA_LOADED=True")
    DATA_LOADED = True
except Exception:
    import traceback as _tb
    DATA_LOADED = False
    _LOAD_ERROR = _tb.format_exc()
    _slog.warning(f"[STARTUP] load_data FAILED: {_LOAD_ERROR}")
else:
    _LOAD_ERROR = None

if not DATA_LOADED:
    st.title("INSIDER THREAT DETECTION")
    st.error("**Failed to load data.** See details below.")
    if _LOAD_ERROR:
        st.code(_LOAD_ERROR, language="text")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Live state → nav request → sidebar → filter state → chrome
# ──────────────────────────────────────────────────────────────
live.init_live_state()

# Consume any programmatic navigation request NOW — before the radio is
# instantiated — so we can write session_state.nav_page freely.
if st.session_state.get("_nav_request"):
    st.session_state.nav_page = st.session_state.pop("_nav_request")

active_page = render_sidebar()

# Filter state needs the dataset date bounds; live records can extend the ceiling.
_ds_live_max = _cached_live_max_date(_DS_MAX)
init_filter_state(_DS_MIN, _DS_MAX, _ds_live_max)

render_chrome()

# ──────────────────────────────────────────────────────────────
# Dispatch to the active page
# ──────────────────────────────────────────────────────────────
_PAGES = {
    "Overview": overview.render,
    "Investigation": investigation.render,
    "Alerts": alerts.render,
    "Channels": channels.render,
}
_PAGES[active_page]()

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
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("---")
