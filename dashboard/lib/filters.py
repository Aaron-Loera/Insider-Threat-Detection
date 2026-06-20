"""Global filter state, the filter dialog, and the per-page filter bar.

The dashboard keeps its filter selections in session_state (flt_* keys) so they
persist across page navigation. init_filter_state() seeds those keys on first run
and records the dataset's date bounds (formerly the _DS_MIN / _DS_MAX / _ds_live_max
module globals) so the dialog can clamp its date picker without the page bodies
having to thread them through. show_filters() and _filter_bar() keep their original
call signatures (no-arg / key-only) so page code is unchanged.
"""

import pandas as pd
import streamlit as st

from lib.aggregations import _cached_filtered_df
from lib.theme import RISK_TIERS

# Dataset date bounds, set once by init_filter_state(). _DS_LIVE_MAX may exceed
# _DS_MAX once a live simulation has appended later-dated records.
_DS_MIN = None
_DS_MAX = None
_DS_LIVE_MAX = None


def init_filter_state(ds_min, ds_max, ds_live_max) -> None:
    """Record dataset date bounds and seed the flt_* session-state defaults.

    Idempotent: the date bounds are refreshed every run (cheap), while the flt_*
    keys are only initialised when absent so user selections survive reruns.
    """
    global _DS_MIN, _DS_MAX, _DS_LIVE_MAX
    _DS_MIN, _DS_MAX, _DS_LIVE_MAX = ds_min, ds_max, ds_live_max

    if "flt_date_start" not in st.session_state:
        st.session_state.flt_date_start = ds_min
    if "flt_date_end" not in st.session_state:
        st.session_state.flt_date_end = ds_max
    if "flt_risk" not in st.session_state:
        st.session_state.flt_risk = list(RISK_TIERS)
    if "flt_min_pctl" not in st.session_state:
        st.session_state.flt_min_pctl = 0.0
    if "flt_max_rows" not in st.session_state:
        st.session_state.flt_max_rows = 500
    if "flt_sort_choice" not in st.session_state:
        st.session_state.flt_sort_choice = "Highest score first"


@st.dialog("Filters")
def show_filters():
    active_page = st.session_state.get("nav_page")
    st.markdown("**Date Range**")
    dr = st.date_input(
        "Date Range",
        value=(st.session_state.flt_date_start, st.session_state.flt_date_end),
        min_value=_DS_MIN,
        max_value=_DS_LIVE_MAX,
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
        st.markdown("**Triage Status**")
        dlg_disp_filter = st.radio(
            "Triage Status",
            options=["Show New Only", "Show All", "Show Investigating", "Show Resolved", "Show Dismissed"],
            index=["Show New Only", "Show All", "Show Investigating", "Show Resolved", "Show Dismissed"].index(
                st.session_state.get("flt_disp_filter", "Show New Only")
            ),
            horizontal=False,
            label_visibility="collapsed",
            key="dlg_disp_filter",
        )
        dlg_view_suppressed = st.checkbox(
            "View Suppressed Alerts",
            value=st.session_state.get("flt_view_suppressed", False),
            key="dlg_view_suppressed",
        )
        st.markdown("---")
        st.markdown("**Sort Alerts By**")
        _sort_opts = [
            "Highest score first",
            "Lowest score first",
            "Highest severity first",
            "Lowest severity first",
            "Most recent first",
            "Oldest first",
            "User A–Z",
            "User Z–A",
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
                st.session_state.flt_disp_filter = dlg_disp_filter
                st.session_state.flt_view_suppressed = dlg_view_suppressed
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
                st.session_state.flt_disp_filter = "Show New Only"
                st.session_state.flt_view_suppressed = False
                st.session_state.flt_sort_choice = "Highest score first"
                st.session_state.flt_min_pctl = 0.0
                st.session_state.flt_max_rows = 500
            st.rerun()


def _get_filtered_df() -> pd.DataFrame:
    """Return merged_df sliced by current session_state filter values."""
    return _cached_filtered_df(
        st.session_state.flt_date_start,
        st.session_state.flt_date_end,
        tuple(sorted(st.session_state.flt_risk)),
    )


def _ov_args() -> tuple:
    """Shorthand for the three filter keys used by all cached functions."""
    return (
        st.session_state.flt_date_start,
        st.session_state.flt_date_end,
        tuple(sorted(st.session_state.flt_risk)),
    )


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
