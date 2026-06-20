"""Alerts page — live-simulation controls, live feed, and the static alert table."""

import html as _html_mod
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from db import get_all_dispositions

from lib import live
from lib.aggregations import _al_top_users
from lib.data import ALERT_STATUS_OPTIONS, _on_status_change, get_feature_groups, load_data
from lib.data import get_alert_detail as _get_alert_detail
from lib.filters import _get_filtered_df, _ov_args, show_filters
from lib.labels import build_alert_summary
from lib.live import _cached_live_rows
from lib.theme import PLOTLY_LAYOUT, RISK_COLORS, RISK_TIERS
from lib.ui import section_header
from ueba.risk import assign_band_from_percentile


def render() -> None:
    _merged_df, *_ = load_data()
    _RAW_FEATURES, CROSS_FLAGS, _CHANNELS = get_feature_groups()
    filtered_df = _get_filtered_df()
    st.markdown(
        "<div class='page-header-block'>"
        "<h1 class='page-title'>Alerts</h1>"
        "<p class='page-subtitle'>Sortable, filterable list of anomaly detection alerts with behavioral context.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Live-simulation control row ────────────────────────────
    # The subprocess start/stop/pause/resume logic lives in lib.live (which also
    # picks live_replay.py vs live_simulation.py based on the cloud mount). The
    # buttons just call those and rerun.
    ctrl_start, ctrl_pause = st.columns([3, 2])
    with ctrl_start:
        if not st.session_state.live_mode:
            if st.button("▶ START LIVE SIMULATION", key="start_live", use_container_width=True):
                live.start_simulation()
                st.rerun()
        else:
            if st.button("⏹ STOP LIVE SIMULATION", key="stop_live", use_container_width=True):
                live.stop_simulation()
                st.rerun()
    with ctrl_pause:
            if not st.session_state.live_paused:
                if st.button("⏸   PAUSE", key="pause_live", use_container_width=True):
                    live.pause_simulation()
                    st.rerun()
            else:
                if st.button("▶   RESUME", key="resume_live", use_container_width=True):
                    live.resume_simulation()
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
