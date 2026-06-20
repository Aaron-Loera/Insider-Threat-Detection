"""Investigation page — per-user deep dive with a live-updating fragment.

The dynamic per-user content lives in a @st.fragment(run_every="2s") so Streamlit
reruns just that subtree every 2s (no time.sleep, no full-page rerun). The page
shell (render) builds the risk-ordered user selector and invokes the fragment.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from db import get_all_dispositions

from config import LIVE_OUTPUT
from lib.aggregations import _peer_channel_avgs, _pop_channel_avgs
from lib.data import get_alert_detail as _get_alert_detail
from lib.data import get_feature_groups, load_data, load_ueba_a
from lib.filters import _filter_bar
from lib.labels import (
    parse_top_contributors,
    parse_top_contributors_with_values,
    prettify_feature_name,
)
from lib.live import _get_live_user_data
from lib.theme import CHANNEL_COLOR_MAP, PLOTLY_LAYOUT, RISK_COLORS
from lib.ui import section_header
from ueba.risk import assign_band_from_percentile


@st.fragment(run_every="2s")
def _render_investigation_content() -> None:
    _user: str | None = st.session_state.get("inv_user_select")
    if _user is None:
        return

    _inv_merged, _, _, _, _ = load_data()
    _inv_ueba_a = load_ueba_a()
    RAW_FEATURES, CROSS_FLAGS, CHANNELS = get_feature_groups()

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


def render() -> None:
    _merged_df, user_risk, all_users, _DS_MIN, _DS_MAX = load_data()
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
