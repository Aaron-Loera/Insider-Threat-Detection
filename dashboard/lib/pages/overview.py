"""Overview page — KPIs, risk distribution, trend, score histogram, cross-flags."""

import pandas as pd
import plotly.express as px
import streamlit as st
from db import get_all_dispositions

from lib.aggregations import (
    _ov_daily_alerts,
    _ov_flag_counts,
    _ov_histogram_sample,
    _ov_kpis,
    _ov_risk_counts,
)
from lib.data import get_feature_groups, load_data
from lib.filters import _filter_bar, _get_filtered_df, _ov_args
from lib.theme import PLOTLY_LAYOUT, RISK_COLORS
from lib.ui import _render_card_carousel, section_header
from ueba.risk import assign_band_from_percentile


def render() -> None:
    _merged_df, user_risk, *_ = load_data()
    _RAW_FEATURES, CROSS_FLAGS, _CHANNELS = get_feature_groups()
    filtered_df = _get_filtered_df()
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
