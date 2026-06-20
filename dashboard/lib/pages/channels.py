"""Channels page — feature distributions across activity channels."""

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.aggregations import _ch_box_sample, _ch_totals, _channel_time_series, _corr_matrix
from lib.data import get_feature_groups, load_data
from lib.filters import _filter_bar, _ov_args
from lib.theme import CHANNEL_COLOR_MAP, PLOTLY_LAYOUT, RISK_COLORS, RISK_TIERS
from lib.ui import section_header


def render() -> None:
    merged_df, *_ = load_data()
    RAW_FEATURES, _CROSS_FLAGS, _CHANNELS = get_feature_groups()
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
