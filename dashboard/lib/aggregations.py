"""Cached aggregations over the filtered dashboard frame.

All functions take explicit cache-key args (date range, risk levels, …) and read
no session state. They obtain the underlying data by calling the cached loaders in
lib.data *internally* rather than closing over app.py module globals — load_data()
and get_feature_groups() are @st.cache_resource, so each call returns the same
in-process object and is never re-hashed. Cache keys stay identical to the former
inline versions, so invalidation semantics are unchanged.
"""

import pandas as pd
import streamlit as st

from lib.data import get_feature_groups, load_data, load_peer_baselines
from lib.theme import MAX_PLOT_POINTS


@st.cache_data(show_spinner=False)
def _cached_filtered_df(date_start, date_end, risk_levels: tuple) -> pd.DataFrame:
    """Return a cached slice of merged_df. Only recomputes when filters change."""
    merged_df, *_ = load_data()
    mask = (
        merged_df["ae_risk_band"].isin(risk_levels)
        & (merged_df["day"].dt.date >= date_start)
        & (merged_df["day"].dt.date <= date_end)
    )
    return merged_df[mask].copy()


@st.cache_data(show_spinner=False)
def _pop_channel_avgs() -> dict[str, float]:
    """Pre-compute population channel averages from the full dataset (run once)."""
    merged_df, *_ = load_data()
    _, _, channels = get_feature_groups()
    result: dict[str, float] = {}
    for channel, feats in channels.items():
        valid = [f for f in feats if f in merged_df.columns]
        if valid:
            result[channel] = float(merged_df[valid].mean().sum())
    return result


def _peer_channel_avgs(department: str, day_min=None, day_max=None) -> dict[str, float]:
    """Return per-channel peer averages for selected department and time window."""
    peer_baselines_df = load_peer_baselines()
    if peer_baselines_df is None or department is None:
        return {}

    df = peer_baselines_df.copy()

    # safer department match
    dept_mask = (
        df["department"].astype(str).str.upper()
        == str(department).upper()
    )
    df = df[dept_mask]

    # align to selected user's time window
    if day_min is not None:
        df = df[df["day"] >= pd.to_datetime(day_min)]

    if day_max is not None:
        df = df[df["day"] <= pd.to_datetime(day_max)]

    if df.empty:
        return {}

    _, _, channels = get_feature_groups()
    result = {}
    for channel, feats in channels.items():
        valid = [f for f in feats if f in df.columns]
        if valid:
            result[channel] = float(df[valid].mean().sum())

    return result


@st.cache_data(show_spinner=False)
def _channel_time_series(date_start, date_end, risk_levels: tuple) -> pd.DataFrame:
    """Cached channel-volume-by-day aggregation for the Channels tab."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    _, _, channels = get_feature_groups()
    parts = []
    for channel, feats in channels.items():
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


# ── Cached Channels aggregations ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _ch_totals(date_start, date_end, risk_levels: tuple) -> dict[str, float]:
    """Channel volume totals for the Channels tab donut chart."""
    fdf = _cached_filtered_df(date_start, date_end, risk_levels)
    _, _, channels = get_feature_groups()
    result: dict[str, float] = {}
    for channel, feats in channels.items():
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
