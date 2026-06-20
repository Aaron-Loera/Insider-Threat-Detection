"""Unit tests for lib.aggregations against the synthetic dashboard tree.

These exercise the cache-closure rewiring: each aggregation calls load_data()
internally (no app.py globals), so importing and calling them under the agg_env
fixture (UEBA_BASE_DIR → synthetic tree) returns values derived from that frame.
"""

import pandas as pd


def _load_lib(agg_env):
    """Import the lib modules fresh under the redirected config and return them."""
    import lib.aggregations as agg
    import lib.data as data
    return agg, data


def test_cached_filtered_df_respects_date_and_risk(agg_env):
    agg, data = _load_lib(agg_env)
    merged, *_ = data.load_data()
    ds_min, ds_max = merged["day"].min().date(), merged["day"].max().date()

    full = agg._cached_filtered_df(ds_min, ds_max, ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
    assert len(full) == len(merged)  # all rows, all bands, full range

    # Restricting to one band drops rows of other bands.
    only_low = agg._cached_filtered_df(ds_min, ds_max, ("LOW",))
    assert set(only_low["ae_risk_band"].astype(str).unique()) <= {"LOW"}
    assert len(only_low) <= len(full)

    # A single-day window keeps only that day.
    one_day = agg._cached_filtered_df(ds_min, ds_min, ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
    assert (one_day["day"].dt.date == ds_min).all()


def test_ov_kpis_consistent_with_filtered_frame(agg_env):
    agg, data = _load_lib(agg_env)
    merged, *_ = data.load_data()
    ds_min, ds_max = merged["day"].min().date(), merged["day"].max().date()
    risk = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    kpis = agg._ov_kpis(ds_min, ds_max, risk)
    fdf = agg._cached_filtered_df(ds_min, ds_max, risk)

    assert kpis["total_records"] == len(fdf)
    assert kpis["total_users"] == fdf["user"].nunique()
    assert 0.0 <= kpis["detection_rate"] <= 100.0
    # critical/high/medium user counts never exceed total users.
    assert kpis["critical_users"] <= kpis["total_users"]
    assert kpis["high_users"] <= kpis["total_users"]


def test_al_top_users_sorted_desc_and_capped(agg_env):
    agg, data = _load_lib(agg_env)
    merged, *_ = data.load_data()
    ds_min, ds_max = merged["day"].min().date(), merged["day"].max().date()
    risk = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    top = agg._al_top_users(ds_min, ds_max, risk)
    assert len(top) <= 10
    # Sorted by max_percentile descending.
    pctls = top["max_percentile"].tolist()
    assert pctls == sorted(pctls, reverse=True)
    # Matches a direct groupby-max on the filtered frame for the same users.
    fdf = agg._cached_filtered_df(ds_min, ds_max, risk)
    expected = fdf.groupby("user", observed=True)["ae_percentile_rank"].max()
    for row in top.itertuples():
        assert row.max_percentile == expected[row.user]


def test_ov_risk_counts_sums_to_total(agg_env):
    agg, data = _load_lib(agg_env)
    merged, *_ = data.load_data()
    ds_min, ds_max = merged["day"].min().date(), merged["day"].max().date()
    risk = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    rc = agg._ov_risk_counts(ds_min, ds_max, risk)
    fdf = agg._cached_filtered_df(ds_min, ds_max, risk)
    assert rc["Count"].sum() == len(fdf)
    assert set(rc.columns) == {"Risk Level", "Count"}


def test_ch_totals_non_negative_and_keyed_by_channel(agg_env):
    agg, data = _load_lib(agg_env)
    merged, *_ = data.load_data()
    ds_min, ds_max = merged["day"].min().date(), merged["day"].max().date()
    risk = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    totals = agg._ch_totals(ds_min, ds_max, risk)
    _, _, channels = data.get_feature_groups()
    assert set(totals) <= set(channels)
    assert all(v >= 0 for v in totals.values())


def test_get_feature_groups_filters_to_present_columns(agg_env):
    _, data = _load_lib(agg_env)
    merged, *_ = data.load_data()
    raw, flags, channels = data.get_feature_groups()
    cols = set(merged.columns)
    assert all(f in cols for f in raw)
    assert all(f in cols for f in flags)
    for feats in channels.values():
        assert all(f in cols for f in feats)
        assert feats  # no empty channels


def test_corr_matrix_square_over_requested_features(agg_env):
    agg, data = _load_lib(agg_env)
    merged, *_ = data.load_data()
    ds_min, ds_max = merged["day"].min().date(), merged["day"].max().date()
    raw, _, _ = data.get_feature_groups()
    feats = tuple(raw[:5])

    corr = agg._corr_matrix(ds_min, ds_max, ("LOW", "MEDIUM", "HIGH", "CRITICAL"), feats)
    assert isinstance(corr, pd.DataFrame)
    assert list(corr.columns) == list(feats)
    assert corr.shape == (len(feats), len(feats))
