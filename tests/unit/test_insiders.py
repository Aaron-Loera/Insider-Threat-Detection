"""Tests for get_insiders / build_insider_mask / prepare_ae_training_data (ueba.models.data_prep)."""

import pandas as pd

from ueba.models.data_prep import build_insider_mask, get_insiders, prepare_ae_training_data


def test_get_insiders_filters_by_version(insiders_csv):
    insiders = get_insiders(insiders_csv, "4.2")
    assert len(insiders) == 2
    assert set(insiders.columns) == {"user", "start_day", "end_day", "scenario"}


def test_get_insiders_normalizes_users_and_days(insiders_csv):
    insiders = get_insiders(insiders_csv, "4.2")
    # Users are stripped + lowercased; start/end are normalized to midnight.
    assert "acm2278" in insiders["user"].values
    row = insiders[insiders["user"] == "acm2278"].iloc[0]
    assert row["start_day"] == pd.Timestamp("2010-08-01")
    assert row["end_day"] == pd.Timestamp("2010-09-15")


def test_get_insiders_accepts_float_version(insiders_csv):
    assert len(get_insiders(insiders_csv, 4.2)) == 2
    assert len(get_insiders(insiders_csv, "5.2")) == 1


def test_get_insiders_return_all(insiders_csv):
    insiders, full = get_insiders(insiders_csv, "4.2", return_all=True)
    assert len(insiders) == 2
    assert len(full) == 3


def test_insider_mask_window_edges_inclusive(ueba_df, insider_windows):
    mask = build_insider_mask(ueba_df, insider_windows)
    flagged = ueba_df[mask]

    u3 = flagged[flagged["user"] == "user03"]["day"]
    assert u3.min() == pd.Timestamp("2010-01-10")  # start day included
    assert u3.max() == pd.Timestamp("2010-01-15")  # end day included
    assert len(u3) == 6


def test_insider_mask_only_flags_window_users(ueba_df, insider_windows):
    mask = build_insider_mask(ueba_df, insider_windows)
    assert set(ueba_df[mask]["user"].unique()) == {"user03", "user07"}


def test_insider_mask_empty_windows(ueba_df):
    empty = pd.DataFrame(columns=["user", "start_day", "end_day"])
    assert not build_insider_mask(ueba_df, empty).any()


def test_prepare_ae_training_data_excludes_insiders(ueba_df, insider_windows):
    train_fit, train_val, train_normal, gated_df, insider_mask = prepare_ae_training_data(
        ueba_df.copy(), insider_windows, val_ratio=0.15
    )
    # Baseline gate applied: every surviving row has >= 14 days of history.
    assert gated_df["baseline_complete"].all()
    # Insider (user, day) rows are excluded from the normal-behavior pool.
    flagged = gated_df[insider_mask]
    merged = train_normal.merge(flagged[["user", "day"]], on=["user", "day"], how="inner")
    assert merged.empty
    # Fit/validation are a partition of the normal pool.
    assert len(train_fit) + len(train_val) == len(train_normal)
