"""Tests for chronological_split (ueba.features.preprocessing)."""

import pandas as pd
import pytest

from ueba.features.preprocessing import chronological_split


def test_split_preserves_all_rows(ueba_df):
    train, test = chronological_split(df=ueba_df.copy(), split_ratio=0.9)
    assert len(train) + len(test) == len(ueba_df)


def test_split_is_chronological(ueba_df):
    train, test = chronological_split(df=ueba_df.copy(), split_ratio=0.9)
    assert train["day"].max() < test["day"].min()


def test_split_ratio_drives_cutoff_day(ueba_df):
    # 30 unique days, ratio 0.9 -> cutoff index int(30*0.9)=27 -> the cutoff day
    # itself is included in train, so train holds 28 unique days and test 2.
    train, test = chronological_split(df=ueba_df.copy(), split_ratio=0.9)
    assert train["day"].nunique() == 28
    assert test["day"].nunique() == 2


def test_split_stable_under_unsorted_input(ueba_df):
    shuffled = ueba_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    train_a, test_a = chronological_split(df=ueba_df.copy(), split_ratio=0.8)
    train_b, test_b = chronological_split(df=shuffled.copy(), split_ratio=0.8)
    assert len(train_a) == len(train_b)
    assert set(test_a["day"]) == set(test_b["day"])


def test_split_normalizes_user_and_day(ueba_df):
    df = ueba_df.copy()
    df["user"] = df["user"].str.upper() + " "
    train, test = chronological_split(df=df, split_ratio=0.9)
    combined = pd.concat([train, test])
    assert combined["user"].str.islower().all()
    assert (combined["day"].dt.normalize() == combined["day"]).all()


def test_split_requires_input():
    with pytest.raises(ValueError):
        chronological_split()
