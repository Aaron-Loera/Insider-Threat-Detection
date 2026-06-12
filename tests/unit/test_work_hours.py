"""Tests for inference-time work-hour envelopes (ueba.features.work_hours)."""

import pandas as pd
import pytest

from ueba.features.work_hours import apply_off_hours_flags, missing_users


@pytest.fixture
def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": ["early01", "default02"],
            "start_hour": [6, 9],
            "end_hour": [14, 17],
            "schedule_complete": [True, False],
        }
    )


def test_per_user_envelope_applied(schedule):
    events = pd.DataFrame({"user": ["early01"] * 3, "hour": [5, 10, 15]})
    flags = apply_off_hours_flags(events, schedule)
    # Envelope 6-14: hour 5 early, 10 inside, 15 late
    assert list(flags) == [True, False, True]


def test_unknown_user_falls_back_to_population_default(schedule):
    events = pd.DataFrame({"user": ["ghost99"] * 3, "hour": [8, 12, 18]})
    flags = apply_off_hours_flags(events, schedule)
    # Fallback 9-17: 8 early, 12 inside, 18 late
    assert list(flags) == [True, False, True]


def test_none_table_applies_default_everywhere():
    events = pd.DataFrame({"user": ["anyone"] * 2, "hour": [8, 12]})
    assert list(apply_off_hours_flags(events, None)) == [True, False]


def test_envelope_bounds_are_inclusive(schedule):
    events = pd.DataFrame({"user": ["early01"] * 2, "hour": [6, 14]})
    assert list(apply_off_hours_flags(events, schedule)) == [False, False]


def test_hour_derived_from_timestamp(schedule):
    events = pd.DataFrame(
        {
            "user": ["early01", "early01"],
            "timestamp": pd.to_datetime(["2010-01-01 05:30:00", "2010-01-01 10:15:00"]),
        }
    )
    assert list(apply_off_hours_flags(events, schedule)) == [True, False]


def test_missing_hour_and_timestamp_raises(schedule):
    with pytest.raises(ValueError):
        apply_off_hours_flags(pd.DataFrame({"user": ["u"]}), schedule)


def test_missing_users_reports_cold_start(schedule):
    # default02 has schedule_complete=False -> also cold-start
    assert missing_users(["early01", "default02", "ghost99"], schedule) == {"default02", "ghost99"}
    assert missing_users(["anyone"], None) == {"anyone"}
