"""Per-user work-hours envelopes at inference time.

Preprocessing derives each user's business-hour envelope from logon history
(compute_user_work_hours in ueba.features.preprocessing) and persists it to
user_work_hours.parquet. CLAUDE.md has always said the table "must be
reapplied at inference time" — this module is that reapplication path
(CLEANUP_REPORT gap 2):

- apply_off_hours_flags() flags raw events against the per-user envelope with
  the population fallback for unknown users — the same `_compute_off_hours`
  logic preprocessing uses, exposed for inference-side feature building.
- missing_users() lets a scorer detect cold-start users whose off-hours
  features were necessarily built with the population default, so the
  condition is logged instead of silent.
"""

import os

import pandas as pd

from ueba import config
from ueba.constants import WORK_HOURS
from ueba.features.preprocessing import _compute_off_hours


def load_user_work_hours(path: str | None = None) -> pd.DataFrame | None:
    """Load the persisted per-user schedule table; None when absent.

    Columns: [user, start_hour, end_hour, schedule_complete].
    """
    path = path or config.USER_WORK_HOURS_PATH
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def apply_off_hours_flags(
    events_df: pd.DataFrame,
    user_work_hours: pd.DataFrame | None,
    default_hours: tuple = WORK_HOURS,
) -> pd.Series:
    """Boolean Series — True where an event falls outside its user's envelope.

    Args:
        events_df: Event-level frame with a "user" column and either an integer
            "hour" column or a datetime "timestamp" column to derive it from.
        user_work_hours: Table from load_user_work_hours(); None applies the
            population default to all rows.
        default_hours: Fallback (start_hour, end_hour) for unknown users.
    """
    if "hour" in events_df.columns:
        hour = events_df["hour"]
    elif "timestamp" in events_df.columns:
        hour = pd.to_datetime(events_df["timestamp"]).dt.hour
    else:
        raise ValueError("events_df needs an 'hour' or 'timestamp' column")
    return _compute_off_hours(hour, events_df["user"], user_work_hours, default_hours)


def missing_users(users, user_work_hours: pd.DataFrame | None) -> set:
    """Users with no derived schedule (cold-start) — their off-hours features
    were built with the population default rather than a personal envelope."""
    if user_work_hours is None:
        return set(users)
    complete = user_work_hours[user_work_hours["schedule_complete"]]["user"]
    return set(users) - set(complete)
