"""
AlertSequencer.py — Temporal alert sequence detection.

Groups consecutive or near-consecutive elevated-risk days (MEDIUM or above,
based on composite_risk_band) per user into sequences, assigns unique sequence
IDs, computes sequence-level metadata, and back-populates alert_sequence_id in
the main alert DataFrame.
"""

import os

import numpy as np
import pandas as pd

# Risk bands that qualify a day as elevated (MEDIUM, HIGH, CRITICAL)
_ELEVATED_BANDS: frozenset[str] = frozenset({"MEDIUM", "HIGH", "CRITICAL"})

# Default output path for the sequence metadata table
_DEFAULT_SEQUENCES_PATH = os.path.join(
    "explainability", "alert_table", "alert_sequences.parquet"
)


def detect_sequences(
    alert_df: pd.DataFrame,
    gap_days: int = 2,
    max_span_days: int = 30,
    sequences_path: str = _DEFAULT_SEQUENCES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detects multi-day alert sequences per user and back-populates alert_sequence_id.

    For each user the elevated-risk alerts (composite_risk_band in MEDIUM/HIGH/CRITICAL)
    are sorted chronologically. A new sequence starts whenever the gap to the previous
    elevated alert exceeds gap_days OR adding the next alert would cause the sequence to
    span more than max_span_days calendar days. Only sequences with at least two alerts
    receive an alert_sequence_id — single-day spikes are left with alert_sequence_id=None
    so they do not earn the sequence membership bonus in compute_priority().

    Sequence IDs have the form ``SEQ-<user>-<YYYYMMDD>`` where the date is the
    sequence's first day, making them human-readable and collision-free within a user.

    Args:
        alert_df: Alert DataFrame produced by AlertObjectBuilder.build_alert_df().
                  Must contain: user, day, composite_risk_band, ae_percentile_rank,
                  if_percentile_rank columns.
        gap_days: Maximum calendar-day gap between consecutive elevated alerts that
                  still keeps them in the same sequence. Alerts exactly gap_days apart
                  are kept in the same sequence (strictly-greater comparison).
        max_span_days: Hard cap on the calendar-day span of any single sequence.
                       When adding the next alert would push (current_day - start_day)
                       above this limit, a new sequence is started instead.
        sequences_path: Destination path for the sequence metadata parquet file.
                        Relative to the working directory (project root when run from
                        a notebook).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            alert_df_out  — copy of the input with alert_sequence_id back-populated
                            for alerts that belong to a multi-day sequence.
            sequences_df  — one row per detected sequence with columns:
                            alert_sequence_id, user, start_date, end_date,
                            duration_days, alert_count, max_percentile, is_escalating.
    """
    df = alert_df.copy()
    df["day"] = pd.to_datetime(df["day"])

    # Per-alert peak signal used for sequence-level percentile stats
    df["_peak_pct"] = np.maximum(
        df["ae_percentile_rank"].fillna(0).values,
        df["if_percentile_rank"].fillna(0).values,
    )

    # Reset any prior sequence assignments so re-runs are idempotent
    df["alert_sequence_id"] = None

    elevated_pos = df.index[df["composite_risk_band"].isin(_ELEVATED_BANDS)].tolist()
    elevated = df.loc[elevated_pos].sort_values(["user", "day"])

    sequence_records: list[dict] = []

    for user, user_df in elevated.groupby("user", sort=False):
        user_df = user_df.sort_values("day")
        days = user_df["day"].values          # numpy datetime64 array
        indices = user_df.index.values        # original df index labels
        pct_vals_all = user_df["_peak_pct"].values

        # Build variable-length windows that respect both gap_days and max_span_days.
        # Iterate row by row; start a new sequence whenever either limit is breached.
        seq_start_pos = 0
        seq_start_day = days[0]

        for i in range(1, len(days) + 1):
            # Decide whether to close the current sequence (at position i-1 being the last)
            close = False
            if i == len(days):
                close = True  # end of user's data — always flush
            else:
                gap = int((days[i] - days[i - 1]) / np.timedelta64(1, "D"))
                span = int((days[i] - seq_start_day) / np.timedelta64(1, "D"))
                if gap > gap_days or span > max_span_days:
                    close = True

            if close:
                seg_indices = indices[seq_start_pos:i]
                seg_pct = pct_vals_all[seq_start_pos:i]
                seg_days_slice = days[seq_start_pos:i]

                if len(seg_indices) >= 2:
                    start = pd.Timestamp(seg_days_slice[0])
                    end = pd.Timestamp(seg_days_slice[-1])
                    seq_id = f"SEQ-{user}-{start.strftime('%Y%m%d')}"
                    df.loc[seg_indices, "alert_sequence_id"] = seq_id

                    mid = len(seg_pct) // 2
                    first_half_peak = float(seg_pct[:mid].max()) if mid > 0 else 0.0
                    second_half_peak = float(seg_pct[mid:].max())

                    sequence_records.append(
                        {
                            "alert_sequence_id": seq_id,
                            "user": user,
                            "start_date": start,
                            "end_date": end,
                            "duration_days": int((end - start).days),
                            "alert_count": len(seg_indices),
                            "max_percentile": round(float(seg_pct.max()), 4),
                            "is_escalating": second_half_peak > first_half_peak,
                        }
                    )

                if i < len(days):
                    seq_start_pos = i
                    seq_start_day = days[i]

    df = df.drop(columns=["_peak_pct"])

    # Build the sequences table (empty schema preserved when no sequences found)
    if sequence_records:
        sequences_df = pd.DataFrame(sequence_records)
    else:
        sequences_df = pd.DataFrame(
            columns=[
                "alert_sequence_id",
                "user",
                "start_date",
                "end_date",
                "duration_days",
                "alert_count",
                "max_percentile",
                "is_escalating",
            ]
        )

    os.makedirs(os.path.dirname(os.path.abspath(sequences_path)), exist_ok=True)
    sequences_df.to_parquet(sequences_path, index=False)

    n_users = sequences_df["user"].nunique() if not sequences_df.empty else 0
    n_alerts_in_seq = int(df["alert_sequence_id"].notna().sum())
    print(
        f"Detected {len(sequences_df)} sequences across {n_users} users "
        f"({n_alerts_in_seq} alerts back-populated) → {sequences_path}"
    )

    return df, sequences_df
