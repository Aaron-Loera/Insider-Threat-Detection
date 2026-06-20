"""Live-simulation output readers (the polling half of the live feature).

These functions read processed_datasets/live_results.jsonl (config.LIVE_OUTPUT),
which the live_simulation subprocess appends to while running. They are cached with
a 1-second TTL (below the Investigation fragment's 2s run_every) so each refresh
sees fresh rows without re-scanning the whole file every rerun. Only the most recent
_MAX_LIVE_ROWS rows are kept so a long simulation history doesn't blow up parse time.

The simulation *subprocess control* (start/stop/pause/resume) stays inline in app.py
until Phase 7.4 — this module is read-only.
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from config import LIVE_OUTPUT

# Maximum rows to load from the live output file. Reading the full file when it
# contains tens-of-thousands of rows causes a multi-second parse on every cache
# expiry, making the dashboard unusable. Only the most recent _MAX_LIVE_ROWS rows
# are kept; older simulation history is discarded.
_MAX_LIVE_ROWS = 5_000


@st.cache_data(ttl=1, show_spinner=False)
def _cached_live_rows():
    """Read the most recent _MAX_LIVE_ROWS scored rows from the live output file.

    Reads only the tail of the file to avoid re-parsing hundreds of thousands of
    rows on every 2-second cache expiry.  Returns (rows_list, stream_done).
    Clear via _cached_live_rows.clear() when starting a new simulation.
    """
    if not os.path.exists(LIVE_OUTPUT):
        return [], False
    # Quick size check — skip expensive read if file is empty
    if os.path.getsize(LIVE_OUTPUT) == 0:
        return [], False
    rows: list[dict] = []
    stream_done = False
    try:
        with open(LIVE_OUTPUT, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        # Process only the tail to cap memory and parse time
        for line in lines[-_MAX_LIVE_ROWS:]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("_eos"):
                stream_done = True
            else:
                rows.append(obj)
        # Check for _eos marker anywhere in the full tail scan
        if not stream_done:
            for line in lines[-100:]:
                try:
                    if json.loads(line.strip()).get("_eos"):
                        stream_done = True
                        break
                except (json.JSONDecodeError, AttributeError):
                    # Partial line still being written, or a non-object record.
                    continue
    except Exception as _e:
        # Deliberate swallow: the simulator writes this file concurrently, so a
        # mid-write read can fail in several ways (truncation, partial UTF-8
        # sequence). Surface on the next refresh instead of crashing the
        # dashboard — but log it, so failures are no longer invisible.
        import logging as _logging
        _logging.getLogger("ueba.live").warning("live results read failed: %s", _e)
    return rows, stream_done


@st.cache_data(ttl=1, show_spinner=False)
def _get_live_user_data(user: str) -> pd.DataFrame:
    """Load live-scored rows for *user* from LIVE_OUTPUT, normalized to match user_data columns.

    Delegates to _cached_live_rows to avoid a redundant full-file scan; the shared
    1-second TTL (below the fragment's 2s run_every) ensures every fragment rerun
    reads genuinely fresh data from the live output file.
    """
    rows, _ = _cached_live_rows()
    if not rows:
        return pd.DataFrame()
    user_rows = [r for r in rows if r.get("user") == user]
    if not user_rows:
        return pd.DataFrame()
    df = pd.DataFrame(user_rows)
    if "cert_timestamp" not in df.columns:
        return pd.DataFrame()
    df["day"] = pd.to_datetime(df["cert_timestamp"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["day"])
    if df.empty:
        return pd.DataFrame()
    for _col in ("if_anomaly_score", "if_percentile_rank"):
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
    # Keep one row per day: worst-of-day (highest percentile rank)
    df = (
        df.sort_values("if_percentile_rank", ascending=False)
        .drop_duplicates(subset=["day"])
        .copy()
    )
    # Map to column names the Investigation page expects
    df["ae_risk_band"] = (
        df["if_risk_band"].astype(str).str.upper()
        if "if_risk_band" in df.columns
        else "LOW"
    )
    df["ae_percentile_rank"] = df.get("if_percentile_rank", np.nan)
    df["_live_row"] = True
    return df.reset_index(drop=True)


@st.cache_data(ttl=1, show_spinner=False)
def _cached_live_file_stats():
    """Return (row_count, stream_done) for the live output file.

    Delegates to _cached_live_rows so at most one file scan occurs per TTL window
    regardless of how many callers need live-file metadata.
    """
    rows, stream_done = _cached_live_rows()
    return len(rows), stream_done


@st.cache_data(ttl=60, show_spinner=False)
def _cached_live_max_date(ds_max):
    """Scan the live output file at most once per minute to find the latest date in
    live records.  Returns ds_max immediately when the file is absent or empty to
    avoid an unnecessary file read on every page render when live mode is off.
    Clear via _cached_live_max_date.clear() when starting a new simulation.

    ds_max is the static dataset maximum date (formerly the _DS_MAX module global);
    passing it in keeps the cache key explicit and the function closure-free.
    """
    try:
        if not os.path.exists(LIVE_OUTPUT) or os.path.getsize(LIVE_OUTPUT) == 0:
            return ds_max
        rows, _ = _cached_live_rows()
        live_max = ds_max
        for _o in rows:
            _ts = _o.get("cert_timestamp")
            if not _ts:
                continue
            _d = pd.to_datetime(_ts, errors="coerce")
            if pd.notna(_d) and _d.date() > live_max:
                live_max = _d.date()
        return live_max
    except Exception:
        return ds_max
