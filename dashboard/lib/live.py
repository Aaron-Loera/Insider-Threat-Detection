"""Live-simulation output readers and subprocess control.

The readers parse processed_datasets/live_results.jsonl (config.LIVE_OUTPUT), which
the live_simulation/live_replay subprocess appends to while running. They are cached
with a 1-second TTL (below the Investigation fragment's 2s run_every) so each refresh
sees fresh rows without re-scanning the whole file every rerun. Only the most recent
_MAX_LIVE_ROWS rows are kept so a long simulation history doesn't blow up parse time.

This module also owns the shared live session-state defaults (init_live_state),
the cache-clearing used when (re)starting a run (clear_live_caches), and the
start/stop/pause/resume subprocess control (start_simulation / stop_simulation /
pause_simulation / resume_simulation). The Alerts page wires the buttons to these.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

from config import BASE_DIR, LIVE_OUTPUT, LIVE_PAUSE_FLAG, LIVE_SIM_SCRIPT

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


# ── Shared live session state + subprocess control ───────────────────────────

def init_live_state() -> None:
    """Seed the live-simulation session-state keys shared across all pages."""
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = False
    if "live_proc" not in st.session_state:
        st.session_state.live_proc = None  # subprocess.Popen or None
    if "live_paused" not in st.session_state:
        st.session_state.live_paused = False
    if "live_page" not in st.session_state:
        st.session_state.live_page = 0


def clear_live_caches() -> None:
    """Invalidate every live-output cache (call when (re)starting a simulation)."""
    _cached_live_max_date.clear()
    _cached_live_file_stats.clear()
    _cached_live_rows.clear()
    _get_live_user_data.clear()


def _live_script_path() -> str:
    """live_replay.py on Streamlit Cloud (no ML deps), live_simulation.py locally.

    Streamlit Cloud mounts the repo at /mount/src, so its presence is the cloud
    signal. live_replay replays pre-scored rows; live_simulation runs the full
    encoder + isolation-forest pipeline.
    """
    on_cloud = os.path.exists("/mount/src")
    return os.path.join(BASE_DIR, "live_replay.py") if on_cloud else LIVE_SIM_SCRIPT


def start_simulation() -> None:
    """Clear prior output/caches and launch the scoring subprocess."""
    if os.path.exists(LIVE_OUTPUT):
        os.remove(LIVE_OUTPUT)
    if os.path.exists(LIVE_PAUSE_FLAG):
        os.remove(LIVE_PAUSE_FLAG)
    clear_live_caches()
    st.session_state.live_page = 0
    proc = subprocess.Popen(
        [sys.executable, _live_script_path(), "--interval", "0.5"],
        cwd=BASE_DIR,
    )
    st.session_state.live_proc = proc
    st.session_state.live_mode = True
    st.session_state.live_paused = False


def stop_simulation() -> None:
    """Terminate the scoring subprocess and reset live session state."""
    proc = st.session_state.live_proc
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    # Remove pause flag so the process isn't blocked on next start
    if os.path.exists(LIVE_PAUSE_FLAG):
        os.remove(LIVE_PAUSE_FLAG)
    st.session_state.live_proc = None
    st.session_state.live_mode = False
    st.session_state.live_paused = False
    st.session_state.live_page = 0


def pause_simulation() -> None:
    """Signal the subprocess to pause (the flag file's existence is the signal)."""
    with open(LIVE_PAUSE_FLAG, "w", encoding="utf-8"):
        pass
    st.session_state.live_paused = True


def resume_simulation() -> None:
    """Clear the pause flag so the subprocess resumes."""
    if os.path.exists(LIVE_PAUSE_FLAG):
        os.remove(LIVE_PAUSE_FLAG)
    st.session_state.live_paused = False
