"""Fixtures for the Streamlit dashboard smoke test.

The dashboard is a top-to-bottom Streamlit script (dashboard/app.py) that loads a
pre-merged "serving" parquet at import time. These fixtures build a synthetic copy
of that data tree in a temp dir, point the app at it via UEBA_BASE_DIR, and boot it
headlessly through streamlit.testing.v1.AppTest.

The data tree mirrors the serving schema declared in
ueba.serving.build_dashboard_dataset (the single source of truth for what the
dashboard renders). Some optional inputs are deliberately left out to exercise the
app's graceful-degradation paths:
    - ueba_dataset_{V}a.parquet  (PC-level drill-down → load_ueba_a returns None)
    - peer_baselines_{V}.parquet (peer comparison → empty/None)
    - live_results.jsonl         (no live replay running)
"""

import sys

import numpy as np
import pandas as pd
import pytest


def _build_dashboard_frame(n_users: int = 10, n_days: int = 30) -> pd.DataFrame:
    """Synthetic serving frame matching build_dashboard_dataset's output columns.

    Percentile ranks are spread linearly 0-100 so all four risk bands appear, and
    baseline_complete is mostly True (the first few days per user are the cold-start
    window the dashboard gates out).
    """
    from ueba import risk
    from ueba.serving.build_dashboard_dataset import (
        BASE_CHANNELS,
        CROSS_FLAGS,
    )

    rng = np.random.default_rng(7)
    users = [f"user{i:02d}" for i in range(n_users)]
    days = pd.date_range("2024-01-01", periods=n_days, freq="D")
    idx = pd.MultiIndex.from_product([users, days], names=["user", "day"])
    df = pd.DataFrame(index=idx).reset_index()
    n = len(df)

    # Behavioral channels: small non-negative counts; a couple of float channels.
    for col in BASE_CHANNELS:
        if col.startswith("http_upload") or col.startswith("http_download"):
            df[col] = (rng.random(n) * 5).astype("float64")
        else:
            df[col] = rng.integers(0, 20, n).astype("int64")

    # Cross-channel flags: 0/1.
    for col in CROSS_FLAGS:
        df[col] = rng.integers(0, 2, n).astype("int64")

    # Profile enrichment (per-user constants).
    df["employee_name"] = "Synthetic Person"
    df["department"] = "Sales"
    df["role"] = "Salesman"
    df["supervisor"] = "Boss Person"
    df["functional_unit"] = "Sales"
    df["is_active"] = True
    df["role_sensitivity"] = np.float32(0.4)

    # Cold-start gate: first 5 days per user are incomplete.
    df["baseline_complete"] = df.groupby("user").cumcount() >= 5

    # Scores / bands. Spread percentiles 0-100 across rows so every band shows up.
    pct = np.linspace(0, 100, n)
    rng.shuffle(pct)
    df["ae_percentile_rank"] = pct.astype("float64")
    df["if_percentile_rank"] = np.clip(pct + rng.normal(0, 3, n), 0, 100).astype("float64")
    df["if_anomaly_score"] = (df["if_percentile_rank"] / 100.0).astype("float64")
    df["composite_score"] = df["ae_percentile_rank"].astype("float64")
    df["ae_risk_band"] = risk.assign_bands_from_percentiles(df["ae_percentile_rank"].to_numpy())
    df["if_risk_band"] = risk.assign_bands_from_percentiles(df["if_percentile_rank"].to_numpy())
    df["composite_risk_band"] = df["ae_risk_band"]
    return df


@pytest.fixture
def dashboard_data_tree(tmp_path):
    """Write a synthetic dashboard data tree under tmp_path; return its root.

    Lays down only what the smoke test needs: the slim serving parquet and a tiny
    per-(user, day) alert-detail table (so fetch_alert_detail never tries to hit
    HuggingFace). Optional inputs are intentionally absent (see module docstring).
    """
    from ueba import config

    root = tmp_path / "tree"
    mv = config.MODEL_VERSION

    ds_dir = root / "processed_datasets" / f"ueba_dataset_{mv}"
    ds_dir.mkdir(parents=True)
    frame = _build_dashboard_frame()
    frame.to_parquet(ds_dir / f"ueba_dataset_{mv}_dashboard.parquet", index=False)

    alert_dir = root / "explainability" / "alert_table" / f"alert_table_{mv}"
    alert_dir.mkdir(parents=True)
    detail = frame[["user", "day"]].copy()
    detail["top_contributors"] = "[('logon_count', 1.0)]"
    detail["explanation"] = "Synthetic explanation."
    detail.to_parquet(alert_dir / f"alert_table_{mv}.parquet", index=False)

    return root


@pytest.fixture
def app_test(monkeypatch, dashboard_data_tree):
    """Boot dashboard/app.py headless against the synthetic tree, authenticated.

    Redirects config to the synthetic tree via UEBA_BASE_DIR, purges already-imported
    config/lib modules so they re-resolve against it, points the disposition DB at a
    temp file, clears Streamlit's in-process caches (they bleed across AppTest runs),
    and enables the dev auth bypass via at.secrets.
    """
    import os

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UEBA_BASE_DIR", str(dashboard_data_tree))
    # Never reach HuggingFace from a test: the synthetic tree supplies every file
    # load_data and the alert-detail loader need, so any network call is a bug.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _purge_config_modules():
        # config (and ueba.config it re-exports) snapshot BASE_DIR from the
        # environment at import time. Drop the cached objects so they re-resolve
        # against whatever UEBA_BASE_DIR is set right now — needed both before the
        # run (to pick up the temp tree) and on teardown (so later tests, e.g.
        # test_shims, re-import a config pointing back at the real repo root).
        for mod in [m for m in sys.modules if m in ("config", "ueba.config")
                    or m == "lib" or m.startswith("lib.")]:
            del sys.modules[mod]

    _purge_config_modules()

    # The app does `from db import …`, resolving `db` from the dashboard dir it
    # puts on sys.path. Pre-import that same module identity (not dashboard.db,
    # which is a distinct object — see tests/unit/test_db.py) and redirect its
    # DB_PATH so init_db/upsert during the run write to a temp file, not the
    # repo's dashboard/alert_state.db. AppTest reuses this sys.modules["db"].
    dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dashboard")
    if dashboard_dir not in sys.path:
        sys.path.insert(0, dashboard_dir)
    import db as _db
    monkeypatch.setattr(_db, "DB_PATH", str(dashboard_data_tree / "alert_state.db"))

    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()

    at = AppTest.from_file("dashboard/app.py", default_timeout=60)
    at.secrets["dev"] = {"bypass_auth": True}
    yield at

    # Teardown: the in-process Streamlit caches and the env-pinned config module
    # both bleed across AppTest runs. monkeypatch restores UEBA_BASE_DIR; we must
    # also evict the config snapshot and clear caches so the next test is clean.
    st.cache_data.clear()
    st.cache_resource.clear()
    _purge_config_modules()


@pytest.fixture
def agg_env(monkeypatch, dashboard_data_tree):
    """Point the lib loaders/aggregations at the synthetic tree for direct calls.

    Like app_test but without booting the Streamlit app — for unit-testing the
    lib.aggregations functions directly. Redirects config via UEBA_BASE_DIR, forces
    HF offline, ensures the dashboard dir is importable (so `from config import …`
    and `from db import …` inside the lib modules resolve), and clears Streamlit's
    caches before and after so cached frames don't leak across tests.
    """
    import os

    import streamlit as st

    monkeypatch.setenv("UEBA_BASE_DIR", str(dashboard_data_tree))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dashboard")
    if dashboard_dir not in sys.path:
        sys.path.insert(0, dashboard_dir)

    def _purge():
        for mod in [m for m in sys.modules if m in ("config", "ueba.config")
                    or m == "lib" or m.startswith("lib.")]:
            del sys.modules[mod]

    _purge()
    st.cache_data.clear()
    st.cache_resource.clear()
    yield dashboard_data_tree
    st.cache_data.clear()
    st.cache_resource.clear()
    _purge()
