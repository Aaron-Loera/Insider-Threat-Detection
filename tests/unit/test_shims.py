"""The back-compat shims must keep every pre-migration import path working.

dashboard/app.py imports `config` from the repo root; the notebooks import
`prepare_data` and `scripts.*`. These tests pin that contract so a future
cleanup can't silently break the dashboard or the notebooks.
"""

import warnings

import pytest


def test_root_config_shim_exposes_dashboard_imports():
    # The exact names dashboard/app.py imports at startup (app.py:1018-1028).
    import config
    import ueba.config as ueba_config
    from config import (  # noqa: F401
        ANALYST_TABLE_CSV,
        ANALYST_TABLE_PARQUET,
        CALIB_ALERT_TABLE_PARQUET,
        HF_DATASET_BASE_URL,
        HF_DATASET_REPO,
        LIVE_OUTPUT,
        LIVE_PAUSE_FLAG,
        LIVE_SIM_SCRIPT,
        MODEL_VERSION,
        PEER_BASELINES_PATH,
        UEBA_A_CSV,
        UEBA_A_PARQUET,
        UEBA_CSV,
        UEBA_PARQUET,
    )

    assert config.BASE_DIR == ueba_config.BASE_DIR
    assert config.MODEL_VERSION == ueba_config.MODEL_VERSION


def test_config_base_dir_is_repo_root():
    import os

    import config

    assert os.path.exists(os.path.join(config.BASE_DIR, "pyproject.toml"))
    assert os.path.basename(config.BASE_DIR) != "ueba"  # the src/ueba/__file__ bug


def test_prepare_data_shim():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from prepare_data import build_insider_mask, chronological_split, get_insiders, to_model_matrix  # noqa: F401


def test_scripts_shims_expose_classes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from scripts.AlertObjectBuilder import AlertObjectBuilder
        from scripts.Preprocessing import WORK_HOURS, chronological_split  # noqa: F401
        from scripts.risk_bands import assign_band_from_percentile

    from ueba.alerts.builder import AlertObjectBuilder as canonical_builder
    from ueba.risk import assign_band_from_percentile as canonical_assign

    assert AlertObjectBuilder is canonical_builder
    assert assign_band_from_percentile is canonical_assign
    assert WORK_HOURS == (9, 17)


def test_constants_extracted_to_ueba_constants():
    from ueba import constants
    from ueba.features import preprocessing

    assert preprocessing.WORK_HOURS is constants.WORK_HOURS
    assert preprocessing.INTERNAL_EMAIL_DOMAIN is constants.INTERNAL_EMAIL_DOMAIN
    assert preprocessing.JOB_DOMAINS is constants.JOB_DOMAINS


def test_live_simulation_shim_importable():
    # live_simulation needs websockets/joblib; skip cleanly where absent (CI).
    pytest.importorskip("websockets")
    pytest.importorskip("joblib")
    import live_simulation

    assert hasattr(live_simulation, "LiveScorer")
    assert hasattr(live_simulation, "main")
