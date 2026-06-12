"""Shared fixtures for the unit test suite.

All fixtures build small synthetic frames (seeded RNG, ~10 users x 30 days).
Real CERT data is never read here: the suite must run in CI where no
datasets or model artifacts exist.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make repo-root modules (config, prepare_data, scripts/, dashboard/)
# importable regardless of where pytest is invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

N_USERS = 10
N_DAYS = 30
START_DAY = "2010-01-01"


@pytest.fixture
def ueba_df() -> pd.DataFrame:
    """Synthetic (user, day) UEBA frame: 10 users x 30 days, seeded.

    Contains a few numeric behavioral features plus the identity/gating
    columns listed in config.NON_FEATURE_COLS so tests can verify they are
    excluded from model matrices.
    """
    rng = np.random.default_rng(42)
    users = [f"user{i:02d}" for i in range(N_USERS)]
    days = pd.date_range(START_DAY, periods=N_DAYS, freq="D")
    idx = pd.MultiIndex.from_product([users, days], names=["user", "day"])
    df = pd.DataFrame(index=idx).reset_index()

    n = len(df)
    df["logon_count"] = rng.integers(0, 20, n)
    df["file_copy_count"] = rng.integers(0, 10, n)
    df["emails_sent"] = rng.integers(0, 30, n)
    df["http_upload_count"] = rng.random(n).astype("float64") * 5

    # Identity / gating columns (must be dropped by to_model_matrix)
    df["employee_name"] = "Synthetic Person"
    df["role"] = "Salesman"
    df["department"] = "Sales"
    df["role_sensitivity"] = np.float32(0.4)
    df["baseline_complete"] = df.groupby("user").cumcount() >= 14
    return df


@pytest.fixture
def insider_windows() -> pd.DataFrame:
    """Ground-truth insider windows shaped like get_insiders() output."""
    return pd.DataFrame(
        {
            "user": ["user03", "user07"],
            "start_day": pd.to_datetime(["2010-01-10", "2010-01-20"]),
            "end_day": pd.to_datetime(["2010-01-15", "2010-01-25"]),
            "scenario": [1, 2],
        }
    )


@pytest.fixture
def insiders_csv(tmp_path) -> str:
    """A tiny insiders.csv shaped like the CERT answers file."""
    path = tmp_path / "insiders.csv"
    path.write_text(
        "dataset,scenario,user,start,end\n"
        "4.2,1, ACM2278 ,2010-08-01 07:00:00,2010-09-15 18:00:00\n"
        "4.2,2,CDE1846,2011-02-01 07:00:00,2011-03-01 18:00:00\n"
        "5.2,1,XYZ0001,2010-08-01 07:00:00,2010-09-15 18:00:00\n",
        encoding="utf-8",
    )
    return str(path)
