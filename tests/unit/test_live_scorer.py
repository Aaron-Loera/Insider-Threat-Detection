"""Tests for LiveScorer.score_row with injected stub components.

No tensorflow, no model artifacts: the Phase 5 injectability refactor lets
the full scoring path (select -> scale -> embed -> score -> rank -> band)
run against stubs, so the serving contract is pinned in CI.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("websockets")  # imported by the serving module

from ueba.serving import live_simulation  # noqa: E402
from ueba.serving.live_simulation import LiveScorer  # noqa: E402

FEATURE_COLS = ["logon_count", "file_copy_count", "emails_sent", "http_upload_count"]


@pytest.fixture(autouse=True)
def isolate_from_local_artifacts(tmp_path, monkeypatch):
    """Tests must not depend on whatever model artifacts exist on this machine:
    point the default-load paths at an empty directory so absolute_thresholds=None
    deterministically exercises the percentile fallback."""
    monkeypatch.setattr(live_simulation, "CALIBRATION_THRESHOLD_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setattr(live_simulation, "FEATURE_COLS_PATH", str(tmp_path / "absent_cols.json"))


class IdentityScaler:
    def transform(self, X):
        return X


class StubEncoder:
    """Pass-through 'embedding' — keeps tensorflow out of the test."""

    def predict(self, X, verbose=0):
        return X


class StubForest:
    """Returns a fixed anomaly score regardless of the embedding."""

    def __init__(self, score: float):
        self.score = score

    def anomaly_score(self, embeddings):
        return np.full(len(embeddings), self.score)


def make_scorer(score: float, absolute_thresholds: dict | None = None, user_work_hours=None) -> LiveScorer:
    return LiveScorer(
        scaler=IdentityScaler(),
        encoder=StubEncoder(),
        iforest=StubForest(score),
        ref_scores=np.arange(100, dtype="float64"),  # percentile == score value
        feature_cols=FEATURE_COLS,
        absolute_thresholds=absolute_thresholds,
        user_work_hours=user_work_hours,
    )


def make_row(**overrides) -> pd.DataFrame:
    row = {
        "user": "user01",
        "day": "2010-01-05",
        "logon_count": 3.0,
        "file_copy_count": 1.0,
        "emails_sent": 7.0,
        "http_upload_count": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_payload_contract():
    payload = make_scorer(score=50.0).score_row(make_row())
    for key in ("user", "day", "cert_timestamp", "if_anomaly_score", "if_percentile_rank", "if_risk_band", "_score_ms"):
        assert key in payload
    assert payload["user"] == "user01"
    assert payload["if_anomaly_score"] == 50.0
    assert payload["if_percentile_rank"] == 50.0
    assert payload["if_risk_band"] == "LOW"


@pytest.mark.parametrize(
    ("score", "band"),
    [(50.0, "LOW"), (85.0, "MEDIUM"), (92.0, "HIGH"), (95.0, "CRITICAL"), (1000.0, "CRITICAL")],
)
def test_percentile_fallback_banding(score, band):
    payload = make_scorer(score).score_row(make_row())
    assert payload["if_risk_band"] == band


def test_calibrated_absolute_thresholds_take_precedence():
    thresholds = {"LOW": 10.0, "MEDIUM": 20.0, "HIGH": 30.0, "CRITICAL": None}
    # Score 25 sits at the 25th percentile (LOW by percentile) but in the
    # calibrated HIGH band — absolute thresholds must win.
    payload = make_scorer(25.0, absolute_thresholds=thresholds).score_row(make_row())
    assert payload["if_risk_band"] == "HIGH"


def test_nan_passthrough_becomes_none():
    payload = make_scorer(50.0).score_row(make_row(usb_insert_count=float("nan")))
    assert payload["usb_insert_count"] is None
    assert payload["logon_count"] == 3.0


def test_cold_start_user_warned_once(capsys):
    schedule = pd.DataFrame(
        {"user": ["someone_else"], "start_hour": [9], "end_hour": [17], "schedule_complete": [True]}
    )
    scorer = make_scorer(50.0, user_work_hours=schedule)
    scorer.score_row(make_row())
    scorer.score_row(make_row())
    out = capsys.readouterr().out
    assert out.count("no derived work-hour envelope") == 1
    assert "user01" in out


def test_scheduled_user_not_warned(capsys):
    schedule = pd.DataFrame(
        {"user": ["user01"], "start_hour": [7], "end_hour": [15], "schedule_complete": [True]}
    )
    make_scorer(50.0, user_work_hours=schedule).score_row(make_row())
    assert "no derived work-hour envelope" not in capsys.readouterr().out
