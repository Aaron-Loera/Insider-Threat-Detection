"""Tests for AlertObjectBuilder baselines, percentiles, and risk banding.

Banding now delegates to scripts/risk_bands.py (the unified semantics —
boundary percentiles belong to the higher band). Exhaustive boundary tests
live in test_risk.py; this module checks the builder-level wiring.
"""

import numpy as np
import pandas as pd
import pytest

from ueba.alerts.builder import AlertObjectBuilder


@pytest.fixture
def builder() -> AlertObjectBuilder:
    b = AlertObjectBuilder()
    baseline = np.arange(100, dtype="float64")  # values 0..99
    b.fit_ae_baseline(baseline)
    b.fit_if_baseline(baseline)
    return b


def test_unfitted_baseline_raises():
    b = AlertObjectBuilder()
    with pytest.raises(ValueError):
        b.compute_ae_percentile(1.0)
    with pytest.raises(ValueError):
        b.compute_if_percentile(1.0)


def test_percentile_range_and_extremes(builder):
    assert builder.compute_if_percentile(-1.0) == 0.0
    assert builder.compute_if_percentile(1000.0) == 100.0
    assert builder.compute_if_percentile(50.0) == 50.0


def test_percentile_monotonic(builder):
    values = [0.5, 10.0, 25.0, 70.0, 99.5]
    ranks = [builder.compute_if_percentile(v) for v in values]
    assert ranks == sorted(ranks)


def test_fit_baseline_sorts_input():
    b = AlertObjectBuilder()
    b.fit_if_baseline(np.array([5.0, 1.0, 3.0]))
    assert list(b.if_baseline) == [1.0, 3.0, 5.0]


def test_ae_and_if_percentiles_agree_on_same_baseline(builder):
    for v in (12.3, 45.6, 78.9):
        assert builder.compute_ae_percentile(v) == builder.compute_if_percentile(v)


def test_band_interior_values(builder):
    assert builder.assign_risk_band(50.0) == "LOW"
    assert builder.assign_risk_band(85.0) == "MEDIUM"
    assert builder.assign_risk_band(92.0) == "HIGH"
    assert builder.assign_risk_band(97.0) == "CRITICAL"


def test_band_boundaries_use_unified_semantics(builder):
    # Boundary percentile belongs to the HIGHER band (changed from the legacy
    # <= semantics in the risk-band unification — CLEANUP_REPORT gap 1).
    assert builder.assign_risk_band(80.0) == "MEDIUM"
    assert builder.assign_risk_band(90.0) == "HIGH"
    assert builder.assign_risk_band(95.0) == "CRITICAL"


def test_legacy_threshold_format_still_accepted():
    legacy = AlertObjectBuilder(percentile_thresholds={"LOW": 80, "MEDIUM": 90, "HIGH": 95, "CRITICAL": 100})
    default = AlertObjectBuilder()
    assert legacy.percentile_thresholds == default.percentile_thresholds


def _explanation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": ["user01", "user02", "user03", "user04"],
            "day": pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04"]),
            # Baseline is 0..99, so these sit at the 25th/85th/92nd/99th percentiles
            "total_reconstruction_error": [25.0, 85.0, 92.0, 99.5],
            "if_anomaly_score": [25.0, 85.0, 92.0, 99.5],
            "contribution_file_copy_count": [0.5, 0.7, 0.2, 0.9],
            "contribution_emails_sent": [0.3, 0.1, 0.6, 0.05],
            "contribution_usb_insert_count": [0.2, 0.2, 0.2, 0.05],
        }
    )


def test_build_alert_df_percentile_mode_matches_row_wise(builder):
    df = _explanation_df()
    alerts = builder.build_alert_df(df, w1=0.5, w2=0.5)

    assert len(alerts) == len(df)
    # Vectorized bands must equal the scalar path for every row
    for i, row in alerts.iterrows():
        assert row["ae_risk_band"] == builder.assign_risk_band(row["ae_percentile_rank"])
        assert row["if_risk_band"] == builder.assign_risk_band(row["if_percentile_rank"])
        assert row["composite_risk_band"] == builder.assign_risk_band(row["composite_score"])
    # Spot-check expected bands: 25th->LOW, 85th->MEDIUM, 92nd->HIGH, 99.5->CRITICAL
    assert list(alerts["if_risk_band"]) == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    # Top contributor extraction picks the max contribution column
    assert alerts["top_contributors"].iloc[3][0][0] == "file_copy_count"


def test_build_alert_df_absolute_mode():
    b = AlertObjectBuilder(
        ae_absolute_thresholds={"LOW": 30.0, "MEDIUM": 60.0, "HIGH": 90.0, "CRITICAL": None},
        if_absolute_thresholds={"LOW": 30.0, "MEDIUM": 60.0, "HIGH": 90.0, "CRITICAL": None},
    )
    baseline = np.arange(100, dtype="float64")
    b.fit_ae_baseline(baseline)
    b.fit_if_baseline(baseline)

    alerts = b.build_alert_df(_explanation_df(), w1=0.5, w2=0.5)
    # Raw scores 25/85/92/99.5 against ceilings 30/60/90/inf
    assert list(alerts["if_risk_band"]) == ["LOW", "HIGH", "CRITICAL", "CRITICAL"]
    for i, row in alerts.iterrows():
        assert row["if_risk_band"] == b.assign_risk_band_from_score(
            row["if_anomaly_score"], b.if_absolute_thresholds
        )


def test_band_from_absolute_score():
    b = AlertObjectBuilder()
    thresholds = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0, "CRITICAL": None}
    assert b.assign_risk_band_from_score(0.5, thresholds) == "LOW"
    assert b.assign_risk_band_from_score(1.0, thresholds) == "LOW"  # inclusive upper edge
    assert b.assign_risk_band_from_score(1.5, thresholds) == "MEDIUM"
    assert b.assign_risk_band_from_score(2.5, thresholds) == "HIGH"
    assert b.assign_risk_band_from_score(99.0, thresholds) == "CRITICAL"
