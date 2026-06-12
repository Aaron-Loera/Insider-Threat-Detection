"""Tests for AlertObjectBuilder baselines, percentiles, and risk banding.

Note on banding tests: only band-interior percentiles are pinned here
(50, 85, 92, 97). Exact-boundary values (80/90/95) intentionally are NOT,
because the boundary semantics are unified in the risk-band consolidation
phase (docs/CLEANUP_REPORT.md, gap 1). Interior values map identically
under both the legacy and unified semantics.
"""

import numpy as np
import pytest

from scripts.AlertObjectBuilder import AlertObjectBuilder


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


def test_band_from_absolute_score():
    b = AlertObjectBuilder()
    thresholds = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0, "CRITICAL": None}
    assert b.assign_risk_band_from_score(0.5, thresholds) == "LOW"
    assert b.assign_risk_band_from_score(1.0, thresholds) == "LOW"  # inclusive upper edge
    assert b.assign_risk_band_from_score(1.5, thresholds) == "MEDIUM"
    assert b.assign_risk_band_from_score(2.5, thresholds) == "HIGH"
    assert b.assign_risk_band_from_score(99.0, thresholds) == "CRITICAL"
