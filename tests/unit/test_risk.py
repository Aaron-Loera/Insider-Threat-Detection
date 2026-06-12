"""Tests pinning the unified risk-band semantics (scripts/risk_bands.py).

These boundaries ARE the contract: a percentile equal to a threshold belongs
to the higher band (p >= 95 -> CRITICAL), matching CLAUDE.md and the live
scorer's historical behavior. The offline AlertObjectBuilder previously used
<= semantics and intentionally changed at exact boundaries (CLEANUP_REPORT
gap 1).
"""

import numpy as np
import pytest

from ueba import risk as risk_bands

# ── Percentile banding ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("percentile", "band"),
    [
        (0.0, "LOW"),
        (79.99, "LOW"),
        (80.0, "MEDIUM"),   # boundary -> higher band
        (80.01, "MEDIUM"),
        (89.99, "MEDIUM"),
        (90.0, "HIGH"),     # boundary -> higher band
        (90.01, "HIGH"),
        (94.99, "HIGH"),
        (95.0, "CRITICAL"), # boundary -> higher band
        (95.01, "CRITICAL"),
        (100.0, "CRITICAL"),
    ],
)
def test_percentile_band_boundaries(percentile, band):
    assert risk_bands.assign_band_from_percentile(percentile) == band


def test_vectorized_matches_scalar_banding():
    percentiles = np.array([0.0, 79.99, 80.0, 89.99, 90.0, 94.99, 95.0, 100.0])
    vector = risk_bands.assign_bands_from_percentiles(percentiles)
    scalar = [risk_bands.assign_band_from_percentile(p) for p in percentiles]
    assert list(vector) == scalar


def test_normalize_accepts_legacy_upper_bound_format():
    legacy = {"LOW": 80, "MEDIUM": 90, "HIGH": 95, "CRITICAL": 100}
    assert risk_bands.normalize_percentile_thresholds(legacy) == {
        "MEDIUM": 80.0,
        "HIGH": 90.0,
        "CRITICAL": 95.0,
    }


def test_normalize_passes_canonical_format_through():
    canonical = {"MEDIUM": 70.0, "HIGH": 85.0, "CRITICAL": 99.0}
    assert risk_bands.normalize_percentile_thresholds(canonical) == canonical
    assert risk_bands.assign_band_from_percentile(85.0, canonical) == "HIGH"


# ── Absolute-score banding ────────────────────────────────────────────────────

ABS = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0, "CRITICAL": None}


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0.5, "LOW"),
        (1.0, "LOW"),       # score equal to a ceiling stays in that band
        (1.5, "MEDIUM"),
        (2.0, "MEDIUM"),
        (3.0, "HIGH"),
        (3.5, "CRITICAL"),
        (1e9, "CRITICAL"),  # above every ceiling -> highest band
    ],
)
def test_absolute_band_boundaries(score, band):
    assert risk_bands.assign_band_from_score(score, ABS) == band


def test_absolute_banding_ignores_dict_insertion_order():
    shuffled = {"CRITICAL": None, "HIGH": 3.0, "LOW": 1.0, "MEDIUM": 2.0}
    for score in (0.5, 1.5, 2.5, 99.0):
        assert risk_bands.assign_band_from_score(score, shuffled) == risk_bands.assign_band_from_score(score, ABS)


def test_vectorized_matches_scalar_absolute():
    scores = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 1e9])
    vector = risk_bands.assign_bands_from_scores(scores, ABS)
    scalar = [risk_bands.assign_band_from_score(s, ABS) for s in scores]
    assert list(vector) == scalar


def test_absolute_banding_accepts_json_infinity():
    # json.load parses "Infinity" to float inf — must behave like None
    inf_thresholds = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0, "CRITICAL": float("inf")}
    assert risk_bands.assign_band_from_score(99.0, inf_thresholds) == "CRITICAL"


# ── Percentile ranking ────────────────────────────────────────────────────────

def test_percentile_rank_equals_both_legacy_formulas():
    rng = np.random.default_rng(11)
    baseline = np.sort(rng.normal(size=500))
    values = rng.normal(size=50)

    for v in values:
        unified = risk_bands.percentile_rank(v, baseline)
        legacy_searchsorted = np.searchsorted(baseline, v) / len(baseline) * 100   # AlertObjectBuilder
        legacy_mean = float(np.mean(baseline < v) * 100)                           # live_simulation
        assert unified == pytest.approx(legacy_searchsorted)
        assert unified == pytest.approx(legacy_mean)


def test_percentile_rank_scalar_and_vector_forms():
    baseline = np.arange(100, dtype="float64")
    scalar = risk_bands.percentile_rank(50.0, baseline)
    assert isinstance(scalar, float) and scalar == 50.0

    vector = risk_bands.percentile_rank(np.array([-1.0, 50.0, 1000.0]), baseline)
    assert list(vector) == [0.0, 50.0, 100.0]


def test_percentile_rank_ties_count_strictly_below():
    baseline = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    # Exactly one value (1.0) is strictly below 2.0 -> 20%
    assert risk_bands.percentile_rank(2.0, baseline) == pytest.approx(20.0)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_band_order_and_colors_cover_all_bands():
    assert risk_bands.BAND_ORDER == ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert set(risk_bands.BAND_COLORS) == set(risk_bands.BAND_ORDER)
