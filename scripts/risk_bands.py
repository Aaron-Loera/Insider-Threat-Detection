"""Single source of truth for risk-band assignment and percentile ranking.

Every component that maps an anomaly signal to a LOW/MEDIUM/HIGH/CRITICAL band
must go through this module. Before unification the offline alert builder used
``percentile <= threshold`` (upper-bound) semantics while the live scorer used
``percentile >= threshold`` (lower-bound), so the two paths disagreed at the
exact 80/90/95 boundaries. The canonical semantics are the documented ones
(CLAUDE.md / dashboard): a percentile equal to a threshold belongs to the
HIGHER band — p >= 95 is CRITICAL.

Absolute-score banding (calibrated thresholds from calibration_thresholds.json)
keeps its historical inclusive-upper-bound semantics: a score equal to a band's
calibrated ceiling stays in that band. Both prior implementations already
agreed on this.
"""

import numpy as np

# Lowest → highest severity. Index position is used for vectorized banding.
BAND_ORDER: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Canonical dashboard palette (see CLAUDE.md "Risk Scoring").
BAND_COLORS: dict[str, str] = {
    "CRITICAL": "#ff1744",
    "HIGH": "#e84545",
    "MEDIUM": "#d4a017",
    "LOW": "#3a86a8",
}

# Lower-bound percentile for each non-LOW band: p >= value → at least that band.
DEFAULT_PERCENTILE_THRESHOLDS: dict[str, float] = {
    "MEDIUM": 80.0,
    "HIGH": 90.0,
    "CRITICAL": 95.0,
}


def normalize_percentile_thresholds(thresholds: dict | None) -> dict[str, float]:
    """Return lower-bound-format thresholds {MEDIUM, HIGH, CRITICAL}.

    Accepts either the canonical lower-bound format or the legacy
    AlertObjectBuilder upper-bound format ``{"LOW": 80, "MEDIUM": 90,
    "HIGH": 95, "CRITICAL": 100}`` (detected by the presence of a "LOW" key).
    The band edges are the same numbers in both formats — only the boundary
    membership semantics differ, and those are fixed by this module.
    """
    if thresholds is None:
        return dict(DEFAULT_PERCENTILE_THRESHOLDS)
    if "LOW" in thresholds:  # legacy upper-bound format
        return {
            "MEDIUM": float(thresholds["LOW"]),
            "HIGH": float(thresholds["MEDIUM"]),
            "CRITICAL": float(thresholds["HIGH"]),
        }
    return {k: float(thresholds[k]) for k in ("MEDIUM", "HIGH", "CRITICAL")}


def assign_band_from_percentile(percentile: float, thresholds: dict | None = None) -> str:
    """Map a 0-100 percentile to a band. Boundary belongs to the higher band."""
    t = normalize_percentile_thresholds(thresholds)
    if percentile >= t["CRITICAL"]:
        return "CRITICAL"
    if percentile >= t["HIGH"]:
        return "HIGH"
    if percentile >= t["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def assign_bands_from_percentiles(percentiles: np.ndarray, thresholds: dict | None = None) -> np.ndarray:
    """Vectorized assign_band_from_percentile. Returns an array of band labels."""
    t = normalize_percentile_thresholds(thresholds)
    bins = [t["MEDIUM"], t["HIGH"], t["CRITICAL"]]
    # right=False → bins[i-1] <= x < bins[i], i.e. a value equal to a threshold
    # lands in the higher band, matching the scalar function.
    return np.asarray(BAND_ORDER)[np.digitize(percentiles, bins, right=False)]


def _sorted_absolute_items(absolute_thresholds: dict) -> list[tuple[str, float]]:
    """Band items sorted by ascending threshold; None (unbounded) treated as +inf."""
    return sorted(
        ((label, float("inf") if value is None else float(value)) for label, value in absolute_thresholds.items()),
        key=lambda kv: kv[1],
    )


def assign_band_from_score(score: float, absolute_thresholds: dict) -> str:
    """Map a raw score to a band via calibrated absolute thresholds.

    A score equal to a band's ceiling stays in that band (inclusive upper
    bound); scores above every ceiling get the highest band.
    """
    items = _sorted_absolute_items(absolute_thresholds)
    for label, ceiling in items:
        if score <= ceiling:
            return label
    return items[-1][0]


def assign_bands_from_scores(scores: np.ndarray, absolute_thresholds: dict) -> np.ndarray:
    """Vectorized assign_band_from_score. Returns an array of band labels."""
    items = _sorted_absolute_items(absolute_thresholds)
    labels = np.array([label for label, _ in items])
    bins = [ceiling for _, ceiling in items]
    # right=True → score <= ceiling stays in that band, matching the scalar
    # function; clip keeps above-all-ceilings scores in the highest band.
    idx = np.digitize(scores, bins, right=True).clip(0, len(items) - 1)
    return labels[idx]


def percentile_rank(values, sorted_baseline: np.ndarray):
    """Percentile rank (0-100) of value(s) against a SORTED baseline.

    Defined as the fraction of baseline observations strictly below the value —
    identical to both legacy formulas (``np.searchsorted(baseline, v)`` and
    ``np.mean(baseline < v) * 100``) but O(log n) per value. The caller is
    responsible for sorting the baseline once (np.sort) at fit/load time.
    """
    ranks = np.searchsorted(sorted_baseline, values, side="left") / len(sorted_baseline) * 100.0
    if np.ndim(ranks) == 0:
        return float(ranks)
    return ranks
