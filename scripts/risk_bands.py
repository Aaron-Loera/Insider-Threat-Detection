"""Back-compat shim — moved to src/ueba/risk.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.risk_bands", "ueba.risk")

from ueba.risk import *  # noqa: E402,F401,F403
from ueba.risk import (  # noqa: E402,F401
    BAND_COLORS,
    BAND_ORDER,
    DEFAULT_PERCENTILE_THRESHOLDS,
    assign_band_from_percentile,
    assign_band_from_score,
    assign_bands_from_percentiles,
    assign_bands_from_scores,
    normalize_percentile_thresholds,
    percentile_rank,
)
