"""Back-compat shim — moved to src/ueba/viz/anomaly_score_distribution.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.AnomalyScoreDistribution", "ueba.viz.anomaly_score_distribution")

from ueba.viz.anomaly_score_distribution import *  # noqa: E402,F401,F403
