"""Back-compat shim — moved to src/ueba/viz/hybrid_risk_scatter.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.HybridRiskScatter", "ueba.viz.hybrid_risk_scatter")

from ueba.viz.hybrid_risk_scatter import *  # noqa: E402,F401,F403
