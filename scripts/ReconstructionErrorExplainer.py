"""Back-compat shim — moved to src/ueba/alerts/explainer.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.ReconstructionErrorExplainer", "ueba.alerts.explainer")

from ueba.alerts.explainer import *  # noqa: E402,F401,F403
