"""Back-compat shim — moved to src/ueba/alerts/builder.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.AlertObjectBuilder", "ueba.alerts.builder")

from ueba.alerts.builder import *  # noqa: E402,F401,F403
from ueba.alerts.builder import AlertObjectBuilder  # noqa: E402,F401
