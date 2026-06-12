"""Back-compat shim — moved to src/ueba/models/isolation_forest.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.UEBAIsolationForest", "ueba.models.isolation_forest")

from ueba.models.isolation_forest import *  # noqa: E402,F401,F403
from ueba.models.isolation_forest import UEBAIsolationForest  # noqa: E402,F401
