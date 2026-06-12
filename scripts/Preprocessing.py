"""Back-compat shim — moved to src/ueba/features/preprocessing.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.Preprocessing", "ueba.features.preprocessing")

from ueba.features.preprocessing import *  # noqa: E402,F401,F403
