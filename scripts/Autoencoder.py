"""Back-compat shim — moved to src/ueba/models/autoencoder.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.Autoencoder", "ueba.models.autoencoder")

from ueba.models.autoencoder import *  # noqa: E402,F401,F403
from ueba.models.autoencoder import Autoencoder  # noqa: E402,F401
