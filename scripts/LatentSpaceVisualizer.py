"""Back-compat shim — moved to src/ueba/viz/latent_space_visualizer.py."""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.LatentSpaceVisualizer", "ueba.viz.latent_space_visualizer")

from ueba.viz.latent_space_visualizer import *  # noqa: E402,F401,F403
