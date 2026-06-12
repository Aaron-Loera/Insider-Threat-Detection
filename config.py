"""Back-compat shim — the configuration now lives in src/ueba/config.py.

dashboard/app.py (the Streamlit Cloud entrypoint) and the notebooks import
``config`` from the repo root. This shim re-exports everything from
ueba.config, working in two situations:

  1. the ueba package is installed (``pip install -e .`` — the normal path,
     requirements.txt includes it for Streamlit Cloud), or
  2. the package is NOT installed (bare clone) — src/ is put on sys.path so
     the source tree resolves directly. This fallback keeps the dashboard
     alive even if the editable install step fails on a redeploy.
"""

import os as _os
import sys as _sys

try:
    import ueba  # noqa: F401
except ImportError:  # not pip-installed — fall back to the source tree
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(_os.path.realpath(__file__))), "src"))

from ueba.config import *  # noqa: F401,F403,E402
