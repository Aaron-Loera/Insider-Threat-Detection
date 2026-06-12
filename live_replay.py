"""Back-compat CLI shim — moved to src/ueba/serving/live_replay.py.

The dashboard may launch this path as a subprocess on Streamlit Cloud, so it
must stay runnable here:

    python live_replay.py [--interval 0.5] [--output <path>] [--port 8765]

Equivalent canonical invocation: python -m ueba.serving.live_replay
"""

import os as _os
import sys as _sys

try:
    import ueba  # noqa: F401
except ImportError:  # not pip-installed — fall back to the source tree
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(_os.path.realpath(__file__))), "src"))

from ueba.serving.live_replay import main, run  # noqa: E402,F401

if __name__ == "__main__":
    main()
