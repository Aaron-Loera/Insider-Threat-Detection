"""Back-compat CLI shim — moved to src/ueba/serving/live_simulation.py.

The dashboard launches this exact path as a subprocess (config.LIVE_SIM_SCRIPT),
so it must stay runnable here:

    python live_simulation.py [--interval 0.5] [--input <path>] [--port 8765]

Equivalent canonical invocation: python -m ueba.serving.live_simulation
"""

import os as _os
import sys as _sys

try:
    import ueba  # noqa: F401
except ImportError:  # not pip-installed — fall back to the source tree
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(_os.path.realpath(__file__))), "src"))

from ueba.serving.live_simulation import (  # noqa: E402,F401
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    PAUSE_FLAG,
    LiveScorer,
    main,
)

if __name__ == "__main__":
    main()
