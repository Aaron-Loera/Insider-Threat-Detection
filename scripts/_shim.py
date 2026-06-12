"""Support for the deprecated scripts/* import locations.

Ensures the ueba package is importable (source-tree fallback when not
pip-installed) and emits the deprecation warning. Each scripts/*.py shim
calls ensure_ueba() before importing from its new ueba.* location.
"""

import os
import sys
import warnings

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__)))), "src")


def ensure_ueba(old: str, new: str) -> None:
    try:
        import ueba  # noqa: F401
    except ImportError:  # not pip-installed — fall back to the source tree
        sys.path.insert(0, _SRC)
    warnings.warn(
        f"{old} is deprecated; import {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
