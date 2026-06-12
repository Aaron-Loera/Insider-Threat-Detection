"""Back-compat shim — moved to src/ueba/models/data_prep.py.

Import from ueba.models.data_prep in new code.
"""

import os as _os
import sys as _sys
import warnings as _warnings

try:
    import ueba  # noqa: F401
except ImportError:  # not pip-installed — fall back to the source tree
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(_os.path.realpath(__file__))), "src"))

from ueba.models.data_prep import (  # noqa: E402,F401
    ENCODER_PATH,
    IF_PATH,
    SCALER_PATH,
    build_insider_mask,
    chronological_split,
    get_insiders,
    get_scores,
    prepare_ae_training_data,
    to_model_matrix,
)

__all__ = [
    "chronological_split",
    "get_insiders",
    "build_insider_mask",
    "prepare_ae_training_data",
    "to_model_matrix",
    "get_scores",
]

_warnings.warn(
    "prepare_data is deprecated; import from ueba.models.data_prep instead.",
    DeprecationWarning,
    stacklevel=2,
)
