"""Back-compat CLI shim — moved to src/ueba/serving/build_dashboard_dataset.py.

Usage (unchanged):
    python scripts/build_dashboard_dataset.py [--version 6]
Equivalent canonical invocation: python -m ueba.serving.build_dashboard_dataset
"""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.build_dashboard_dataset", "ueba.serving.build_dashboard_dataset")

from ueba.serving.build_dashboard_dataset import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
