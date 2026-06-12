"""Back-compat CLI shim — moved to src/ueba/serving/upload_to_hf.py.

Usage (unchanged):
    python scripts/upload_to_hf.py [--version 6] [--dry-run] [--skip-model-repo]
Equivalent canonical invocation: python -m ueba.serving.upload_to_hf
"""

from scripts._shim import ensure_ueba

ensure_ueba("scripts.upload_to_hf", "ueba.serving.upload_to_hf")

from ueba.serving.upload_to_hf import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
