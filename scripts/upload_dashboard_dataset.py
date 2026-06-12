"""Upload the slim dashboard parquet to the HuggingFace dataset repo.

This pushes ONLY ``ueba_dataset_{V}_dashboard.parquet`` (the file the Streamlit
dashboard downloads at startup) to the dataset repo configured in config.py
(default: InsiderGuard-AI/ueba-v6). Build the file first with
``scripts/build_dashboard_dataset.py``.

Token resolution (same convention as scripts/upload_to_hf.py):
    --token TOKEN  ->  HF_TOKEN env var  ->  .hf_token file at project root

Usage:
    python scripts/upload_dashboard_dataset.py
    python scripts/upload_dashboard_dataset.py --version 6 --dry-run
    python scripts/upload_dashboard_dataset.py --dataset-repo InsiderGuard-AI/ueba-v6
"""
from __future__ import annotations

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import requests
from huggingface_hub import HfApi, configure_http_backend
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _configure_retry_session() -> None:
    """Retry the small LFS completion POST on CDN TCP resets without re-sending
    the full payload (same hardening used by scripts/upload_to_hf.py)."""
    retry = Retry(total=8, read=8, backoff_factor=3.0, allowed_methods=None, raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)

    def _factory() -> requests.Session:
        s = requests.Session()
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    configure_http_backend(_factory)


def _resolve_token(token_arg: str | None) -> str:
    token = token_arg or os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = os.path.join(BASE_DIR, ".hf_token")
    if os.path.exists(token_file):
        with open(token_file) as f:
            return f.read().strip()
    sys.exit(
        "No HF token found. Pass --token, set the HF_TOKEN env var, "
        "or create a .hf_token file at the project root."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="dataset version (default: config.MODEL_VERSION)")
    ap.add_argument("--dataset-repo", default=None, help="override HF dataset repo id")
    ap.add_argument("--token", default=None, help="HF write token (falls back to HF_TOKEN env, then .hf_token)")
    ap.add_argument("--dry-run", action="store_true", help="resolve everything but do not upload")
    args = ap.parse_args()

    try:
        import config
        mv = args.version or config.MODEL_VERSION
        repo = args.dataset_repo or config.HF_DATASET_REPO
    except Exception:
        mv = args.version or "6"
        repo = args.dataset_repo or "InsiderGuard-AI/ueba-v6"

    fname = f"ueba_dataset_{mv}_dashboard.parquet"
    local = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{mv}", fname)
    if not os.path.exists(local):
        sys.exit(f"[ERROR] {local} not found — run scripts/build_dashboard_dataset.py first.")

    size_mb = os.path.getsize(local) / 1e6
    print(f"File : {local} ({size_mb:.1f} MB)")
    print(f"Repo : datasets/{repo}")
    print(f"Path : {fname}")

    if args.dry_run:
        print("[dry-run] skipping upload.")
        return

    token = _resolve_token(args.token)
    _configure_retry_session()
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=fname,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Add slim dashboard serving dataset ({fname})",
    )
    print(f"Uploaded -> https://huggingface.co/datasets/{repo}/blob/main/{fname}")


if __name__ == "__main__":
    main()
