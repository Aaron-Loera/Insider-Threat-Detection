"""DEPRECATED — superseded by scripts/upload_to_hf.py.

This script uploaded the v5 pre-merged parquet to the old DSKittens/ueba-dashboard-dat
repo.  All uploads now go through upload_to_hf.py, which targets InsiderGuard-AI/ueba-v{V}
and handles both dataset and model artifacts in a single run.  Do not use this script.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from huggingface_hub import HfApi

HF_REPO   = "DSKittens/ueba-dashboard-dat"
HF_TOKEN  = os.environ.get("HF_TOKEN") or open(os.path.join(BASE_DIR, ".hf_token")).read().strip()
LOCAL_SRC = os.path.join(BASE_DIR, "explainability", "alert_table", "merged_dataset_5.parquet")

api = HfApi(token=HF_TOKEN)
print(f"Uploading {LOCAL_SRC} ({os.path.getsize(LOCAL_SRC)/1e6:.1f} MB) …")
api.upload_file(
    path_or_fileobj=LOCAL_SRC,
    path_in_repo="merged_dataset_5.parquet",
    repo_id=HF_REPO,
    repo_type="dataset",
)
print("Done.")
