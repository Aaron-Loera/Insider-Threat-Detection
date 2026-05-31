"""Upload UEBA artifacts to HuggingFace — dataset splits to a Dataset repo, model
weights and inference artifacts to a Model repo — in a single run.

Usage:
    python scripts/upload_to_hf.py [--version 6]
                                   [--dataset-repo InsiderGuard-AI/ueba-v6]
                                   [--model-repo   InsiderGuard-AI/ueba-models-v6]
                                   [--token TOKEN] [--dry-run] [--skip-model-repo]

Both repos must already exist on HuggingFace before running.
Repo defaults are derived from --version so no edits are needed for future versions.
After a successful upload, update dashboard/app.py _HF_BASE to point at the dataset repo.
"""
import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import HfApi, configure_http_backend

# Manifest tuple shape: (local_abs_path, path_in_repo, required, dest)
# dest is "dataset" or "model" — controls which HF repo receives the file.
_DATASET = "dataset"
_MODEL   = "model"


def _configure_hf_retry_session() -> None:
    """Patch the huggingface_hub HTTP session to retry on connection-reset errors.

    The LFS multipart completion POST (sent after the full file body has been
    transferred) is vulnerable to WinError 10054 TCP resets from the HF CDN.
    Retrying at the urllib3 level means only the small completion POST is
    re-sent — the 158 MB payload does NOT need to be re-uploaded.
    """
    retry = Retry(
        total=8,
        read=8,            # retry on read/connection errors (covers WinError 10054)
        backoff_factor=3.0,
        allowed_methods=None,  # allow retrying POST (the completion POST)
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    def _session_factory() -> requests.Session:
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    configure_http_backend(_session_factory)
    print("    [info] urllib3 retry adapter configured (8 retries on connection errors)")


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


def _build_manifest(V: str) -> list[tuple[str, str, bool, str]]:
    """Return list of (local_abs_path, path_in_repo, required, dest) tuples."""
    enc_dir = f"encoder_model_{V}"
    if_dir  = f"iforest_model_{V}"
    ds_dir  = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}")

    return [
        # ── Dataset repo ────────────────────────────────────────────────────
        # Dashboard data — required for app.py fallback download
        (config.ANALYST_TABLE_PARQUET,
         f"alert_table_{V}.parquet",                    True,  _DATASET),
        (config.CALIB_ALERT_TABLE_PARQUET,
         f"alert_table_{V}_calib.parquet",              False, _DATASET),
        (os.path.join(ds_dir, f"ueba_dataset_{V}b.parquet"),
         f"ueba_dataset_{V}b.parquet",                  True,  _DATASET),

        # Dataset splits
        (config.UEBA_PATH,
         f"ueba_dataset_{V}_train.parquet",             False, _DATASET),
        (config.UEBA_CALIBRATION_PATH,
         f"ueba_dataset_{V}_calibration.parquet",       False, _DATASET),
        (config.UEBA_CALIB_EVAL_PATH,
         f"ueba_dataset_{V}_calibration_eval.parquet",  False, _DATASET),
        (os.path.join(ds_dir, f"ueba_dataset_{V}_test_stream.parquet"),
         f"ueba_dataset_{V}_test_stream.parquet",       False, _DATASET),
        (config.UEBA_A_PARQUET,
         f"ueba_dataset_{V}a.parquet",                  False, _DATASET),
        (os.path.join(ds_dir, "user_work_hours.parquet"),
         "user_work_hours.parquet",                     False, _DATASET),

        # ── Model repo ──────────────────────────────────────────────────────
        # Encoder artifacts
        (config.ENCODER_PATH,
         f"{enc_dir}/encoder_model.keras",              False, _MODEL),
        (config.SCALER_PATH,
         f"{enc_dir}/feature_scaler.pkl",               False, _MODEL),
        (os.path.join(BASE_DIR, "encoders", enc_dir, "feature_cols.json"),
         f"{enc_dir}/feature_cols.json",                False, _MODEL),
        (config.AE_BASELINE_PATH,
         f"{enc_dir}/ae_baseline_clean.npy",            False, _MODEL),
        (config.CALIBRATION_THRESHOLD_PATH,
         f"{enc_dir}/calibration_thresholds.json",      False, _MODEL),

        # Isolation Forest artifacts
        (config.IF_PATH,
         f"{if_dir}/iforest_model.pkl",                 False, _MODEL),
        (config.IF_BASELINE_PATH,
         f"{if_dir}/if_baseline_clean.npy",             False, _MODEL),
        (config.IF_SCORES_PATH,
         f"{if_dir}/anomaly_scores.npy",                False, _MODEL),
    ]


def _print_manifest(
    manifest: list[tuple[str, str, bool, str]],
    dataset_repo: str,
    model_repo: str,
    skip_model_repo: bool,
) -> tuple[int, int]:
    """Print manifest grouped by destination repo. Returns (dataset_bytes, model_bytes)."""
    sections = [(_DATASET, dataset_repo), (_MODEL, model_repo)]
    ds_total = model_total = 0

    for dest, repo in sections:
        entries = [(l, r, req) for l, r, req, d in manifest if d == dest]
        if not entries:
            continue
        if dest == _MODEL and skip_model_repo:
            print(f"\n  Model repo ({repo}) — skipped (--skip-model-repo)")
            continue

        print(f"\n  {'Dataset' if dest == _DATASET else 'Model'} repo -> {repo}")
        for local, remote, required in entries:
            if not os.path.exists(local):
                tag = "MISSING (required)" if required else "missing (optional)"
                print(f"    {'':>9}  [{tag}]  {remote}")
            else:
                size = os.path.getsize(local)
                if dest == _DATASET:
                    ds_total += size
                else:
                    model_total += size
                req = " *" if required else ""
                print(f"    {size / 1e6:>8.1f} MB  {remote}{req}")

    return ds_total, model_total


def _upload_with_retry(
    api: HfApi,
    local: str,
    remote: str,
    repo_id: str,
    repo_type: str,
    max_attempts: int = 5,
    base_delay: float = 5.0,
) -> None:
    """Upload a single file, retrying on transient network errors with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            return
        except Exception as exc:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"\n    [warn] attempt {attempt}/{max_attempts} failed ({exc}); retrying in {delay:.0f}s …",
                  end=" ", flush=True)
            time.sleep(delay)


def _run(args: argparse.Namespace) -> None:
    V            = args.version
    dataset_repo = args.dataset_repo or f"InsiderGuard-AI/ueba-v{V}"
    model_repo   = args.model_repo   or f"InsiderGuard-AI/ueba-models-v{V}"

    token    = _resolve_token(args.token)
    manifest = _build_manifest(V)

    # Pre-flight: abort if any required dataset file is missing
    missing_required = [
        (local, remote)
        for local, remote, required, dest in manifest
        if required and dest == _DATASET and not os.path.exists(local)
    ]
    if missing_required:
        for local, _ in missing_required:
            print(f"  [error] Required file not found: {local}")
        sys.exit("Aborting — one or more required files are missing.")

    total_files = len(manifest) if not args.skip_model_repo else sum(
        1 for *_, dest in manifest if dest == _DATASET
    )

    print(f"\nDataset repo : {dataset_repo}")
    print(f"Model repo   : {model_repo}{'  (skipped)' if args.skip_model_repo else ''}")
    print(f"Version      : {V}")
    print(f"Dry run      : {args.dry_run}")
    print(f"\nManifest ({total_files} files):  (* = required for dashboard fallback)")

    ds_bytes, model_bytes = _print_manifest(
        manifest, dataset_repo, model_repo, args.skip_model_repo
    )
    print(f"\n  Dataset total : {ds_bytes / 1e6:.1f} MB")
    if not args.skip_model_repo:
        print(f"  Model total   : {model_bytes / 1e6:.1f} MB")
    print(f"  Grand total   : {(ds_bytes + model_bytes) / 1e6:.1f} MB")

    if args.dry_run:
        print("\nDry run complete -- nothing uploaded.")
        return

    print()
    _configure_hf_retry_session()
    api = HfApi(token=token)
    uploaded_ds = uploaded_model = 0

    for local, remote, _, dest in manifest:
        if dest == _MODEL and args.skip_model_repo:
            continue
        if not os.path.exists(local):
            print(f"  [warn] skipping missing optional file: {remote}")
            continue
        repo_id = dataset_repo if dest == _DATASET else model_repo
        size_mb = os.path.getsize(local) / 1e6
        print(f"  uploading {remote} ({size_mb:.1f} MB) → {dest} repo …", end=" ", flush=True)
        _upload_with_retry(api, local, remote, repo_id, dest)
        print("done")
        if dest == _DATASET:
            uploaded_ds += 1
        else:
            uploaded_model += 1

    print(f"\nUploaded {uploaded_ds} file(s) to dataset repo  : {dataset_repo}")
    if not args.skip_model_repo:
        print(f"Uploaded {uploaded_model} file(s) to model repo    : {model_repo}")
    print(
        f"\nNext steps:"
        f"\n  1. Update dashboard/app.py _HF_BASE to:"
        f"\n       https://huggingface.co/datasets/{dataset_repo}/resolve/main"
        f"\n  2. Update live_simulation.py / config.py HF model base to:"
        f"\n       https://huggingface.co/{model_repo}/resolve/main"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload UEBA artifacts to HuggingFace (dataset + model repos).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", default=config.MODEL_VERSION,
        help="Dataset/model version suffix (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-repo", default=None,
        help="HF dataset repo ID (default: InsiderGuard-AI/ueba-v{version})",
    )
    parser.add_argument(
        "--repo", dest="dataset_repo",
        help="Alias for --dataset-repo (backward compat)",
    )
    parser.add_argument(
        "--model-repo", default=None,
        help="HF model repo ID (default: InsiderGuard-AI/ueba-models-v{version})",
    )
    parser.add_argument(
        "--token", default=None,
        help="HF write token; falls back to HF_TOKEN env var, then .hf_token file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the upload manifest without uploading anything",
    )
    parser.add_argument(
        "--skip-model-repo", action="store_true",
        help="Skip all model repo uploads — push dataset files only",
    )
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
