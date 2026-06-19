"""Shared HuggingFace read helpers for the dataset and model repos.

Every runtime path that needs a v{N} artifact (the dashboard, live_replay's
cloud fallback, per-user explanation lookups) should go through
get_dataset_file / get_model_file instead of re-implementing its own
local-path check + hf_hub_download. Both default to config.MODEL_VERSION, so
nothing here needs to change for v7, v8, etc.
"""
from __future__ import annotations

import os

from ueba import config


def _resolve_token() -> str | None:
    """Streamlit secrets first (dashboard context), then HF_TOKEN env var
    (live_replay / CLI context, where streamlit may not even be installed)."""
    try:
        import streamlit as st
        try:
            section = st.secrets.get("huggingface", {})
            token = dict(section).get("token") if section else None
            if token:
                return token
        except Exception:
            pass
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


def _get_repo_file(
    filename: str,
    *,
    repo_id: str,
    repo_type: str,
    local_path: str | None,
) -> str:
    if local_path and os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=_resolve_token(),
    )


def get_dataset_file(
    filename: str,
    *,
    version: str | None = None,
    local_path: str | None = None,
) -> str:
    """Return a local path to `filename` in the ueba-v{version} dataset repo.

    Checks `local_path` (where the pipeline would have written it) first and
    only downloads from HuggingFace — with HF's own on-disk caching — when
    that file is missing, e.g. a Streamlit Cloud deploy that never ran the
    local pipeline.
    """
    V = version or config.MODEL_VERSION
    repo_id = f"{config.HF_ORG}/ueba-v{V}"
    return _get_repo_file(filename, repo_id=repo_id, repo_type="dataset", local_path=local_path)


def get_model_file(
    filename: str,
    *,
    version: str | None = None,
    local_path: str | None = None,
) -> str:
    """Return a local path to `filename` in the ueba-models-v{version} repo.

    Same local-cache-first behavior as get_dataset_file. Not currently wired
    into live_simulation.py (which only reads local model paths); this exists
    so a future "bootstrap a fresh clone" workflow can pull the trained
    encoder/scaler/IF ensemble instead of retraining from scratch.
    """
    V = version or config.MODEL_VERSION
    repo_id = f"{config.HF_ORG}/ueba-models-v{V}"
    return _get_repo_file(filename, repo_id=repo_id, repo_type="model", local_path=local_path)
