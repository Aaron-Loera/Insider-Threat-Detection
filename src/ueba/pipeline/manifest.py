"""Fail-fast artifact manifest for the offline pipeline.

Converts the historical silent-skip failure mode (a stage quietly doing
nothing because an upstream artifact is missing — CLEANUP_REPORT gap 3) into
a loud, attributable error: require() names the missing path AND the stage
that produces it. record() keeps a journal of what each stage produced so
`python -m ueba.pipeline status` can audit the artifact tree.
"""

import datetime
import hashlib
import json
import os
import subprocess

from ueba import config


class MissingArtifactError(RuntimeError):
    """Raised when a stage's required input artifacts are absent."""


def require(artifacts: list[tuple[str, str]], allow_missing: bool = False) -> list[str]:
    """Fail unless every (path, producing_stage) artifact exists.

    Returns the list of missing paths (only ever non-empty when
    allow_missing=True, the explicit escape hatch).
    """
    missing = [(p, s) for p, s in artifacts if not os.path.exists(p)]
    if missing and not allow_missing:
        lines = ["Missing required artifacts:"]
        for path, stage in missing:
            lines.append(f"  {path}")
            lines.append(f"      -> produced by stage '{stage}' (run: python -m ueba.pipeline {stage})")
        raise MissingArtifactError("\n".join(lines))
    return [p for p, _ in missing]


def _sha1_head(path: str, n_bytes: int = 1024 * 1024) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read(n_bytes)).hexdigest()[:12]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=config.BASE_DIR, timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load() -> dict:
    path = config.PIPELINE_MANIFEST_PATH
    if not os.path.exists(path):
        return {"model_version": config.MODEL_VERSION, "artifacts": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def record(stage: str, paths: list[str]) -> None:
    """Journal the artifacts a stage just produced (keyed by repo-relative path)."""
    data = load()
    commit = _git_commit()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    for path in paths:
        if not os.path.exists(path):
            continue  # optional outputs may legitimately be absent
        rel = os.path.relpath(path, config.BASE_DIR)
        data["artifacts"][rel] = {
            "stage": stage,
            "bytes": os.path.getsize(path),
            "sha1_head": _sha1_head(path),
            "git_commit": commit,
            "model_version": config.MODEL_VERSION,
            "recorded_at": now,
        }
    os.makedirs(os.path.dirname(config.PIPELINE_MANIFEST_PATH), exist_ok=True)
    with open(config.PIPELINE_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def validate_recorded() -> list[str]:
    """Return problems with previously recorded artifacts (missing / resized)."""
    problems = []
    data = load()
    for rel, meta in data.get("artifacts", {}).items():
        path = os.path.join(config.BASE_DIR, rel)
        if not os.path.exists(path):
            problems.append(f"{rel}: recorded by '{meta['stage']}' but no longer on disk")
        elif os.path.getsize(path) != meta["bytes"]:
            problems.append(
                f"{rel}: size changed since recorded by '{meta['stage']}' "
                f"({meta['bytes']:,} -> {os.path.getsize(path):,} bytes)"
            )
    return problems
