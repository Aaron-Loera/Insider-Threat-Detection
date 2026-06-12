"""Tests for the pipeline artifact manifest (ueba.pipeline.manifest)."""

import os

import pytest

from ueba import config
from ueba.pipeline import manifest


def test_require_passes_when_artifacts_exist(tmp_path):
    p = tmp_path / "artifact.parquet"
    p.write_text("x")
    assert manifest.require([(str(p), "preprocess")]) == []


def test_require_raises_naming_the_producing_stage(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(manifest.MissingArtifactError) as exc:
        manifest.require([(missing, "explain --split test_stream")])
    msg = str(exc.value)
    assert "nope.parquet" in msg
    assert "explain --split test_stream" in msg
    assert "python -m ueba.pipeline" in msg


def test_require_allow_missing_returns_paths(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    assert manifest.require([(missing, "train-ae")], allow_missing=True) == [missing]


@pytest.fixture
def tmp_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PIPELINE_MANIFEST_PATH", str(tmp_path / "pipeline_manifest.json"))
    return tmp_path


def test_record_and_load_roundtrip(tmp_manifest):
    artifact = tmp_manifest / "data" / "out.parquet"
    artifact.parent.mkdir()
    artifact.write_bytes(b"hello parquet")

    manifest.record("preprocess", [str(artifact)])

    data = manifest.load()
    rel = os.path.join("data", "out.parquet")
    assert rel in data["artifacts"]
    entry = data["artifacts"][rel]
    assert entry["stage"] == "preprocess"
    assert entry["bytes"] == len(b"hello parquet")


def test_record_skips_absent_optional_outputs(tmp_manifest):
    manifest.record("train-ae", [str(tmp_manifest / "never_created.npy")])
    assert manifest.load()["artifacts"] == {}


def test_validate_recorded_flags_deleted_and_resized(tmp_manifest):
    artifact = tmp_manifest / "model.pkl"
    artifact.write_bytes(b"12345")
    manifest.record("train-if", [str(artifact)])

    assert manifest.validate_recorded() == []

    artifact.write_bytes(b"123456789")  # resized
    problems = manifest.validate_recorded()
    assert len(problems) == 1 and "size changed" in problems[0]

    artifact.unlink()  # deleted
    problems = manifest.validate_recorded()
    assert len(problems) == 1 and "no longer on disk" in problems[0]
