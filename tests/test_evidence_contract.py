from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts.verify_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[1]


def copy_evidence_contract(destination: Path) -> Path:
    shutil.copytree(ROOT / "evidence", destination / "evidence")
    shutil.copytree(ROOT / "scripts", destination / "scripts")
    shutil.copytree(ROOT / "claims", destination / "claims")
    shutil.copy2(ROOT / "SOURCE_ARTIFACTS.yaml", destination / "SOURCE_ARTIFACTS.yaml")
    return destination


@pytest.fixture
def repository_copy() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=".evidence-test-", dir=ROOT) as temporary:
        yield copy_evidence_contract(Path(temporary) / "repository")


def test_repository_evidence_passes() -> None:
    assert validate_evidence(ROOT) == []


def test_tampered_normalized_evidence_is_rejected(repository_copy: Path) -> None:
    evidence = repository_copy / "evidence" / "normalized" / "gatemem.csv"
    evidence.write_text(evidence.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert any(
        "gatemem.csv SHA-256 mismatch" in error for error in validate_evidence(repository_copy)
    )


def test_source_hash_must_match_the_frozen_index(repository_copy: Path) -> None:
    manifest_path = repository_copy / "evidence" / "manifests" / "gatemem.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any(
        "source SHA-256 disagrees with index" in error
        for error in validate_evidence(repository_copy)
    )


def test_transformation_script_is_hash_bound(repository_copy: Path) -> None:
    script = repository_copy / "scripts" / "import_frozen_evidence.py"
    script.write_text(script.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert any(
        "transformation script SHA-256 mismatch" in error
        for error in validate_evidence(repository_copy)
    )


def test_manifest_paths_cannot_escape_repository(repository_copy: Path) -> None:
    manifest_path = repository_copy / "evidence" / "manifests" / "gatemem.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["normalized_file"] = "../../outside.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any("escapes the repository" in error for error in validate_evidence(repository_copy))


def test_normalized_values_must_match_claim_contract(repository_copy: Path) -> None:
    evidence = repository_copy / "evidence" / "normalized" / "mechanism_smoke.csv"
    content = evidence.read_text(encoding="utf-8")
    evidence.write_text(
        content.replace(",0.11889880952380952,,,", ",0.12,,,"),
        encoding="utf-8",
    )
    manifest_path = repository_copy / "evidence" / "manifests" / "mechanism_smoke.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from verify_agent_memory.provenance import sha256_file

    manifest["normalized_sha256"] = sha256_file(evidence)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any(
        "estimate disagrees with claims/claims.yaml" in error
        for error in validate_evidence(repository_copy)
    )
