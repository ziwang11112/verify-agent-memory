from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from scripts.export_anonymous_artifact import export_anonymous_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_anonymous_export_redacts_identity_and_repairs_manifest(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "artifact"
    script = repository / "scripts" / "import_evidence.py"
    manifest = repository / "evidence" / "manifests" / "sample.json"
    readme = repository / "README.md"
    workflow = repository / ".github" / "workflows" / "ci.yml"
    env_file = repository / ".env"
    for path in (script, manifest, readme, workflow, env_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    private_handle = "zi" + "wang11112"
    private_name = "zi" + " wang"
    private_username = "zi" + "wan"
    private_email = "zw" + "ang@" + "ua" + "lr.edu"
    private_institution = "University of " + "Arkansas at Little Rock"
    private_institution_short = "UA" + "LR"
    private_workspace = "D:\\" + "agent" + "-mem"
    script.write_text(f'SOURCE = "{private_handle}/verify-agent-memory"\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "source_repository": f"{private_handle}/verify-agent-memory",
                "transformation_script": "scripts/import_evidence.py",
                "transformation_script_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    readme.write_text(
        "\n".join(
            (
                f"Approved-by: {private_name}",
                f"Contact: {private_email}",
                f"Institution: {private_institution} ({private_institution_short})",
                f"Home: C:\\Users\\{private_username}",
                f"Workspace: {private_workspace}\\verify-agent-memory",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    workflow.write_text("name: ci\n", encoding="utf-8")
    env_file.write_text("TOKEN=not-exported\n", encoding="utf-8")

    report = export_anonymous_artifact(
        repository,
        output,
        tracked_paths=(
            "scripts/import_evidence.py",
            "evidence/manifests/sample.json",
            "README.md",
            ".github/workflows/ci.yml",
            ".env",
        ),
        require_clean=False,
    )

    assert private_handle not in (output / "scripts" / "import_evidence.py").read_text()
    exported_readme = (output / "README.md").read_text(encoding="utf-8")
    assert private_name not in exported_readme.lower()
    assert private_username not in exported_readme.lower()
    assert private_email not in exported_readme.lower()
    assert private_institution not in exported_readme
    assert private_institution_short.lower() not in exported_readme.lower()
    assert private_workspace not in exported_readme
    assert not (output / ".github").exists()
    assert not (output / ".env").exists()
    exported_manifest = json.loads(
        (output / "evidence" / "manifests" / "sample.json").read_text(encoding="utf-8")
    )
    assert exported_manifest["source_repository"] == "anonymous/verify-agent-memory"
    assert exported_manifest["transformation_script_sha256"] == _sha256(
        output / "scripts" / "import_evidence.py"
    )
    assert report["history_included"] is False
    assert report["repaired_evidence_manifests"] == ["evidence/manifests/sample.json"]
    assert (output / "ANONYMIZATION_REPORT.json").is_file()


def test_anonymous_export_rejects_secret_shaped_content(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "artifact"
    source = repository / "README.md"
    source.parent.mkdir(parents=True)
    source.write_text("token=gsk_" + "X" * 24, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Groq API key"):
        export_anonymous_artifact(
            repository,
            output,
            tracked_paths=("README.md",),
            require_clean=False,
        )


def test_anonymous_export_rejects_identity_in_non_utf8_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "artifact"
    source = repository / "fixture.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"\xffprivate=" + ("zi" + "wang11112").encode("ascii"))

    with pytest.raises(RuntimeError, match="private repository handle"):
        export_anonymous_artifact(
            repository,
            output,
            tracked_paths=("fixture.bin",),
            require_clean=False,
        )


def test_exported_anonymizer_preserves_its_identity_rules(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "artifact"
    export_anonymous_artifact(
        repository,
        output,
        tracked_paths=("scripts/export_anonymous_artifact.py",),
        require_clean=False,
    )

    exported_path = output / "scripts" / "export_anonymous_artifact.py"
    spec = importlib.util.spec_from_file_location("exported_anonymizer", exported_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load exported anonymizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private_email = "zw" + "ang@" + "ua" + "lr.edu"

    redacted, changed = module._redact(f"contact={private_email}".encode())

    assert changed is True
    assert private_email.encode("utf-8") not in redacted
