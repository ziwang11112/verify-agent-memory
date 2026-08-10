from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fetch_public_sources import PublicSource, load_manifest, verify_source

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_public_source_manifest_is_valid() -> None:
    manifest = load_manifest(ROOT / "data" / "upstream_sources.json")

    assert [source.name for source in manifest.sources] == ["gatemem", "rhelm", "memops"]
    assert all(len(source.revision) == 40 for source in manifest.sources)
    assert all(len(source.tree) == 40 for source in manifest.sources)


def test_public_source_manifest_rejects_unsafe_names(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_destination": "data/raw/upstream",
                "redistribution_boundary": "test boundary",
                "sources": [
                    {
                        "name": "../escape",
                        "repository": "https://github.com/example/example.git",
                        "revision": "0" * 40,
                        "tree": "1" * 40,
                        "role": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe source name"):
        load_manifest(path)


def test_verify_source_fails_closed_for_missing_checkout(tmp_path: Path) -> None:
    source = PublicSource(
        name="fixture",
        repository="https://github.com/example/example.git",
        revision="0" * 40,
        tree="1" * 40,
        role="test",
    )

    with pytest.raises(FileNotFoundError, match="not a Git checkout"):
        verify_source(source, tmp_path / "fixture")
