"""Fetch and verify exact public benchmark source revisions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "upstream_sources.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class PublicSource:
    name: str
    repository: str
    revision: str
    tree: str
    role: str


@dataclass(frozen=True)
class SourceManifest:
    default_destination: str
    redistribution_boundary: str
    sources: tuple[PublicSource, ...]


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> SourceManifest:
    """Load and validate the public-source manifest without network access."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("public-source manifest must use schema_version 1")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("public-source manifest must contain sources")

    sources: list[PublicSource] = []
    seen: set[str] = set()
    for position, item in enumerate(raw_sources):
        if not isinstance(item, Mapping):
            raise ValueError(f"sources[{position}] must be an object")
        source = PublicSource(
            name=_text(item.get("name"), f"sources[{position}].name"),
            repository=_text(item.get("repository"), f"sources[{position}].repository"),
            revision=_text(item.get("revision"), f"sources[{position}].revision"),
            tree=_text(item.get("tree"), f"sources[{position}].tree"),
            role=_text(item.get("role"), f"sources[{position}].role"),
        )
        if not SAFE_NAME.fullmatch(source.name):
            raise ValueError(f"unsafe source name: {source.name!r}")
        if source.name in seen:
            raise ValueError(f"duplicate source name: {source.name}")
        if not HEX40.fullmatch(source.revision) or not HEX40.fullmatch(source.tree):
            raise ValueError(f"{source.name} revision and tree must be lowercase Git hashes")
        if not source.repository.startswith(
            "https://github.com/"
        ) or not source.repository.endswith(".git"):
            raise ValueError(f"{source.name} repository must be an HTTPS GitHub clone URL")
        seen.add(source.name)
        sources.append(source)

    destination = _text(payload.get("default_destination"), "default_destination")
    destination_path = Path(destination)
    if destination_path.is_absolute() or ".." in destination_path.parts:
        raise ValueError("default_destination must be a safe relative path")
    return SourceManifest(
        default_destination=destination,
        redistribution_boundary=_text(
            payload.get("redistribution_boundary"), "redistribution_boundary"
        ),
        sources=tuple(sources),
    )


def _destination(root: Path, source: PublicSource) -> Path:
    resolved_root = root.resolve()
    destination = (resolved_root / source.name).resolve()
    if not destination.is_relative_to(resolved_root):
        raise ValueError(f"source destination escapes output root: {source.name}")
    return destination


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def verify_source(source: PublicSource, destination: Path) -> dict[str, object]:
    """Verify a local checkout against its exact commit and tree identities."""
    if not (destination / ".git").is_dir():
        raise FileNotFoundError(f"{source.name} is not a Git checkout: {destination}")
    revision = _git("rev-parse", "HEAD", cwd=destination)
    tree = _git("rev-parse", "HEAD^{tree}", cwd=destination)
    if revision != source.revision:
        raise ValueError(f"{source.name} revision mismatch: {revision}")
    if tree != source.tree:
        raise ValueError(f"{source.name} tree mismatch: {tree}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=destination)
    if dirty:
        raise ValueError(f"{source.name} checkout has tracked modifications")
    return {
        "name": source.name,
        "destination": str(destination),
        "revision": revision,
        "tree": tree,
        "verified": True,
    }


def fetch_source(source: PublicSource, destination: Path) -> dict[str, object]:
    """Clone or update one checkout, then fail unless the exact identities match."""
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise FileExistsError(f"refusing to replace non-Git path: {destination}")
        remote = _git("remote", "get-url", "origin", cwd=destination)
        if remote.rstrip("/") != source.repository.rstrip("/"):
            raise ValueError(f"{source.name} origin mismatch: {remote}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--filter=blob:none", "--no-checkout", source.repository, str(destination))
    _git("fetch", "--depth=1", "origin", source.revision, cwd=destination)
    _git("checkout", "--detach", source.revision, cwd=destination)
    return verify_source(source, destination)


def _selected_sources(
    manifest: SourceManifest,
    requested: Sequence[str],
) -> tuple[PublicSource, ...]:
    index = {source.name: source for source in manifest.sources}
    unknown = sorted(set(requested) - index.keys())
    if unknown:
        raise ValueError(f"unknown sources: {unknown!r}")
    return tuple(index[name] for name in requested) if requested else manifest.sources


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "validate", help="validate the checked-in manifest without network access"
    )
    for command in ("fetch", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-root", type=Path)
        child.add_argument("--source", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest.resolve())
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_count": len(manifest.sources),
                    "sources": [asdict(source) for source in manifest.sources],
                    "valid": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (ROOT / manifest.default_destination).resolve()
    )
    sources = _selected_sources(manifest, args.source)
    operation = fetch_source if args.command == "fetch" else verify_source
    receipts = [operation(source, _destination(output_root, source)) for source in sources]
    print(json.dumps({"command": args.command, "sources": receipts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
