"""Publish or verify content-free supplemental natural-corpus diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from verify_agent_memory.provenance import sha256_file

DATA_FILES = (
    "natural_top_k_pareto.csv",
    "natural_admissibility_attribution.csv",
    "natural_admissibility_attribution_deltas.csv",
    "natural_namespace_corruption_seed_rows.csv",
    "natural_namespace_corruption_summary.csv",
    "natural_namespace_break_even.json",
    "natural_governance_corruption_seed_rows.csv",
    "natural_governance_corruption_summary.csv",
    "natural_governance_break_even.json",
)
SOURCE_MANIFESTS = (
    "natural_top_k_pareto_manifest.json",
    "natural_admissibility_attribution_manifest.json",
    "natural_namespace_break_even_manifest.json",
    "natural_governance_break_even_manifest.json",
)
FORBIDDEN_COLUMNS = {
    "query_id",
    "memory_id",
    "ranked_memory_ids",
    "raw_text",
    "text",
    "embedding",
    "vector",
    "prompt",
    "response",
    "answer",
}
EXPECTED_ROWS = {
    "natural_top_k_pareto.csv": 45,
    "natural_admissibility_attribution.csv": 5,
    "natural_admissibility_attribution_deltas.csv": 12,
    "natural_namespace_corruption_seed_rows.csv": 160,
    "natural_namespace_corruption_summary.csv": 16,
    "natural_governance_corruption_seed_rows.csv": 560,
    "natural_governance_corruption_summary.csv": 56,
}


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _validate_csv(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not fields or fields.intersection(FORBIDDEN_COLUMNS):
            raise ValueError(f"{path.name} contains forbidden or absent columns")
        rows = list(reader)
    expected = EXPECTED_ROWS.get(path.name)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{path.name} has {len(rows)} rows; expected {expected}")
    return len(rows)


def _source_output_hashes(input_dir: Path) -> dict[str, str]:
    top_k = _object(input_dir / "natural_top_k_pareto_manifest.json")
    attribution = _object(input_dir / "natural_admissibility_attribution_manifest.json")
    namespace = _object(input_dir / "natural_namespace_break_even_manifest.json")
    governance = _object(input_dir / "natural_governance_break_even_manifest.json")
    return {
        "natural_top_k_pareto.csv": str(top_k["output_sha256"]),
        "natural_admissibility_attribution.csv": str(attribution["main_output_sha256"]),
        "natural_admissibility_attribution_deltas.csv": str(attribution["delta_output_sha256"]),
        "natural_namespace_corruption_seed_rows.csv": str(namespace["seed_rows_sha256"]),
        "natural_namespace_corruption_summary.csv": str(namespace["summary_sha256"]),
        "natural_namespace_break_even.json": str(namespace["break_even_sha256"]),
        "natural_governance_corruption_seed_rows.csv": str(governance["seed_rows_sha256"]),
        "natural_governance_corruption_summary.csv": str(governance["summary_sha256"]),
        "natural_governance_break_even.json": str(governance["break_even_sha256"]),
    }


def publish(input_dir: Path, output_dir: Path) -> Path:
    """Validate local aggregate outputs and publish a hash-bound package."""
    expected_hashes = _source_output_hashes(input_dir)
    for name in DATA_FILES:
        path = input_dir / name
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"local supplemental output failed its source receipt: {name}")
        if path.suffix == ".csv":
            _validate_csv(path)
    for name in SOURCE_MANIFESTS:
        manifest = _object(input_dir / name)
        for field in ("provider_calls", "reader_calls", "judge_calls", "paid_calls"):
            if manifest.get(field) != 0:
                raise ValueError(f"{name} records nonzero {field}")
        if manifest.get("contains_query_or_memory_ids") is not False:
            raise ValueError(f"{name} is not marked identifier-free")
        if manifest.get("contains_raw_text_embeddings_or_responses") is not False:
            raise ValueError(f"{name} is not marked content-free")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (*DATA_FILES, *SOURCE_MANIFESTS):
        _write_text_lf(output_dir / name, (input_dir / name).read_text(encoding="utf-8"))
    files = {
        name: {"sha256": sha256_file(output_dir / name)}
        for name in (*DATA_FILES, *SOURCE_MANIFESTS)
    }
    for name, expected in EXPECTED_ROWS.items():
        files[name]["row_count"] = expected
    source_manifests = [_object(input_dir / name) for name in SOURCE_MANIFESTS]
    manifest = {
        "schema_version": 1,
        "status": "post_hoc_supplemental_diagnostics_not_official_benchmark_result",
        "population": {
            "groups": 87,
            "memories": 182_908,
            "queries": 3_767,
            "sources": ["RHELM", "MemOps"],
        },
        "analysis_commits": sorted({str(row["new_repository_commit"]) for row in source_manifests}),
        "frozen_execution_commit": "8e34e3d41c56e1699696bc27be95cdac7c9528e5",
        "embedding_checkpoint_sha256": (
            "d333477116735f49b7628ee5d324e92a9f608190079a7b1d699308f5ec4f1ae6"
        ),
        "ranking_parameters_changed": False,
        "evaluation_retuning": False,
        "contains_query_or_memory_ids": False,
        "contains_raw_text_embeddings_or_responses": False,
        "provider_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
        "paid_calls": 0,
        "publisher": "scripts/publish_supplemental_results.py",
        "publisher_sha256": sha256_file(Path(__file__).resolve()),
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    _write_text_lf(
        manifest_path,
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest_path


def verify(output_dir: Path) -> None:
    """Verify a published package without opening any private source artifact."""
    manifest = _object(output_dir / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("supplemental manifest schema is unsupported")
    for field in ("provider_calls", "reader_calls", "judge_calls", "paid_calls"):
        if manifest.get(field) != 0:
            raise ValueError(f"supplemental manifest has nonzero {field}")
    if manifest.get("contains_query_or_memory_ids") is not False:
        raise ValueError("supplemental package is not marked identifier-free")
    if manifest.get("contains_raw_text_embeddings_or_responses") is not False:
        raise ValueError("supplemental package is not marked content-free")
    if manifest.get("publisher_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("supplemental publisher hash drifted")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {*DATA_FILES, *SOURCE_MANIFESTS}:
        raise ValueError("supplemental manifest file set drifted")
    for name, raw_record in files.items():
        if not isinstance(name, str) or not isinstance(raw_record, Mapping):
            raise TypeError("supplemental file record is malformed")
        path = output_dir / name
        if not path.is_file() or raw_record.get("sha256") != sha256_file(path):
            raise ValueError(f"supplemental file hash drifted: {name}")
        if path.suffix == ".csv":
            rows = _validate_csv(path)
            if raw_record.get("row_count") != rows:
                raise ValueError(f"supplemental row count drifted: {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "verify"))
    parser.add_argument("--input-dir", type=Path, default=root / "tmp")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "supplemental_natural",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "publish":
        path = publish(args.input_dir.resolve(), args.output_dir.resolve())
        print(json.dumps({"manifest": path.as_posix(), "status": "published"}))
    else:
        verify(args.output_dir.resolve())
        print(json.dumps({"status": "valid"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
