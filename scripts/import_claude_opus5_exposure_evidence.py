"""Normalize the Claude Opus 5 exposure replication into claim-bound evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.import_counterfactual_exposure_evidence import (
    CELL_CONTRASTS,
    GAP_CONTRAST,
    _csv,
    _one,
    _write_csv,
)
from scripts.publish_claude_opus5_exposure_results import (
    EXECUTION_COMMIT,
    validate_published,
)
from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "claude_opus5_exposure_replication"
DEFAULT_NORMALIZED = ROOT / "evidence" / "normalized" / "claude_opus5_exposure_replication.csv"
DEFAULT_MANIFEST = ROOT / "evidence" / "manifests" / "claude_opus5_exposure_replication.json"
SOURCE_REPOSITORY = "ziwang11112/verify-agent-memory"
MODEL = "claude-opus-5"


def normalized_rows(results_dir: Path) -> tuple[dict[str, object], ...]:
    """Return five reader-specific effects without pooling models."""
    validate_published(results_dir)
    intervals = _csv(results_dir / "bootstrap_ci.csv")
    population = "16 controlled scenarios with paired exposed and withheld reader requests"
    notes = (
        "controlled_prompt_intervention;separate_fourth_reader_replication;"
        "providers_not_pooled;no_judge;not_official_benchmark"
    )
    rows = []
    for contrast in (*CELL_CONTRASTS, GAP_CONTRAST):
        source = _one(
            intervals,
            provider="Anthropic",
            model=MODEL,
            scope="overall",
            contrast=contrast,
        )
        rows.append(
            {
                "claim_id": "C12",
                "family": "counterfactual_exposure_replication",
                "population": population,
                "source": MODEL,
                "contrast": "admissible_minus_inadmissible"
                if contrast == GAP_CONTRAST
                else "exposed_minus_withheld",
                "metric": "selectivity_gap" if contrast == GAP_CONTRAST else f"{contrast}_effect",
                "estimate": source["estimate"],
                "ci95_lower": source["ci_lower"],
                "ci95_upper": source["ci_upper"],
                "n": source["unit_count"],
                "notes": notes,
            }
        )
    return tuple(rows)


def generate(
    results_dir: Path,
    normalized_path: Path,
    manifest_path: Path,
) -> Mapping[str, object]:
    rows = normalized_rows(results_dir)
    if len(rows) != 5:
        raise ValueError(f"expected 5 normalized rows, found {len(rows)}")
    _write_csv(normalized_path, rows)
    source_names = (
        "manifest.json",
        "cell_metrics.csv",
        "bootstrap_ci.csv",
        "provider_usage.csv",
    )
    manifest = {
        "schema_version": 1,
        "contains_raw_text_or_private_content": False,
        "normalized_file": normalized_path.relative_to(ROOT).as_posix(),
        "normalized_sha256": sha256_file(normalized_path),
        "row_count": len(rows),
        "source_artifacts": [
            {
                "path": (results_dir / name).relative_to(ROOT).as_posix(),
                "sha256": sha256_file(results_dir / name),
            }
            for name in source_names
        ],
        "source_repository": SOURCE_REPOSITORY,
        "source_snapshot_commit": EXECUTION_COMMIT,
        "transformation_script": Path(__file__).relative_to(ROOT).as_posix(),
        "transformation_script_sha256": sha256_file(Path(__file__)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate(
        args.results_dir.resolve(),
        args.normalized.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
