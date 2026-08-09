"""Normalize the natural route-to-reader evaluation into claim-bound evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.import_counterfactual_exposure_evidence import _write_csv
from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TWO_READER_RESULTS = ROOT / "results" / "natural_end_to_end_two_reader_judged"
DEFAULT_GPT_RESULTS = ROOT / "results" / "natural_end_to_end_gpt_luna_judged"
DEFAULT_NORMALIZED = ROOT / "evidence" / "normalized" / "natural_end_to_end.csv"
DEFAULT_MANIFEST = ROOT / "evidence" / "manifests" / "natural_end_to_end.json"
SOURCE_REPOSITORY = "ziwang11112/verify-agent-memory"
SOURCE_SNAPSHOT_COMMIT = "c608fa639c03eec3f9665c27ea9eb9c83dd0a22d"

PRIMARY_METRICS = {
    "evidence_recall",
    "feasible",
    "penalized_admissibility_upper_risk",
    "answer_correct",
    "answer_quality",
    "over_refusal",
    "protected_disclosure",
    "stale_disclosure",
}
POLICY_METRICS = {"penalized_admissibility_upper_risk", "answer_correct"}
TEXT_VERIFIER_METRICS = {"penalized_admissibility_upper_risk", "answer_correct"}
SOURCE_FILENAMES = (
    "manifest.json",
    "main_table.csv",
    "paired_deltas.csv",
    "execution_receipt.json",
)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def _validate_package(path: Path, *, expected_status: str) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != expected_status:
        raise ValueError(f"unexpected result status in {path}")
    if manifest.get("sample_case_count") != 1523:
        raise ValueError(f"unexpected sample size in {path}")
    if manifest.get("reader_estimates_pooled") is not False:
        raise ValueError(f"reader estimates must remain separate in {path}")
    receipt = json.loads((path / "execution_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("complete_bundle") is not True:
        raise ValueError(f"incomplete result bundle in {path}")
    if receipt.get("benchmark_payload_or_response_content_included") is not False:
        raise ValueError(f"result package is not content-free: {path}")


def _selected(row: Mapping[str, str], *, include_diagnostic_routes: bool) -> bool:
    arm = row["arm"]
    reference = row["reference"]
    metric = row["metric"]
    if arm == "namespace_dense" and reference == "global_dense":
        return metric in PRIMARY_METRICS
    if arm == "namespace_policy_gate" and reference == "namespace_dense":
        return metric in POLICY_METRICS
    if include_diagnostic_routes and arm == "namespace_text_verifier":
        return reference == "namespace_dense" and metric in TEXT_VERIFIER_METRICS
    return False


def normalized_rows(
    two_reader_results: Path,
    gpt_results: Path,
) -> tuple[dict[str, object], ...]:
    """Return reader-separated paired deltas without pooling models or routes."""
    _validate_package(
        two_reader_results,
        expected_status="complete_nonofficial_cost_aware_semantic_judge_sample",
    )
    _validate_package(
        gpt_results,
        expected_status="complete_nonofficial_gpt_luna_primary_route_replication",
    )
    population = (
        "frozen 1,523-case natural sample: all 523 RHELM cases and an "
        "outcome-independent 1,000-case MemOps sample"
    )
    rows: list[dict[str, object]] = []
    for result_dir, sequential_replication, include_diagnostic_routes in (
        (two_reader_results, False, True),
        (gpt_results, True, False),
    ):
        notes = (
            "natural_same_population;reader_estimates_not_pooled;"
            "shared_claude_haiku_judge;nonofficial_sample;"
            "no_general_disclosure_gain"
        )
        if sequential_replication:
            notes += ";sequential_reader_replication"
        else:
            notes += ";prespecified_two_reader_execution"
        for source in _read_csv(result_dir / "paired_deltas.csv"):
            if not _selected(source, include_diagnostic_routes=include_diagnostic_routes):
                continue
            rows.append(
                {
                    "claim_id": "C13",
                    "family": "natural_end_to_end",
                    "population": population,
                    "source": source["reader_model"],
                    "contrast": f"{source['arm']}_minus_{source['reference']}",
                    "metric": source["metric"],
                    "estimate": source["mean_delta_arm_minus_reference"],
                    "ci95_lower": source["bootstrap_ci95_lower"],
                    "ci95_upper": source["bootstrap_ci95_upper"],
                    "n": source["paired_query_count"],
                    "notes": notes,
                }
            )
    rows.sort(key=lambda row: (str(row["source"]), str(row["contrast"]), str(row["metric"])))
    if len(rows) != 34:
        raise ValueError(f"expected 34 normalized rows, found {len(rows)}")
    return tuple(rows)


def generate(
    two_reader_results: Path,
    gpt_results: Path,
    normalized_path: Path,
    manifest_path: Path,
) -> Mapping[str, object]:
    rows = normalized_rows(two_reader_results, gpt_results)
    _write_csv(normalized_path, rows)
    source_paths = tuple(
        result_dir / filename
        for result_dir in (two_reader_results, gpt_results)
        for filename in SOURCE_FILENAMES
    )
    manifest = {
        "schema_version": 1,
        "contains_raw_text_or_private_content": False,
        "normalized_file": normalized_path.relative_to(ROOT).as_posix(),
        "normalized_sha256": sha256_file(normalized_path),
        "row_count": len(rows),
        "source_artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "source_repository": SOURCE_REPOSITORY,
        "source_snapshot_commit": SOURCE_SNAPSHOT_COMMIT,
        "transformation_script": Path(__file__).relative_to(ROOT).as_posix(),
        "transformation_script_sha256": sha256_file(Path(__file__)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-reader-results", type=Path, default=DEFAULT_TWO_READER_RESULTS)
    parser.add_argument("--gpt-results", type=Path, default=DEFAULT_GPT_RESULTS)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate(
        args.two_reader_results.resolve(),
        args.gpt_results.resolve(),
        args.normalized.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
