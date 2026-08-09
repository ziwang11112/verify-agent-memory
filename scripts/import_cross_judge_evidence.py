"""Normalize the post-hoc natural cross-judge audit into claim-bound evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.import_counterfactual_exposure_evidence import _write_csv
from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "natural_cross_judge_audit"
DEFAULT_NORMALIZED = ROOT / "evidence" / "normalized" / "cross_judge_audit.csv"
DEFAULT_MANIFEST = ROOT / "evidence" / "manifests" / "cross_judge_audit.json"
SOURCE_REPOSITORY = "ziwang11112/verify-agent-memory"
SOURCE_SNAPSHOT_COMMIT = "3007717b6dc1dc038e6f79c946f533b2369caffe"
JUDGE_MODEL = "gpt-5.1-2025-11-13"
SOURCE_FILENAMES = (
    "manifest.json",
    "answer_correct_agreement.csv",
    "answer_correct_confusion.csv",
    "answer_quality_agreement.csv",
    "secondary_label_agreement.csv",
    "sample_profile.csv",
    "execution_receipt.json",
    "audit_summary.md",
)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def _one(
    rows: Sequence[Mapping[str, str]],
    **selectors: str,
) -> Mapping[str, str]:
    selected = [
        row for row in rows if all(row.get(field) == value for field, value in selectors.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one cross-judge row for {selectors}, found {len(selected)}")
    return selected[0]


def _validate_package(results_dir: Path) -> None:
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "status": "complete_posthoc_outcome_independent_cross_judge_audit",
        "sample_count": 200,
        "reader_estimates_pooled": False,
        "independently_preregistered_replication": False,
        "official_benchmark_claim": False,
        "raw_content_included": False,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"unexpected {field} in cross-judge manifest")

    receipt = json.loads((results_dir / "execution_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("complete_bundle") is not True
        or receipt.get("raw_payload_or_response_content_included") is not False
    ):
        raise ValueError("cross-judge receipt is incomplete or contains raw content")
    if receipt.get("implementation_commit") != SOURCE_SNAPSHOT_COMMIT:
        raise ValueError("cross-judge receipt is bound to an unexpected implementation commit")
    if receipt.get("model") != JUDGE_MODEL or receipt.get("request_count") != 200:
        raise ValueError("cross-judge receipt has an unexpected judge or request count")


def _row(
    *,
    contrast: str,
    metric: str,
    estimate: str,
    n: str,
    lower: str = "",
    upper: str = "",
) -> dict[str, object]:
    return {
        "claim_id": "C14",
        "family": "natural_cross_judge_audit",
        "population": (
            "200 exact-deduplicated natural closure outputs, selected without outcomes "
            "across reader, source, and common-route strata"
        ),
        "source": JUDGE_MODEL,
        "contrast": contrast,
        "metric": metric,
        "estimate": estimate,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "n": n,
        "notes": (
            "post_hoc_outcome_independent;not_independently_preregistered;"
            "full_population_not_rescored;one_alternate_judge;"
            "reader_effects_not_reestimated;nonofficial_sample"
        ),
    }


def normalized_rows(results_dir: Path) -> tuple[dict[str, object], ...]:
    """Return the bounded aggregate audit evidence without raw prompts or answers."""
    _validate_package(results_dir)
    agreement = _read_csv(results_dir / "answer_correct_agreement.csv")
    quality = _one(
        _read_csv(results_dir / "answer_quality_agreement.csv"),
        group_type="overall",
        group_value="all",
    )
    secondary = _read_csv(results_dir / "secondary_label_agreement.csv")
    confusion = _one(
        _read_csv(results_dir / "answer_correct_confusion.csv"),
        group_type="overall",
        group_value="all",
    )

    rows: list[dict[str, object]] = []
    agreement_groups = (
        ("overall", "all", "overall"),
        ("reader_panel", "original_two_reader", "original_two_reader"),
        ("reader_panel", "sequential_gpt_reader", "sequential_gpt_reader"),
        ("reader", "DeepSeek", "reader_deepseek"),
        ("reader", "Gemini", "reader_gemini"),
    )
    for group_type, group_value, contrast in agreement_groups:
        source = _one(agreement, group_type=group_type, group_value=group_value)
        metrics = (
            ("exact_agreement", "cohen_kappa", "gwet_ac1")
            if contrast == "overall"
            else ("exact_agreement",)
        )
        for metric in metrics:
            rows.append(
                _row(
                    contrast=contrast,
                    metric=metric,
                    estimate=source[metric],
                    lower=source[f"{metric}_ci_low"],
                    upper=source[f"{metric}_ci_high"],
                    n=source["n"],
                )
            )

    for metric in (
        "exact_agreement",
        "within_one_agreement",
        "mean_absolute_error",
        "mean_strong_minus_cheap",
        "quadratic_weighted_kappa",
    ):
        rows.append(
            _row(
                contrast="answer_quality",
                metric=metric,
                estimate=quality[metric],
                n=quality["n"],
            )
        )

    for field in ("protected_disclosure", "stale_disclosure"):
        source = _one(secondary, group_type="overall", group_value="all", field=field)
        rows.append(
            _row(
                contrast=field,
                metric="exact_agreement",
                estimate=source["exact_agreement"],
                n=source["n"],
            )
        )

    for metric in ("cheap_correct_strong_incorrect", "cheap_incorrect_strong_correct"):
        rows.append(
            _row(
                contrast="directional_disagreement",
                metric=metric,
                estimate=confusion[metric],
                n=confusion["n"],
            )
        )

    if len(rows) != 16:
        raise ValueError(f"expected 16 normalized cross-judge rows, found {len(rows)}")
    return tuple(rows)


def generate(
    results_dir: Path,
    normalized_path: Path,
    manifest_path: Path,
) -> Mapping[str, object]:
    rows = normalized_rows(results_dir)
    _write_csv(normalized_path, rows)
    source_paths = tuple(results_dir / filename for filename in SOURCE_FILENAMES)
    manifest = {
        "schema_version": 1,
        "contains_raw_text_or_private_content": False,
        "normalized_file": normalized_path.relative_to(ROOT).as_posix(),
        "normalized_sha256": sha256_file(normalized_path),
        "row_count": len(rows),
        "source_artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
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
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate(args.results.resolve(), args.normalized.resolve(), args.manifest.resolve())
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
