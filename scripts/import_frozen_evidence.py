"""Import hash-verified, content-free aggregate evidence from the frozen archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml

from verify_agent_memory.provenance import sha256_bytes, sha256_file, verify_sha256

SOURCE_COMMIT = "a28093110325968c26906223e9eb0f1e078f6aad"
NATURAL_EXECUTION_COMMIT = "8e34e3d41c56e1699696bc27be95cdac7c9528e5"
FIELDS = (
    "claim_id",
    "family",
    "population",
    "source",
    "contrast",
    "metric",
    "estimate",
    "ci95_lower",
    "ci95_upper",
    "n",
    "notes",
)


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _integer(value: object, label: str) -> int:
    number = _finite_float(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _csv_rows(content: bytes, label: str) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(f"{label} has invalid CSV headers")
    return [dict(row) for row in reader]


def _one(
    rows: Iterable[Mapping[str, str]],
    *,
    label: str,
    **criteria: str,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(field) == expected for field, expected in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} expected one row, found {len(matches)}")
    return matches[0]


class FrozenSource:
    """Read exact blobs from one commit and enforce the source index hashes."""

    def __init__(self, repository: Path, index_path: Path) -> None:
        self.repository = repository.resolve()
        if not (self.repository / ".git").is_dir():
            raise ValueError(f"source repository is not a Git checkout: {self.repository}")
        with index_path.open(encoding="utf-8") as handle:
            index = yaml.safe_load(handle)
        artifacts = index.get("artifacts") if isinstance(index, Mapping) else None
        if not isinstance(artifacts, list):
            raise ValueError("source artifact index is invalid")
        self._index = {
            str(item["source_path"]): str(item["source_sha256"])
            for item in artifacts
            if isinstance(item, Mapping)
        }
        self.used: dict[str, str] = {}

    def read(self, path: str) -> bytes:
        expected = self._index.get(path)
        if expected is None:
            raise ValueError(f"source path is not allowlisted: {path}")
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "show",
                f"{SOURCE_COMMIT}:{path}",
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"git show failed for {path}: {message}")
        verify_sha256(completed.stdout, expected, label=path)
        self.used[path] = expected
        return completed.stdout


def _evidence_row(
    *,
    claim_id: str,
    family: str,
    population: str,
    source: str,
    contrast: str,
    metric: str,
    estimate: float,
    lower: float | None = None,
    upper: float | None = None,
    n: int | None = None,
    notes: str = "",
) -> dict[str, object]:
    values = (estimate, lower, upper)
    if any(value is not None and not math.isfinite(value) for value in values):
        raise ValueError("normalized evidence values must be finite")
    if (
        lower is not None
        and upper is not None
        and (lower > upper or not lower <= estimate <= upper)
    ):
        raise ValueError("normalized confidence interval is invalid")
    return {
        "claim_id": claim_id,
        "family": family,
        "population": population,
        "source": source,
        "contrast": contrast,
        "metric": metric,
        "estimate": estimate,
        "ci95_lower": "" if lower is None else lower,
        "ci95_upper": "" if upper is None else upper,
        "n": "" if n is None else n,
        "notes": notes,
    }


def _gatemem_rows(source: FrozenSource) -> list[dict[str, object]]:
    reader_specs = (
        (
            "reader_a",
            "reports/stage3_gatemem_clean_reader_replication",
        ),
        (
            "reader_b",
            "reports/stage3_gatemem_independent_reader_ra",
        ),
    )
    output: list[dict[str, object]] = []
    for public_reader, directory in reader_specs:
        manifest_path = f"{directory}/reader_manifest.json"
        manifest = json.loads(source.read(manifest_path))
        model = str(manifest["model"])
        exposure_path = f"{directory}/exposure_use_pairs.csv"
        exposure = _one(
            _csv_rows(source.read(exposure_path), exposure_path),
            label=f"{public_reader} pooled exposure",
            population="pooled",
        )
        output.append(
            _evidence_row(
                claim_id="C2",
                family="gatemem",
                population="exposure-discordant privacy or safety checkpoints",
                source=model,
                contrast="exposed_minus_unexposed",
                metric="answer_leakage_risk_difference",
                estimate=_finite_float(
                    exposure["risk_difference_exposed_minus_unexposed"],
                    "risk difference",
                ),
                lower=_finite_float(exposure["ci95_lower"], "risk lower"),
                upper=_finite_float(exposure["ci95_upper"], "risk upper"),
                n=_integer(exposure["discordant_checkpoints"], "discordant checkpoints"),
                notes="association_only;same_provider;reader_estimates_not_pooled",
            )
        )

        prediction_path = f"{directory}/incremental_prediction.csv"
        predictions = _csv_rows(source.read(prediction_path), prediction_path)
        for metric in ("brier", "log_loss"):
            row = _one(
                predictions,
                label=f"{public_reader} {metric}",
                metric=metric,
            )
            output.append(
                _evidence_row(
                    claim_id="C2",
                    family="gatemem",
                    population="held-out checkpoint rows",
                    source=model,
                    contrast="exposure_augmented_minus_baseline",
                    metric=f"{metric}_delta",
                    estimate=_finite_float(
                        row["delta_augmented_minus_baseline"],
                        f"{metric} delta",
                    ),
                    lower=_finite_float(row["ci95_lower"], f"{metric} lower"),
                    upper=_finite_float(row["ci95_upper"], f"{metric} upper"),
                    n=_integer(row["rows"], f"{metric} rows"),
                    notes="association_only;same_provider;reader_estimates_not_pooled",
                )
            )

    tradeoff_path = "reports/stage3_gatemem_clean_reader_replication/utility_and_refusal.csv"
    tradeoff = _csv_rows(source.read(tradeoff_path), tradeoff_path)
    metric_map = {
        "primary_answer_leakage": "answer_leakage_delta",
        "utility_accuracy": "utility_accuracy_delta",
        "over_refusal_rate": "over_refusal_delta",
    }
    for source_metric, public_metric in metric_map.items():
        row = _one(
            tradeoff,
            label=f"G1-G0 {source_metric}",
            comparison="G1-G0",
            metric=source_metric,
        )
        output.append(
            _evidence_row(
                claim_id="C3",
                family="gatemem",
                population="frozen clean-reader eligible checkpoints",
                source="gpt-4o-mini-2024-07-18",
                contrast="G1_minus_G0",
                metric=public_metric,
                estimate=_finite_float(
                    row["delta_first_minus_second"],
                    f"{source_metric} delta",
                ),
                lower=_finite_float(row["ci95_lower"], f"{source_metric} lower"),
                upper=_finite_float(row["ci95_upper"], f"{source_metric} upper"),
                n=_integer(row["checkpoints"], f"{source_metric} checkpoints"),
                notes="non_causal;bounded_utility_population",
            )
        )
    return output


def _mechanism_smoke_rows(source: FrozenSource) -> list[dict[str, object]]:
    path = "reports/stage3_admissibility_method_comparison/main_table.csv"
    rows = _csv_rows(source.read(path), path)
    method_map = {
        "flat_dense": "global_dense",
        "namespace_dense": "namespace_dense",
    }
    metric_map = {
        "mean_evidence_recall": "evidence_recall",
        "contamination_at_matched_recall": "measured_contamination",
        "wrong_namespace_leakage_rate": "wrong_scope_leakage",
    }
    output: list[dict[str, object]] = []
    for source_method, public_method in method_map.items():
        row = _one(
            rows,
            label=f"mechanism smoke {source_method}",
            split="eval",
            method=source_method,
            source_group="all",
        )
        for source_metric, public_metric in metric_map.items():
            output.append(
                _evidence_row(
                    claim_id="C4",
                    family="mechanism_smoke",
                    population="16 curated evaluation packets",
                    source="public-source audit subset",
                    contrast=public_method,
                    metric=public_metric,
                    estimate=_finite_float(
                        row[source_metric],
                        f"{source_method} {source_metric}",
                    ),
                    n=_integer(row["query_count"], "smoke query count"),
                    notes="mechanism_smoke_only;not_natural_corpus_estimate",
                )
            )
    return output


def _source_macro(
    rows: Sequence[Mapping[str, str]],
    *,
    arm: str,
    metric: str,
) -> float:
    base_sources = ("rhelm", "memops")
    selected = [row for row in rows if row.get("arm") == arm and row.get("source") in base_sources]
    if len(selected) != len(base_sources) or {row.get("source") for row in selected} != set(
        base_sources
    ):
        raise ValueError(f"{arm} does not have exactly the two expected sources")
    return sum(_finite_float(row[metric], f"{arm} {metric}") for row in selected) / 2


def _paired_evidence(
    paired: Sequence[Mapping[str, str]],
    *,
    claim_id: str,
    method: str,
    reference: str,
    metric: str,
    public_contrast: str,
    public_metric: str,
    notes: str,
) -> dict[str, object]:
    row = _one(
        paired,
        label=f"{method} minus {reference} {metric}",
        method=method,
        reference=reference,
        source="source_macro",
        stratum="all",
        metric=metric,
    )
    return _evidence_row(
        claim_id=claim_id,
        family="natural_evaluation",
        population="87 groups; 182908 memories; 3767 queries",
        source="RHELM and MemOps public sources",
        contrast=public_contrast,
        metric=public_metric,
        estimate=_finite_float(
            row["mean_delta_method_minus_reference"],
            f"{method} {metric} delta",
        ),
        lower=_finite_float(row["bootstrap_ci95_lower"], f"{method} {metric} lower"),
        upper=_finite_float(row["bootstrap_ci95_upper"], f"{method} {metric} upper"),
        n=_integer(row["paired_query_count"], f"{method} paired queries"),
        notes=notes,
    )


def _validate_natural_execution_manifest(
    manifest: Mapping[str, object],
    *,
    main_content: bytes,
    paired_content: bytes,
) -> int:
    selected_settings = manifest.get("selected_settings")
    if not isinstance(selected_settings, Mapping):
        raise ValueError("natural execution manifest lacks selected settings")
    expected_arms = {
        "flat_bm25",
        "flat_bm25_dense_rrf",
        "flat_dense",
        "namespace_current_only",
        "namespace_dense",
        "namespace_source_intent_lifecycle",
        "ncr_a5",
        "ncr_threshold",
        "recency_dense",
    }
    if set(selected_settings) != expected_arms:
        raise ValueError("natural execution manifest has unexpected method coverage")

    query_rows = _integer(manifest.get("query_rows"), "natural query rows")
    expected_rows = query_rows * len(expected_arms)
    if _integer(manifest.get("route_rows"), "natural route rows") != expected_rows:
        raise ValueError("natural route rows do not cover every method and query")
    if _integer(manifest.get("score_rows"), "natural score rows") != expected_rows:
        raise ValueError("natural score rows do not cover every method and query")
    if manifest.get("commit") != NATURAL_EXECUTION_COMMIT:
        raise ValueError("natural execution commit mismatch")

    for field in ("provider_calls", "reader_calls", "judge_calls", "paid_calls"):
        if _integer(manifest.get(field), field) != 0:
            raise ValueError(f"natural execution manifest has nonzero {field}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("natural execution manifest lacks artifact hashes")
    expected_hashes = {
        "main_table.csv": sha256_bytes(main_content),
        "paired_deltas.csv": sha256_bytes(paired_content),
    }
    for name, expected in expected_hashes.items():
        if artifacts.get(name) != expected:
            raise ValueError(f"natural execution manifest hash mismatch for {name}")
    return query_rows


def _natural_rows(source: FrozenSource) -> list[dict[str, object]]:
    manifest_path = "reports/stage3_natural_corpus_admissibility_eval/execution_manifest.json"
    main_path = "reports/stage3_natural_corpus_admissibility_eval/main_table.csv"
    paired_path = "reports/stage3_natural_corpus_admissibility_eval/paired_deltas.csv"
    manifest = json.loads(source.read(manifest_path))
    main_content = source.read(main_path)
    paired_content = source.read(paired_path)
    query_rows = _validate_natural_execution_manifest(
        manifest,
        main_content=main_content,
        paired_content=paired_content,
    )
    main = _csv_rows(main_content, main_path)
    paired = _csv_rows(paired_content, paired_path)
    population = "87 groups; 182908 memories; 3767 queries"
    output = [
        _evidence_row(
            claim_id="C5",
            family="natural_evaluation",
            population=population,
            source="RHELM and MemOps public sources",
            contrast="global_dense",
            metric="evidence_recall",
            estimate=_source_macro(
                main,
                arm="flat_dense",
                metric="mean_evidence_recall",
            ),
            n=query_rows,
            notes="source_macro;not_official_benchmark_submission",
        ),
        _evidence_row(
            claim_id="C5",
            family="natural_evaluation",
            population=population,
            source="RHELM and MemOps public sources",
            contrast="namespace_dense",
            metric="evidence_recall",
            estimate=_source_macro(
                main,
                arm="namespace_dense",
                metric="mean_evidence_recall",
            ),
            n=query_rows,
            notes="source_macro;trusted_released_namespace",
        ),
        _evidence_row(
            claim_id="C5",
            family="natural_evaluation",
            population=population,
            source="RHELM and MemOps public sources",
            contrast="namespace_dense",
            metric="wrong_scope_leakage",
            estimate=_source_macro(
                main,
                arm="namespace_dense",
                metric="mean_wrong_namespace_leakage_rate",
            ),
            n=query_rows,
            notes="trusted_released_namespace",
        ),
    ]

    public_arms = {
        "flat_bm25": "global_bm25",
        "flat_dense": "global_dense",
        "flat_bm25_dense_rrf": "global_bm25_dense_rrf",
        "recency_dense": "global_recency_dense",
        "namespace_dense": "namespace_dense",
        "namespace_current_only": "namespace_current_only",
        "namespace_source_intent_lifecycle": "released_intent_lifecycle_upper_bound",
        "ncr_threshold": "threshold_router",
        "ncr_a5": "cluster_router",
    }
    arm_metrics = {
        "mean_evidence_recall": "evidence_recall",
        "feasible_rate": "feasible_rate",
        "conservative_contamination_upper": "penalized_contamination_upper",
        "mean_candidates_scored": "mean_candidates_scored",
    }
    for source_arm, public_arm in public_arms.items():
        for source_metric, public_metric in arm_metrics.items():
            if public_arm == "global_dense" and public_metric == "evidence_recall":
                continue
            if public_arm == "namespace_dense" and public_metric == "evidence_recall":
                continue
            notes = "source_macro;descriptive_full_arm_table;not_official_benchmark_submission"
            if public_arm == "released_intent_lifecycle_upper_bound":
                notes += ";released_field_upper_bound"
            if public_metric == "penalized_contamination_upper":
                notes += ";infeasible_or_unresolved_equals_one"
            if public_metric == "mean_candidates_scored":
                notes += ";diagnostic_only;not_production_latency"
            output.append(
                _evidence_row(
                    claim_id="C5",
                    family="natural_evaluation",
                    population=population,
                    source="RHELM and MemOps public sources",
                    contrast=public_arm,
                    metric=public_metric,
                    estimate=_source_macro(
                        main,
                        arm=source_arm,
                        metric=source_metric,
                    ),
                    n=query_rows,
                    notes=notes,
                )
            )

    for source_arm, public_arm in (
        ("flat_dense", "global_dense"),
        ("namespace_dense", "namespace_dense"),
    ):
        for source_metric, public_metric in {
            "mean_contamination_at_matched_recall": "known_contamination",
            "mean_contamination_label_coverage": "label_coverage",
            "mean_contamination_lower_bound": "lower_bound",
            "mean_contamination_upper_bound": "upper_bound",
        }.items():
            output.append(
                _evidence_row(
                    claim_id="C5",
                    family="natural_evaluation",
                    population=population,
                    source="RHELM and MemOps public sources",
                    contrast=f"{public_arm}_matched_prefix",
                    metric=public_metric,
                    estimate=_source_macro(
                        main,
                        arm=source_arm,
                        metric=source_metric,
                    ),
                    n=query_rows,
                    notes=(
                        "source_macro;feasible_matched_prefix_only;"
                        "descriptive_incomplete_label_diagnostic"
                    ),
                )
            )

    for source_metric, public_metric in {
        "evidence_recall": "recall_delta",
        "feasible_rate": "feasible_rate_delta",
        "conservative_contamination_upper": "conservative_contamination_delta",
    }.items():
        output.append(
            _paired_evidence(
                paired,
                claim_id="C5",
                method="namespace_dense",
                reference="flat_dense",
                metric=source_metric,
                public_contrast="namespace_dense_minus_global_dense",
                public_metric=public_metric,
                notes=(
                    "source_macro;trusted_released_namespace;infeasible_or_unresolved_equals_one"
                    if public_metric == "conservative_contamination_delta"
                    else "source_macro;trusted_released_namespace"
                ),
            )
        )

    for method, public_method in (
        ("ncr_threshold", "threshold_router"),
        ("ncr_a5", "cluster_router"),
    ):
        for source_metric, public_metric in {
            "evidence_recall": "recall_delta",
            "conservative_contamination_upper": "conservative_contamination_delta",
        }.items():
            output.append(
                _paired_evidence(
                    paired,
                    claim_id="C6",
                    method=method,
                    reference="namespace_dense",
                    metric=source_metric,
                    public_contrast=f"{public_method}_minus_namespace_dense",
                    public_metric=public_metric,
                    notes="diagnostic_router;no_incremental_utility_established",
                )
            )

    output.extend(
        (
            _evidence_row(
                claim_id="C6",
                family="natural_evaluation",
                population=population,
                source="RHELM and MemOps public sources",
                contrast="threshold_router",
                metric="fallback_rate",
                estimate=_source_macro(
                    main,
                    arm="ncr_threshold",
                    metric="fallback_rate",
                ),
                n=query_rows,
                notes="diagnostic_router;namespace_local_fallback",
            ),
            _evidence_row(
                claim_id="C6",
                family="natural_evaluation",
                population=population,
                source="RHELM and MemOps public sources",
                contrast="cluster_router",
                metric="mean_route_width",
                estimate=_source_macro(
                    main,
                    arm="ncr_a5",
                    metric="mean_route_width",
                ),
                n=query_rows,
                notes="diagnostic_router;selected_clusters",
            ),
        )
    )

    for source_metric, public_metric in {
        "evidence_recall": "recall_delta",
        "conservative_contamination_upper": "conservative_contamination_delta",
        "prohibited_stale_exposure_rate": "prohibited_stale_exposure_delta",
        "prohibited_superseded_exposure_rate": "prohibited_superseded_exposure_delta",
    }.items():
        output.append(
            _paired_evidence(
                paired,
                claim_id="C7",
                method="namespace_source_intent_lifecycle",
                reference="namespace_dense",
                metric=source_metric,
                public_contrast=("released_intent_lifecycle_upper_bound_minus_namespace_dense"),
                public_metric=public_metric,
                notes="released_field_upper_bound;not_deployable_blind_inference",
            )
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(row[field]) for field in FIELDS)))


def _write_manifest(
    *,
    repository_root: Path,
    normalized_path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, object]],
    source_artifacts: Mapping[str, str],
) -> None:
    manifest = {
        "schema_version": 1,
        "source_repository": "ziwang11112/bomi",
        "source_snapshot_commit": SOURCE_COMMIT,
        "normalized_file": normalized_path.relative_to(repository_root).as_posix(),
        "normalized_sha256": sha256_file(normalized_path),
        "row_count": len(rows),
        "contains_raw_text_or_private_content": False,
        "transformation_script": "scripts/import_frozen_evidence.py",
        "transformation_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_artifacts": [
            {"path": path, "sha256": digest} for path, digest in sorted(source_artifacts.items())
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    manifest_path.write_bytes(content)


def import_evidence(
    *,
    source_repository: Path,
    repository_root: Path,
    source_index: Path,
) -> tuple[Path, ...]:
    source = FrozenSource(source_repository, source_index)
    families = (
        ("gatemem", _gatemem_rows),
        ("mechanism_smoke", _mechanism_smoke_rows),
        ("natural_evaluation", _natural_rows),
    )
    written: list[Path] = []
    for family, builder in families:
        before = set(source.used)
        rows = builder(source)
        used_paths = set(source.used) - before
        normalized_path = repository_root / "evidence" / "normalized" / f"{family}.csv"
        manifest_path = repository_root / "evidence" / "manifests" / f"{family}.json"
        _write_csv(normalized_path, rows)
        _write_manifest(
            repository_root=repository_root,
            normalized_path=normalized_path,
            manifest_path=manifest_path,
            rows=rows,
            source_artifacts={path: source.used[path] for path in used_paths},
        )
        written.extend((normalized_path, manifest_path))
    return tuple(written)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=root.parent / "bomi-codex-starter",
    )
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--source-index",
        type=Path,
        default=root / "SOURCE_ARTIFACTS.yaml",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    written = import_evidence(
        source_repository=args.source_repo,
        repository_root=args.repository_root.resolve(),
        source_index=args.source_index,
    )
    digest = sha256_bytes("\n".join(path.as_posix() for path in written).encode())
    print(f"imported {len(written) // 2} evidence families; receipt={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
