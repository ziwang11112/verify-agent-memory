from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_natural_end_to_end_experiment as base  # noqa: E402
from scripts import run_natural_gpt_reader_replication as gpt  # noqa: E402
from scripts import run_natural_two_reader_judge as two_reader  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results" / "natural_end_to_end_case_audit"
CASE_TOKEN_DOMAIN = "verify-agent-memory|natural-case-audit-v1|case|"
GROUP_TOKEN_DOMAIN = "verify-agent-memory|natural-case-audit-v1|group|"
HEX64 = frozenset("0123456789abcdef")

SCORE_FIELDS = (
    "evidence_recall",
    "feasible",
    "penalized_admissibility_upper_risk",
    "route_width",
    "wrong_namespace_exposure",
    "policy_disallowed_exposure",
    "lifecycle_incompatible_exposure",
    "unresolved_exposure",
    "answer_correct",
    "answer_quality",
    "contains_reference_answer",
    "protected_disclosure_evaluable",
    "protected_disclosure",
    "literal_protected_disclosure",
    "stale_disclosure",
)
PAIRED_METRICS = (
    "evidence_recall",
    "feasible",
    "penalized_admissibility_upper_risk",
    "answer_correct",
    "answer_quality",
    "non_answer",
    "protected_disclosure",
    "stale_disclosure",
)
COMPARISONS = (
    ("namespace_dense", "global_dense"),
    ("namespace_policy_gate", "namespace_dense"),
    ("namespace_text_verifier", "namespace_dense"),
    ("released_field_oracle", "namespace_dense"),
)


def _token(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}{value}".encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _sanitize_row(
    row: Mapping[str, object],
    *,
    case_order: int,
    anchor_total: int,
    reader_record_sha256: str,
    judge_record_sha256: str,
) -> dict[str, object]:
    case_id = str(row["case_id"])
    source = str(row["source"])
    group_id = str(row["group_id"])
    if case_order < 0 or anchor_total < 1:
        raise ValueError("natural case must have at least one evidence anchor")
    if not _is_sha256(reader_record_sha256) or not _is_sha256(judge_record_sha256):
        raise ValueError("private record bindings must be SHA-256 values")
    return {
        "case_token": _token(CASE_TOKEN_DOMAIN, case_id),
        "group_token": _token(GROUP_TOKEN_DOMAIN, f"{source}|{group_id}"),
        "source": source,
        "reader_provider": row["reader_provider"],
        "reader_model": row["reader_model"],
        "arm": row["arm"],
        "case_order": case_order,
        "anchor_total": anchor_total,
        **{field: row[field] for field in SCORE_FIELDS},
        "non_answer": int(bool(row["over_refusal"])),
        "reader_record_sha256": reader_record_sha256,
        "judge_record_sha256": judge_record_sha256,
    }


def _score_gpt_plan(
    execution: base.Protocol,
    cases: Sequence[base.NaturalEndToEndCase],
    judge_specs: Mapping[str, base.CallSpec],
    judge_assignments: Mapping[tuple[str, str], str],
    reader_responses: Mapping[str, base.ReaderResponse],
    reader_assignments: Mapping[tuple[str, str], str],
    verifier_scores: Mapping[str, Mapping[str, float]],
    *,
    runtime: Path,
) -> list[dict[str, object]]:
    judge_responses = base._load_judge_responses(
        protocol=execution,
        runtime=runtime,
        specs=judge_specs,
    )
    rows = []
    for case in cases:
        for arm in gpt.PRIMARY_ARMS:
            candidates = base.route_candidates(
                case,
                arm,
                verifier_scores=verifier_scores[case.case_id],
                verifier_threshold=execution.verifier_threshold,
            )
            route_score = base.matched_route_score(
                case,
                candidates,
                target_recall=execution.target_recall,
            )
            response = reader_responses[reader_assignments[(case.case_id, arm)]]
            deterministic = base.deterministic_answer_metrics(case, response)
            judge = judge_responses[judge_assignments[(case.case_id, arm)]]
            rows.append(
                {
                    "reader_provider": "OpenAI",
                    "reader_model": "gpt-5.6-luna",
                    "case_id": case.case_id,
                    "source": case.source,
                    "group_id": case.group_id,
                    "arm": arm,
                    "evidence_recall": route_score.evidence_recall,
                    "feasible": int(route_score.feasible),
                    "penalized_admissibility_upper_risk": (
                        route_score.penalized_admissibility_upper_risk
                    ),
                    "route_width": route_score.route_width,
                    "wrong_namespace_exposure": route_score.wrong_scope_exposure_rate,
                    "policy_disallowed_exposure": route_score.policy_disallowed_exposure_rate,
                    "lifecycle_incompatible_exposure": (
                        route_score.lifecycle_incompatible_exposure_rate
                    ),
                    "unresolved_exposure": route_score.unresolved_exposure_rate,
                    "answer_correct": int(judge.answer_correct),
                    "answer_quality": judge.answer_quality / 10,
                    "over_refusal": int(bool(deterministic["over_refusal"])),
                    "contains_reference_answer": int(
                        bool(deterministic["contains_reference_answer"])
                    ),
                    "protected_disclosure_evaluable": int(case.protected_disclosure_evaluable),
                    "protected_disclosure": (
                        int(judge.protected_disclosure)
                        if case.protected_disclosure_evaluable
                        else None
                    ),
                    "literal_protected_disclosure": deterministic["literal_protected_disclosure"],
                    "stale_disclosure": (
                        int(judge.stale_disclosure) if case.source == "memops" else None
                    ),
                }
            )
    return rows


def _build_two_reader_rows() -> tuple[list[dict[str, object]], list[base.NaturalEndToEndCase]]:
    judge_protocol = two_reader.load_judge_protocol(two_reader.DEFAULT_JUDGE_PROTOCOL)
    execution = two_reader.bind_recovered_judge(
        judge_protocol,
        base.load_protocol(two_reader.DEFAULT_EXECUTION_PROTOCOL),
    )
    selected, specs, judge_assignments, responses, reader_assignments, verifier_scores = (
        two_reader._build_plan(
            judge_protocol=judge_protocol,
            execution_protocol=execution,
            cases_path=two_reader.DEFAULT_CASES,
            source_runtime=two_reader.DEFAULT_SOURCE_RUNTIME,
        )
    )
    scored = two_reader._score_rows(
        selected=selected,
        execution_protocol=execution,
        judge_runtime=two_reader.DEFAULT_RUNTIME,
        judge_specs=specs,
        judge_assignments=judge_assignments,
        responses_by_provider=responses,
        reader_assignments=reader_assignments,
        verifier_scores=verifier_scores,
    )
    case_by_id = {case.case_id: case for case in selected}
    case_order = {case.case_id: index for index, case in enumerate(selected)}
    binding_by_provider = {binding.provider: binding for binding in execution.readers}
    public_rows = []
    for row in scored:
        case_id = str(row["case_id"])
        arm = str(row["arm"])
        provider = str(row["reader_provider"])
        reader_request = reader_assignments[(case_id, arm)]
        judge_request = judge_assignments[(provider, case_id, arm)]
        reader_path = base._response_path(
            two_reader.DEFAULT_SOURCE_RUNTIME,
            "reader",
            binding_by_provider[provider],
            reader_request,
        )
        judge_path = base._response_path(
            two_reader.DEFAULT_RUNTIME,
            "judge",
            execution.judge,
            judge_request,
        )
        public_rows.append(
            _sanitize_row(
                row,
                case_order=case_order[case_id],
                anchor_total=case_by_id[case_id].anchor_total,
                reader_record_sha256=base._sha256_file(reader_path),
                judge_record_sha256=base._sha256_file(judge_path),
            )
        )
    return public_rows, selected


def _build_gpt_rows() -> tuple[list[dict[str, object]], list[base.NaturalEndToEndCase]]:
    reader_protocol = gpt.load_protocol()
    judge_protocol = gpt.load_judge_protocol()
    (
        execution,
        cases,
        judge_specs,
        judge_assignments,
        reader_responses,
        reader_assignments,
        verifier_scores,
    ) = gpt._judge_plan(reader_protocol, judge_protocol)
    runtime = ROOT / str(judge_protocol.outputs["runtime"])
    scored = _score_gpt_plan(
        execution,
        cases,
        judge_specs,
        judge_assignments,
        reader_responses,
        reader_assignments,
        verifier_scores,
        runtime=runtime,
    )
    case_by_id = {case.case_id: case for case in cases}
    case_order = {case.case_id: index for index, case in enumerate(cases)}
    reader_binding = execution.readers[0]
    public_rows = []
    for row in scored:
        case_id = str(row["case_id"])
        arm = str(row["arm"])
        reader_request = reader_assignments[(case_id, arm)]
        judge_request = judge_assignments[(case_id, arm)]
        public_rows.append(
            _sanitize_row(
                row,
                case_order=case_order[case_id],
                anchor_total=case_by_id[case_id].anchor_total,
                reader_record_sha256=base._sha256_file(
                    base._response_path(runtime, "reader", reader_binding, reader_request)
                ),
                judge_record_sha256=base._sha256_file(
                    base._response_path(runtime, "judge", execution.judge, judge_request)
                ),
            )
        )
    return public_rows, list(cases)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if value in (True, "True"):
        return 1.0
    if value in (False, "False"):
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("public score must be finite")
    return result


def _source_delta(
    rows: Sequence[Mapping[str, object]],
    *,
    provider: str,
    source: str,
    arm: str,
    reference: str,
    metric: str,
) -> tuple[float, float, float, int, int]:
    selected = [
        row for row in rows if row["reader_provider"] == provider and row["source"] == source
    ]
    by_key = {(str(row["case_token"]), str(row["arm"])): row for row in selected}
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for case_token, row_arm in sorted(by_key):
        if row_arm != arm:
            continue
        row = by_key[(case_token, arm)]
        comparator = by_key.get((case_token, reference))
        if comparator is None:
            continue
        value = _optional_float(row[metric])
        reference_value = _optional_float(comparator[metric])
        if value is None or reference_value is None:
            continue
        grouped[str(row["group_token"])].append(value - reference_value)
    if not grouped:
        raise ValueError("source-specific contrast has no evaluable pairs")
    clusters = [(sum(values), len(values)) for _group, values in sorted(grouped.items())]

    def aggregate(sample: Sequence[tuple[float, int]]) -> float:
        return sum(total for total, _count in sample) / sum(count for _total, count in sample)

    point = aggregate(clusters)
    seed_text = f"{provider}|{source}|{arm}|{reference}|{metric}|source-ci-v1"
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    replicates = []
    for _ in range(10_000):
        sample = [clusters[rng.randrange(len(clusters))] for _cluster in clusters]
        replicates.append(aggregate(sample))
    replicates.sort()
    lower = replicates[round((len(replicates) - 1) * 0.025)]
    upper = replicates[round((len(replicates) - 1) * 0.975)]
    return point, lower, upper, sum(count for _total, count in clusters), len(clusters)


def source_specific_deltas(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    providers = sorted({str(row["reader_provider"]) for row in rows})
    model_by_provider = {str(row["reader_provider"]): str(row["reader_model"]) for row in rows}
    arms_by_provider = {
        provider: {str(row["arm"]) for row in rows if row["reader_provider"] == provider}
        for provider in providers
    }
    for provider in providers:
        for source in ("rhelm", "memops"):
            for arm, reference in COMPARISONS:
                if {arm, reference} - arms_by_provider[provider]:
                    continue
                for metric in PAIRED_METRICS:
                    try:
                        point, lower, upper, pair_count, group_count = _source_delta(
                            rows,
                            provider=provider,
                            source=source,
                            arm=arm,
                            reference=reference,
                            metric=metric,
                        )
                    except ValueError:
                        continue
                    output.append(
                        {
                            "reader_provider": provider,
                            "reader_model": model_by_provider[provider],
                            "source": source,
                            "arm": arm,
                            "reference": reference,
                            "metric": metric,
                            "mean_delta_arm_minus_reference": point,
                            "bootstrap_ci95_lower": lower,
                            "bootstrap_ci95_upper": upper,
                            "paired_case_count": pair_count,
                            "namespace_group_count": group_count,
                            "bootstrap_replicates": 10000,
                        }
                    )
    return output


def population_summary(cases: Sequence[base.NaturalEndToEndCase]) -> list[dict[str, object]]:
    output = []
    for source in ("rhelm", "memops"):
        selected = [case for case in cases if case.source == source]
        anchors = sorted(case.anchor_total for case in selected)
        midpoint = len(anchors) // 2
        median = (
            anchors[midpoint]
            if len(anchors) % 2
            else (anchors[midpoint - 1] + anchors[midpoint]) / 2
        )
        protected_count = sum(case.protected_disclosure_evaluable for case in selected)
        output.append(
            {
                "source": source,
                "case_count": len(selected),
                "namespace_group_count": len({case.group_id for case in selected}),
                "anchor_total_mean": sum(anchors) / len(anchors),
                "anchor_total_median": median,
                "anchor_total_min": min(anchors),
                "anchor_total_max": max(anchors),
                "protected_target_evaluable_count": protected_count,
                "protected_target_evaluable_rate": protected_count / len(selected),
            }
        )
    return output


def _read_public_scores(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {
                key: value for key, value in raw.items() if key not in SCORE_FIELDS
            }
            for field in SCORE_FIELDS:
                row[field] = _optional_float(raw[field])
            row["anchor_total"] = int(raw["anchor_total"])
            row["case_order"] = int(raw["case_order"])
            row["non_answer"] = int(raw["non_answer"])
            rows.append(row)
    return rows


def _legacy_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "reader_provider": row["reader_provider"],
            "reader_model": row["reader_model"],
            "case_id": row["case_token"],
            "source": row["source"],
            "group_id": row["group_token"],
            "arm": row["arm"],
            **{field: row[field] for field in SCORE_FIELDS},
            "over_refusal": row["non_answer"],
        }
        for row in rows
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_rows_close(
    actual: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
) -> None:
    if len(actual) != len(expected):
        raise ValueError("recomputed result row count drifted")
    for actual_row, expected_row in zip(actual, expected, strict=True):
        if tuple(actual_row) != tuple(expected_row):
            raise ValueError("recomputed result columns drifted")
        for key in actual_row:
            left = actual_row[key]
            right = expected_row[key]
            if left is None and right in (None, ""):
                continue
            try:
                left_number = float(left)  # type: ignore[arg-type]
                right_number = float(right)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                if str(left) != str(right):
                    raise ValueError(f"recomputed result drifted: {key}") from None
            else:
                if not math.isclose(left_number, right_number, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"recomputed numeric result drifted: {key}")


def _validate_public_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if len(rows) != 19_799:
        raise ValueError("natural case-audit row count drifted")
    forbidden_columns = {
        "case_id",
        "group_id",
        "query",
        "query_text",
        "answer",
        "expected_answer",
        "prompt",
        "payload",
        "response",
        "reason",
    }
    if forbidden_columns & set(rows[0]):
        raise ValueError("public case audit contains a raw-content column")
    for row in rows:
        for field in (
            "case_token",
            "group_token",
            "reader_record_sha256",
            "judge_record_sha256",
        ):
            if not _is_sha256(row[field]):
                raise ValueError(f"invalid public hash field: {field}")
        if row["source"] not in {"rhelm", "memops"}:
            raise ValueError("unexpected public source")
        if (
            int(row["case_order"]) < 0
            or int(row["anchor_total"]) < 1
            or int(row["non_answer"]) not in {0, 1}
        ):
            raise ValueError("invalid public score value")


def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    manifest = base._read_json(output / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("raw_content_included") is not False:
        raise ValueError("natural case-audit manifest drifted")
    artifacts = base._mapping(manifest.get("artifacts"), "case-audit artifacts")
    for name, expected in artifacts.items():
        path = output / str(name)
        if not path.is_file() or base._sha256_file(path) != expected:
            raise ValueError(f"natural case-audit artifact drifted: {name}")
    rows = _read_public_scores(output / "case_scores.csv")
    _validate_public_rows(rows)
    legacy = _legacy_rows(rows)

    original = [row for row in legacy if row["reader_provider"] in {"DeepSeek", "Gemini"}]
    gpt_rows = [row for row in legacy if row["reader_provider"] == "OpenAI"]
    _assert_rows_close(
        base._aggregate_rows(original),
        _read_csv(ROOT / "results/natural_end_to_end_two_reader_judged/main_table.csv"),
    )
    _assert_rows_close(
        base._paired_rows(original),
        _read_csv(ROOT / "results/natural_end_to_end_two_reader_judged/paired_deltas.csv"),
    )
    _assert_rows_close(
        base._aggregate_rows(gpt_rows),
        _read_csv(ROOT / "results/natural_end_to_end_gpt_luna_judged/main_table.csv"),
    )
    _assert_rows_close(
        gpt._paired_rows(gpt_rows),
        _read_csv(ROOT / "results/natural_end_to_end_gpt_luna_judged/paired_deltas.csv"),
    )
    _assert_rows_close(
        source_specific_deltas(rows),
        _read_csv(output / "source_specific_deltas.csv"),
    )
    return {
        "status": "verified",
        "case_score_rows": len(rows),
        "provider_calls_made": 0,
        "raw_content_read": False,
    }


def _readme() -> str:
    return """# Natural End-to-End Case Audit

This package exposes the finest safe public derivative of the frozen natural
route-to-reader evaluation. `case_scores.csv` contains one row per
reader--case--route assignment. Raw case and namespace identifiers are replaced by
domain-separated SHA-256 tokens. The file contains route scores, parsed reader action,
judge labels, and SHA-256 bindings to the private reader and judge records; it contains
no query, memory, reference-answer, prompt, model-answer, or judge-rationale text.

`source_specific_deltas.csv` reports paired source-specific effects with 10,000
namespace-group bootstrap replicates. `population_summary.csv` reports anchor-count
and protected-target coverage without source text.

Run `python -m scripts.publish_natural_case_audit verify` to reconstruct the checked-in
aggregate tables and paired intervals from `case_scores.csv`. This verification path
does not read private files or call a provider. The legacy aggregate field
`over_refusal` is represented here by the accurate name `non_answer`: it equals one
exactly when the reader action is not `answer`; it is not a gold-action-aware
over-refusal measure.

The original provider records and benchmark payloads are intentionally absent.
RHELM and MemOps source text must be obtained from the pinned upstream owners, and the
current license audit does not authorize redistribution of incorporated benchmark
text. Consequently this package supports result and bootstrap regeneration, plus
hash verification against a separately held raw bundle; it does not independently
replay the historical payload-to-score boundary.
"""


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    two_rows, two_cases = _build_two_reader_rows()
    gpt_rows, gpt_cases = _build_gpt_rows()
    if {case.case_id for case in two_cases} != {case.case_id for case in gpt_cases}:
        raise ValueError("sequential reader case set differs from the primary panel")
    rows = sorted(
        [*two_rows, *gpt_rows],
        key=lambda row: (
            str(row["reader_provider"]),
            int(row["case_order"]),
            str(row["arm"]),
        ),
    )
    _validate_public_rows(rows)
    base._write_csv(output / "case_scores.csv", rows)
    base._write_csv(output / "source_specific_deltas.csv", source_specific_deltas(rows))
    base._write_csv(output / "population_summary.csv", population_summary(two_cases))
    readme_path = output / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(_readme(), encoding="ascii", newline="\n")
    source_inputs = {
        path.relative_to(ROOT).as_posix(): base._sha256_file(path)
        for path in (
            two_reader.DEFAULT_JUDGE_PROTOCOL,
            gpt.DEFAULT_PROTOCOL,
            gpt.DEFAULT_JUDGE_PROTOCOL,
            ROOT / "results/natural_end_to_end_two_reader_judged/manifest.json",
            ROOT / "results/natural_end_to_end_gpt_luna_judged/manifest.json",
        )
    }
    manifest = {
        "schema_version": 1,
        "status": "content_free_case_level_derivative",
        "case_score_rows": len(rows),
        "sample_case_count": len(two_cases),
        "reader_models_reported_separately": True,
        "provider_calls_made": 0,
        "raw_content_included": False,
        "raw_inputs_required_to_rebuild_case_scores": True,
        "public_case_scores_rebuild_aggregate_results": True,
        "legacy_metric_alias": {"over_refusal": "non_answer"},
        "source_inputs": source_inputs,
        "artifacts": {
            name: base._sha256_file(output / name)
            for name in (
                "case_scores.csv",
                "source_specific_deltas.csv",
                "population_summary.csv",
                "README.md",
            )
        },
    }
    base._write_json(output / "manifest.json", manifest)
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish or verify the content-free natural case-level audit."
    )
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output) if args.command == "build" else verify(args.output)
    print(base._canonical_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
