"""Run the frozen zero-call deterministic triage for the two-reader bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_natural_end_to_end_experiment as run  # noqa: E402
from verify_agent_memory.natural_end_to_end import (  # noqa: E402
    ARMS,
    NaturalEndToEndCase,
    deterministic_answer_metrics,
    load_cases,
    matched_route_score,
    route_candidates,
)

DEFAULT_ANALYSIS_PROTOCOL = run.TWO_READER_ANALYSIS_PROTOCOL
DEFAULT_RUNTIME = ROOT / "tmp" / "natural_end_to_end" / "two_reader_runtime"
DEFAULT_OUTPUT = ROOT / "results" / "natural_end_to_end_two_reader_deterministic"

METRICS = (
    "evidence_recall",
    "feasible",
    "penalized_admissibility_upper_risk",
    "route_width",
    "wrong_namespace_exposure",
    "policy_disallowed_exposure",
    "lifecycle_incompatible_exposure",
    "unresolved_exposure",
    "contains_reference_answer",
    "over_refusal",
    "literal_protected_disclosure",
)


def _file_set_sha256(path: Path) -> str:
    entries = "".join(
        f"{item.name}:{run._sha256_file(item)}\n"
        for item in sorted(path.iterdir(), key=lambda item: item.name)
        if item.is_file()
    )
    return hashlib.sha256(entries.encode("utf-8")).hexdigest()


def load_analysis_protocol(path: Path) -> Mapping[str, Any]:
    raw = run._read_json(path)
    expected = {
        "schema_version": 1,
        "analysis_protocol_id": "natural-heldout-two-reader-deterministic-v1",
        "status": "frozen_before_reader_outcome_scoring",
        "provider_calls_authorized": 0,
        "official_benchmark_claim": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"deterministic analysis protocol drifted: {key}")
    metrics = tuple(
        run._sequence(raw.get("deterministic_metrics"), "analysis.deterministic_metrics")
    )
    if metrics != METRICS:
        raise ValueError("deterministic metric list drifted")
    return raw


def _verify_hash(path: Path, expected: object, label: str) -> None:
    if not path.is_file() or run._sha256_file(path) != expected:
        raise ValueError(f"frozen deterministic input drifted: {label}")


def validate_frozen_inputs(
    analysis: Mapping[str, Any],
    *,
    runtime_dir: Path,
) -> tuple[run.Protocol, tuple[NaturalEndToEndCase, ...]]:
    inputs = run._mapping(analysis.get("inputs"), "analysis.inputs")
    execution_protocol_path = ROOT / run._string(
        inputs.get("execution_protocol_path"),
        "analysis.inputs.execution_protocol_path",
    )
    _verify_hash(
        execution_protocol_path,
        inputs.get("execution_protocol_sha256"),
        "execution_protocol",
    )
    protocol = run.load_protocol(execution_protocol_path)
    execution_commit = run._string(
        inputs.get("execution_implementation_commit"),
        "analysis.inputs.execution_implementation_commit",
    )
    cases_path = ROOT / run._string(
        inputs.get("case_bundle_path"),
        "analysis.inputs.case_bundle_path",
    )
    _verify_hash(cases_path, inputs.get("case_bundle_sha256"), "case_bundle")
    materialization_path = ROOT / run._string(
        inputs.get("materialization_manifest_path"),
        "analysis.inputs.materialization_manifest_path",
    )
    _verify_hash(
        materialization_path,
        inputs.get("materialization_manifest_sha256"),
        "materialization_manifest",
    )
    _verify_hash(
        runtime_dir / "checkpoint_import.json",
        inputs.get("checkpoint_import_sha256"),
        "checkpoint_import",
    )
    verifier_completion = run._completion_path(
        runtime_dir,
        "verifier",
        protocol.verifier,
    )
    _verify_hash(
        verifier_completion,
        inputs.get("verifier_completion_sha256"),
        "verifier_completion",
    )
    if run._completed_stage_commit(protocol, runtime_dir, "verifier", protocol.verifier) != (
        execution_commit
    ):
        raise ValueError("verifier execution commit drifted")

    declarations = run._mapping(
        inputs.get("reader_completions"),
        "analysis.inputs.reader_completions",
    )
    for binding in protocol.readers:
        declaration = run._mapping(
            declarations.get(binding.provider),
            f"analysis.inputs.reader_completions.{binding.provider}",
        )
        completion = run._completion_path(runtime_dir, "reader", binding)
        _verify_hash(completion, declaration.get("sha256"), f"{binding.provider}_completion")
        if run._completed_stage_commit(protocol, runtime_dir, "reader", binding) != (
            execution_commit
        ):
            raise ValueError(f"{binding.provider} execution commit drifted")
        receipt = run._read_json(completion)
        if receipt.get("request_count") != declaration.get("request_count"):
            raise ValueError(f"{binding.provider} request count drifted")
        response_dir = run._stage_root(runtime_dir, "reader", binding) / "responses"
        if len(tuple(response_dir.glob("*.json"))) != declaration.get("request_count"):
            raise ValueError(f"{binding.provider} response file count drifted")
        failure_dir = run._stage_root(runtime_dir, "reader", binding) / "failures"
        if _file_set_sha256(failure_dir) != declaration.get("failure_file_set_sha256"):
            raise ValueError(f"{binding.provider} failure file set drifted")
        failure_rows = [run._read_json(path) for path in failure_dir.glob("*.json")]
        if len(failure_rows) != declaration.get("failure_count"):
            raise ValueError(f"{binding.provider} failure count drifted")
        terminal = sum(row.get("retry_eligible") is False for row in failure_rows)
        if terminal != declaration.get("terminal_failure_count"):
            raise ValueError(f"{binding.provider} terminal failure count drifted")

    cases = load_cases(cases_path)
    if len(cases) != 3767:
        raise ValueError("deterministic case count drifted")
    return protocol, cases


def _score_rows(
    cases: Sequence[NaturalEndToEndCase],
    protocol: run.Protocol,
    runtime_dir: Path,
) -> list[dict[str, object]]:
    reader_specs, assignments, verifier_scores = run._reader_plan(
        cases,
        protocol,
        runtime_dir,
    )
    rows = []
    for binding in protocol.readers:
        responses = run._load_reader_responses(
            protocol=protocol,
            runtime=runtime_dir,
            binding=binding,
            specs=reader_specs,
        )
        for case in cases:
            for arm in ARMS:
                candidates = route_candidates(
                    case,
                    arm,
                    verifier_scores=verifier_scores[case.case_id],
                    verifier_threshold=protocol.verifier_threshold,
                )
                route_score = matched_route_score(
                    case,
                    candidates,
                    target_recall=protocol.target_recall,
                )
                response = responses[assignments[(case.case_id, arm)]]
                answer = deterministic_answer_metrics(case, response)
                rows.append(
                    {
                        "reader_provider": binding.provider,
                        "reader_model": binding.model,
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
                        "policy_disallowed_exposure": (route_score.policy_disallowed_exposure_rate),
                        "lifecycle_incompatible_exposure": (
                            route_score.lifecycle_incompatible_exposure_rate
                        ),
                        "unresolved_exposure": route_score.unresolved_exposure_rate,
                        "contains_reference_answer": int(bool(answer["contains_reference_answer"])),
                        "over_refusal": int(bool(answer["over_refusal"])),
                        "protected_disclosure_evaluable": int(case.protected_disclosure_evaluable),
                        "literal_protected_disclosure": answer["literal_protected_disclosure"],
                    }
                )
    return rows


def _mean(rows: Sequence[Mapping[str, object]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row[metric] is not None]
    return sum(values) / len(values) if values else None


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["reader_provider"]),
                str(row["reader_model"]),
                str(row["arm"]),
                str(row["source"]),
            )
        ].append(row)
    output = []
    for (provider, model, arm, source), selected in sorted(grouped.items()):
        output.append(
            {
                "reader_provider": provider,
                "reader_model": model,
                "arm": arm,
                "source": source,
                "query_count": len(selected),
                "protected_disclosure_coverage": sum(
                    int(row["protected_disclosure_evaluable"]) for row in selected
                )
                / len(selected),
                **{metric: _mean(selected, metric) for metric in METRICS},
            }
        )

    by_reader_arm: defaultdict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in output:
        by_reader_arm[
            (str(row["reader_provider"]), str(row["reader_model"]), str(row["arm"]))
        ].append(row)
    for (provider, model, arm), sources in sorted(by_reader_arm.items()):
        if {str(row["source"]) for row in sources} != {"rhelm", "memops"}:
            raise RuntimeError("deterministic source macro requires both sources")
        output.append(
            {
                "reader_provider": provider,
                "reader_model": model,
                "arm": arm,
                "source": "equal_source_macro",
                "query_count": sum(int(row["query_count"]) for row in sources),
                "protected_disclosure_coverage": sum(
                    int(row["query_count"]) * float(row["protected_disclosure_coverage"])
                    for row in sources
                )
                / sum(int(row["query_count"]) for row in sources),
                **{
                    metric: (
                        sum(float(row[metric]) for row in sources) / len(sources)
                        if all(row[metric] is not None for row in sources)
                        else None
                    )
                    for metric in METRICS
                },
            }
        )
    return output


def _paired_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["reader_provider"]), str(row["case_id"]), str(row["arm"])): row for row in rows
    }
    providers = sorted({str(row["reader_provider"]) for row in rows})
    models = {str(row["reader_provider"]): str(row["reader_model"]) for row in rows}
    comparisons = tuple((arm, "global_dense") for arm in ARMS if arm != "global_dense")
    output = []
    for provider in providers:
        for arm, reference in comparisons:
            for metric in METRICS:
                point, lower, upper, query_count, group_count = run._paired_delta(
                    by_key,
                    provider=provider,
                    arm=arm,
                    reference=reference,
                    metric=metric,
                )
                output.append(
                    {
                        "reader_provider": provider,
                        "reader_model": models[provider],
                        "arm": arm,
                        "reference": reference,
                        "metric": metric,
                        "mean_delta_arm_minus_reference": point,
                        "bootstrap_ci95_lower": lower,
                        "bootstrap_ci95_upper": upper,
                        "paired_query_count": query_count,
                        "namespace_group_count": group_count,
                        "bootstrap_replicates": 10000,
                    }
                )
    return output


def _evaluate_gate(
    analysis: Mapping[str, Any],
    paired: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    comparison = run._mapping(analysis.get("comparison"), "analysis.comparison")
    gate = run._mapping(
        analysis.get("judge_continuation_gate"),
        "analysis.judge_continuation_gate",
    )
    primary = tuple(
        str(value)
        for value in run._sequence(
            comparison.get("primary_constrained_arms"),
            "analysis.comparison.primary_constrained_arms",
        )
    )
    points = {
        (str(row["reader_provider"]), str(row["arm"]), str(row["metric"])): float(
            row["mean_delta_arm_minus_reference"]
        )
        for row in paired
    }
    providers = sorted({str(row["reader_provider"]) for row in paired})
    route_gate = run._mapping(gate.get("route_gate"), "analysis.gate.route_gate")
    route_details = {}
    for arm in primary:
        values = [
            (
                points[(provider, arm, "wrong_namespace_exposure")],
                points[(provider, arm, "penalized_admissibility_upper_risk")],
            )
            for provider in providers
        ]
        if len(set(values)) != 1:
            raise RuntimeError("route-only deterministic deltas differ across readers")
        wrong_scope, risk = values[0]
        route_details[arm] = {
            "wrong_namespace_exposure_delta": wrong_scope,
            "penalized_admissibility_upper_risk_delta": risk,
            "passed": wrong_scope
            < float(route_gate["wrong_namespace_exposure_delta_strictly_below"])
            and risk < float(route_gate["penalized_admissibility_upper_risk_delta_strictly_below"]),
        }
    route_passed = any(bool(row["passed"]) for row in route_details.values())

    reader_gate = run._mapping(gate.get("per_reader_gate"), "analysis.gate.per_reader_gate")
    reader_details = {}
    for provider in providers:
        arms = {}
        for arm in primary:
            utility = points[(provider, arm, "contains_reference_answer")]
            over_refusal = points[(provider, arm, "over_refusal")]
            arms[arm] = {
                "contains_reference_answer_delta": utility,
                "over_refusal_delta": over_refusal,
                "passed": utility >= float(reader_gate["contains_reference_answer_delta_at_least"])
                and over_refusal <= float(reader_gate["over_refusal_delta_at_most"]),
            }
        reader_details[provider] = {
            "arms": arms,
            "passed": any(bool(row["passed"]) for row in arms.values()),
        }
    readers_passed = all(bool(row["passed"]) for row in reader_details.values())
    passed = route_passed and readers_passed
    return {
        "passed": passed,
        "route_gate_passed": route_passed,
        "reader_gate_passed": readers_passed,
        "route_details": route_details,
        "reader_details": reader_details,
    }


def _summary_text(
    aggregates: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> str:
    macro = [row for row in aggregates if row["source"] == "equal_source_macro"]
    lines = [
        "# Two-reader deterministic triage",
        "",
        "This zero-call analysis reports Gemini and DeepSeek separately. It uses literal ",
        "reference containment as a deterministic utility proxy; it is not semantic answer ",
        "accuracy and is not an official RHELM or MemOps result.",
        "",
        "| reader | arm | ref contains | over-refusal | recall | risk | wrong namespace |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(macro, key=lambda item: (str(item["reader_provider"]), str(item["arm"]))):
        lines.append(
            f"| {row['reader_provider']} | {row['arm']} | "
            f"{float(row['contains_reference_answer']):.4f} | "
            f"{float(row['over_refusal']):.4f} | "
            f"{float(row['evidence_recall']):.4f} | "
            f"{float(row['penalized_admissibility_upper_risk']):.4f} | "
            f"{float(row['wrong_namespace_exposure']):.4f} |"
        )
    lines.extend(
        [
            "",
            f"Deterministic continuation gate: **{'PASS' if gate['passed'] else 'STOP'}**.",
            "",
            "The gate was frozen before outcome scoring. Passing only authorizes blinded semantic ",
            "judging; it is not a paper claim and does not establish answer utility.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    *,
    analysis_protocol_path: Path,
    runtime_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError("deterministic output directory already exists")
    gate_path = runtime_dir / "deterministic_gate.json"
    if gate_path.exists():
        raise RuntimeError("deterministic gate already exists")
    analysis = load_analysis_protocol(analysis_protocol_path)
    protocol, cases = validate_frozen_inputs(analysis, runtime_dir=runtime_dir)
    rows = _score_rows(cases, protocol, runtime_dir)
    aggregates = _aggregate_rows(rows)
    paired = _paired_rows(rows)
    gate = _evaluate_gate(analysis, paired)
    run._write_csv(output_dir / "main_table.csv", aggregates)
    run._write_csv(output_dir / "paired_deltas.csv", paired)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(
        _summary_text(aggregates, gate),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete_zero_call_deterministic_triage",
        "analysis_protocol_id": analysis["analysis_protocol_id"],
        "analysis_protocol_sha256": run._sha256_file(analysis_protocol_path),
        "analysis_implementation_commit": run._git_head(),
        "execution_protocol_sha256": protocol.protocol_sha256,
        "execution_implementation_commit": analysis["inputs"]["execution_implementation_commit"],
        "case_count": len(cases),
        "row_count": len(rows),
        "reader_estimates_pooled": False,
        "provider_calls_made": 0,
        "gate": gate,
        "artifacts": {
            name: run._sha256_file(output_dir / name)
            for name in ("main_table.csv", "paired_deltas.csv", "summary.md")
        },
        "official_benchmark_result": False,
    }
    run._write_json(output_dir / "manifest.json", manifest)
    gate_receipt = {
        "schema_version": 1,
        "protocol_sha256": protocol.protocol_sha256,
        "analysis_protocol_id": analysis["analysis_protocol_id"],
        "analysis_protocol_sha256": run._sha256_file(analysis_protocol_path),
        "analysis_implementation_commit": run._git_head(),
        "execution_implementation_commit": analysis["inputs"]["execution_implementation_commit"],
        "status": ("deterministic_gate_passed" if gate["passed"] else "deterministic_gate_failed"),
        "deterministic_manifest_sha256": run._sha256_file(output_dir / "manifest.json"),
        "provider_calls_made": 0,
    }
    run._write_json(gate_path, gate_receipt)
    return {**gate_receipt, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-protocol", type=Path, default=DEFAULT_ANALYSIS_PROTOCOL)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(
                analysis_protocol_path=args.analysis_protocol,
                runtime_dir=args.runtime,
                output_dir=args.output,
            ),
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
