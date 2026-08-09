"""Run the frozen cost-aware semantic judge over paired natural cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_natural_end_to_end_experiment as base  # noqa: E402

DEFAULT_JUDGE_PROTOCOL = ROOT / "experiments" / "natural_end_to_end_two_reader_judge_protocol.json"
DEFAULT_EXECUTION_PROTOCOL = ROOT / "experiments" / "natural_end_to_end_two_reader_protocol.json"
DEFAULT_CASES = ROOT / "tmp" / "natural_end_to_end" / "cases.jsonl.gz"
DEFAULT_SOURCE_RUNTIME = ROOT / "tmp" / "natural_end_to_end" / "two_reader_runtime"
DEFAULT_RUNTIME = ROOT / "tmp" / "natural_end_to_end" / "two_reader_judge_runtime"
DEFAULT_OUTPUT = ROOT / "results" / "natural_end_to_end_two_reader_judged"
DEFAULT_DOTENV = ROOT.parent / "bomi-codex-starter" / ".env"
CONTRACT_PATHS = (
    *base.CONTRACT_PATHS,
    "experiments/natural_end_to_end_two_reader_judge_protocol.json",
    "scripts/run_natural_two_reader_judge.py",
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


@dataclass(frozen=True)
class JudgeProtocol:
    raw: dict[str, Any]
    path: Path
    sha256: str
    source: dict[str, Any]
    gate: dict[str, Any]
    sampling: dict[str, Any]
    judge: dict[str, Any]
    budget: dict[str, Any]
    outputs: dict[str, Any]

    @property
    def fixture_cap_usd(self) -> float:
        return _number(self.budget["fixture_hard_cap_usd"], "fixture_hard_cap_usd")

    @property
    def plan_cap_usd(self) -> float:
        return _number(
            self.budget["planned_calls_conservative_cap_usd"],
            "planned_calls_conservative_cap_usd",
        )

    @property
    def total_cap_usd(self) -> float:
        return _number(
            self.budget["total_incremental_hard_cap_usd"],
            "total_incremental_hard_cap_usd",
        )


def load_judge_protocol(path: Path) -> JudgeProtocol:
    raw = _mapping(base._read_json(path), "judge protocol")
    expected_top = {
        "schema_version",
        "protocol_id",
        "status",
        "frozen_date",
        "objective",
        "source_execution",
        "continuation_gate",
        "sampling",
        "judge",
        "budget",
        "uncertainty",
        "outputs",
        "official_rhelm_or_memops_claim",
    }
    if set(raw) != expected_top:
        raise ValueError("judge protocol has missing or unknown top-level fields")
    if raw["schema_version"] != 1:
        raise ValueError("judge protocol schema drifted")
    if raw["protocol_id"] != "natural-heldout-two-reader-semantic-judge-v1":
        raise ValueError("judge protocol identity drifted")
    if raw["status"] != "frozen_before_any_semantic_judge_outcome":
        raise ValueError("judge protocol is not frozen before semantic scoring")
    if raw["official_rhelm_or_memops_claim"] is not False:
        raise ValueError("judge protocol cannot authorize an official benchmark claim")
    source = _mapping(raw["source_execution"], "source_execution")
    gate = _mapping(raw["continuation_gate"], "continuation_gate")
    sampling = _mapping(raw["sampling"], "sampling")
    judge = _mapping(raw["judge"], "judge")
    budget = _mapping(raw["budget"], "budget")
    outputs = _mapping(raw["outputs"], "outputs")
    if sampling.get("outcome_or_route_dependent_selection") is not False:
        raise ValueError("semantic-judge sampling must be outcome independent")
    if sampling.get("selection_fields") != ["source", "case_id"]:
        raise ValueError("semantic-judge selection fields drifted")
    if sampling.get("arms") != list(base.ARMS):
        raise ValueError("semantic-judge arm panel drifted")
    if sampling.get("reader_providers") != ["Gemini", "DeepSeek"]:
        raise ValueError("semantic-judge reader panel drifted")
    if judge.get("provider") != "Anthropic" or judge.get("model") != "claude-haiku-4-5":
        raise ValueError("semantic judge binding drifted")
    if judge.get("arm_and_reader_blinded") is not True:
        raise ValueError("semantic judge must remain arm- and reader-blind")
    if judge.get("semantic_or_output_repair") is not False:
        raise ValueError("semantic judge output repair is forbidden")
    if judge.get("selective_rerun") is not False:
        raise ValueError("semantic judge selective reruns are forbidden")
    protocol = JudgeProtocol(
        raw=raw,
        path=path,
        sha256=base._sha256_file(path),
        source=source,
        gate=gate,
        sampling=sampling,
        judge=judge,
        budget=budget,
        outputs=outputs,
    )
    if protocol.fixture_cap_usd != 0.01:
        raise ValueError("judge fixture cap drifted")
    if protocol.plan_cap_usd != 45.0 or protocol.total_cap_usd != 60.0:
        raise ValueError("semantic judge budget drifted")
    return protocol


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_contract(protocol: JudgeProtocol) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *CONTRACT_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("judge execution requires committed, clean contract paths")
    ancestor = _string(
        protocol.gate["analysis_implementation_commit"],
        "analysis_implementation_commit",
    )
    descendant = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if descendant.returncode != 0:
        raise RuntimeError("judge implementation must descend from the passed analysis commit")
    return _git_head()


def _assert_file(path: Path, expected_sha256: object, label: str) -> None:
    expected = _string(expected_sha256, f"{label}.sha256")
    if not path.is_file() or base._sha256_file(path) != expected:
        raise ValueError(f"{label} hash drifted")


def _validate_source(
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    *,
    cases_path: Path,
    source_runtime: Path,
) -> dict[str, object]:
    source = judge_protocol.source
    execution_path = ROOT / _string(source["protocol_path"], "source protocol path")
    _assert_file(execution_path, source["protocol_sha256"], "source execution protocol")
    if execution_protocol.protocol_sha256 != source["protocol_sha256"]:
        raise ValueError("loaded execution protocol differs from judge source")
    _assert_file(cases_path, source["case_bundle_sha256"], "case bundle")

    gate = judge_protocol.gate
    analysis_path = ROOT / _string(gate["analysis_protocol_path"], "analysis protocol path")
    manifest_path = ROOT / _string(
        gate["deterministic_manifest_path"], "deterministic manifest path"
    )
    runtime_gate_path = ROOT / _string(gate["runtime_gate_path"], "runtime gate path")
    _assert_file(analysis_path, gate["analysis_protocol_sha256"], "analysis protocol")
    _assert_file(
        manifest_path,
        gate["deterministic_manifest_sha256"],
        "deterministic manifest",
    )
    _assert_file(runtime_gate_path, gate["runtime_gate_sha256"], "deterministic gate")
    gate_receipt = _mapping(base._read_json(runtime_gate_path), "deterministic gate")
    expected_gate = {
        "status": gate["required_status"],
        "provider_calls_made": gate["provider_calls_in_gate"],
        "analysis_implementation_commit": gate["analysis_implementation_commit"],
        "analysis_protocol_sha256": gate["analysis_protocol_sha256"],
        "deterministic_manifest_sha256": gate["deterministic_manifest_sha256"],
        "protocol_sha256": source["protocol_sha256"],
    }
    for key, value in expected_gate.items():
        if gate_receipt.get(key) != value:
            raise ValueError(f"deterministic continuation gate drifted: {key}")

    completion_hashes = _mapping(source["reader_completions"], "reader completions")
    for binding in execution_protocol.readers:
        path = base._completion_path(source_runtime, "reader", binding)
        _assert_file(path, completion_hashes[binding.provider], f"{binding.provider} completion")
        completion = _mapping(base._read_json(path), f"{binding.provider} completion")
        if completion.get("complete_bundle") is not True or completion.get("request_count") != 7250:
            raise ValueError(f"{binding.provider} reader bundle is incomplete")
    if execution_protocol.judge.provider != judge_protocol.judge["provider"]:
        raise ValueError("source judge provider drifted")
    if execution_protocol.judge.model != judge_protocol.judge["model"]:
        raise ValueError("source judge model drifted")
    if (
        execution_protocol.maximum_output_tokens_judge
        != judge_protocol.judge["maximum_output_tokens"]
    ):
        raise ValueError("source judge output-token contract drifted")
    return {
        "source_protocol_sha256": execution_protocol.protocol_sha256,
        "deterministic_gate_sha256": gate["runtime_gate_sha256"],
        "reader_completion_count": len(execution_protocol.readers),
    }


def select_cases(
    cases: list[base.NaturalEndToEndCase],
    protocol: JudgeProtocol,
) -> list[base.NaturalEndToEndCase]:
    rhelm = [case for case in cases if case.source == "rhelm"]
    memops = [case for case in cases if case.source == "memops"]
    rhelm_contract = _mapping(protocol.sampling["rhelm"], "sampling.rhelm")
    memops_contract = _mapping(protocol.sampling["memops"], "sampling.memops")
    if len(rhelm) != _integer(rhelm_contract["population_count"], "rhelm population"):
        raise ValueError("RHELM population count drifted")
    if len(memops) != _integer(memops_contract["population_count"], "MemOps population"):
        raise ValueError("MemOps population count drifted")
    salt = _string(memops_contract["salt"], "MemOps sampling salt")
    memops.sort(
        key=lambda case: (
            hashlib.sha256(f"{salt}{case.case_id}".encode()).hexdigest(),
            case.case_id,
        )
    )
    selected_memops = memops[: _integer(memops_contract["sample_count"], "MemOps sample")]
    selected = sorted([*rhelm, *selected_memops], key=lambda case: case.case_id)
    case_id_hash = base._sha256_object([case.case_id for case in selected])
    if case_id_hash != protocol.sampling["sample_case_id_set_sha256"]:
        raise ValueError("semantic-judge sample identity drifted")
    if len(selected) != _integer(protocol.sampling["sample_case_count"], "sample count"):
        raise ValueError("semantic-judge sample count drifted")
    group_count = len({case.group_id for case in selected_memops})
    if group_count != _integer(
        memops_contract["sample_namespace_group_count"], "sample namespace groups"
    ):
        raise ValueError("MemOps sample namespace coverage drifted")
    return selected


def _build_plan(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    cases_path: Path,
    source_runtime: Path,
) -> tuple[
    list[base.NaturalEndToEndCase],
    dict[str, base.CallSpec],
    dict[tuple[str, str, str], str],
    dict[str, dict[str, base.ReaderResponse]],
    dict[tuple[str, str], str],
    dict[str, dict[str, float]],
]:
    cases = base.load_cases(cases_path)
    selected = select_cases(cases, judge_protocol)
    specs, assignments, responses, reader_assignments, verifier_scores = base._judge_plan(
        cases,
        execution_protocol,
        source_runtime,
    )
    selected_ids = {case.case_id for case in selected}
    selected_assignments = {
        key: request_id
        for key, request_id in assignments.items()
        if key[1] in selected_ids and key[2] in base.ARMS
    }
    expected_assignment_count = len(selected) * len(base.ARMS) * len(execution_protocol.readers)
    if len(selected_assignments) != expected_assignment_count:
        raise RuntimeError("semantic-judge pairing is incomplete")
    request_ids = set(selected_assignments.values())
    selected_specs = {request_id: specs[request_id] for request_id in request_ids}
    if len(selected_specs) != judge_protocol.judge["expected_unique_request_count"]:
        raise ValueError("semantic-judge unique request count drifted")
    if (
        base._sha256_object(sorted(request_ids))
        != judge_protocol.judge["expected_request_id_set_sha256"]
    ):
        raise ValueError("semantic-judge request identity drifted")
    return (
        selected,
        selected_specs,
        selected_assignments,
        responses,
        reader_assignments,
        verifier_scores,
    )


def _plan_path(runtime: Path) -> Path:
    return runtime / "judge_plan.json"


def _fixture_attestation_path(runtime: Path) -> Path:
    return runtime / "judge_fixture_attestation.json"


def _sample_completion_path(runtime: Path) -> Path:
    return runtime / "sample_completion.json"


def write_plan(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    cases_path: Path,
    source_runtime: Path,
    runtime: Path,
) -> dict[str, object]:
    _validate_source(
        judge_protocol,
        execution_protocol,
        cases_path=cases_path,
        source_runtime=source_runtime,
    )
    selected, specs, *_ = _build_plan(
        judge_protocol=judge_protocol,
        execution_protocol=execution_protocol,
        cases_path=cases_path,
        source_runtime=source_runtime,
    )
    stats = base._plan_stats(
        specs,
        execution_protocol.judge,
        system_prompt=execution_protocol.judge_prompt,
        maximum_output_tokens=execution_protocol.maximum_output_tokens_judge,
    )
    if float(stats["conservative_cost_bound_usd"]) > judge_protocol.plan_cap_usd:
        raise RuntimeError("semantic-judge plan exceeds its frozen conservative cap")
    receipt = {
        "schema_version": 1,
        "judge_protocol_sha256": judge_protocol.sha256,
        "execution_protocol_sha256": execution_protocol.protocol_sha256,
        "implementation_commit": _git_head(),
        "provider_calls_made": 0,
        "sample_case_count": len(selected),
        "sample_case_id_set_sha256": judge_protocol.sampling["sample_case_id_set_sha256"],
        "assignment_count": len(selected) * len(base.ARMS) * len(execution_protocol.readers),
        "request_id_set_sha256": judge_protocol.judge["expected_request_id_set_sha256"],
        **stats,
    }
    base._write_json(_plan_path(runtime), receipt)
    return receipt


def _require_plan(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    cases_path: Path,
    source_runtime: Path,
    runtime: Path,
    implementation_commit: str,
) -> tuple[
    dict[str, object],
    list[base.NaturalEndToEndCase],
    dict[str, base.CallSpec],
    dict[tuple[str, str, str], str],
    dict[str, dict[str, base.ReaderResponse]],
    dict[tuple[str, str], str],
    dict[str, dict[str, float]],
]:
    path = _plan_path(runtime)
    if not path.is_file():
        raise RuntimeError("run plan before any semantic-judge provider call")
    receipt = _mapping(base._read_json(path), "judge plan receipt")
    expected = {
        "judge_protocol_sha256": judge_protocol.sha256,
        "execution_protocol_sha256": execution_protocol.protocol_sha256,
        "implementation_commit": implementation_commit,
        "provider_calls_made": 0,
        "sample_case_count": judge_protocol.sampling["sample_case_count"],
        "sample_case_id_set_sha256": judge_protocol.sampling["sample_case_id_set_sha256"],
        "request_id_set_sha256": judge_protocol.judge["expected_request_id_set_sha256"],
        "request_count": judge_protocol.judge["expected_unique_request_count"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"semantic-judge plan receipt drifted: {key}")
    if _number(receipt["conservative_cost_bound_usd"], "plan bound") > (
        judge_protocol.plan_cap_usd
    ):
        raise ValueError("semantic-judge plan receipt exceeds the frozen cap")
    built = _build_plan(
        judge_protocol=judge_protocol,
        execution_protocol=execution_protocol,
        cases_path=cases_path,
        source_runtime=source_runtime,
    )
    return receipt, *built


def _fixture_spec(execution_protocol: base.Protocol) -> base.CallSpec:
    case = base._fixture_case()
    return base.CallSpec(
        request_id=case.case_id,
        payload=base.judge_payload(
            case,
            base.ReaderResponse(action="answer", answer="123 Market Street"),
        ),
        schema=base.judge_response_schema(),
    )


def run_fixture(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    cases_path: Path,
    source_runtime: Path,
    runtime: Path,
    dotenv: Path,
) -> dict[str, object]:
    implementation_commit = _require_clean_contract(judge_protocol)
    plan, *_ = _require_plan(
        judge_protocol=judge_protocol,
        execution_protocol=execution_protocol,
        cases_path=cases_path,
        source_runtime=source_runtime,
        runtime=runtime,
        implementation_commit=implementation_commit,
    )
    bound = base._conservative_call_cost(
        _fixture_spec(execution_protocol),
        execution_protocol.judge,
        system_prompt=execution_protocol.judge_prompt,
        maximum_output_tokens=execution_protocol.maximum_output_tokens_judge,
    )
    if bound > judge_protocol.fixture_cap_usd:
        raise RuntimeError("semantic-judge fixture exceeds its frozen cap")
    receipt = base.run_fixture(
        protocol=execution_protocol,
        runtime=runtime,
        stage="judge",
        binding=execution_protocol.judge,
        dotenv=dotenv,
    )
    fixture_path = base._stage_root(runtime, "fixtures", execution_protocol.judge) / "judge.json"
    cost = base._record_cost(receipt)
    if cost > judge_protocol.fixture_cap_usd:
        raise RuntimeError("semantic-judge fixture exceeded its frozen cap")
    attestation = {
        "schema_version": 1,
        "judge_protocol_sha256": judge_protocol.sha256,
        "execution_protocol_sha256": execution_protocol.protocol_sha256,
        "implementation_commit": implementation_commit,
        "plan_sha256": base._sha256_file(_plan_path(runtime)),
        "fixture_sha256": base._sha256_file(fixture_path),
        "fixture_conservative_bound_usd": bound,
        "fixture_cost_usd": cost,
        "fixture_hard_cap_usd": judge_protocol.fixture_cap_usd,
        "planned_calls_conservative_bound_usd": plan["conservative_cost_bound_usd"],
    }
    base._write_json(_fixture_attestation_path(runtime), attestation)
    return attestation


def _require_fixture(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    runtime: Path,
    implementation_commit: str,
) -> dict[str, object]:
    attestation_path = _fixture_attestation_path(runtime)
    if not attestation_path.is_file():
        raise RuntimeError("run the capped synthetic judge fixture before execution")
    attestation = _mapping(base._read_json(attestation_path), "judge fixture attestation")
    fixture_path = base._stage_root(runtime, "fixtures", execution_protocol.judge) / "judge.json"
    expected = {
        "judge_protocol_sha256": judge_protocol.sha256,
        "execution_protocol_sha256": execution_protocol.protocol_sha256,
        "implementation_commit": implementation_commit,
        "plan_sha256": base._sha256_file(_plan_path(runtime)),
        "fixture_sha256": base._sha256_file(fixture_path),
        "fixture_hard_cap_usd": judge_protocol.fixture_cap_usd,
    }
    for key, value in expected.items():
        if attestation.get(key) != value:
            raise ValueError(f"semantic-judge fixture attestation drifted: {key}")
    if _number(attestation["fixture_cost_usd"], "fixture cost") > (judge_protocol.fixture_cap_usd):
        raise ValueError("semantic-judge fixture cost exceeds its cap")
    return attestation


def _failure_cost(runtime: Path, execution_protocol: base.Protocol) -> float:
    root = base._stage_root(runtime, "judge", execution_protocol.judge) / "failures"
    total = 0.0
    for path in root.glob("*.json") if root.is_dir() else ():
        failure = _mapping(base._read_json(path), "judge failure")
        if (
            failure.get("protocol_sha256") == execution_protocol.protocol_sha256
            and failure.get("provider") == execution_protocol.judge.provider
            and failure.get("model") == execution_protocol.judge.model
        ):
            total += _number(failure.get("cost_bound_usd", 0), "failure cost")
    return total


def execute(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    cases_path: Path,
    source_runtime: Path,
    runtime: Path,
    dotenv: Path,
    workers: int,
) -> dict[str, object]:
    implementation_commit = _require_clean_contract(judge_protocol)
    plan, _cases, specs, *_ = _require_plan(
        judge_protocol=judge_protocol,
        execution_protocol=execution_protocol,
        cases_path=cases_path,
        source_runtime=source_runtime,
        runtime=runtime,
        implementation_commit=implementation_commit,
    )
    fixture = _require_fixture(
        judge_protocol=judge_protocol,
        execution_protocol=execution_protocol,
        runtime=runtime,
        implementation_commit=implementation_commit,
    )
    fixture_cost = _number(fixture["fixture_cost_usd"], "fixture cost")
    response_cap = judge_protocol.total_cap_usd - fixture_cost
    completion = base._execute_specs(
        protocol=execution_protocol,
        runtime=runtime,
        stage="judge",
        binding=execution_protocol.judge,
        specs=specs,
        system_prompt=execution_protocol.judge_prompt,
        maximum_output_tokens=execution_protocol.maximum_output_tokens_judge,
        dotenv=dotenv,
        workers=workers,
        parser=base._judge_parser,
        incremental_cost_cap_usd=response_cap,
    )
    failure_cost = _failure_cost(runtime, execution_protocol)
    total_cost = fixture_cost + float(completion["cost_usd"]) + failure_cost
    if total_cost > judge_protocol.total_cap_usd:
        raise RuntimeError("semantic-judge total cost exceeded its hard cap")
    completion_path = base._completion_path(runtime, "judge", execution_protocol.judge)
    sample_completion = {
        "schema_version": 1,
        "judge_protocol_sha256": judge_protocol.sha256,
        "execution_protocol_sha256": execution_protocol.protocol_sha256,
        "implementation_commit": implementation_commit,
        "plan_sha256": base._sha256_file(_plan_path(runtime)),
        "fixture_attestation_sha256": base._sha256_file(_fixture_attestation_path(runtime)),
        "provider_completion_sha256": base._sha256_file(completion_path),
        "sample_case_count": judge_protocol.sampling["sample_case_count"],
        "request_count": completion["request_count"],
        "request_id_set_sha256": judge_protocol.judge["expected_request_id_set_sha256"],
        "fixture_cost_usd": fixture_cost,
        "response_cost_usd": completion["cost_usd"],
        "conservative_failure_cost_usd": failure_cost,
        "total_budget_consumption_usd": total_cost,
        "total_incremental_hard_cap_usd": judge_protocol.total_cap_usd,
        "complete_bundle": True,
        "planned_calls_conservative_bound_usd": plan["conservative_cost_bound_usd"],
    }
    base._write_json(_sample_completion_path(runtime), sample_completion)
    return sample_completion


def _require_sample_completion(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    runtime: Path,
) -> dict[str, object]:
    path = _sample_completion_path(runtime)
    if not path.is_file():
        raise RuntimeError("semantic-judge sample is not complete")
    receipt = _mapping(base._read_json(path), "sample completion")
    completion_path = base._completion_path(runtime, "judge", execution_protocol.judge)
    expected = {
        "judge_protocol_sha256": judge_protocol.sha256,
        "execution_protocol_sha256": execution_protocol.protocol_sha256,
        "provider_completion_sha256": base._sha256_file(completion_path),
        "sample_case_count": judge_protocol.sampling["sample_case_count"],
        "request_count": judge_protocol.judge["expected_unique_request_count"],
        "request_id_set_sha256": judge_protocol.judge["expected_request_id_set_sha256"],
        "total_incremental_hard_cap_usd": judge_protocol.total_cap_usd,
        "complete_bundle": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"semantic-judge completion drifted: {key}")
    if _number(receipt["total_budget_consumption_usd"], "total cost") > (
        judge_protocol.total_cap_usd
    ):
        raise ValueError("semantic-judge completion exceeds its hard cap")
    return receipt


def _score_rows(
    *,
    selected: list[base.NaturalEndToEndCase],
    execution_protocol: base.Protocol,
    judge_runtime: Path,
    judge_specs: dict[str, base.CallSpec],
    judge_assignments: dict[tuple[str, str, str], str],
    responses_by_provider: dict[str, dict[str, base.ReaderResponse]],
    reader_assignments: dict[tuple[str, str], str],
    verifier_scores: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    judge_responses = base._load_judge_responses(
        protocol=execution_protocol,
        runtime=judge_runtime,
        specs=judge_specs,
    )
    model_by_provider = {binding.provider: binding.model for binding in execution_protocol.readers}
    rows = []
    for provider in sorted(responses_by_provider):
        reader_responses = responses_by_provider[provider]
        for case in selected:
            for arm in base.ARMS:
                candidates = base.route_candidates(
                    case,
                    arm,
                    verifier_scores=verifier_scores[case.case_id],
                    verifier_threshold=execution_protocol.verifier_threshold,
                )
                route_score = base.matched_route_score(
                    case,
                    candidates,
                    target_recall=execution_protocol.target_recall,
                )
                response = reader_responses[reader_assignments[(case.case_id, arm)]]
                deterministic = base.deterministic_answer_metrics(case, response)
                judge = judge_responses[judge_assignments[(provider, case.case_id, arm)]]
                rows.append(
                    {
                        "reader_provider": provider,
                        "reader_model": model_by_provider[provider],
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
                        "literal_protected_disclosure": deterministic[
                            "literal_protected_disclosure"
                        ],
                        "stale_disclosure": (
                            int(judge.stale_disclosure) if case.source == "memops" else None
                        ),
                    }
                )
    return rows


def score(
    *,
    judge_protocol: JudgeProtocol,
    execution_protocol: base.Protocol,
    cases_path: Path,
    source_runtime: Path,
    runtime: Path,
    output: Path,
) -> dict[str, object]:
    _validate_source(
        judge_protocol,
        execution_protocol,
        cases_path=cases_path,
        source_runtime=source_runtime,
    )
    completion = _require_sample_completion(
        judge_protocol=judge_protocol,
        execution_protocol=execution_protocol,
        runtime=runtime,
    )
    selected, specs, assignments, responses, reader_assignments, verifier_scores = _build_plan(
        judge_protocol=judge_protocol,
        execution_protocol=execution_protocol,
        cases_path=cases_path,
        source_runtime=source_runtime,
    )
    rows = _score_rows(
        selected=selected,
        execution_protocol=execution_protocol,
        judge_runtime=runtime,
        judge_specs=specs,
        judge_assignments=assignments,
        responses_by_provider=responses,
        reader_assignments=reader_assignments,
        verifier_scores=verifier_scores,
    )
    aggregates = base._aggregate_rows(rows)
    paired = base._paired_rows(rows)
    base._write_csv(output / "main_table.csv", aggregates)
    base._write_csv(output / "paired_deltas.csv", paired)
    output.mkdir(parents=True, exist_ok=True)
    summary = base._summary_text(aggregates).replace(
        "# Natural end-to-end route-to-reader evaluation",
        "# Cost-aware natural end-to-end route-to-reader evaluation",
        1,
    )
    sample_note = (
        "\nThe semantic judge covers all 523 RHELM cases and a frozen, outcome-independent "
        "simple random sample of 1,000/3,244 MemOps cases. Every selected case is paired "
        "across both readers and all five routes; all 80 MemOps namespace groups are "
        "represented. Estimates describe this prespecified judge sample, not an official "
        "benchmark submission.\n"
    )
    (output / "summary.md").write_text(
        summary.replace("\n\nThis is", f"\n{sample_note}\nThis is", 1),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": 1,
        "judge_protocol_id": judge_protocol.raw["protocol_id"],
        "status": "complete_nonofficial_cost_aware_semantic_judge_sample",
        "judge_protocol_sha256": judge_protocol.sha256,
        "execution_protocol_sha256": execution_protocol.protocol_sha256,
        "case_bundle_sha256": base._sha256_file(cases_path),
        "sample_case_count": len(selected),
        "sample_case_id_set_sha256": judge_protocol.sampling["sample_case_id_set_sha256"],
        "source_sample_counts": {
            source: sum(case.source == source for case in selected)
            for source in ("rhelm", "memops")
        },
        "reader_models": [
            {"provider": binding.provider, "model": binding.model}
            for binding in execution_protocol.readers
        ],
        "reader_estimates_pooled": False,
        "judge": {
            "provider": execution_protocol.judge.provider,
            "model": execution_protocol.judge.model,
        },
        "sample_completion_sha256": base._sha256_file(_sample_completion_path(runtime)),
        "total_budget_consumption_usd": completion["total_budget_consumption_usd"],
        "artifacts": {
            name: base._sha256_file(output / name)
            for name in ("main_table.csv", "paired_deltas.csv", "summary.md")
        },
        "official_benchmark_result": False,
    }
    base._write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-protocol", type=Path, default=DEFAULT_JUDGE_PROTOCOL)
    parser.add_argument("--execution-protocol", type=Path, default=DEFAULT_EXECUTION_PROTOCOL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--source-runtime", type=Path, default=DEFAULT_SOURCE_RUNTIME)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("command", choices=("validate", "plan", "fixture", "execute", "score"))
    args = parser.parse_args()

    judge_protocol = load_judge_protocol(args.judge_protocol)
    execution_protocol = base.load_protocol(args.execution_protocol)
    if args.command == "validate":
        result = _validate_source(
            judge_protocol,
            execution_protocol,
            cases_path=args.cases,
            source_runtime=args.source_runtime,
        )
        selected = select_cases(base.load_cases(args.cases), judge_protocol)
        result = {
            **result,
            "judge_protocol_sha256": judge_protocol.sha256,
            "sample_case_count": len(selected),
            "provider_calls_made": 0,
        }
    elif args.command == "plan":
        result = write_plan(
            judge_protocol=judge_protocol,
            execution_protocol=execution_protocol,
            cases_path=args.cases,
            source_runtime=args.source_runtime,
            runtime=args.runtime,
        )
    elif args.command == "fixture":
        result = run_fixture(
            judge_protocol=judge_protocol,
            execution_protocol=execution_protocol,
            cases_path=args.cases,
            source_runtime=args.source_runtime,
            runtime=args.runtime,
            dotenv=args.dotenv,
        )
    elif args.command == "execute":
        result = execute(
            judge_protocol=judge_protocol,
            execution_protocol=execution_protocol,
            cases_path=args.cases,
            source_runtime=args.source_runtime,
            runtime=args.runtime,
            dotenv=args.dotenv,
            workers=args.workers,
        )
    else:
        result = score(
            judge_protocol=judge_protocol,
            execution_protocol=execution_protocol,
            cases_path=args.cases,
            source_runtime=args.source_runtime,
            runtime=args.runtime,
            output=args.output,
        )
    print(json.dumps(result, allow_nan=False, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
