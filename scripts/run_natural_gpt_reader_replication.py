"""Run the frozen GPT-5.6 Luna natural reader replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_natural_end_to_end_experiment as base  # noqa: E402
from scripts import run_natural_two_reader_judge as source_judge  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "experiments" / "natural_end_to_end_gpt_reader_protocol.json"
DEFAULT_JUDGE_PROTOCOL = ROOT / "experiments" / "natural_end_to_end_gpt_judge_protocol.json"
DEFAULT_DOTENV = ROOT.parent / "bomi-codex-starter" / ".env"
DEFAULT_OUTPUT = ROOT / "results" / "natural_end_to_end_gpt_luna_judged"
PRIMARY_ARMS = ("global_dense", "namespace_dense", "namespace_policy_gate")
CONTRACT_PATHS = (
    "experiments/natural_end_to_end_gpt_reader_protocol.json",
    "experiments/natural_end_to_end_gpt_judge_protocol.json",
    "experiments/natural_end_to_end_two_reader_protocol.json",
    "experiments/natural_end_to_end_two_reader_judge_protocol.json",
    "experiments/prompts/natural_end_to_end_reader_v1.txt",
    "results/natural_end_to_end_two_reader_judged/manifest.json",
    "results/natural_end_to_end_two_reader_judged/paired_deltas.csv",
    "scripts/run_natural_end_to_end_experiment.py",
    "scripts/run_natural_gpt_reader_replication.py",
    "scripts/run_natural_two_reader_judge.py",
    "src/verify_agent_memory/natural_end_to_end.py",
)


@dataclass(frozen=True)
class ReplicationProtocol:
    path: Path
    raw: dict[str, Any]
    sha256: str
    source: dict[str, Any]
    continuation_gate: dict[str, Any]
    sample: dict[str, Any]
    reader: dict[str, Any]
    execution: dict[str, Any]
    reporting: dict[str, Any]

    @property
    def reader_binding(self) -> base.ProviderBinding:
        return base.ProviderBinding(
            provider=str(self.reader["provider"]),
            model=str(self.reader["model"]),
            api_surface=str(self.reader["api_surface"]),
            input_usd_per_million=float(self.reader["input_usd_per_million"]),
            output_usd_per_million=float(self.reader["output_usd_per_million"]),
            hard_cap_usd=float(self.reader["conservative_all_requests_hard_cap_usd"]),
            controls={"effort": str(self.reader["effort"])},
        )

    @property
    def incremental_cap_usd(self) -> float:
        return float(self.reader["incremental_hard_cap_usd"])

    @property
    def fixture_cap_usd(self) -> float:
        return float(self.reader["fixture_hard_cap_usd"])


@dataclass(frozen=True)
class JudgeProtocol:
    path: Path
    raw: dict[str, Any]
    sha256: str
    source_reader: dict[str, Any]
    sample: dict[str, Any]
    judge: dict[str, Any]
    budget: dict[str, Any]
    execution: dict[str, Any]
    outputs: dict[str, Any]

    @property
    def fixture_cap_usd(self) -> float:
        return float(self.budget["fixture_hard_cap_usd"])

    @property
    def total_cap_usd(self) -> float:
        return float(self.budget["total_incremental_hard_cap_usd"])


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _assert_file(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} must bind a SHA-256")
    if not path.is_file() or base._sha256_file(path) != expected:
        raise ValueError(f"{label} hash drifted")


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> ReplicationProtocol:
    raw = _object(json.loads(path.read_text(encoding="utf-8")), "protocol")
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != (
        "natural-heldout-route-to-reader-gpt-luna-replication-v1"
    ):
        raise ValueError("GPT reader protocol identity drifted")
    if raw.get("status") != "frozen_nonofficial_gpt_only_reader_replication":
        raise ValueError("GPT reader protocol status drifted")
    source = _object(raw.get("source_execution"), "source_execution")
    gate = _object(raw.get("continuation_gate"), "continuation_gate")
    sample = _object(raw.get("sample"), "sample")
    reader = _object(raw.get("reader"), "reader")
    execution = _object(raw.get("execution"), "execution")
    reporting = _object(raw.get("reporting"), "reporting")
    protocol = ReplicationProtocol(
        path=path,
        raw=raw,
        sha256=base._sha256_file(path),
        source=source,
        continuation_gate=gate,
        sample=sample,
        reader=reader,
        execution=execution,
        reporting=reporting,
    )
    if tuple(sample.get("arms", ())) != PRIMARY_ARMS:
        raise ValueError("GPT reader route scope drifted")
    if sample.get("case_count") != 1523 or sample.get("assignment_count") != 4569:
        raise ValueError("GPT reader sample size drifted")
    if sample.get("expected_unique_reader_request_count") != 3157:
        raise ValueError("GPT reader request count drifted")
    expected_reader = {
        "provider": "OpenAI",
        "model": "gpt-5.6-luna",
        "api_surface": "responses",
        "effort": "none",
        "input_usd_per_million": 1.0,
        "output_usd_per_million": 6.0,
        "conservative_all_requests_hard_cap_usd": 110.0,
        "incremental_hard_cap_usd": 45.0,
        "fixture_hard_cap_usd": 0.02,
        "maximum_output_tokens": 2048,
        "response_schema": "strict_action_and_answer_json",
        "models_reported_separately": True,
    }
    if reader != expected_reader:
        raise ValueError("GPT reader binding or budget drifted")
    if execution.get("outcome_selective_rerun") is not False:
        raise ValueError("outcome-selective rerun must remain disabled")
    if execution.get("partial_or_prefix_scoring") is not False:
        raise ValueError("partial reader scoring must remain disabled")
    if reporting.get("reader_estimates_pooled") is not False:
        raise ValueError("reader pooling must remain disabled")
    return protocol


def load_judge_protocol(path: Path = DEFAULT_JUDGE_PROTOCOL) -> JudgeProtocol:
    raw = _object(json.loads(path.read_text(encoding="utf-8")), "judge protocol")
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != (
        "natural-heldout-gpt-luna-cross-provider-judge-v1"
    ):
        raise ValueError("GPT judge protocol identity drifted")
    if raw.get("status") != "frozen_after_complete_gpt_reader_before_judge_execution":
        raise ValueError("GPT judge protocol status drifted")
    protocol = JudgeProtocol(
        path=path,
        raw=raw,
        sha256=base._sha256_file(path),
        source_reader=_object(raw.get("source_reader"), "source_reader"),
        sample=_object(raw.get("sample"), "sample"),
        judge=_object(raw.get("judge"), "judge"),
        budget=_object(raw.get("budget"), "budget"),
        execution=_object(raw.get("execution"), "execution"),
        outputs=_object(raw.get("outputs"), "outputs"),
    )
    if tuple(protocol.sample.get("arms", ())) != PRIMARY_ARMS:
        raise ValueError("GPT judge route scope drifted")
    if protocol.sample.get("case_count") != 1523:
        raise ValueError("GPT judge sample count drifted")
    expected_judge = {
        "provider": "Anthropic",
        "model": "claude-haiku-4-5-20251001",
        "thinking": "disabled",
        "effort": "omitted",
        "maximum_output_tokens": 512,
        "input_usd_per_million": 1.0,
        "output_usd_per_million": 5.0,
        "expected_unique_request_count": 3397,
        "expected_request_id_set_sha256": (
            "5d4237295a917968ff95f18033cd75ba1f9b4707f1b008d16a7c3670d52fac07"
        ),
    }
    for key, value in expected_judge.items():
        if protocol.judge.get(key) != value:
            raise ValueError(f"GPT judge binding drifted: {key}")
    if protocol.fixture_cap_usd != 0.01 or protocol.total_cap_usd != 10.0:
        raise ValueError("GPT judge budget drifted")
    if protocol.execution.get("outcome_selective_rerun") is not False:
        raise ValueError("GPT judge outcome-selective rerun must remain disabled")
    if protocol.outputs.get("reader_estimates_pooled") is not False:
        raise ValueError("GPT reader estimates cannot be pooled")
    return protocol


def _source_paths(protocol: ReplicationProtocol) -> tuple[Path, Path, Path, Path]:
    source_protocol = ROOT / str(protocol.source["protocol_path"])
    cases = ROOT / str(protocol.source["case_bundle_path"])
    source_runtime = ROOT / str(protocol.source["source_runtime"])
    source_judge_protocol = ROOT / str(protocol.sample["source_judge_protocol_path"])
    return source_protocol, cases, source_runtime, source_judge_protocol


def _validate_continuation_gate(protocol: ReplicationProtocol) -> None:
    gate = protocol.continuation_gate
    manifest = ROOT / str(gate["judged_manifest_path"])
    paired = ROOT / str(gate["paired_deltas_path"])
    _assert_file(manifest, gate["judged_manifest_sha256"], "judged manifest")
    _assert_file(paired, gate["paired_deltas_sha256"], "paired deltas")
    rows = list(csv.DictReader(paired.open(encoding="utf-8", newline="")))
    for provider in gate["required_readers"]:
        by_metric = {
            row["metric"]: row
            for row in rows
            if row["reader_provider"] == provider
            and row["arm"] == "namespace_dense"
            and row["reference"] == "global_dense"
        }
        if float(by_metric["answer_correct"]["bootstrap_ci95_lower"]) <= 0:
            raise ValueError(f"{provider} answer-accuracy continuation gate failed")
        if float(by_metric["penalized_admissibility_upper_risk"]["bootstrap_ci95_upper"]) >= 0:
            raise ValueError(f"{provider} risk continuation gate failed")


def _load_sources(
    protocol: ReplicationProtocol,
) -> tuple[base.Protocol, source_judge.JudgeProtocol, list[base.NaturalEndToEndCase], Path]:
    source_path, cases_path, source_runtime, judge_path = _source_paths(protocol)
    _assert_file(source_path, protocol.source["protocol_sha256"], "source protocol")
    _assert_file(cases_path, protocol.source["case_bundle_sha256"], "case bundle")
    _assert_file(
        ROOT / str(protocol.source["reader_prompt_path"]),
        protocol.source["reader_prompt_sha256"],
        "reader prompt",
    )
    _assert_file(
        judge_path,
        protocol.sample["source_judge_protocol_sha256"],
        "source judge protocol",
    )
    source = base.load_protocol(source_path)
    judge = source_judge.load_judge_protocol(judge_path)
    selected = source_judge.select_cases(base.load_cases(cases_path), judge)
    if len(selected) != protocol.sample["case_count"]:
        raise ValueError("GPT reader selected case count drifted")
    if (
        base._sha256_object([case.case_id for case in selected])
        != (protocol.sample["case_id_set_sha256"])
    ):
        raise ValueError("GPT reader selected case identity drifted")
    return source, judge, selected, source_runtime


def _execution_protocol(
    protocol: ReplicationProtocol,
    source: base.Protocol,
) -> base.Protocol:
    reader = protocol.reader_binding
    return replace(
        source,
        path=protocol.path,
        protocol_id=str(protocol.raw["protocol_id"]),
        protocol_sha256=protocol.sha256,
        readers=(reader,),
        maximum_output_tokens_reader=int(protocol.reader["maximum_output_tokens"]),
        maximum_transport_retries=int(protocol.execution["maximum_transport_retries"]),
        maximum_model_contract_recovery_attempts=int(
            protocol.execution["maximum_model_contract_recovery_attempts"]
        ),
        reader_incremental_hard_caps={reader.provider: protocol.incremental_cap_usd},
        reader_fixture_hard_caps={reader.provider: protocol.fixture_cap_usd},
    )


def _reader_plan(
    protocol: ReplicationProtocol,
) -> tuple[
    base.Protocol,
    list[base.NaturalEndToEndCase],
    dict[str, base.CallSpec],
    dict[tuple[str, str], str],
    dict[str, dict[str, float]],
]:
    source, _judge, selected, source_runtime = _load_sources(protocol)
    specs, assignments, verifier_scores = base._reader_plan(selected, source, source_runtime)
    selected_assignments = {
        (case.case_id, arm): assignments[(case.case_id, arm)]
        for case in selected
        for arm in PRIMARY_ARMS
    }
    request_ids = set(selected_assignments.values())
    selected_specs = {request_id: specs[request_id] for request_id in request_ids}
    if len(selected_assignments) != protocol.sample["assignment_count"]:
        raise ValueError("GPT reader assignment count drifted")
    if len(selected_specs) != protocol.sample["expected_unique_reader_request_count"]:
        raise ValueError("GPT reader unique request count drifted")
    if (
        base._sha256_object(sorted(request_ids))
        != (protocol.sample["expected_reader_request_id_set_sha256"])
    ):
        raise ValueError("GPT reader request identity drifted")
    return (
        _execution_protocol(protocol, source),
        selected,
        selected_specs,
        selected_assignments,
        verifier_scores,
    )


def _require_clean_contract() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *CONTRACT_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("GPT reader execution requires committed, clean contract paths")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate(protocol: ReplicationProtocol) -> dict[str, object]:
    _validate_continuation_gate(protocol)
    execution, selected, specs, assignments, _scores = _reader_plan(protocol)
    return {
        "protocol_sha256": protocol.sha256,
        "source_protocol_sha256": execution.raw.get("protocol_id")
        and protocol.source["protocol_sha256"],
        "reader_provider": execution.readers[0].provider,
        "reader_model": execution.readers[0].model,
        "sample_case_count": len(selected),
        "route_arms": list(PRIMARY_ARMS),
        "assignment_count": len(assignments),
        "unique_reader_request_count": len(specs),
        "incremental_hard_cap_usd": protocol.incremental_cap_usd,
        "provider_calls_made": 0,
    }


def plan_reader(protocol: ReplicationProtocol) -> dict[str, object]:
    execution, _selected, specs, _assignments, _scores = _reader_plan(protocol)
    binding = execution.readers[0]
    return {
        **base._plan_stats(
            specs,
            binding,
            system_prompt=execution.reader_prompt,
            maximum_output_tokens=execution.maximum_output_tokens_reader,
        ),
        "incremental_hard_cap_usd": protocol.incremental_cap_usd,
        "fixture_hard_cap_usd": protocol.fixture_cap_usd,
    }


def fixture_reader(
    protocol: ReplicationProtocol,
    *,
    runtime: Path,
    dotenv: Path,
) -> dict[str, object]:
    _require_clean_contract()
    execution, _selected, _specs, _assignments, _scores = _reader_plan(protocol)
    return base.run_fixture(
        protocol=execution,
        runtime=runtime,
        stage="reader",
        binding=execution.readers[0],
        dotenv=dotenv,
    )


def execute_reader(
    protocol: ReplicationProtocol,
    *,
    runtime: Path,
    dotenv: Path,
    workers: int,
) -> dict[str, object]:
    _require_clean_contract()
    execution, _selected, specs, _assignments, _scores = _reader_plan(protocol)
    binding = execution.readers[0]
    return base._execute_specs(
        protocol=execution,
        runtime=runtime,
        stage="reader",
        binding=binding,
        specs=specs,
        system_prompt=execution.reader_prompt,
        maximum_output_tokens=execution.maximum_output_tokens_reader,
        dotenv=dotenv,
        workers=workers,
        parser=base._reader_parser,
        incremental_cost_cap_usd=protocol.incremental_cap_usd,
    )


def _judge_plan(
    reader_protocol: ReplicationProtocol,
    judge_protocol: JudgeProtocol,
) -> tuple[
    base.Protocol,
    list[base.NaturalEndToEndCase],
    dict[str, base.CallSpec],
    dict[tuple[str, str], str],
    dict[str, base.ReaderResponse],
    dict[tuple[str, str], str],
    dict[str, dict[str, float]],
]:
    source_reader_path = ROOT / str(judge_protocol.source_reader["protocol_path"])
    _assert_file(
        source_reader_path,
        judge_protocol.source_reader["protocol_sha256"],
        "GPT reader protocol",
    )
    if reader_protocol.sha256 != judge_protocol.source_reader["protocol_sha256"]:
        raise ValueError("loaded GPT reader protocol differs from judge source")
    execution, cases, reader_specs, reader_assignments, verifier_scores = _reader_plan(
        reader_protocol
    )
    runtime = ROOT / str(judge_protocol.source_reader["runtime"])
    completion_path = ROOT / str(judge_protocol.source_reader["completion_path"])
    _assert_file(
        completion_path,
        judge_protocol.source_reader["completion_sha256"],
        "GPT reader completion",
    )
    completion = _object(base._read_json(completion_path), "GPT reader completion")
    expected_completion = {
        "protocol_sha256": reader_protocol.sha256,
        "implementation_commit": judge_protocol.source_reader["execution_implementation_commit"],
        "provider": "OpenAI",
        "model": "gpt-5.6-luna",
        "request_count": judge_protocol.source_reader["request_count"],
        "response_set_sha256": judge_protocol.source_reader["response_set_sha256"],
        "complete_bundle": True,
    }
    for key, value in expected_completion.items():
        if completion.get(key) != value:
            raise ValueError(f"GPT reader completion drifted: {key}")
    reader_binding = execution.readers[0]
    responses = base._load_reader_responses(
        protocol=execution,
        runtime=runtime,
        binding=reader_binding,
        specs=reader_specs,
    )
    judge_specs: dict[str, base.CallSpec] = {}
    judge_assignments = {}
    for case in cases:
        for arm in PRIMARY_ARMS:
            response = responses[reader_assignments[(case.case_id, arm)]]
            payload = base.judge_payload(case, response)
            request_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            previous = judge_specs.setdefault(
                request_id,
                base.CallSpec(
                    request_id=request_id,
                    payload=payload,
                    schema=base.judge_response_schema(),
                ),
            )
            if previous.payload != payload:
                raise RuntimeError("GPT judge prompt SHA-256 collision")
            judge_assignments[(case.case_id, arm)] = request_id
    if len(judge_assignments) != judge_protocol.sample["assignment_count"]:
        raise ValueError("GPT judge assignment count drifted")
    if len(judge_specs) != judge_protocol.judge["expected_unique_request_count"]:
        raise ValueError("GPT judge unique request count drifted")
    if (
        base._sha256_object(sorted(judge_specs))
        != (judge_protocol.judge["expected_request_id_set_sha256"])
    ):
        raise ValueError("GPT judge request identity drifted")
    source_judge_path = ROOT / str(judge_protocol.judge["source_protocol_path"])
    _assert_file(
        source_judge_path,
        judge_protocol.judge["source_protocol_sha256"],
        "source judge protocol",
    )
    _assert_file(
        ROOT / str(judge_protocol.judge["prompt_path"]),
        judge_protocol.judge["prompt_sha256"],
        "judge prompt",
    )
    recovered = source_judge.bind_recovered_judge(
        source_judge.load_judge_protocol(source_judge_path),
        base.load_protocol(base.TWO_READER_PROTOCOL),
    )
    judge_execution = replace(
        execution,
        path=judge_protocol.path,
        protocol_id=str(judge_protocol.raw["protocol_id"]),
        protocol_sha256=judge_protocol.sha256,
        judge=recovered.judge,
        judge_prompt=recovered.judge_prompt,
        maximum_output_tokens_judge=int(judge_protocol.judge["maximum_output_tokens"]),
        maximum_transport_retries=int(judge_protocol.execution["maximum_transport_retries"]),
        maximum_model_contract_recovery_attempts=int(
            judge_protocol.execution["maximum_model_contract_recovery_attempts"]
        ),
    )
    return (
        judge_execution,
        cases,
        judge_specs,
        judge_assignments,
        responses,
        reader_assignments,
        verifier_scores,
    )


def validate_judge(
    reader_protocol: ReplicationProtocol,
    judge_protocol: JudgeProtocol,
) -> dict[str, object]:
    execution, cases, specs, assignments, *_ = _judge_plan(
        reader_protocol,
        judge_protocol,
    )
    return {
        "judge_protocol_sha256": judge_protocol.sha256,
        "reader_protocol_sha256": reader_protocol.sha256,
        "reader_model": "gpt-5.6-luna",
        "judge_provider": execution.judge.provider,
        "judge_model": execution.judge.model,
        "sample_case_count": len(cases),
        "assignment_count": len(assignments),
        "unique_judge_request_count": len(specs),
        "total_incremental_hard_cap_usd": judge_protocol.total_cap_usd,
        "provider_calls_made": 0,
    }


def plan_judge(
    reader_protocol: ReplicationProtocol,
    judge_protocol: JudgeProtocol,
) -> dict[str, object]:
    execution, _cases, specs, *_ = _judge_plan(reader_protocol, judge_protocol)
    plan = base._plan_stats(
        specs,
        execution.judge,
        system_prompt=execution.judge_prompt,
        maximum_output_tokens=execution.maximum_output_tokens_judge,
    )
    expected = float(judge_protocol.budget["conservative_all_requests_bound_usd"])
    if abs(float(plan["conservative_cost_bound_usd"]) - expected) > 1e-9:
        raise ValueError("GPT judge conservative plan bound drifted")
    return {
        **plan,
        "fixture_hard_cap_usd": judge_protocol.fixture_cap_usd,
        "total_incremental_hard_cap_usd": judge_protocol.total_cap_usd,
    }


def fixture_judge(
    reader_protocol: ReplicationProtocol,
    judge_protocol: JudgeProtocol,
    *,
    runtime: Path,
    dotenv: Path,
) -> dict[str, object]:
    _require_clean_contract()
    execution, *_ = _judge_plan(reader_protocol, judge_protocol)
    fixture = base._fixture_case()
    spec = base.CallSpec(
        request_id=fixture.case_id,
        payload=base.judge_payload(
            fixture,
            base.ReaderResponse(action="answer", answer="123 Market Street"),
        ),
        schema=base.judge_response_schema(),
    )
    conservative = base._conservative_call_cost(
        spec,
        execution.judge,
        system_prompt=execution.judge_prompt,
        maximum_output_tokens=execution.maximum_output_tokens_judge,
    )
    if conservative > judge_protocol.fixture_cap_usd:
        raise RuntimeError("GPT judge fixture exceeds its hard cap")
    with source_judge._recovered_anthropic_adapter():
        receipt = base.run_fixture(
            protocol=execution,
            runtime=runtime,
            stage="judge",
            binding=execution.judge,
            dotenv=dotenv,
        )
    if base._record_cost(receipt) > judge_protocol.fixture_cap_usd:
        raise RuntimeError("GPT judge fixture exceeded its hard cap")
    return receipt


def _require_judge_fixture(
    judge_protocol: JudgeProtocol,
    execution: base.Protocol,
    runtime: Path,
) -> dict[str, Any]:
    path = base._stage_root(runtime, "fixtures", execution.judge) / "judge.json"
    if not path.is_file():
        raise RuntimeError("run the GPT judge fixture before execution")
    receipt = _object(base._read_json(path), "GPT judge fixture")
    expected = {
        "protocol_sha256": judge_protocol.sha256,
        "implementation_commit": _require_clean_contract(),
        "stage": "judge",
        "provider": execution.judge.provider,
        "model": execution.judge.model,
        "synthetic_fixture": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"GPT judge fixture drifted: {key}")
    if base._record_cost(receipt) > judge_protocol.fixture_cap_usd:
        raise ValueError("GPT judge fixture cost exceeds its cap")
    return receipt


def _judge_failure_cost(
    judge_protocol: JudgeProtocol,
    execution: base.Protocol,
    runtime: Path,
) -> float:
    root = base._stage_root(runtime, "judge", execution.judge) / "failures"
    total = 0.0
    for path in root.glob("*.json") if root.is_dir() else ():
        failure = _object(base._read_json(path), "GPT judge failure")
        if (
            failure.get("protocol_sha256") == judge_protocol.sha256
            and failure.get("provider") == execution.judge.provider
            and failure.get("model") == execution.judge.model
        ):
            total += float(failure.get("cost_bound_usd", 0))
    return total


def execute_judge(
    reader_protocol: ReplicationProtocol,
    judge_protocol: JudgeProtocol,
    *,
    runtime: Path,
    dotenv: Path,
    workers: int,
) -> dict[str, object]:
    _require_clean_contract()
    execution, _cases, specs, *_ = _judge_plan(reader_protocol, judge_protocol)
    fixture = _require_judge_fixture(judge_protocol, execution, runtime)
    fixture_cost = base._record_cost(fixture)
    response_cap = judge_protocol.total_cap_usd - fixture_cost
    with source_judge._recovered_anthropic_adapter():
        completion = base._execute_specs(
            protocol=execution,
            runtime=runtime,
            stage="judge",
            binding=execution.judge,
            specs=specs,
            system_prompt=execution.judge_prompt,
            maximum_output_tokens=execution.maximum_output_tokens_judge,
            dotenv=dotenv,
            workers=workers,
            parser=base._judge_parser,
            incremental_cost_cap_usd=response_cap,
        )
    failure_cost = _judge_failure_cost(judge_protocol, execution, runtime)
    total_cost = fixture_cost + float(completion["cost_usd"]) + failure_cost
    if total_cost > judge_protocol.total_cap_usd:
        raise RuntimeError("GPT judge total cost exceeded its hard cap")
    return {
        **completion,
        "fixture_cost_usd": fixture_cost,
        "conservative_failure_cost_usd": failure_cost,
        "total_incremental_cost_usd": total_cost,
        "total_incremental_hard_cap_usd": judge_protocol.total_cap_usd,
    }


def _scored_rows(
    reader_protocol: ReplicationProtocol,
    judge_protocol: JudgeProtocol,
    *,
    runtime: Path,
) -> tuple[list[dict[str, object]], base.Protocol, dict[str, base.CallSpec]]:
    (
        execution,
        cases,
        judge_specs,
        judge_assignments,
        reader_responses,
        reader_assignments,
        verifier_scores,
    ) = _judge_plan(reader_protocol, judge_protocol)
    judge_responses = base._load_judge_responses(
        protocol=execution,
        runtime=runtime,
        specs=judge_specs,
    )
    rows = []
    for case in cases:
        for arm in PRIMARY_ARMS:
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
    return rows, execution, judge_specs


def _paired_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["reader_provider"]), str(row["case_id"]), str(row["arm"])): row for row in rows
    }
    comparisons = (
        ("namespace_dense", "global_dense"),
        ("namespace_policy_gate", "namespace_dense"),
    )
    metrics = (
        "evidence_recall",
        "feasible",
        "penalized_admissibility_upper_risk",
        "answer_correct",
        "answer_quality",
        "over_refusal",
        "protected_disclosure",
        "stale_disclosure",
    )
    output = []
    for arm, reference in comparisons:
        for metric in metrics:
            point, lower, upper, query_count, group_count = base._paired_delta(
                by_key,
                provider="OpenAI",
                arm=arm,
                reference=reference,
                metric=metric,
            )
            output.append(
                {
                    "reader_provider": "OpenAI",
                    "reader_model": "gpt-5.6-luna",
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


def _paired_lookup(
    paired: list[dict[str, object]],
    arm: str,
    reference: str,
    metric: str,
) -> dict[str, object]:
    matches = [
        row
        for row in paired
        if row["arm"] == arm and row["reference"] == reference and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError("GPT paired result lookup is not unique")
    return matches[0]


def _delta_text(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta_arm_minus_reference']):+.4f} "
        f"[{float(row['bootstrap_ci95_lower']):+.4f}, "
        f"{float(row['bootstrap_ci95_upper']):+.4f}]"
    )


def score_judge(
    reader_protocol: ReplicationProtocol,
    judge_protocol: JudgeProtocol,
    *,
    runtime: Path,
    output: Path,
) -> dict[str, object]:
    rows, execution, judge_specs = _scored_rows(
        reader_protocol,
        judge_protocol,
        runtime=runtime,
    )
    aggregates = base._aggregate_rows(rows)
    paired = _paired_rows(rows)
    base._write_csv(output / "main_table.csv", aggregates)
    base._write_csv(output / "paired_deltas.csv", paired)
    namespace_accuracy = _paired_lookup(
        paired,
        "namespace_dense",
        "global_dense",
        "answer_correct",
    )
    namespace_risk = _paired_lookup(
        paired,
        "namespace_dense",
        "global_dense",
        "penalized_admissibility_upper_risk",
    )
    namespace_refusal = _paired_lookup(
        paired,
        "namespace_dense",
        "global_dense",
        "over_refusal",
    )
    policy_accuracy = _paired_lookup(
        paired,
        "namespace_policy_gate",
        "namespace_dense",
        "answer_correct",
    )
    policy_risk = _paired_lookup(
        paired,
        "namespace_policy_gate",
        "namespace_dense",
        "penalized_admissibility_upper_risk",
    )
    summary = "\n".join(
        [
            "# GPT-5.6 Luna natural route-to-reader replication",
            "",
            "This non-official replication evaluates the frozen 1,523-case sample under "
            "global dense, namespace dense, and namespace plus released-policy gating. "
            "GPT estimates are reported separately from Gemini and DeepSeek.",
            "",
            "## Paired findings",
            "",
            "- Namespace dense versus global dense: answer accuracy "
            f"{_delta_text(namespace_accuracy)}; "
            f"penalized admissibility upper risk {_delta_text(namespace_risk)}; over-refusal "
            f"{_delta_text(namespace_refusal)}.",
            f"- Released-policy gate versus namespace dense: answer accuracy "
            f"{_delta_text(policy_accuracy)}; penalized admissibility upper risk "
            f"{_delta_text(policy_risk)}.",
            "- Confidence intervals use 10,000 paired namespace-group bootstrap replicates "
            "with equal-source macro aggregation.",
            "",
            "This is a prespecified robustness replication, not an official RHELM or MemOps "
            "submission. It does not rerun the text-verifier or released-field-oracle "
            "diagnostic routes and makes no pooled cross-reader claim.",
            "",
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    reader_completion_path = ROOT / str(judge_protocol.source_reader["completion_path"])
    reader_completion = _object(
        base._read_json(reader_completion_path),
        "GPT reader completion",
    )
    judge_completion_path = base._completion_path(runtime, "judge", execution.judge)
    judge_completion = _object(
        base._read_json(judge_completion_path),
        "GPT judge completion",
    )
    fixture_path = base._stage_root(runtime, "fixtures", execution.judge) / "judge.json"
    fixture_cost = base._record_cost(base._read_json(fixture_path))
    failure_cost = _judge_failure_cost(judge_protocol, execution, runtime)
    receipt = {
        "schema_version": 1,
        "reader_protocol_sha256": reader_protocol.sha256,
        "judge_protocol_sha256": judge_protocol.sha256,
        "reader_provider": "OpenAI",
        "reader_model": "gpt-5.6-luna",
        "reader_request_count": reader_completion["request_count"],
        "reader_cost_usd": reader_completion["total_incremental_cost_usd"],
        "reader_response_set_sha256": reader_completion["response_set_sha256"],
        "judge_provider": execution.judge.provider,
        "judge_model": execution.judge.model,
        "judge_request_count": judge_completion["request_count"],
        "judge_fixture_cost_usd": fixture_cost,
        "judge_response_cost_usd": judge_completion["cost_usd"],
        "judge_conservative_failure_cost_usd": failure_cost,
        "judge_total_cost_usd": fixture_cost + float(judge_completion["cost_usd"]) + failure_cost,
        "sample_case_count": judge_protocol.sample["case_count"],
        "complete_bundle": True,
        "benchmark_payload_or_response_content_included": False,
        "official_benchmark_result": False,
    }
    base._write_json(output / "execution_receipt.json", receipt)
    manifest = {
        "schema_version": 1,
        "status": "complete_nonofficial_gpt_luna_primary_route_replication",
        "reader_protocol_sha256": reader_protocol.sha256,
        "judge_protocol_sha256": judge_protocol.sha256,
        "sample_case_count": judge_protocol.sample["case_count"],
        "arms": list(PRIMARY_ARMS),
        "unique_reader_request_count": reader_completion["request_count"],
        "unique_judge_request_count": len(judge_specs),
        "reader_estimates_pooled": False,
        "artifacts": {
            name: base._sha256_file(output / name)
            for name in (
                "main_table.csv",
                "paired_deltas.csv",
                "summary.md",
                "execution_receipt.json",
            )
        },
        "official_rhelm_or_memops_claim": False,
    }
    base._write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--judge-protocol", type=Path, default=DEFAULT_JUDGE_PROTOCOL)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "plan-reader",
            "fixture-reader",
            "execute-reader",
            "validate-judge",
            "plan-judge",
            "fixture-judge",
            "execute-judge",
            "score",
        ),
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    runtime = args.runtime or ROOT / str(protocol.execution["runtime"])
    if args.command == "validate":
        result = validate(protocol)
    elif args.command == "plan-reader":
        result = plan_reader(protocol)
    elif args.command == "fixture-reader":
        result = fixture_reader(protocol, runtime=runtime, dotenv=args.dotenv)
    elif args.command == "execute-reader":
        result = execute_reader(
            protocol,
            runtime=runtime,
            dotenv=args.dotenv,
            workers=args.workers,
        )
    else:
        judge_protocol = load_judge_protocol(args.judge_protocol)
        judge_runtime = args.runtime or ROOT / str(judge_protocol.outputs["runtime"])
        if args.command == "validate-judge":
            result = validate_judge(protocol, judge_protocol)
        elif args.command == "plan-judge":
            result = plan_judge(protocol, judge_protocol)
        elif args.command == "fixture-judge":
            result = fixture_judge(
                protocol,
                judge_protocol,
                runtime=judge_runtime,
                dotenv=args.dotenv,
            )
        elif args.command == "execute-judge":
            result = execute_judge(
                protocol,
                judge_protocol,
                runtime=judge_runtime,
                dotenv=args.dotenv,
                workers=args.workers,
            )
        else:
            result = score_judge(
                protocol,
                judge_protocol,
                runtime=judge_runtime,
                output=args.output,
            )
    print(json.dumps(result, allow_nan=False, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
