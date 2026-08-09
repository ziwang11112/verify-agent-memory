"""Run the frozen GPT-5.6 Luna natural reader replication."""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_DOTENV = ROOT.parent / "bomi-codex-starter" / ".env"
PRIMARY_ARMS = ("global_dense", "namespace_dense", "namespace_policy_gate")
CONTRACT_PATHS = (
    "experiments/natural_end_to_end_gpt_reader_protocol.json",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "command",
        choices=("validate", "plan-reader", "fixture-reader", "execute-reader"),
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
    else:
        result = execute_reader(
            protocol,
            runtime=runtime,
            dotenv=args.dotenv,
            workers=args.workers,
        )
    print(json.dumps(result, allow_nan=False, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
