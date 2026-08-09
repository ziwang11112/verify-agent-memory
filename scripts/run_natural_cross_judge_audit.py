"""Run the outcome-independent 200-output natural cross-judge robustness audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_natural_end_to_end_experiment as base  # noqa: E402
from scripts import run_natural_gpt_reader_replication as gpt  # noqa: E402
from scripts import run_natural_two_reader_judge as two_reader  # noqa: E402
from verify_agent_memory.judge_agreement import (  # noqa: E402
    AuditAssignment,
    SelectedAuditUnit,
    binary_agreement,
    bootstrap_binary_agreement,
    quality_agreement,
    select_unique_stratified,
)

DEFAULT_PROTOCOL = ROOT / "experiments" / "natural_cross_judge_audit_protocol.json"
DEFAULT_SAMPLE_MANIFEST = ROOT / "experiments" / "manifests" / "natural_cross_judge_sample.json"
DEFAULT_RUNTIME = ROOT / "tmp" / "natural_cross_judge_audit"
DEFAULT_OUTPUT = ROOT / "results" / "natural_cross_judge_audit"
DEFAULT_DOTENV = ROOT.parent / "bomi-codex-starter" / ".env"
PRIMARY_ARMS = gpt.PRIMARY_ARMS
CONTRACT_PATHS = (
    *base.CONTRACT_PATHS,
    "docs/NATURAL_CROSS_JUDGE_AUDIT.md",
    "docs/NATURAL_CROSS_JUDGE_RECOVERY.md",
    "experiments/natural_cross_judge_audit_protocol.json",
    "experiments/natural_cross_judge_recovery_protocol.json",
    "experiments/manifests/natural_cross_judge_sample.json",
    "scripts/run_natural_cross_judge_audit.py",
    "src/verify_agent_memory/judge_agreement.py",
    "tests/test_natural_cross_judge_audit.py",
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class AuditProtocol:
    path: Path
    raw: dict[str, Any]
    sha256: str
    source: dict[str, Any]
    sampling: dict[str, Any]
    judge: dict[str, Any]
    budget: dict[str, Any]
    analysis: dict[str, Any]
    outputs: dict[str, Any]

    @property
    def binding(self) -> base.ProviderBinding:
        return base.ProviderBinding(
            provider=_string(self.judge["provider"], "judge.provider"),
            model=_string(self.judge["model"], "judge.model"),
            api_surface=_string(self.judge["api_surface"], "judge.api_surface"),
            input_usd_per_million=_number(self.judge["input_usd_per_million"], "judge.input price"),
            output_usd_per_million=_number(
                self.judge["output_usd_per_million"], "judge.output price"
            ),
            hard_cap_usd=self.total_cap_usd,
            controls={"effort": _string(self.judge["effort"], "judge.effort")},
        )

    @property
    def fixture_cap_usd(self) -> float:
        return _number(self.budget["fixture_hard_cap_usd"], "fixture cap")

    @property
    def total_cap_usd(self) -> float:
        return _number(self.budget["total_incremental_hard_cap_usd"], "total cap")

    @property
    def call_reservation_usd(self) -> float:
        return _number(self.budget["per_call_reservation_usd"], "call reservation")

    @property
    def prior_budget_consumption_usd(self) -> float:
        return _number(
            self.budget.get("prior_budget_consumption_usd", 0.0),
            "prior budget consumption",
        )


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> AuditProtocol:
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "audit protocol")
    protocol_id = raw.get("protocol_id")
    expected_top = {
        "schema_version",
        "protocol_id",
        "status",
        "frozen_date",
        "objective",
        "interpretation",
        "source_bundles",
        "sampling",
        "judge",
        "budget",
        "analysis",
        "outputs",
    }
    if protocol_id == "natural-cross-judge-ceiling-recovery-v1":
        expected_top.add("recovery")
    if set(raw) != expected_top:
        raise ValueError("cross-judge protocol has missing or unknown top-level fields")
    if raw.get("schema_version") != 1 or protocol_id not in {
        "natural-cross-judge-posthoc-audit-v1",
        "natural-cross-judge-ceiling-recovery-v1",
    }:
        raise ValueError("cross-judge protocol identity drifted")
    expected_status = {
        "natural-cross-judge-posthoc-audit-v1": ("frozen_before_any_strong_judge_provider_call"),
        "natural-cross-judge-ceiling-recovery-v1": (
            "frozen_after_zero_label_ceiling_failure_before_gpt51_provider_call"
        ),
    }[str(protocol_id)]
    if raw.get("status") != expected_status:
        raise ValueError("cross-judge protocol is not frozen")
    interpretation = _mapping(raw["interpretation"], "interpretation")
    required_false = (
        "independently_preregistered_replication",
        "reader_estimates_pooled",
        "official_benchmark_claim",
        "output_repair",
        "selective_rerun",
    )
    if any(interpretation.get(key) is not False for key in required_false):
        raise ValueError("cross-judge interpretation boundary drifted")
    if interpretation.get("posthoc_outcome_independent_robustness_audit") is not True:
        raise ValueError("post-hoc audit label is required")

    protocol = AuditProtocol(
        path=path,
        raw=raw,
        sha256=base._sha256_file(path),
        source=_mapping(raw["source_bundles"], "source_bundles"),
        sampling=_mapping(raw["sampling"], "sampling"),
        judge=_mapping(raw["judge"], "judge"),
        budget=_mapping(raw["budget"], "budget"),
        analysis=_mapping(raw["analysis"], "analysis"),
        outputs=_mapping(raw["outputs"], "outputs"),
    )
    if tuple(protocol.sampling.get("arms", ())) != PRIMARY_ARMS:
        raise ValueError("cross-judge route panel drifted")
    if protocol.sampling.get("reader_providers") != ["DeepSeek", "Gemini", "OpenAI"]:
        raise ValueError("cross-judge reader panel drifted")
    if protocol.sampling.get("sources") != ["memops", "rhelm"]:
        raise ValueError("cross-judge source panel drifted")
    if protocol.sampling.get("selection_uses_outcomes_or_answer_content") is not False:
        raise ValueError("cross-judge selection must remain outcome independent")
    if _integer(protocol.sampling.get("sample_count"), "sample count", minimum=1) != 200:
        raise ValueError("cross-judge sample count drifted")
    expected_model = {
        "natural-cross-judge-posthoc-audit-v1": "gpt-5-pro-2025-10-06",
        "natural-cross-judge-ceiling-recovery-v1": "gpt-5.1-2025-11-13",
    }[str(protocol_id)]
    if protocol.binding.provider != "OpenAI" or protocol.binding.model != expected_model:
        raise ValueError("strong judge binding drifted")
    if protocol.binding.controls != {"effort": "high"}:
        raise ValueError("strong judge reasoning control drifted")
    if protocol.judge.get("strict_json_schema") is not True:
        raise ValueError("strong judge must use strict structured output")
    expected_output_tokens = {
        "natural-cross-judge-posthoc-audit-v1": 512,
        "natural-cross-judge-ceiling-recovery-v1": 4096,
    }[str(protocol_id)]
    if (
        _integer(protocol.judge.get("maximum_output_tokens"), "output ceiling")
        != expected_output_tokens
    ):
        raise ValueError("strong judge output ceiling drifted")
    if _integer(protocol.judge.get("maximum_transport_retries"), "retries") != 0:
        raise ValueError("transport retries are forbidden")
    expected_fixture_cap = {
        "natural-cross-judge-posthoc-audit-v1": 0.25,
        "natural-cross-judge-ceiling-recovery-v1": 0.1,
    }[str(protocol_id)]
    expected_reservation = {
        "natural-cross-judge-posthoc-audit-v1": 0.096,
        "natural-cross-judge-ceiling-recovery-v1": 0.04384,
    }[str(protocol_id)]
    if protocol.fixture_cap_usd != expected_fixture_cap or protocol.total_cap_usd != 20.0:
        raise ValueError("cross-judge budget drifted")
    if protocol.call_reservation_usd != expected_reservation:
        raise ValueError("cross-judge call reservation drifted")
    if (
        protocol.prior_budget_consumption_usd
        + protocol.fixture_cap_usd
        + 200 * protocol.call_reservation_usd
        > protocol.total_cap_usd
    ):
        raise ValueError("cross-judge reservations exceed the hard cap")
    if protocol_id == "natural-cross-judge-ceiling-recovery-v1":
        _validate_recovery_contract(protocol)
    return protocol


def _assert_hash(path: Path, expected: object, label: str) -> None:
    expected_hash = _string(expected, f"{label} SHA-256")
    if len(expected_hash) != 64 or not path.is_file() or base._sha256_file(path) != expected_hash:
        raise ValueError(f"{label} hash drifted")


def _validate_recovery_contract(protocol: AuditProtocol) -> None:
    recovery = _mapping(protocol.raw.get("recovery"), "recovery")
    source_protocol_path = ROOT / _string(
        recovery.get("source_protocol_path"), "recovery source protocol path"
    )
    _assert_hash(
        source_protocol_path,
        recovery.get("source_protocol_sha256"),
        "recovery source protocol",
    )
    for prefix in ("failure", "fixture"):
        _string(recovery.get(f"{prefix}_path"), f"recovery {prefix} path")
        digest = _string(recovery.get(f"{prefix}_sha256"), f"recovery {prefix} hash")
        if len(digest) != 64:
            raise ValueError(f"recovery {prefix} hash must be a SHA-256")
    if recovery.get("accepted_benchmark_response_count") != 0:
        raise ValueError("model migration is allowed only before an accepted benchmark label")
    if recovery.get("selection_uses_outcomes") is not False:
        raise ValueError("cross-judge recovery cannot use outcomes")
    if recovery.get("provider_response_content_inspected") is not False:
        raise ValueError("cross-judge recovery cannot inspect an incomplete provider response")
    accounted = _number(recovery.get("fixture_cost_usd"), "prior fixture cost") + _number(
        recovery.get("failure_cost_bound_usd"), "prior failure bound"
    )
    if abs(accounted - protocol.prior_budget_consumption_usd) > 1e-12:
        raise ValueError("prior recovery budget is not fully accounted")


def _validate_recovery_artifacts(protocol: AuditProtocol) -> None:
    if protocol.raw["protocol_id"] != "natural-cross-judge-ceiling-recovery-v1":
        return
    recovery = _mapping(protocol.raw.get("recovery"), "recovery")
    failure_path = ROOT / _string(recovery.get("failure_path"), "recovery failure path")
    _assert_hash(failure_path, recovery.get("failure_sha256"), "recovery failure")
    fixture_path = ROOT / _string(recovery.get("fixture_path"), "recovery fixture path")
    _assert_hash(fixture_path, recovery.get("fixture_sha256"), "recovery fixture")
    failure = _mapping(base._read_json(failure_path), "recovery failure")
    expected_failure = {
        "protocol_sha256": recovery.get("source_protocol_sha256"),
        "implementation_commit": recovery.get("source_implementation_commit"),
        "request_id": recovery.get("failed_request_id"),
        "response_accepted": False,
        "automatic_rerun_allowed": False,
    }
    for key, value in expected_failure.items():
        if failure.get(key) != value:
            raise ValueError(f"cross-judge recovery failure drifted: {key}")


def _validate_source_completion(source: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / _string(source.get("completion_path"), "source completion path")
    _assert_hash(path, source.get("completion_sha256"), "source completion")
    completion = _mapping(base._read_json(path), "source completion")
    expected = {
        "complete_bundle": True,
        "provider": "Anthropic",
        "model": "claude-haiku-4-5-20251001",
        "request_count": source.get("request_count"),
        "response_set_sha256": source.get("response_set_sha256"),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"source completion drifted: {key}")
    return completion


@dataclass(frozen=True)
class SourcePlans:
    specs: dict[str, base.CallSpec]
    assignments: tuple[AuditAssignment, ...]
    two_execution: base.Protocol
    two_specs: dict[str, base.CallSpec]
    gpt_execution: base.Protocol
    gpt_specs: dict[str, base.CallSpec]


def _source_plans(protocol: AuditProtocol) -> SourcePlans:
    two_source = _mapping(protocol.source["two_reader"], "two_reader source")
    gpt_source = _mapping(protocol.source["gpt_reader"], "gpt_reader source")
    for source in (two_source, gpt_source):
        _validate_source_completion(source)

    two_judge_path = ROOT / _string(
        two_source["judge_protocol_path"], "two-reader judge protocol path"
    )
    two_execution_path = ROOT / _string(
        two_source["execution_protocol_path"], "two-reader execution protocol path"
    )
    _assert_hash(
        two_judge_path,
        two_source["judge_protocol_sha256"],
        "two-reader judge protocol",
    )
    _assert_hash(
        two_execution_path,
        two_source["execution_protocol_sha256"],
        "two-reader execution protocol",
    )
    two_judge = two_reader.load_judge_protocol(two_judge_path)
    two_execution = two_reader.bind_recovered_judge(
        two_judge,
        base.load_protocol(two_execution_path),
    )
    (
        two_cases,
        two_specs,
        two_assignments,
        *_two_rest,
    ) = two_reader._build_plan(
        judge_protocol=two_judge,
        execution_protocol=two_execution,
        cases_path=two_reader.DEFAULT_CASES,
        source_runtime=ROOT / _string(two_source["reader_runtime"], "reader runtime"),
    )
    source_by_case = {case.case_id: case.source for case in two_cases}
    assignments = [
        AuditAssignment(
            request_id=request_id,
            case_id=case_id,
            reader=reader,
            source=source_by_case[case_id],
            arm=arm,
        )
        for (reader, case_id, arm), request_id in two_assignments.items()
        if arm in PRIMARY_ARMS
    ]

    gpt_reader_path = ROOT / _string(gpt_source["reader_protocol_path"], "GPT reader protocol path")
    gpt_judge_path = ROOT / _string(gpt_source["judge_protocol_path"], "GPT judge protocol path")
    _assert_hash(
        gpt_reader_path,
        gpt_source["reader_protocol_sha256"],
        "GPT reader protocol",
    )
    _assert_hash(
        gpt_judge_path,
        gpt_source["judge_protocol_sha256"],
        "GPT judge protocol",
    )
    (
        gpt_execution,
        gpt_cases,
        gpt_specs,
        gpt_assignments,
        *_gpt_rest,
    ) = gpt._judge_plan(
        gpt.load_protocol(gpt_reader_path),
        gpt.load_judge_protocol(gpt_judge_path),
    )
    gpt_source_by_case = {case.case_id: case.source for case in gpt_cases}
    assignments.extend(
        AuditAssignment(
            request_id=request_id,
            case_id=case_id,
            reader="OpenAI",
            source=gpt_source_by_case[case_id],
            arm=arm,
        )
        for (case_id, arm), request_id in gpt_assignments.items()
    )

    specs = dict(two_specs)
    for request_id, spec in gpt_specs.items():
        prior = specs.setdefault(request_id, spec)
        if prior.payload != spec.payload or prior.schema != spec.schema:
            raise RuntimeError("shared judge request identity has inconsistent payload")
    return SourcePlans(
        specs=specs,
        assignments=tuple(assignments),
        two_execution=two_execution,
        two_specs=two_specs,
        gpt_execution=gpt_execution,
        gpt_specs=gpt_specs,
    )


def _quotas(protocol: AuditProtocol) -> dict[tuple[str, str, str], int]:
    rows = protocol.sampling.get("stratum_quotas")
    if not isinstance(rows, list):
        raise TypeError("sampling.stratum_quotas must be a list")
    quotas = {}
    for raw_row in rows:
        row = _mapping(raw_row, "stratum quota")
        key = (
            _string(row.get("reader"), "quota reader"),
            _string(row.get("source"), "quota source"),
            _string(row.get("arm"), "quota arm"),
        )
        if key in quotas:
            raise ValueError("duplicate stratum quota")
        quotas[key] = _integer(row.get("count"), "quota count", minimum=1)
    expected = {
        (reader, source, arm)
        for reader in protocol.sampling["reader_providers"]
        for source in protocol.sampling["sources"]
        for arm in protocol.sampling["arms"]
    }
    if set(quotas) != expected or sum(quotas.values()) != protocol.sampling["sample_count"]:
        raise ValueError("stratum quotas do not cover the frozen sample panel")
    return quotas


def derive_sample(protocol: AuditProtocol) -> tuple[dict[str, Any], dict[str, base.CallSpec]]:
    plans = _source_plans(protocol)
    selected = select_unique_stratified(
        plans.assignments,
        _quotas(protocol),
        seed=_string(protocol.sampling["seed"], "sampling seed"),
    )
    request_ids = {unit.request_id for unit in selected}
    specs = {request_id: plans.specs[request_id] for request_id in request_ids}
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol.sampling.get("manifest_protocol_id", protocol.raw["protocol_id"]),
        "selection_type": "posthoc_outcome_independent_exact_payload_sample",
        "sampling_seed": protocol.sampling["seed"],
        "selection_fields": protocol.sampling["selection_fields"],
        "sample_count": len(selected),
        "request_id_set_sha256": base._sha256_object(sorted(request_ids)),
        "stratum_counts": [
            {"reader": key[0], "source": key[1], "arm": key[2], "count": value}
            for key, value in sorted(Counter(unit.anchor_stratum for unit in selected).items())
        ],
        "units": [asdict(unit) for unit in selected],
        "answer_reference_or_response_content_included": False,
        "judge_outcomes_included": False,
    }
    return manifest, specs


def freeze_sample(protocol: AuditProtocol, path: Path) -> dict[str, Any]:
    manifest, _specs = derive_sample(protocol)
    expected_set_hash = _string(
        protocol.sampling["expected_request_id_set_sha256"], "expected request set hash"
    )
    if manifest["request_id_set_sha256"] != expected_set_hash:
        raise ValueError("derived sample request set differs from frozen protocol")
    payload = base._canonical_bytes(manifest) + b"\n"
    expected_manifest_hash = _string(
        protocol.sampling["sample_manifest_sha256"], "sample manifest hash"
    )
    if hashlib.sha256(payload).hexdigest() != expected_manifest_hash:
        raise ValueError("derived sample manifest differs from frozen protocol")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() != payload:
        raise ValueError("existing sample manifest differs from frozen selection")
    path.write_bytes(payload)
    return manifest


def _load_frozen_sample(
    protocol: AuditProtocol,
    path: Path = DEFAULT_SAMPLE_MANIFEST,
) -> tuple[tuple[SelectedAuditUnit, ...], dict[str, base.CallSpec], SourcePlans]:
    _assert_hash(path, protocol.sampling["sample_manifest_sha256"], "sample manifest")
    manifest = _mapping(base._read_json(path), "sample manifest")
    derived, specs = derive_sample(protocol)
    if base._canonical_bytes(manifest) != base._canonical_bytes(derived):
        raise ValueError("sample manifest is not the deterministic protocol selection")
    units = tuple(
        SelectedAuditUnit(
            request_id=_string(row["request_id"], "sample request ID"),
            case_id=_string(row["case_id"], "sample case ID"),
            source=_string(row["source"], "sample source"),
            anchor_reader=_string(row["anchor_reader"], "sample reader"),
            anchor_arm=_string(row["anchor_arm"], "sample arm"),
            associated_readers=tuple(row["associated_readers"]),
            associated_arms=tuple(row["associated_arms"]),
            assignment_count=_integer(row["assignment_count"], "assignment count", minimum=1),
        )
        for raw_row in manifest["units"]
        if (row := _mapping(raw_row, "sample unit"))
    )
    return units, specs, _source_plans(protocol)


def _execution_protocol(protocol: AuditProtocol) -> base.Protocol:
    source = two_reader.bind_recovered_judge(
        two_reader.load_judge_protocol(two_reader.DEFAULT_JUDGE_PROTOCOL),
        base.load_protocol(two_reader.DEFAULT_EXECUTION_PROTOCOL),
    )
    prompt_path = ROOT / _string(protocol.judge["prompt_path"], "judge prompt path")
    _assert_hash(prompt_path, protocol.judge["prompt_sha256"], "judge prompt")
    return replace(
        source,
        path=protocol.path,
        raw=protocol.raw,
        protocol_id=str(protocol.raw["protocol_id"]),
        protocol_sha256=protocol.sha256,
        judge=protocol.binding,
        judge_prompt=prompt_path.read_text(encoding="utf-8"),
        maximum_output_tokens_judge=int(protocol.judge["maximum_output_tokens"]),
        timeout_seconds=int(protocol.judge["timeout_seconds"]),
        maximum_transport_retries=0,
        maximum_model_contract_recovery_attempts=0,
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_contract() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *CONTRACT_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("cross-judge provider execution requires committed clean contracts")
    return _git_head()


def validate(protocol: AuditProtocol, sample_manifest: Path) -> dict[str, Any]:
    _validate_recovery_artifacts(protocol)
    units, specs, plans = _load_frozen_sample(protocol, sample_manifest)
    return {
        "protocol_sha256": protocol.sha256,
        "sample_manifest_sha256": base._sha256_file(sample_manifest),
        "sample_count": len(units),
        "unique_request_count": len(specs),
        "source_unique_request_count": len(plans.specs),
        "stratum_count": len({unit.anchor_stratum for unit in units}),
        "judge_provider": protocol.binding.provider,
        "judge_model": protocol.binding.model,
        "fixture_hard_cap_usd": protocol.fixture_cap_usd,
        "total_incremental_hard_cap_usd": protocol.total_cap_usd,
        "posthoc_outcome_independent": True,
    }


def _fixture_path(runtime: Path, binding: base.ProviderBinding) -> Path:
    return base._stage_root(runtime, "fixtures", binding) / "judge.json"


def run_fixture(protocol: AuditProtocol, *, runtime: Path, dotenv: Path) -> dict[str, Any]:
    implementation_commit = _require_clean_contract()
    _validate_recovery_artifacts(protocol)
    execution = _execution_protocol(protocol)
    path = _fixture_path(runtime, execution.judge)
    if path.is_file():
        receipt = _mapping(base._read_json(path), "fixture receipt")
        expected = {
            "protocol_sha256": protocol.sha256,
            "implementation_commit": implementation_commit,
            "provider": "OpenAI",
            "model": protocol.binding.model,
            "synthetic_fixture": True,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"cross-judge fixture drifted: {key}")
        return {**receipt, "checkpoint_reused": True}

    case = base._fixture_case()
    spec = base.CallSpec(
        request_id=case.case_id,
        payload=base.judge_payload(
            case,
            base.ReaderResponse(action="answer", answer="123 Market Street"),
        ),
        schema=base.judge_response_schema(),
    )
    credentials = base.provider_runtime._load_dotenv(dotenv)
    api_key = credentials.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is absent or empty")
    record = base._call_provider(
        protocol=execution,
        stage="fixture-cross-judge",
        binding=execution.judge,
        spec=spec,
        system_prompt=execution.judge_prompt,
        maximum_output_tokens=execution.maximum_output_tokens_judge,
        api_key=api_key,
        parser=base._judge_parser,
        implementation_commit=implementation_commit,
    )
    receipt = {
        **record,
        "stage": "cross_judge",
        "response": None,
        "response_sha256": base._sha256_object(record["response"]),
        "synthetic_fixture": True,
        "fixture_hard_cap_usd": protocol.fixture_cap_usd,
    }
    if float(receipt["cost_usd"]) > protocol.fixture_cap_usd:
        raise RuntimeError("synthetic cross-judge fixture exceeded its hard cap")
    base._write_json(path, receipt)
    return receipt


def _failure_paths(runtime: Path, binding: base.ProviderBinding) -> list[Path]:
    root = base._stage_root(runtime, "cross_judge", binding) / "failures"
    return sorted(root.glob("*.json")) if root.is_dir() else []


def execute(
    protocol: AuditProtocol,
    *,
    sample_manifest: Path,
    runtime: Path,
    dotenv: Path,
) -> dict[str, Any]:
    implementation_commit = _require_clean_contract()
    execution = _execution_protocol(protocol)
    _units, specs, _plans = _load_frozen_sample(protocol, sample_manifest)
    fixture = run_fixture(protocol, runtime=runtime, dotenv=dotenv)
    fixture_cost = float(fixture["cost_usd"])

    existing = {}
    for request_id, spec in specs.items():
        path = base._response_path(runtime, "cross_judge", execution.judge, request_id)
        if path.is_file():
            existing[request_id] = base._load_record(
                path,
                protocol=execution,
                stage="cross_judge",
                binding=execution.judge,
                spec=spec,
                parser=base._judge_parser,
                implementation_commit=implementation_commit,
            )
    prior_failures = _failure_paths(runtime, execution.judge)
    if prior_failures:
        raise RuntimeError("a prior cross-judge failure requires a separate recovery contract")

    credentials = base.provider_runtime._load_dotenv(dotenv)
    api_key = credentials.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is absent or empty")
    spent = (
        protocol.prior_budget_consumption_usd
        + fixture_cost
        + sum(float(record["cost_usd"]) for record in existing.values())
    )
    pending = sorted(set(specs) - set(existing))
    if spent + len(pending) * protocol.call_reservation_usd > protocol.total_cap_usd:
        raise RuntimeError("remaining call reservations exceed the cross-judge hard cap")

    for request_id in pending:
        if spent + protocol.call_reservation_usd > protocol.total_cap_usd:
            raise RuntimeError("cross-judge hard cap cannot reserve the next request")
        spec = specs[request_id]
        try:
            record = base._call_provider(
                protocol=execution,
                stage="cross_judge",
                binding=execution.judge,
                spec=spec,
                system_prompt=execution.judge_prompt,
                maximum_output_tokens=execution.maximum_output_tokens_judge,
                api_key=api_key,
                parser=base._judge_parser,
                implementation_commit=implementation_commit,
            )
        except Exception as error:  # noqa: BLE001 - checkpoint and freeze paid progress.
            failure = {
                "schema_version": 1,
                "protocol_sha256": protocol.sha256,
                "implementation_commit": implementation_commit,
                "provider": execution.judge.provider,
                "model": execution.judge.model,
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "cost_bound_usd": protocol.call_reservation_usd,
                "response_accepted": False,
                "automatic_rerun_allowed": False,
            }
            base._write_json(
                base._failure_path(runtime, "cross_judge", execution.judge, request_id),
                failure,
            )
            raise RuntimeError("cross-judge execution failed and is frozen") from error
        if float(record["cost_usd"]) > protocol.call_reservation_usd:
            raise RuntimeError("one cross-judge response exceeded its reserved cost")
        base._write_json(
            base._response_path(runtime, "cross_judge", execution.judge, request_id),
            record,
        )
        existing[request_id] = record
        spent += float(record["cost_usd"])
        completed = len(existing)
        if completed % 10 == 0 or completed == len(specs):
            print(
                json.dumps(
                    {
                        "stage": "cross_judge",
                        "completed": completed,
                        "total": len(specs),
                        "spent_usd": round(spent, 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if set(existing) != set(specs):
        raise RuntimeError("cross-judge response bundle is incomplete")
    completion = {
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "sample_manifest_sha256": base._sha256_file(sample_manifest),
        "implementation_commit": implementation_commit,
        "provider": execution.judge.provider,
        "model": execution.judge.model,
        "request_count": len(existing),
        "input_tokens": sum(int(row["usage"]["input_tokens"]) for row in existing.values()),
        "output_tokens": sum(int(row["usage"]["output_tokens"]) for row in existing.values()),
        "response_cost_usd": sum(float(row["cost_usd"]) for row in existing.values()),
        "fixture_cost_usd": fixture_cost,
        "prior_budget_consumption_usd": protocol.prior_budget_consumption_usd,
        "total_incremental_cost_usd": spent,
        "total_incremental_hard_cap_usd": protocol.total_cap_usd,
        "response_set_sha256": base._sha256_object(
            [
                {
                    "request_id": request_id,
                    "response_sha256": base._sha256_object(existing[request_id]["response"]),
                }
                for request_id in sorted(existing)
            ]
        ),
        "complete_bundle": True,
        "raw_payload_or_response_content_included": False,
    }
    base._write_json(runtime / "completion.json", completion)
    return completion


def _source_judgment(
    unit: SelectedAuditUnit,
    plans: SourcePlans,
    protocol: AuditProtocol,
) -> base.JudgeResponse:
    if unit.anchor_reader == "OpenAI":
        source = _mapping(protocol.source["gpt_reader"], "GPT source")
        execution = plans.gpt_execution
        specs = plans.gpt_specs
    else:
        source = _mapping(protocol.source["two_reader"], "two-reader source")
        execution = plans.two_execution
        specs = plans.two_specs
    spec = specs[unit.request_id]
    completion = _validate_source_completion(source)
    record = base._load_record(
        base._response_path(
            ROOT / _string(source["judge_runtime"], "source judge runtime"),
            "judge",
            execution.judge,
            unit.request_id,
        ),
        protocol=execution,
        stage="judge",
        binding=execution.judge,
        spec=spec,
        parser=base._judge_parser,
        implementation_commit=str(completion["implementation_commit"]),
    )
    return base.JudgeResponse.from_mapping(record["response"])


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _groups(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups = [("overall", "all", list(rows))]
    fields = {
        "reader": "anchor_reader",
        "source": "source",
        "route": "anchor_arm",
    }
    for group_type, field in fields.items():
        for value in sorted({str(row[field]) for row in rows}):
            groups.append((group_type, value, [row for row in rows if row[field] == value]))
    for key in sorted({(row["anchor_reader"], row["source"], row["anchor_arm"]) for row in rows}):
        groups.append(
            (
                "stratum",
                "/".join(key),
                [
                    row
                    for row in rows
                    if (row["anchor_reader"], row["source"], row["anchor_arm"]) == key
                ],
            )
        )
    return groups


def analyze(
    protocol: AuditProtocol,
    *,
    sample_manifest: Path,
    runtime: Path,
    output: Path,
) -> dict[str, Any]:
    _require_clean_contract()
    _validate_recovery_artifacts(protocol)
    units, specs, plans = _load_frozen_sample(protocol, sample_manifest)
    execution = _execution_protocol(protocol)
    completion_path = runtime / "completion.json"
    completion = _mapping(base._read_json(completion_path), "cross-judge completion")
    expected_completion = {
        "protocol_sha256": protocol.sha256,
        "sample_manifest_sha256": base._sha256_file(sample_manifest),
        "provider": "OpenAI",
        "model": protocol.binding.model,
        "request_count": 200,
        "complete_bundle": True,
    }
    for key, value in expected_completion.items():
        if completion.get(key) != value:
            raise ValueError(f"cross-judge completion drifted: {key}")

    paired = []
    for unit in units:
        cheap = _source_judgment(unit, plans, protocol)
        record = base._load_record(
            base._response_path(runtime, "cross_judge", execution.judge, unit.request_id),
            protocol=execution,
            stage="cross_judge",
            binding=execution.judge,
            spec=specs[unit.request_id],
            parser=base._judge_parser,
            implementation_commit=str(completion["implementation_commit"]),
        )
        strong = base.JudgeResponse.from_mapping(record["response"])
        paired.append(
            {
                **asdict(unit),
                "cheap_answer_correct": cheap.answer_correct,
                "strong_answer_correct": strong.answer_correct,
                "cheap_answer_quality": cheap.answer_quality,
                "strong_answer_quality": strong.answer_quality,
                "cheap_protected_disclosure": cheap.protected_disclosure,
                "strong_protected_disclosure": strong.protected_disclosure,
                "cheap_stale_disclosure": cheap.stale_disclosure,
                "strong_stale_disclosure": strong.stale_disclosure,
            }
        )

    private_path = runtime / "private_scored_pairs.jsonl"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in paired),
        encoding="utf-8",
    )
    bootstrap_iterations = _integer(
        protocol.analysis["bootstrap_iterations"], "bootstrap iterations", minimum=1
    )
    bootstrap_seed = _integer(protocol.analysis["bootstrap_seed"], "bootstrap seed")
    agreement_rows = []
    quality_rows = []
    confusion_rows = []
    secondary_rows = []
    for group_index, (group_type, group_value, group) in enumerate(_groups(paired)):
        cheap = [bool(row["cheap_answer_correct"]) for row in group]
        strong = [bool(row["strong_answer_correct"]) for row in group]
        metrics = binary_agreement(cheap, strong)
        intervals = bootstrap_binary_agreement(
            cheap,
            strong,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + group_index,
        )
        agreement_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "n": metrics["n"],
                "exact_agreement": metrics["exact_agreement"],
                "exact_agreement_ci_low": intervals["exact_agreement"][0],
                "exact_agreement_ci_high": intervals["exact_agreement"][1],
                "cohen_kappa": metrics["cohen_kappa"],
                "cohen_kappa_ci_low": intervals["cohen_kappa"][0],
                "cohen_kappa_ci_high": intervals["cohen_kappa"][1],
                "gwet_ac1": metrics["gwet_ac1"],
                "gwet_ac1_ci_low": intervals["gwet_ac1"][0],
                "gwet_ac1_ci_high": intervals["gwet_ac1"][1],
            }
        )
        confusion_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "n": metrics["n"],
                "both_correct": metrics["both_true"],
                "both_incorrect": metrics["both_false"],
                "cheap_correct_strong_incorrect": metrics["first_true_second_false"],
                "cheap_incorrect_strong_correct": metrics["first_false_second_true"],
            }
        )
        quality_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                **quality_agreement(
                    [int(row["cheap_answer_quality"]) for row in group],
                    [int(row["strong_answer_quality"]) for row in group],
                ),
            }
        )
        for field in ("protected_disclosure", "stale_disclosure"):
            secondary = binary_agreement(
                [bool(row[f"cheap_{field}"]) for row in group],
                [bool(row[f"strong_{field}"]) for row in group],
            )
            secondary_rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "field": field,
                    "n": secondary["n"],
                    "exact_agreement": secondary["exact_agreement"],
                    "cohen_kappa": secondary["cohen_kappa"],
                    "gwet_ac1": secondary["gwet_ac1"],
                    "cheap_positive_strong_negative": secondary["first_true_second_false"],
                    "cheap_negative_strong_positive": secondary["first_false_second_true"],
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "answer_correct_agreement.csv", agreement_rows)
    _write_csv(output / "answer_correct_confusion.csv", confusion_rows)
    _write_csv(output / "answer_quality_agreement.csv", quality_rows)
    _write_csv(output / "secondary_label_agreement.csv", secondary_rows)
    profile = [
        {
            "reader": key[0],
            "source": key[1],
            "route": key[2],
            "selected_unique_payloads": value,
        }
        for key, value in sorted(Counter(unit.anchor_stratum for unit in units).items())
    ]
    _write_csv(output / "sample_profile.csv", profile)

    overall = agreement_rows[0]
    quality = quality_rows[0]
    passed = float(overall["exact_agreement"]) >= float(protocol.analysis["agreement_gate"])
    recovery_note = (
        "The original GPT-5 Pro attempt accepted zero benchmark labels before a "
        "512-token completion-ceiling failure; the frozen recovery migrated to GPT-5.1 "
        "before outcome inspection.\n\n"
        if protocol.raw["protocol_id"] == "natural-cross-judge-ceiling-recovery-v1"
        else ""
    )
    summary = f"""# Natural cross-judge audit

## Result

The blinded `{protocol.binding.model}` audit agreed with the frozen Claude Haiku 4.5 labels on
**{float(overall["exact_agreement"]):.3f}** of 200 exact-deduplicated outputs
(item-bootstrap 95% CI **[{float(overall["exact_agreement_ci_low"]):.3f},
{float(overall["exact_agreement_ci_high"]):.3f}]**). Cohen's kappa was
**{float(overall["cohen_kappa"]):.3f}** and Gwet's AC1 was
**{float(overall["gwet_ac1"]):.3f}**. The pre-existing continuation threshold of
0.85 was **{"met" if passed else "not met"}**.

For the 0--10 answer-quality score, exact agreement was
**{float(quality["exact_agreement"]):.3f}**, agreement within one point was
**{float(quality["within_one_agreement"]):.3f}**, and mean absolute error was
**{float(quality["mean_absolute_error"]):.3f}**.

## Scope

{recovery_note}This is a post-hoc but outcome-independent robustness audit, not an independently
preregistered replication. The 200 payloads were selected without inspecting reader
answers or either judge's labels, using near-equal quotas over three readers, two
sources, and the three common primary routes. Exact duplicate judge payloads were
called once. Reader estimates remain separate and no model pooling is introduced.

The primary endpoint is semantic answer-correctness agreement. Protected- and
stale-disclosure agreement is secondary because many sampled payloads are negative or
not disclosure-evaluable. No answer, reference answer, prompt, or judge rationale is
included in the public artifacts.
"""
    (output / "audit_summary.md").write_text(summary, encoding="utf-8")
    public_receipt = {
        key: completion[key]
        for key in (
            "schema_version",
            "protocol_sha256",
            "sample_manifest_sha256",
            "implementation_commit",
            "provider",
            "model",
            "request_count",
            "input_tokens",
            "output_tokens",
            "response_cost_usd",
            "fixture_cost_usd",
            "prior_budget_consumption_usd",
            "total_incremental_cost_usd",
            "total_incremental_hard_cap_usd",
            "response_set_sha256",
            "complete_bundle",
        )
    }
    public_receipt.update(
        {
            "private_scored_pair_sha256": base._sha256_file(private_path),
            "raw_payload_or_response_content_included": False,
            "official_benchmark_result": False,
        }
    )
    base._write_json(output / "execution_receipt.json", public_receipt)
    artifacts = {
        path.name: base._sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "status": "complete_posthoc_outcome_independent_cross_judge_audit",
        "protocol_sha256": protocol.sha256,
        "sample_manifest_sha256": base._sha256_file(sample_manifest),
        "sample_count": len(units),
        "primary_metric": "answer_correct_exact_agreement",
        "primary_value": overall["exact_agreement"],
        "agreement_gate": protocol.analysis["agreement_gate"],
        "agreement_gate_met": passed,
        "reader_estimates_pooled": False,
        "independently_preregistered_replication": False,
        "official_benchmark_claim": False,
        "raw_content_included": False,
        "artifacts": artifacts,
    }
    base._write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("derive-sample", "freeze-sample", "validate", "fixture", "execute", "analyze"),
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "derive-sample":
        manifest, _specs = derive_sample(protocol)
        result = {
            "sample_count": manifest["sample_count"],
            "request_id_set_sha256": manifest["request_id_set_sha256"],
            "sample_manifest_sha256": hashlib.sha256(
                base._canonical_bytes(manifest) + b"\n"
            ).hexdigest(),
            "stratum_counts": manifest["stratum_counts"],
        }
    elif args.command == "freeze-sample":
        result = freeze_sample(protocol, args.sample_manifest)
    elif args.command == "validate":
        result = validate(protocol, args.sample_manifest)
    elif args.command == "fixture":
        result = run_fixture(protocol, runtime=args.runtime, dotenv=args.dotenv)
    elif args.command == "execute":
        result = execute(
            protocol,
            sample_manifest=args.sample_manifest,
            runtime=args.runtime,
            dotenv=args.dotenv,
        )
    else:
        result = analyze(
            protocol,
            sample_manifest=args.sample_manifest,
            runtime=args.runtime,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
