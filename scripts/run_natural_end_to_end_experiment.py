"""Validate, execute, and score the natural route-to-reader experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_inferred_admissibility_experiment as provider_runtime  # noqa: E402
from verify_agent_memory import inferred_admissibility as verifier_contract  # noqa: E402
from verify_agent_memory.natural_end_to_end import (  # noqa: E402
    ARMS,
    JudgeResponse,
    NaturalEndToEndCase,
    ReaderResponse,
    build_reader_requests,
    deterministic_answer_metrics,
    judge_payload,
    judge_response_schema,
    load_cases,
    matched_route_score,
    reader_response_schema,
    route_candidates,
    truncate_utf8,
)

DEFAULT_PROTOCOL = ROOT / "experiments" / "natural_end_to_end_protocol.json"
TWO_READER_PROTOCOL = ROOT / "experiments" / "natural_end_to_end_two_reader_protocol.json"
DEFAULT_CASES = ROOT / "tmp" / "natural_end_to_end" / "cases.jsonl.gz"
DEFAULT_MATERIALIZATION = ROOT / "tmp" / "natural_end_to_end" / "materialization_manifest.json"
DEFAULT_RUNTIME = ROOT / "tmp" / "natural_end_to_end" / "provider_runtime"
DEFAULT_OUTPUT = ROOT / "results" / "natural_end_to_end"
DEFAULT_DOTENV = ROOT.parent / "bomi-codex-starter" / ".env"
CONTRACT_PATHS = (
    "experiments/natural_end_to_end_protocol.json",
    "experiments/natural_end_to_end_two_reader_protocol.json",
    "experiments/prompts/inferred_admissibility_v1.txt",
    "experiments/prompts/natural_end_to_end_judge_v1.txt",
    "experiments/prompts/natural_end_to_end_reader_v1.txt",
    "scripts/run_inferred_admissibility_experiment.py",
    "scripts/import_natural_verifier_bundle.py",
    "scripts/import_natural_two_reader_checkpoints.py",
    "scripts/run_natural_end_to_end_experiment.py",
    "src/verify_agent_memory/inferred_admissibility.py",
    "src/verify_agent_memory/natural_end_to_end.py",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer at least {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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
        raise RuntimeError("provider execution requires committed, clean contract paths")
    return _git_head()


@dataclass(frozen=True)
class ProviderBinding:
    provider: str
    model: str
    api_surface: str
    input_usd_per_million: float
    output_usd_per_million: float
    hard_cap_usd: float
    controls: Mapping[str, str]

    def runtime_binding(self) -> provider_runtime.ProviderBinding:
        return provider_runtime.ProviderBinding(**asdict(self))


@dataclass(frozen=True)
class Protocol:
    path: Path
    raw: Mapping[str, Any]
    protocol_id: str
    protocol_sha256: str
    reader_prompt: str
    verifier_prompt: str
    judge_prompt: str
    verifier: ProviderBinding
    readers: tuple[ProviderBinding, ...]
    judge: ProviderBinding
    verifier_threshold: float
    target_recall: float
    verifier_memory_max_bytes: int
    maximum_output_tokens_verifier: int
    maximum_output_tokens_reader: int
    maximum_output_tokens_judge: int
    timeout_seconds: int
    maximum_transport_retries: int
    reader_incremental_hard_caps: Mapping[str, float]


def _prompt(value: object, label: str) -> str:
    row = _mapping(value, label)
    path = ROOT / _string(row.get("path"), f"{label}.path")
    expected_hash = _string(row.get("sha256"), f"{label}.sha256")
    if not path.is_file() or _sha256_file(path) != expected_hash:
        raise ValueError(f"{label} file hash drifted")
    return path.read_text(encoding="utf-8")


def _binding(value: object, label: str) -> ProviderBinding:
    row = _mapping(value, label)
    known = {
        "provider",
        "model",
        "api_surface",
        "input_usd_per_million",
        "output_usd_per_million",
        "hard_cap_usd",
        "maximum_calls",
        "prompt",
        "violation_threshold",
        "threshold_source",
        "arm_and_reader_blinded",
        "maximum_calls_before_exact_prompt_deduplication",
        "execution_locked_until_deterministic_gate",
    }
    return ProviderBinding(
        provider=_string(row.get("provider"), f"{label}.provider"),
        model=_string(row.get("model"), f"{label}.model"),
        api_surface=_string(row.get("api_surface"), f"{label}.api_surface"),
        input_usd_per_million=_number(
            row.get("input_usd_per_million"),
            f"{label}.input_usd_per_million",
        ),
        output_usd_per_million=_number(
            row.get("output_usd_per_million"),
            f"{label}.output_usd_per_million",
        ),
        hard_cap_usd=_number(row.get("hard_cap_usd"), f"{label}.hard_cap_usd"),
        controls={
            key: _string(item, f"{label}.{key}") for key, item in row.items() if key not in known
        },
    )


def load_protocol(path: Path) -> Protocol:
    raw = _read_json(path)
    protocol_id = _string(raw.get("protocol_id"), "protocol.protocol_id")
    if raw.get("schema_version") != 1 or protocol_id not in {
        "natural-heldout-route-to-reader-v2",
        "natural-heldout-route-to-reader-two-reader-v3",
    }:
        raise ValueError("natural end-to-end protocol identity drifted")
    expected_status = {
        "natural-heldout-route-to-reader-v2": (
            "frozen_nonofficial_same_population_end_to_end_evaluation"
        ),
        "natural-heldout-route-to-reader-two-reader-v3": (
            "frozen_nonofficial_cost_aware_two_reader_completion"
        ),
    }[protocol_id]
    if raw.get("status") != expected_status:
        raise ValueError("natural end-to-end protocol status drifted")
    source = _mapping(raw.get("source_contract"), "protocol.source_contract")
    if source.get("query_counts") != {"rhelm": 523, "memops": 3244, "total": 3767}:
        raise ValueError("natural end-to-end query counts drifted")
    retrieval = _mapping(raw.get("retrieval"), "protocol.retrieval")
    if tuple(_sequence(retrieval.get("arms"), "protocol.retrieval.arms")) != ARMS:
        raise ValueError("natural end-to-end route arms drifted")
    if retrieval.get("gate_semantics") != (
        "delete_only_from_frozen_top20_without_backfill_or_reranking"
    ):
        raise ValueError("natural end-to-end gate semantics drifted")
    verifier = _mapping(raw.get("text_verifier"), "protocol.text_verifier")
    reader = _mapping(raw.get("reader"), "protocol.reader")
    judge = _mapping(raw.get("judge"), "protocol.judge")
    execution = _mapping(raw.get("execution"), "protocol.execution")
    readers = tuple(
        _binding(item, f"protocol.reader.providers[{index}]")
        for index, item in enumerate(_sequence(reader.get("providers"), "reader.providers"))
    )
    expected_readers = {
        "natural-heldout-route-to-reader-v2": (
            "OpenAI",
            "Anthropic",
            "Gemini",
            "DeepSeek",
        ),
        "natural-heldout-route-to-reader-two-reader-v3": ("Gemini", "DeepSeek"),
    }[protocol_id]
    if tuple(binding.provider for binding in readers) != expected_readers:
        raise ValueError("natural end-to-end reader panel drifted")
    judge_binding = _binding(judge, "protocol.judge")
    expected_judge = {
        "natural-heldout-route-to-reader-v2": ("OpenAI", "gpt-5.6-sol"),
        "natural-heldout-route-to-reader-two-reader-v3": (
            "Anthropic",
            "claude-haiku-4-5",
        ),
    }[protocol_id]
    if (judge_binding.provider, judge_binding.model) != expected_judge:
        raise ValueError("natural end-to-end judge binding drifted")
    raw_incremental_caps = _mapping(
        execution.get("reader_incremental_hard_cap_usd", {}),
        "protocol.execution.reader_incremental_hard_cap_usd",
    )
    incremental_caps = {
        provider: _number(value, f"reader_incremental_hard_cap_usd.{provider}")
        for provider, value in raw_incremental_caps.items()
    }
    if protocol_id == "natural-heldout-route-to-reader-v2" and incremental_caps:
        raise ValueError("v2 protocol cannot define incremental reader caps")
    if protocol_id == "natural-heldout-route-to-reader-two-reader-v3":
        if incremental_caps != {"Gemini": 29.0, "DeepSeek": 16.0}:
            raise ValueError("two-reader incremental caps must bind Gemini $29 and DeepSeek $16")
        if execution.get("maximum_model_contract_recovery_attempts") != 1:
            raise ValueError("two-reader model-contract recovery limit drifted")
        if execution.get("outcome_selective_rerun") is not False:
            raise ValueError("two-reader outcome-selective rerun rule drifted")
    if sum(binding.hard_cap_usd for binding in readers) + _number(
        verifier.get("hard_cap_usd"),
        "text_verifier.hard_cap_usd",
    ) + _number(judge.get("hard_cap_usd"), "judge.hard_cap_usd") > _number(
        execution.get("total_hard_cap_usd"),
        "execution.total_hard_cap_usd",
    ):
        raise ValueError("stage hard caps exceed the total hard cap")
    return Protocol(
        path=path,
        raw=raw,
        protocol_id=protocol_id,
        protocol_sha256=_sha256_file(path),
        reader_prompt=_prompt(reader.get("prompt"), "protocol.reader.prompt"),
        verifier_prompt=_prompt(verifier.get("prompt"), "protocol.text_verifier.prompt"),
        judge_prompt=_prompt(judge.get("prompt"), "protocol.judge.prompt"),
        verifier=_binding(verifier, "protocol.text_verifier"),
        readers=readers,
        judge=judge_binding,
        verifier_threshold=_number(
            verifier.get("violation_threshold"),
            "text_verifier.violation_threshold",
        ),
        target_recall=_number(retrieval.get("target_recall"), "retrieval.target_recall"),
        verifier_memory_max_bytes=_integer(
            retrieval.get("verifier_memory_max_utf8_bytes"),
            "retrieval.verifier_memory_max_utf8_bytes",
            minimum=1,
        ),
        maximum_output_tokens_verifier=_integer(
            execution.get("maximum_output_tokens_verifier"),
            "execution.maximum_output_tokens_verifier",
            minimum=1,
        ),
        maximum_output_tokens_reader=_integer(
            execution.get("maximum_output_tokens_reader"),
            "execution.maximum_output_tokens_reader",
            minimum=1,
        ),
        maximum_output_tokens_judge=_integer(
            execution.get("maximum_output_tokens_judge"),
            "execution.maximum_output_tokens_judge",
            minimum=1,
        ),
        timeout_seconds=_integer(
            execution.get("timeout_seconds"),
            "execution.timeout_seconds",
            minimum=1,
        ),
        maximum_transport_retries=_integer(
            execution.get("maximum_transport_retries"),
            "execution.maximum_transport_retries",
        ),
        reader_incremental_hard_caps=incremental_caps,
    )


def _provider_slug(binding: ProviderBinding) -> str:
    return f"{binding.provider}-{binding.model}".casefold().replace("/", "-")


@dataclass(frozen=True)
class CallSpec:
    request_id: str
    payload: str
    schema: Mapping[str, object]


def _verifier_case(
    case: NaturalEndToEndCase,
    maximum_bytes: int,
) -> verifier_contract.InferenceCase:
    candidates = tuple(
        verifier_contract.InferenceCandidate(
            candidate_key=f"c{index:02d}",
            rank=index,
            text=truncate_utf8(candidate.text, maximum_bytes),
            visible_order=candidate.visible_order,
            required_evidence=candidate.required_evidence,
            released_policy_allowed=candidate.policy_allowed,
            released_lifecycle_compatible=candidate.lifecycle_compatible,
        )
        for index, candidate in enumerate(case.namespace_candidates, start=1)
    )
    return verifier_contract.InferenceCase(
        case_id=case.case_id,
        source=case.source,
        group_id=case.group_id,
        role="analysis",
        query_text=case.query_text,
        query_visible_time=case.query_visible_time,
        released_query_intent=case.query_intent,
        anchor_total=case.anchor_total,
        candidates=candidates,
    )


def _verifier_specs(
    cases: Sequence[NaturalEndToEndCase],
    protocol: Protocol,
) -> tuple[dict[str, CallSpec], dict[str, verifier_contract.InferenceCase]]:
    specs = {}
    inference_cases = {}
    for case in cases:
        inference_case = _verifier_case(case, protocol.verifier_memory_max_bytes)
        inference_cases[case.case_id] = inference_case
        specs[case.case_id] = CallSpec(
            request_id=case.case_id,
            payload=verifier_contract.prompt_payload(inference_case),
            schema=verifier_contract.response_json_schema(len(inference_case.candidates)),
        )
    return specs, inference_cases


def _stage_root(runtime: Path, stage: str, binding: ProviderBinding) -> Path:
    return runtime / stage / _provider_slug(binding)


def _response_path(runtime: Path, stage: str, binding: ProviderBinding, request_id: str) -> Path:
    return _stage_root(runtime, stage, binding) / "responses" / f"{request_id}.json"


def _completion_path(runtime: Path, stage: str, binding: ProviderBinding) -> Path:
    return _stage_root(runtime, stage, binding) / "completion.json"


def _failure_path(runtime: Path, stage: str, binding: ProviderBinding, request_id: str) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return _stage_root(runtime, stage, binding) / "failures" / f"{request_id}-{timestamp}.json"


def _lock_path(runtime: Path, stage: str, binding: ProviderBinding) -> Path:
    return runtime / ".locks" / f"{stage}-{_provider_slug(binding)}.lock"


@contextmanager
def _exclusive_stage_lock(
    runtime: Path,
    stage: str,
    binding: ProviderBinding,
):
    """Permit only one writer for a provider stage and fail closed on stale locks."""
    path = _lock_path(runtime, stage, binding)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"provider stage is already locked: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "stage": stage,
                    "provider": binding.provider,
                    "model": binding.model,
                    "pid": os.getpid(),
                    "created_unix_ns": time.time_ns(),
                    "token": token,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
        yield
    finally:
        try:
            current = _read_json(path)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            current = None
        if isinstance(current, Mapping) and current.get("token") == token:
            path.unlink()


def _record_cost(record: Mapping[str, Any]) -> float:
    return _number(record.get("cost_usd"), "response.cost_usd")


def _load_record(
    path: Path,
    *,
    protocol: Protocol,
    stage: str,
    binding: ProviderBinding,
    spec: CallSpec,
    parser: Callable[[object, str], object],
) -> Mapping[str, Any]:
    row = _read_json(path)
    expected = {
        "schema_version": 1,
        "protocol_sha256": protocol.protocol_sha256,
        "implementation_commit": _git_head(),
        "stage": stage,
        "provider": binding.provider,
        "model": binding.model,
        "request_id": spec.request_id,
        "payload_sha256": hashlib.sha256(spec.payload.encode("utf-8")).hexdigest(),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"saved response binding drifted: {path}/{key}")
    parser(row.get("response"), spec.request_id)
    _record_cost(row)
    return row


def _conservative_call_cost(
    spec: CallSpec,
    binding: ProviderBinding,
    *,
    system_prompt: str,
    maximum_output_tokens: int,
) -> float:
    input_bytes = len(system_prompt.encode("utf-8")) + len(spec.payload.encode("utf-8"))
    schema_bytes = len(_canonical_bytes(spec.schema))
    return (
        (input_bytes + schema_bytes) * binding.input_usd_per_million
        + maximum_output_tokens * binding.output_usd_per_million
    ) / 1_000_000


def _retryable_execution_error(error: BaseException) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    message = str(error)
    if message in {"provider transport failure", "provider request failed"}:
        return True
    if not message.startswith("provider HTTP failure: "):
        return False
    try:
        status = int(message.rsplit(": ", 1)[1])
    except ValueError:
        return False
    return status == 429 or status >= 500


def _call_provider(
    *,
    protocol: Protocol,
    stage: str,
    binding: ProviderBinding,
    spec: CallSpec,
    system_prompt: str,
    maximum_output_tokens: int,
    api_key: str,
    parser: Callable[[object, str], object],
    implementation_commit: str,
) -> Mapping[str, Any]:
    provider_call = provider_runtime.PROVIDER_CALLS[binding.provider]
    response, usage, attempts, latency_ms, request_body_sha256 = provider_call(
        binding.runtime_binding(),
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=spec.payload,
        schema=spec.schema,
        max_output_tokens=maximum_output_tokens,
        timeout_seconds=protocol.timeout_seconds,
        max_retries=protocol.maximum_transport_retries,
    )
    parser(response, spec.request_id)
    cost = (
        usage["input_tokens"] * binding.input_usd_per_million
        + usage["output_tokens"] * binding.output_usd_per_million
    ) / 1_000_000
    return {
        "schema_version": 1,
        "protocol_sha256": protocol.protocol_sha256,
        "implementation_commit": implementation_commit,
        "stage": stage,
        "provider": binding.provider,
        "model": binding.model,
        "request_id": spec.request_id,
        "payload_sha256": hashlib.sha256(spec.payload.encode("utf-8")).hexdigest(),
        "request_body_sha256": request_body_sha256,
        "response": response,
        "usage": usage,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "cost_usd": cost,
    }


def _execute_specs(
    *,
    protocol: Protocol,
    runtime: Path,
    stage: str,
    binding: ProviderBinding,
    specs: Mapping[str, CallSpec],
    system_prompt: str,
    maximum_output_tokens: int,
    dotenv: Path,
    workers: int,
    parser: Callable[[object, str], object],
    incremental_cost_cap_usd: float | None = None,
) -> dict[str, object]:
    with _exclusive_stage_lock(runtime, stage, binding):
        return _execute_specs_locked(
            protocol=protocol,
            runtime=runtime,
            stage=stage,
            binding=binding,
            specs=specs,
            system_prompt=system_prompt,
            maximum_output_tokens=maximum_output_tokens,
            dotenv=dotenv,
            workers=workers,
            parser=parser,
            incremental_cost_cap_usd=incremental_cost_cap_usd,
        )


def _execute_specs_locked(
    *,
    protocol: Protocol,
    runtime: Path,
    stage: str,
    binding: ProviderBinding,
    specs: Mapping[str, CallSpec],
    system_prompt: str,
    maximum_output_tokens: int,
    dotenv: Path,
    workers: int,
    parser: Callable[[object, str], object],
    incremental_cost_cap_usd: float | None = None,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    implementation_commit = _require_clean_contract()
    fixture = _stage_root(runtime, "fixtures", binding) / f"{stage}.json"
    if not fixture.is_file():
        raise RuntimeError(f"provider fixture is required before {stage}: {fixture}")
    credentials = provider_runtime._load_dotenv(dotenv)
    credential_key = provider_runtime._credential_key(binding.provider)
    api_key = credentials.get(credential_key)
    if not api_key:
        raise RuntimeError(f"{credential_key} is absent or empty")

    existing: dict[str, Mapping[str, Any]] = {}
    for request_id, spec in specs.items():
        path = _response_path(runtime, stage, binding, request_id)
        if path.is_file():
            existing[request_id] = _load_record(
                path,
                protocol=protocol,
                stage=stage,
                binding=binding,
                spec=spec,
                parser=parser,
            )
    spent = sum(_record_cost(record) for record in existing.values())
    incremental_spent = sum(
        _record_cost(record) for record in existing.values() if "compatibility_source" not in record
    )
    failure_budget_spent = 0.0
    terminal_failure_ids = set()
    failure_root = _stage_root(runtime, stage, binding) / "failures"
    for failure_path in failure_root.glob("*.json") if failure_root.is_dir() else ():
        failure = _read_json(failure_path)
        if (
            failure.get("protocol_sha256") == protocol.protocol_sha256
            and failure.get("provider") == binding.provider
            and failure.get("model") == binding.model
        ):
            failure_budget_spent += _number(
                failure.get("cost_bound_usd", 0),
                f"{failure_path}.cost_bound_usd",
            )
            request_id = failure.get("request_id")
            if (
                isinstance(request_id, str)
                and request_id in specs
                and request_id not in existing
                and failure.get("retry_eligible") is False
            ):
                terminal_failure_ids.add(request_id)
    if terminal_failure_ids:
        raise RuntimeError(
            f"{stage}/{binding.provider} has {len(terminal_failure_ids)} terminal "
            "contract failure(s); an amended protocol is required"
        )
    incremental_budget_spent = incremental_spent + failure_budget_spent
    if incremental_cost_cap_usd is not None and incremental_budget_spent > incremental_cost_cap_usd:
        raise RuntimeError(
            f"recorded incremental {stage} spend ${incremental_budget_spent:.2f} "
            f"exceeds {binding.provider} cap ${incremental_cost_cap_usd:.2f}"
        )
    pending_ids = sorted(set(specs) - set(existing))
    pending_bound = sum(
        _conservative_call_cost(
            specs[request_id],
            binding,
            system_prompt=system_prompt,
            maximum_output_tokens=maximum_output_tokens,
        )
        for request_id in pending_ids
    )
    if spent + pending_bound > binding.hard_cap_usd:
        raise RuntimeError(
            f"conservative {stage} bound ${spent + pending_bound:.2f} exceeds "
            f"{binding.provider} cap ${binding.hard_cap_usd:.2f}"
        )
    if not pending_ids:
        return _complete_stage(protocol, runtime, stage, binding, specs, existing)

    failures = []
    completed = len(existing)
    pending_queue = deque(pending_ids)
    reservations: dict[Future[Mapping[str, Any]], float] = {}

    def submit(
        executor: ThreadPoolExecutor,
        request_id: str,
    ) -> Future[Mapping[str, Any]] | None:
        call_bound = _conservative_call_cost(
            specs[request_id],
            binding,
            system_prompt=system_prompt,
            maximum_output_tokens=maximum_output_tokens,
        )
        if incremental_cost_cap_usd is not None and (
            incremental_budget_spent + sum(reservations.values()) + call_bound
            > incremental_cost_cap_usd
        ):
            return None
        future = executor.submit(
            _call_provider,
            protocol=protocol,
            stage=stage,
            binding=binding,
            spec=specs[request_id],
            system_prompt=system_prompt,
            maximum_output_tokens=maximum_output_tokens,
            api_key=api_key,
            parser=parser,
            implementation_commit=implementation_commit,
        )
        reservations[future] = call_bound
        return future

    def fill(executor: ThreadPoolExecutor, futures: dict[Future[Mapping[str, Any]], str]) -> None:
        while pending_queue and len(futures) < workers:
            request_id = pending_queue[0]
            future = submit(executor, request_id)
            if future is None:
                return
            pending_queue.popleft()
            futures[future] = request_id

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[Mapping[str, Any]], str] = {}
        fill(executor, futures)
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                request_id = futures.pop(future)
                call_bound = reservations.pop(future)
                try:
                    record = future.result()
                    _write_json(_response_path(runtime, stage, binding, request_id), record)
                    existing[request_id] = record
                    completed += 1
                    incremental_budget_spent += _record_cost(record)
                    if completed % 25 == 0 or completed == len(specs):
                        print(
                            json.dumps(
                                {
                                    "stage": stage,
                                    "provider": binding.provider,
                                    "completed": completed,
                                    "total": len(specs),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001 - preserve partial paid progress.
                    retry_eligible = _retryable_execution_error(exc)
                    failure_cost_bound = (
                        0.0
                        if isinstance(exc, RuntimeError)
                        and str(exc) == "provider HTTP failure: 429"
                        else call_bound
                    )
                    incremental_budget_spent += failure_cost_bound
                    failure = {
                        "schema_version": 1,
                        "protocol_sha256": protocol.protocol_sha256,
                        "stage": stage,
                        "provider": binding.provider,
                        "model": binding.model,
                        "request_id": request_id,
                        "implementation_commit": implementation_commit,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "cost_bound_usd": failure_cost_bound,
                        "response_accepted": False,
                        "retry_eligible": retry_eligible,
                        "automatic_rerun_allowed": retry_eligible,
                    }
                    _write_json(_failure_path(runtime, stage, binding, request_id), failure)
                    failures.append(failure)
            if failures:
                continue
            fill(executor, futures)
    if failures:
        raise RuntimeError(
            f"{stage}/{binding.provider} stopped after {len(failures)} failure(s); "
            "successful calls remain checkpointed"
        )
    if pending_queue:
        if incremental_cost_cap_usd is None:
            raise RuntimeError(f"{stage}/{binding.provider} stopped with pending requests")
        raise RuntimeError(
            f"incremental {stage} cap ${incremental_cost_cap_usd:.2f} cannot reserve "
            f"the next {binding.provider} request; successful calls remain checkpointed"
        )
    return _complete_stage(protocol, runtime, stage, binding, specs, existing)


def _complete_stage(
    protocol: Protocol,
    runtime: Path,
    stage: str,
    binding: ProviderBinding,
    specs: Mapping[str, CallSpec],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    if set(records) != set(specs):
        raise RuntimeError("cannot complete a partial provider stage")
    completion = {
        "schema_version": 1,
        "protocol_sha256": protocol.protocol_sha256,
        "implementation_commit": _git_head(),
        "stage": stage,
        "provider": binding.provider,
        "model": binding.model,
        "request_count": len(records),
        "input_tokens": sum(int(record["usage"]["input_tokens"]) for record in records.values()),
        "output_tokens": sum(int(record["usage"]["output_tokens"]) for record in records.values()),
        "cost_usd": sum(_record_cost(record) for record in records.values()),
        "response_set_sha256": _sha256_object(
            [
                {
                    "request_id": request_id,
                    "response_sha256": _sha256_object(records[request_id]["response"]),
                }
                for request_id in sorted(records)
            ]
        ),
        "complete_bundle": True,
    }
    incremental_cap = protocol.reader_incremental_hard_caps.get(binding.provider)
    if stage == "reader" and incremental_cap is not None:
        imported = [record for record in records.values() if "compatibility_source" in record]
        completion.update(
            {
                "imported_response_count": len(imported),
                "incremental_response_count": len(records) - len(imported),
                "incremental_cost_usd": sum(
                    _record_cost(record)
                    for record in records.values()
                    if "compatibility_source" not in record
                ),
                "incremental_hard_cap_usd": incremental_cap,
            }
        )
    _write_json(_completion_path(runtime, stage, binding), completion)
    return completion


def _reader_parser(value: object, _request_id: str) -> ReaderResponse:
    return ReaderResponse.from_mapping(value)


def _judge_parser(value: object, _request_id: str) -> JudgeResponse:
    return JudgeResponse.from_mapping(value)


def _verifier_parser_factory(
    inference_cases: Mapping[str, verifier_contract.InferenceCase],
) -> Callable[[object, str], object]:
    def parse(value: object, request_id: str) -> object:
        case = inference_cases.get(request_id)
        if case is None:
            raise ValueError(f"unknown verifier request {request_id!r}")
        return verifier_contract.prediction_from_mapping(value, case)

    return parse


def _fixture_case() -> NaturalEndToEndCase:
    from verify_agent_memory.natural_end_to_end import NaturalCandidate

    candidates = (
        NaturalCandidate(
            memory_key="old-address",
            rank=1,
            text="The former shipping address was 456 Old Street.",
            visible_order="2025-01-01",
            required_evidence=False,
            scope_allowed=True,
            policy_allowed=True,
            lifecycle_compatible=False,
        ),
        NaturalCandidate(
            memory_key="current-address",
            rank=2,
            text="The current shipping address is 123 Market Street.",
            visible_order="2026-01-01",
            required_evidence=True,
            scope_allowed=True,
            policy_allowed=True,
            lifecycle_compatible=True,
        ),
    )
    return NaturalEndToEndCase(
        case_id="synthetic-natural-end-to-end-fixture",
        source="memops",
        group_id="synthetic-group",
        query_text="Where should the replacement card be sent now?",
        query_visible_time="2026-02-01",
        query_intent="current_state",
        expected_answer="123 Market Street",
        answer_metadata={"evaluation_type": "StateTransition"},
        protected_targets=("456 Old Street",),
        anchor_total=1,
        global_candidates=candidates,
        namespace_candidates=candidates,
    )


def run_fixture(
    *,
    protocol: Protocol,
    runtime: Path,
    stage: str,
    binding: ProviderBinding,
    dotenv: Path,
) -> dict[str, object]:
    """Make one synthetic provider call through the exact stage adapter."""
    if stage not in {"verifier", "reader", "judge"}:
        raise ValueError("fixture stage must be verifier, reader, or judge")
    implementation_commit = _require_clean_contract()
    output = _stage_root(runtime, "fixtures", binding) / f"{stage}.json"
    if output.is_file():
        receipt = dict(_read_json(output))
        if (
            receipt.get("protocol_sha256") != protocol.protocol_sha256
            or receipt.get("stage") != stage
            or receipt.get("provider") != binding.provider
            or receipt.get("model") != binding.model
        ):
            raise ValueError("fixture receipt binding drifted")
        return {**receipt, "checkpoint_reused": True}

    case = _fixture_case()
    if stage == "verifier":
        inference_case = _verifier_case(case, protocol.verifier_memory_max_bytes)
        spec = CallSpec(
            request_id=inference_case.case_id,
            payload=verifier_contract.prompt_payload(inference_case),
            schema=verifier_contract.response_json_schema(len(inference_case.candidates)),
        )
        parser = _verifier_parser_factory({inference_case.case_id: inference_case})
        system_prompt = protocol.verifier_prompt
        maximum_output_tokens = protocol.maximum_output_tokens_verifier
    elif stage == "reader":
        from verify_agent_memory.natural_end_to_end import reader_payload

        spec = CallSpec(
            request_id=case.case_id,
            payload=reader_payload(case, case.namespace_candidates),
            schema=reader_response_schema(),
        )
        parser = _reader_parser
        system_prompt = protocol.reader_prompt
        maximum_output_tokens = protocol.maximum_output_tokens_reader
    else:
        spec = CallSpec(
            request_id=case.case_id,
            payload=judge_payload(
                case,
                ReaderResponse(action="answer", answer="123 Market Street"),
            ),
            schema=judge_response_schema(),
        )
        parser = _judge_parser
        system_prompt = protocol.judge_prompt
        maximum_output_tokens = protocol.maximum_output_tokens_judge

    credentials = provider_runtime._load_dotenv(dotenv)
    key = provider_runtime._credential_key(binding.provider)
    api_key = credentials.get(key)
    if not api_key:
        raise RuntimeError(f"{key} is absent or empty")
    record = _call_provider(
        protocol=protocol,
        stage=f"fixture-{stage}",
        binding=binding,
        spec=spec,
        system_prompt=system_prompt,
        maximum_output_tokens=maximum_output_tokens,
        api_key=api_key,
        parser=parser,
        implementation_commit=implementation_commit,
    )
    receipt = {
        **record,
        "stage": stage,
        "response": None,
        "response_sha256": _sha256_object(record["response"]),
        "synthetic_fixture": True,
    }
    _write_json(output, receipt)
    return receipt


def _load_verifier_scores(
    cases: Sequence[NaturalEndToEndCase],
    protocol: Protocol,
    runtime: Path,
) -> dict[str, dict[str, float]]:
    specs, inference_cases = _verifier_specs(cases, protocol)
    parser = _verifier_parser_factory(inference_cases)
    scores = {}
    for case in cases:
        record = _load_record(
            _response_path(runtime, "verifier", protocol.verifier, case.case_id),
            protocol=protocol,
            stage="verifier",
            binding=protocol.verifier,
            spec=specs[case.case_id],
            parser=parser,
        )
        prediction = verifier_contract.prediction_from_mapping(
            record["response"],
            inference_cases[case.case_id],
        )
        scores[case.case_id] = {
            candidate.memory_key: candidate_prediction.violation_probability
            for candidate, candidate_prediction in zip(
                case.namespace_candidates,
                prediction.candidates,
                strict=True,
            )
        }
    return scores


def _reader_plan(
    cases: Sequence[NaturalEndToEndCase],
    protocol: Protocol,
    runtime: Path,
) -> tuple[dict[str, CallSpec], dict[tuple[str, str], str], dict[str, dict[str, float]]]:
    verifier_scores = _load_verifier_scores(cases, protocol, runtime)
    requests, assignments = build_reader_requests(
        cases,
        verifier_scores=verifier_scores,
        verifier_threshold=protocol.verifier_threshold,
    )
    specs = {
        request.prompt_sha256: CallSpec(
            request_id=request.prompt_sha256,
            payload=request.payload,
            schema=reader_response_schema(),
        )
        for request in requests
    }
    return specs, assignments, verifier_scores


def _reader_binding(protocol: Protocol, provider: str) -> ProviderBinding:
    matches = [binding for binding in protocol.readers if binding.provider == provider]
    if len(matches) != 1:
        raise ValueError(f"unknown reader provider {provider!r}")
    return matches[0]


def _require_judge_gate(protocol: Protocol, runtime: Path) -> None:
    judge = _mapping(protocol.raw.get("judge"), "protocol.judge")
    if judge.get("execution_locked_until_deterministic_gate") is not True:
        return
    path = runtime / "deterministic_gate.json"
    if not path.is_file():
        raise RuntimeError("judge execution is locked pending deterministic_gate.json")
    receipt = _read_json(path)
    expected = {
        "schema_version": 1,
        "protocol_sha256": protocol.protocol_sha256,
        "status": "deterministic_gate_passed",
        "provider_calls_made": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"deterministic judge gate drifted: {key}")


def _load_reader_responses(
    *,
    protocol: Protocol,
    runtime: Path,
    binding: ProviderBinding,
    specs: Mapping[str, CallSpec],
) -> dict[str, ReaderResponse]:
    responses = {}
    for request_id, spec in specs.items():
        record = _load_record(
            _response_path(runtime, "reader", binding, request_id),
            protocol=protocol,
            stage="reader",
            binding=binding,
            spec=spec,
            parser=_reader_parser,
        )
        responses[request_id] = ReaderResponse.from_mapping(record["response"])
    return responses


def _judge_plan(
    cases: Sequence[NaturalEndToEndCase],
    protocol: Protocol,
    runtime: Path,
) -> tuple[
    dict[str, CallSpec],
    dict[tuple[str, str, str], str],
    dict[str, dict[str, ReaderResponse]],
    dict[tuple[str, str], str],
    dict[str, dict[str, float]],
]:
    reader_specs, assignments, verifier_scores = _reader_plan(cases, protocol, runtime)
    case_by_id = {case.case_id: case for case in cases}
    responses_by_provider = {}
    specs = {}
    judge_assignments = {}
    for binding in protocol.readers:
        responses = _load_reader_responses(
            protocol=protocol,
            runtime=runtime,
            binding=binding,
            specs=reader_specs,
        )
        responses_by_provider[binding.provider] = responses
        for case_id, case in case_by_id.items():
            for arm in ARMS:
                response = responses[assignments[(case_id, arm)]]
                payload = judge_payload(case, response)
                request_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                previous = specs.setdefault(
                    request_id,
                    CallSpec(
                        request_id=request_id,
                        payload=payload,
                        schema=judge_response_schema(),
                    ),
                )
                if previous.payload != payload:
                    raise RuntimeError("judge prompt SHA-256 collision")
                judge_assignments[(binding.provider, case_id, arm)] = request_id
    return (
        specs,
        judge_assignments,
        responses_by_provider,
        assignments,
        verifier_scores,
    )


def validate_runtime(
    *,
    protocol: Protocol,
    cases_path: Path,
    materialization_path: Path,
    runtime: Path,
) -> dict[str, object]:
    cases = load_cases(cases_path)
    materialization = _read_json(materialization_path)
    source = _mapping(protocol.raw["source_contract"], "source_contract")
    retrieval = _mapping(protocol.raw["retrieval"], "retrieval")
    if materialization.get("case_bundle_sha256") != _sha256_file(cases_path):
        raise ValueError("materialized case bundle hash drifted")
    if materialization.get("route_checkpoint_sha256") != source.get("route_checkpoint_sha256"):
        raise ValueError("materialized route checkpoint differs from protocol")
    if materialization.get("reader_memory_max_utf8_bytes") != retrieval.get(
        "reader_memory_max_utf8_bytes"
    ):
        raise ValueError("materialized reader truncation differs from protocol")
    if len(cases) != 3767:
        raise ValueError("materialized query count differs from protocol")
    source_counts = {
        source_name: sum(case.source == source_name for case in cases)
        for source_name in ("rhelm", "memops")
    }
    verifier_specs, _ = _verifier_specs(cases, protocol)
    audit: dict[str, object] = {
        "protocol_sha256": protocol.protocol_sha256,
        "case_bundle_sha256": _sha256_file(cases_path),
        "case_count": len(cases),
        "source_counts": source_counts,
        "answer_gold_count": sum(bool(case.expected_answer) for case in cases),
        "protected_disclosure_evaluable": sum(
            case.protected_disclosure_evaluable for case in cases
        ),
        "verifier_request_count": len(verifier_specs),
        "reader_max_request_count_per_provider": len(cases) * len(ARMS),
        "provider_calls_made": 0,
    }
    verifier_completion = _completion_path(runtime, "verifier", protocol.verifier)
    if verifier_completion.is_file():
        reader_specs, assignments, _ = _reader_plan(cases, protocol, runtime)
        audit["reader_unique_request_count_per_provider"] = len(reader_specs)
        audit["reader_exact_prompt_deduplication_count"] = len(assignments) - len(reader_specs)
        audit["reader_input_utf8_bytes"] = sum(
            len(spec.payload.encode("utf-8")) for spec in reader_specs.values()
        )
    return audit


def _plan_stats(
    specs: Mapping[str, CallSpec],
    binding: ProviderBinding,
    *,
    system_prompt: str,
    maximum_output_tokens: int,
) -> dict[str, object]:
    byte_lengths = sorted(len(spec.payload.encode("utf-8")) for spec in specs.values())

    def percentile(fraction: float) -> int:
        return byte_lengths[round((len(byte_lengths) - 1) * fraction)]

    return {
        "provider": binding.provider,
        "model": binding.model,
        "request_count": len(specs),
        "payload_utf8_bytes_total": sum(byte_lengths),
        "payload_utf8_bytes_median": percentile(0.5),
        "payload_utf8_bytes_p95": percentile(0.95),
        "payload_utf8_bytes_max": max(byte_lengths),
        "conservative_cost_bound_usd": sum(
            _conservative_call_cost(
                spec,
                binding,
                system_prompt=system_prompt,
                maximum_output_tokens=maximum_output_tokens,
            )
            for spec in specs.values()
        ),
        "hard_cap_usd": binding.hard_cap_usd,
    }


def _load_judge_responses(
    *,
    protocol: Protocol,
    runtime: Path,
    specs: Mapping[str, CallSpec],
) -> dict[str, JudgeResponse]:
    responses = {}
    for request_id, spec in specs.items():
        record = _load_record(
            _response_path(runtime, "judge", protocol.judge, request_id),
            protocol=protocol,
            stage="judge",
            binding=protocol.judge,
            spec=spec,
            parser=_judge_parser,
        )
        responses[request_id] = JudgeResponse.from_mapping(record["response"])
    return responses


def _score_rows(
    *,
    cases: Sequence[NaturalEndToEndCase],
    protocol: Protocol,
    runtime: Path,
) -> list[dict[str, object]]:
    (
        judge_specs,
        judge_assignments,
        responses_by_provider,
        reader_assignments,
        verifier_scores,
    ) = _judge_plan(cases, protocol, runtime)
    judge_responses = _load_judge_responses(
        protocol=protocol,
        runtime=runtime,
        specs=judge_specs,
    )
    model_by_provider = {binding.provider: binding.model for binding in protocol.readers}
    rows = []
    for provider in sorted(responses_by_provider):
        reader_responses = responses_by_provider[provider]
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
                response = reader_responses[reader_assignments[(case.case_id, arm)]]
                deterministic = deterministic_answer_metrics(case, response)
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
                        "policy_disallowed_exposure": (route_score.policy_disallowed_exposure_rate),
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


METRICS = (
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
    "over_refusal",
    "contains_reference_answer",
    "protected_disclosure",
    "literal_protected_disclosure",
    "stale_disclosure",
)


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
            raise RuntimeError("source macro requires both source rows")
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


def _paired_delta(
    rows_by_key: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    provider: str,
    arm: str,
    reference: str,
    metric: str,
) -> tuple[float, float, float, int, int]:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for (row_provider, case_id, row_arm), row in rows_by_key.items():
        if row_provider != provider or row_arm != arm or row[metric] is None:
            continue
        comparator = rows_by_key[(provider, case_id, reference)]
        if comparator[metric] is None:
            continue
        grouped[(str(row["source"]), str(row["group_id"]))].append(
            float(row[metric]) - float(comparator[metric])
        )
    if not grouped:
        return float("nan"), float("nan"), float("nan"), 0, 0
    source_groups: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    for (source, _group), values in grouped.items():
        source_groups[source].append((sum(values), len(values)))

    def aggregate(sampled: Mapping[str, Sequence[tuple[float, int]]]) -> float:
        source_means = []
        for source in sorted(sampled):
            total = sum(value for value, _count in sampled[source])
            count = sum(count for _value, count in sampled[source])
            if count:
                source_means.append(total / count)
        return sum(source_means) / len(source_means)

    point = aggregate(source_groups)
    rng = random.Random(
        int(hashlib.sha256(f"{provider}|{arm}|{reference}|{metric}".encode()).hexdigest()[:16], 16)
        ^ 20260808
    )
    replicates = []
    for _ in range(10_000):
        sampled = {
            source: [groups[rng.randrange(len(groups))] for _ in groups]
            for source, groups in source_groups.items()
        }
        replicates.append(aggregate(sampled))
    replicates.sort()
    lower = replicates[round((len(replicates) - 1) * 0.025)]
    upper = replicates[round((len(replicates) - 1) * 0.975)]
    return (
        point,
        lower,
        upper,
        sum(len(values) for values in grouped.values()),
        len(grouped),
    )


def _paired_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["reader_provider"]), str(row["case_id"]), str(row["arm"])): row for row in rows
    }
    providers = sorted({str(row["reader_provider"]) for row in rows})
    model_by_provider = {str(row["reader_provider"]): str(row["reader_model"]) for row in rows}
    comparisons = (
        ("namespace_dense", "global_dense"),
        ("namespace_policy_gate", "namespace_dense"),
        ("namespace_text_verifier", "namespace_dense"),
        ("released_field_oracle", "namespace_dense"),
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
    for provider in providers:
        for arm, reference in comparisons:
            for metric in metrics:
                point, lower, upper, query_count, group_count = _paired_delta(
                    by_key,
                    provider=provider,
                    arm=arm,
                    reference=reference,
                    metric=metric,
                )
                output.append(
                    {
                        "reader_provider": provider,
                        "reader_model": model_by_provider[provider],
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


def _summary_text(aggregates: Sequence[Mapping[str, object]]) -> str:
    macro = {
        (str(row["reader_provider"]), str(row["arm"])): row
        for row in aggregates
        if row["source"] == "equal_source_macro"
    }
    memops = {
        (str(row["reader_provider"]), str(row["arm"])): row
        for row in aggregates
        if row["source"] == "memops"
    }
    lines = [
        "# Natural end-to-end route-to-reader evaluation",
        "",
        "This is a non-official same-population diagnostic over frozen public-source ",
        "RHELM and MemOps evaluation queries. Reader estimates are reported separately and ",
        "are never pooled. RHELM has no source-defined protected target, so protected ",
        "disclosure is reported for MemOps only.",
        "",
        "| reader | arm | answer accuracy | over-refusal | evidence recall | "
        "admissibility risk | MemOps protected disclosure |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for provider, arm in sorted(macro):
        row = macro[(provider, arm)]
        protected = memops[(provider, arm)]["protected_disclosure"]
        lines.append(
            f"| {provider} | {arm} | {float(row['answer_correct']):.4f} | "
            f"{float(row['over_refusal']):.4f} | {float(row['evidence_recall']):.4f} | "
            f"{float(row['penalized_admissibility_upper_risk']):.4f} | "
            f"{float(protected):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation must use `paired_deltas.csv`; point estimates alone do not establish ",
            "an improvement. The released-field oracle is an upper bound, not a deployable "
            "method. ",
            "The text verifier uses a threshold selected on public development data and is not ",
            "retuned here. This experiment is not an official RHELM or MemOps benchmark result.",
            "",
        ]
    )
    return "\n".join(lines)


def score(
    *,
    protocol: Protocol,
    cases_path: Path,
    runtime: Path,
    output: Path,
) -> dict[str, object]:
    cases = load_cases(cases_path)
    rows = _score_rows(cases=cases, protocol=protocol, runtime=runtime)
    aggregates = _aggregate_rows(rows)
    paired = _paired_rows(rows)
    _write_csv(output / "main_table.csv", aggregates)
    _write_csv(output / "paired_deltas.csv", paired)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text(_summary_text(aggregates), encoding="utf-8", newline="\n")
    completion_paths = [
        _completion_path(runtime, "verifier", protocol.verifier),
        *[_completion_path(runtime, "reader", binding) for binding in protocol.readers],
        _completion_path(runtime, "judge", protocol.judge),
    ]
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "status": "complete_nonofficial_same_population_end_to_end_evaluation",
        "protocol_sha256": protocol.protocol_sha256,
        "case_bundle_sha256": _sha256_file(cases_path),
        "query_count": len(cases),
        "reader_models": [
            {"provider": binding.provider, "model": binding.model} for binding in protocol.readers
        ],
        "reader_estimates_pooled": False,
        "protected_disclosure_source": "memops_only",
        "protected_disclosure_evaluable_count": sum(
            case.protected_disclosure_evaluable for case in cases
        ),
        "completion_sha256": {
            str(path.relative_to(runtime)).replace("\\", "/"): _sha256_file(path)
            for path in completion_paths
        },
        "artifacts": {
            name: _sha256_file(output / name)
            for name in ("main_table.csv", "paired_deltas.csv", "summary.md")
        },
        "official_benchmark_result": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--materialization", type=Path, default=DEFAULT_MATERIALIZATION)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    parser.add_argument("--workers", type=int, default=4)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--stage", choices=("verifier", "reader", "judge"), required=True)
    fixture.add_argument("--provider")
    subparsers.add_parser("plan-verifier")
    subparsers.add_parser("execute-verifier")
    subparsers.add_parser("plan-readers")
    reader = subparsers.add_parser("execute-reader")
    reader.add_argument("--provider", required=True)
    subparsers.add_parser("plan-judge")
    subparsers.add_parser("execute-judge")
    subparsers.add_parser("score")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    validation = validate_runtime(
        protocol=protocol,
        cases_path=args.cases,
        materialization_path=args.materialization,
        runtime=args.runtime,
    )
    if args.command == "validate":
        result = validation
    else:
        cases = load_cases(args.cases)
        if args.command == "fixture":
            if args.stage == "verifier":
                binding = protocol.verifier
            elif args.stage == "judge":
                _require_judge_gate(protocol, args.runtime)
                binding = protocol.judge
            else:
                if not args.provider:
                    raise ValueError("reader fixture requires --provider")
                binding = _reader_binding(protocol, args.provider)
            result = run_fixture(
                protocol=protocol,
                runtime=args.runtime,
                stage=args.stage,
                binding=binding,
                dotenv=args.dotenv,
            )
        elif args.command in {"plan-verifier", "execute-verifier"}:
            specs, inference_cases = _verifier_specs(cases, protocol)
            if args.command == "plan-verifier":
                result = _plan_stats(
                    specs,
                    protocol.verifier,
                    system_prompt=protocol.verifier_prompt,
                    maximum_output_tokens=protocol.maximum_output_tokens_verifier,
                )
            else:
                result = _execute_specs(
                    protocol=protocol,
                    runtime=args.runtime,
                    stage="verifier",
                    binding=protocol.verifier,
                    specs=specs,
                    system_prompt=protocol.verifier_prompt,
                    maximum_output_tokens=protocol.maximum_output_tokens_verifier,
                    dotenv=args.dotenv,
                    workers=args.workers,
                    parser=_verifier_parser_factory(inference_cases),
                )
        elif args.command in {"plan-readers", "execute-reader"}:
            specs, _assignments, _scores = _reader_plan(cases, protocol, args.runtime)
            if args.command == "plan-readers":
                result = {
                    binding.provider: _plan_stats(
                        specs,
                        binding,
                        system_prompt=protocol.reader_prompt,
                        maximum_output_tokens=protocol.maximum_output_tokens_reader,
                    )
                    for binding in protocol.readers
                }
            else:
                binding = _reader_binding(protocol, args.provider)
                result = _execute_specs(
                    protocol=protocol,
                    runtime=args.runtime,
                    stage="reader",
                    binding=binding,
                    specs=specs,
                    system_prompt=protocol.reader_prompt,
                    maximum_output_tokens=protocol.maximum_output_tokens_reader,
                    dotenv=args.dotenv,
                    workers=args.workers,
                    parser=_reader_parser,
                    incremental_cost_cap_usd=protocol.reader_incremental_hard_caps.get(
                        binding.provider
                    ),
                )
        elif args.command in {"plan-judge", "execute-judge"}:
            _require_judge_gate(protocol, args.runtime)
            specs, _assignments, _responses, _reader_assignments, _scores = _judge_plan(
                cases,
                protocol,
                args.runtime,
            )
            if args.command == "plan-judge":
                result = _plan_stats(
                    specs,
                    protocol.judge,
                    system_prompt=protocol.judge_prompt,
                    maximum_output_tokens=protocol.maximum_output_tokens_judge,
                )
            else:
                result = _execute_specs(
                    protocol=protocol,
                    runtime=args.runtime,
                    stage="judge",
                    binding=protocol.judge,
                    specs=specs,
                    system_prompt=protocol.judge_prompt,
                    maximum_output_tokens=protocol.maximum_output_tokens_judge,
                    dotenv=args.dotenv,
                    workers=args.workers,
                    parser=_judge_parser,
                )
        elif args.command == "score":
            _require_judge_gate(protocol, args.runtime)
            result = score(
                protocol=protocol,
                cases_path=args.cases,
                runtime=args.runtime,
                output=args.output,
            )
        else:  # pragma: no cover - argparse restricts commands.
            raise RuntimeError("unreachable command")
    print(json.dumps(result, allow_nan=False, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
