"""Run the separately authorized provider layer for paired exposure requests."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts import run_counterfactual_exposure_intervention as zero_runtime
from scripts import run_inferred_admissibility_experiment as provider_runtime
from verify_agent_memory.exposure_intervention import (
    EXPOSURE_STATES,
    ExposureUnit,
    ReaderResponse,
    ordered_requests,
    request_payload,
    response_from_mapping,
    response_json_schema,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments" / "counterfactual_exposure_execution_protocol.json"
DEFAULT_RUNTIME_DIR = ROOT / "tmp" / "counterfactual_exposure" / "provider_runtime"
DEFAULT_UNLOCK = DEFAULT_RUNTIME_DIR / "paid_execution_unlock.json"
DEFAULT_DOTENV = provider_runtime.DEFAULT_DOTENV
ProviderBinding = provider_runtime.ProviderBinding
PROVIDER_CALLS = dict(provider_runtime.PROVIDER_CALLS)
CONTRACT_PATH_PREFIXES = ("experiments/", "scripts/", "src/")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
    if not result >= 0 or result == float("inf"):
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        return ()
    return tuple(
        _mapping(json.loads(line), f"{path}:{line_number}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing derivative: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _bound_file(identity: object, label: str) -> tuple[str, Path, str]:
    row = _mapping(identity, label)
    if set(row) != {"path", "sha256"}:
        raise ValueError(f"{label} identity has missing or unknown fields")
    reference = _string(row["path"], f"{label}.path").replace("\\", "/")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path must be repository-relative")
    path = ROOT / relative
    expected_hash = _string(row["sha256"], f"{label}.sha256")
    if not path.is_file() or _sha256_file(path) != expected_hash:
        raise ValueError(f"{label} hash drifted")
    return reference, path, expected_hash


@dataclass(frozen=True)
class ExecutionProtocol:
    path: Path
    raw: Mapping[str, Any]
    base: zero_runtime.Protocol
    providers: tuple[ProviderBinding, ...]
    max_output_tokens: int
    timeout_seconds: int
    max_transport_retries: int
    total_hard_cap_usd: float
    max_calls_per_provider: int


def _request_input_bytes(protocol: zero_runtime.Protocol, unit: ExposureUnit, exposure: str) -> int:
    payload = (
        protocol.prompt_text
        + request_payload(unit, exposure=exposure)
        + json.dumps(response_json_schema(), separators=(",", ":"), sort_keys=True)
    )
    return len(payload.encode("utf-8"))


def _all_request_input_bytes(protocol: zero_runtime.Protocol) -> tuple[int, ...]:
    return tuple(
        _request_input_bytes(protocol, unit, exposure)
        for unit in protocol.units
        for exposure in EXPOSURE_STATES
    )


def _binding_from_mapping(value: object, location: str) -> ProviderBinding:
    row = _mapping(value, location)
    known = {
        "provider",
        "model",
        "api_surface",
        "input_usd_per_million",
        "output_usd_per_million",
        "hard_cap_usd",
    }
    controls = {
        key: _string(item, f"{location}.{key}") for key, item in row.items() if key not in known
    }
    return ProviderBinding(
        provider=_string(row.get("provider"), f"{location}.provider"),
        model=_string(row.get("model"), f"{location}.model"),
        api_surface=_string(row.get("api_surface"), f"{location}.api_surface"),
        input_usd_per_million=_number(
            row.get("input_usd_per_million"), f"{location}.input_usd_per_million"
        ),
        output_usd_per_million=_number(
            row.get("output_usd_per_million"), f"{location}.output_usd_per_million"
        ),
        hard_cap_usd=_number(row.get("hard_cap_usd"), f"{location}.hard_cap_usd"),
        controls=controls,
    )


def load_execution_protocol(path: Path) -> ExecutionProtocol:
    """Load the paid-layer contract without reading credentials or making calls."""
    raw = _read_json(path)
    expected = {
        "schema_version",
        "protocol_id",
        "status",
        "base_protocol",
        "implementation",
        "providers",
        "execution",
        "authorization",
    }
    if set(raw) != expected:
        raise ValueError("execution protocol has missing or unknown fields")
    if raw["schema_version"] != 1:
        raise ValueError("execution protocol schema version drifted")
    if raw["protocol_id"] != "paired-counterfactual-exposure-execution-v1":
        raise ValueError("execution protocol identity drifted")
    if raw["status"] != "implementation_only_paid_execution_not_authorized":
        raise ValueError("execution protocol status drifted")

    _base_reference, base_path, _base_hash = _bound_file(
        raw["base_protocol"], "protocol.base_protocol"
    )
    base = zero_runtime.load_protocol(base_path)
    implementation = _mapping(raw["implementation"], "protocol.implementation")
    if set(implementation) != {"runner", "provider_adapter"}:
        raise ValueError("execution implementation identity drifted")
    _bound_file(implementation["runner"], "protocol.implementation.runner")
    _bound_file(implementation["provider_adapter"], "protocol.implementation.provider_adapter")

    providers = tuple(
        _binding_from_mapping(value, f"protocol.providers[{index}]")
        for index, value in enumerate(_sequence(raw["providers"], "protocol.providers"))
    )
    expected_panel = {
        "OpenAI": ("gpt-5.6-sol", "responses", {"effort": "none"}),
        "Gemini": (
            "gemini-3.6-flash",
            "generate_content",
            {"thinking_level": "minimal"},
        ),
        "DeepSeek": (
            "deepseek-v4-pro",
            "chat_completions",
            {"pricing_basis": "two_times_regular_peak_guard", "thinking": "disabled"},
        ),
    }
    if len(providers) != len(expected_panel) or len({item.provider for item in providers}) != len(
        providers
    ):
        raise ValueError("execution provider panel must contain three distinct providers")
    for binding in providers:
        if binding.provider not in expected_panel:
            raise ValueError(f"unexpected execution provider {binding.provider!r}")
        model, surface, controls = expected_panel[binding.provider]
        if (binding.model, binding.api_surface, dict(binding.controls)) != (
            model,
            surface,
            controls,
        ):
            raise ValueError(f"execution provider contract drifted for {binding.provider}")

    execution = _mapping(raw["execution"], "protocol.execution")
    expected_execution_fields = {
        "request_count_per_provider",
        "fixture_calls_per_provider",
        "max_calls_per_provider",
        "max_output_tokens_per_request",
        "timeout_seconds",
        "max_transport_retries",
        "checkpoint_after_each_request",
        "complete_bundle_required",
        "semantic_or_output_repair",
        "selective_rerun",
        "input_token_bound",
        "input_byte_bound_total",
        "max_input_byte_bound_per_request",
        "total_hard_cap_usd",
    }
    if set(execution) != expected_execution_fields:
        raise ValueError("execution settings have missing or unknown fields")
    request_count = len(base.units) * len(EXPOSURE_STATES)
    fixture_calls = _integer(execution["fixture_calls_per_provider"], "fixture_calls_per_provider")
    max_calls = _integer(execution["max_calls_per_provider"], "max_calls_per_provider")
    if execution["request_count_per_provider"] != request_count:
        raise ValueError("execution request count drifted")
    if fixture_calls != 1 or max_calls != request_count + fixture_calls:
        raise ValueError("execution call cap drifted")
    if (
        execution["checkpoint_after_each_request"] is not True
        or execution["complete_bundle_required"] is not True
        or execution["semantic_or_output_repair"] is not False
        or execution["selective_rerun"] is not False
        or execution["input_token_bound"] != "one_utf8_byte_per_token_conservative_upper_bound"
    ):
        raise ValueError("execution safety controls drifted")
    input_bytes = _all_request_input_bytes(base)
    if execution["input_byte_bound_total"] != sum(input_bytes):
        raise ValueError("execution total input-byte bound drifted")
    if execution["max_input_byte_bound_per_request"] != max(input_bytes):
        raise ValueError("execution per-request input-byte bound drifted")
    max_output_tokens = _integer(
        execution["max_output_tokens_per_request"],
        "max_output_tokens_per_request",
        minimum=1,
    )
    timeout_seconds = _integer(execution["timeout_seconds"], "timeout_seconds", minimum=1)
    max_retries = _integer(execution["max_transport_retries"], "max_transport_retries")
    if max_retries != 0:
        raise ValueError("transport retries must remain zero")
    total_cap = _number(execution["total_hard_cap_usd"], "total_hard_cap_usd")
    if sum(binding.hard_cap_usd for binding in providers) > total_cap:
        raise ValueError("provider hard caps exceed total hard cap")
    for binding in providers:
        conservative_bound = (
            (sum(input_bytes) + max(input_bytes)) * binding.input_usd_per_million
            + max_calls * max_output_tokens * binding.output_usd_per_million
        ) / 1_000_000
        if conservative_bound > binding.hard_cap_usd:
            raise ValueError(f"conservative cost bound exceeds {binding.provider} hard cap")

    authorization = _mapping(raw["authorization"], "protocol.authorization")
    if authorization != {
        "unlock_required": True,
        "exact_commit_required": True,
        "exact_protocol_hash_required": True,
        "explicit_owner_required": True,
        "credential_read_before_unlock": False,
        "provider_call_before_unlock": False,
    }:
        raise ValueError("execution authorization contract drifted")
    return ExecutionProtocol(
        path=path,
        raw=raw,
        base=base,
        providers=providers,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        max_transport_retries=max_retries,
        total_hard_cap_usd=total_cap,
        max_calls_per_provider=max_calls,
    )


def _binding(protocol: ExecutionProtocol, provider: str) -> ProviderBinding:
    matches = [binding for binding in protocol.providers if binding.provider == provider]
    if len(matches) != 1:
        raise ValueError(f"unknown provider {provider!r}")
    return matches[0]


def _head_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _contract_dirty_paths() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = []
    for line in result.stdout.splitlines():
        path = line[3:].replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", maxsplit=1)[1]
        if path.startswith(CONTRACT_PATH_PREFIXES):
            dirty.append(path)
    return tuple(sorted(dirty))


def unlock_template(protocol: ExecutionProtocol, *, owner: str) -> dict[str, object]:
    """Return the exact external unlock payload required for paid commands."""
    owner = _string(owner, "owner")
    return {
        "schema_version": 1,
        "unlock_type": "paired_counterfactual_exposure_paid_execution",
        "commit": _head_commit(),
        "execution_protocol_sha256": _sha256_file(protocol.path),
        "base_protocol_sha256": _sha256_file(protocol.base.path),
        "models": {binding.provider: binding.model for binding in protocol.providers},
        "total_hard_cap_usd": protocol.total_hard_cap_usd,
        "owner": owner,
    }


def validate_unlock(protocol: ExecutionProtocol, path: Path) -> Mapping[str, Any]:
    """Validate an external approval against the exact implementation commit."""
    dirty = _contract_dirty_paths()
    if dirty:
        raise RuntimeError(f"contract-bearing paths are dirty: {dirty}")
    unlock = _read_json(path)
    if unlock != unlock_template(protocol, owner=_string(unlock.get("owner"), "unlock.owner")):
        raise ValueError("paid execution unlock does not match the exact contract")
    return unlock


def validate_execution_protocol(protocol: ExecutionProtocol) -> dict[str, object]:
    """Return a credential-free and call-free execution receipt."""
    input_bytes = _all_request_input_bytes(protocol.base)
    return {
        "status": "valid_implementation_paid_execution_not_authorized",
        "execution_protocol_sha256": _sha256_file(protocol.path),
        "base_protocol_sha256": _sha256_file(protocol.base.path),
        "head_commit": _head_commit(),
        "contract_dirty_paths": list(_contract_dirty_paths()),
        "request_count_per_provider": len(input_bytes),
        "input_byte_bound_total": sum(input_bytes),
        "max_input_byte_bound_per_request": max(input_bytes),
        "max_output_tokens_per_request": protocol.max_output_tokens,
        "providers": [
            {
                "provider": binding.provider,
                "model": binding.model,
                "hard_cap_usd": binding.hard_cap_usd,
            }
            for binding in protocol.providers
        ],
        "total_hard_cap_usd": protocol.total_hard_cap_usd,
        "credential_read": False,
        "provider_call_made": False,
    }


def preflight(
    protocol: ExecutionProtocol,
    *,
    unlock_path: Path,
) -> dict[str, object]:
    """Validate approval and hashes without reading credentials."""
    unlock = validate_unlock(protocol, unlock_path)
    return {
        "status": "authorized_preflight_passed_no_credentials_no_calls",
        "unlock_sha256": _sha256_file(unlock_path),
        "commit": unlock["commit"],
        "execution_protocol_sha256": unlock["execution_protocol_sha256"],
        "credential_read": False,
        "provider_call_made": False,
    }


def _provider_slug(binding: ProviderBinding) -> str:
    return f"{binding.provider}-{binding.model}".casefold().replace("/", "-")


def _fixture_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "fixtures" / f"{_provider_slug(binding)}.json"


def _fixture_failure_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "failures" / f"{_provider_slug(binding)}-fixture.json"


def _checkpoint_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "raw" / f"{_provider_slug(binding)}.jsonl"


def _completion_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "completion" / f"{_provider_slug(binding)}.json"


def _execution_failure_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "failures" / f"{_provider_slug(binding)}-execution.json"


def _normalized_response_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "normalized" / f"{_provider_slug(binding)}.jsonl"


def _score_dir(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "scores" / _provider_slug(binding)


def _ordered_requests(
    protocol: ExecutionProtocol,
    binding: ProviderBinding,
) -> tuple[tuple[ExposureUnit, str], ...]:
    return ordered_requests(
        protocol.base.units,
        model=binding.model,
        seed=protocol.base.request_order_seed,
    )


def _request_hash(unit: ExposureUnit, exposure: str) -> str:
    return _sha256_text(request_payload(unit, exposure=exposure))


def _cap_cost(binding: ProviderBinding, usage: Mapping[str, int]) -> float:
    return (
        usage["input_tokens"] * binding.input_usd_per_million
        + usage["output_tokens"] * binding.output_usd_per_million
    ) / 1_000_000


def _call_bound(
    protocol: ExecutionProtocol,
    binding: ProviderBinding,
    unit: ExposureUnit,
    exposure: str,
) -> float:
    return (
        _request_input_bytes(protocol.base, unit, exposure) * binding.input_usd_per_million
        + protocol.max_output_tokens * binding.output_usd_per_million
    ) / 1_000_000


def _credential(binding: ProviderBinding, dotenv: Path) -> str:
    values = provider_runtime._load_dotenv(dotenv)
    key = provider_runtime._credential_key(binding.provider)
    value = values.get(key)
    if not value:
        raise ValueError(f"credential {key} is missing")
    return value


def _gemini_exposure_request(
    binding: ProviderBinding,
    *,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, object],
    max_output_tokens: int,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[Mapping[str, object], dict[str, int], int, float, str]:
    """Call Gemini while charging all non-prompt tokens, including thinking."""
    body: dict[str, object] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": provider_runtime._limited_provider_schema(schema),
            "thinkingConfig": {"thinkingLevel": binding.controls["thinking_level"].upper()},
        },
    }
    response, attempts, latency_ms = provider_runtime._http_json(
        url=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{binding.model}:generateContent"
        ),
        headers={"x-goog-api-key": api_key},
        body=body,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    candidates = _sequence(response.get("candidates"), "Gemini candidates")
    if len(candidates) != 1:
        raise ValueError("Gemini response must contain exactly one candidate")
    candidate = _mapping(candidates[0], "Gemini candidate")
    if candidate.get("finishReason") != "STOP":
        raise ValueError("Gemini response did not finish normally")
    content = _mapping(candidate.get("content"), "Gemini content")
    parts = _sequence(content.get("parts"), "Gemini content parts")
    text_blocks = [
        _string(part.get("text"), "Gemini text")
        for raw_part in parts
        if (part := _mapping(raw_part, "Gemini content part")).get("thought") is not True
        and part.get("text") is not None
    ]
    if len(text_blocks) != 1:
        raise ValueError("Gemini response must contain exactly one answer text block")
    usage = _mapping(response.get("usageMetadata"), "Gemini usage")
    prompt_tokens = _integer(usage.get("promptTokenCount"), "Gemini prompt tokens")
    total_tokens = _integer(usage.get("totalTokenCount"), "Gemini total tokens")
    if total_tokens < prompt_tokens:
        raise ValueError("Gemini total tokens cannot be less than prompt tokens")
    return (
        _mapping(json.loads(text_blocks[0]), "Gemini JSON output"),
        {
            "input_tokens": prompt_tokens,
            "output_tokens": total_tokens - prompt_tokens,
        },
        attempts,
        latency_ms,
        _sha256_object(body),
    )


PROVIDER_CALLS["Gemini"] = _gemini_exposure_request


def _adapter_sha256(provider: str) -> str:
    if provider != "Gemini":
        return provider_runtime._provider_adapter_sha256(provider)
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            provider_runtime._http_json,
            provider_runtime._limited_provider_schema,
            _gemini_exposure_request,
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _provider_request(
    protocol: ExecutionProtocol,
    binding: ProviderBinding,
    *,
    unit: ExposureUnit,
    exposure: str,
    api_key: str,
) -> tuple[Mapping[str, object], Mapping[str, int], int, float, str]:
    return PROVIDER_CALLS[binding.provider](
        binding,
        api_key=api_key,
        system_prompt=protocol.base.prompt_text,
        user_prompt=request_payload(unit, exposure=exposure),
        schema=response_json_schema(),
        max_output_tokens=protocol.max_output_tokens,
        timeout_seconds=protocol.timeout_seconds,
        max_retries=protocol.max_transport_retries,
    )


def _fixture_expected(
    protocol: ExecutionProtocol,
    binding: ProviderBinding,
    unlock_path: Path,
) -> dict[str, object]:
    unit, exposure = _ordered_requests(protocol, binding)[0]
    return {
        "status": "passed",
        "provider": binding.provider,
        "model": binding.model,
        "commit": _head_commit(),
        "unlock_sha256": _sha256_file(unlock_path),
        "execution_protocol_sha256": _sha256_file(protocol.path),
        "base_protocol_sha256": _sha256_file(protocol.base.path),
        "adapter_sha256": _adapter_sha256(binding.provider),
        "fixture_request_id": unit.request_id(exposure),
        "fixture_payload_sha256": _request_hash(unit, exposure),
    }


def fixture_provider(
    protocol: ExecutionProtocol,
    *,
    binding: ProviderBinding,
    unlock_path: Path,
    dotenv: Path,
    runtime_dir: Path,
) -> Mapping[str, object]:
    """Make one authorized compatibility call without preserving its answer."""
    validate_unlock(protocol, unlock_path)
    failure_path = _fixture_failure_path(runtime_dir, binding)
    if failure_path.exists():
        raise RuntimeError("frozen fixture failure exists; rerun is prohibited")
    receipt_path = _fixture_path(runtime_dir, binding)
    expected = _fixture_expected(protocol, binding, unlock_path)
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        if all(receipt.get(key) == value for key, value in expected.items()):
            return receipt
        raise ValueError("fixture receipt binding drifted")

    unit, exposure = _ordered_requests(protocol, binding)[0]
    if _call_bound(protocol, binding, unit, exposure) > binding.hard_cap_usd:
        raise RuntimeError("fixture conservative cost exceeds provider hard cap")
    api_key = _credential(binding, dotenv)
    try:
        response, usage, attempts, latency_ms, provider_request_hash = _provider_request(
            protocol,
            binding,
            unit=unit,
            exposure=exposure,
            api_key=api_key,
        )
        response_from_mapping(response, request_id=unit.request_id(exposure))
        cost = _cap_cost(binding, usage)
        if cost > binding.hard_cap_usd:
            raise RuntimeError("fixture accounted cost exceeds provider hard cap")
    except Exception as error:
        _write_json(
            failure_path,
            {
                "status": "frozen_fixture_failure",
                "provider": binding.provider,
                "model": binding.model,
                "request_id": unit.request_id(exposure),
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_rerun_allowed": False,
                "response_saved": False,
            },
        )
        raise
    receipt = {
        **expected,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cap_accounted_cost_usd": cost,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "provider_request_sha256": provider_request_hash,
        "response_saved": False,
    }
    _write_json(receipt_path, receipt)
    return receipt


def _validate_fixture(
    protocol: ExecutionProtocol,
    binding: ProviderBinding,
    unlock_path: Path,
    runtime_dir: Path,
) -> Mapping[str, Any]:
    path = _fixture_path(runtime_dir, binding)
    if not path.is_file():
        raise FileNotFoundError(f"fixture receipt not found: {path}")
    receipt = _read_json(path)
    expected = _fixture_expected(protocol, binding, unlock_path)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("fixture receipt binding drifted")
    return receipt


def _validate_checkpoint(
    protocol: ExecutionProtocol,
    binding: ProviderBinding,
    path: Path,
) -> tuple[tuple[Mapping[str, Any], ReaderResponse], ...]:
    rows = _read_jsonl(path)
    expected_order = _ordered_requests(protocol, binding)
    if len(rows) > len(expected_order):
        raise ValueError("checkpoint has more rows than the frozen request set")
    validated = []
    required = {
        "request_id",
        "model",
        "execution_protocol_sha256",
        "base_protocol_sha256",
        "request_payload_sha256",
        "response",
        "usage",
        "attempts",
        "latency_ms",
        "provider_request_sha256",
        "cap_accounted_cost_usd",
    }
    for index, raw in enumerate(rows):
        if set(raw) != required:
            raise ValueError(f"checkpoint row {index} has missing or unknown fields")
        unit, exposure = expected_order[index]
        request_id = unit.request_id(exposure)
        expected = {
            "request_id": request_id,
            "model": binding.model,
            "execution_protocol_sha256": _sha256_file(protocol.path),
            "base_protocol_sha256": _sha256_file(protocol.base.path),
            "request_payload_sha256": _request_hash(unit, exposure),
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise ValueError(f"checkpoint row {index} binding drifted")
        usage = _mapping(raw["usage"], f"checkpoint row {index}.usage")
        if set(usage) != {"input_tokens", "output_tokens"}:
            raise ValueError(f"checkpoint row {index} usage drifted")
        parsed_usage = {
            "input_tokens": _integer(usage["input_tokens"], "input_tokens"),
            "output_tokens": _integer(usage["output_tokens"], "output_tokens"),
        }
        expected_cost = _cap_cost(binding, parsed_usage)
        observed_cost = _number(raw["cap_accounted_cost_usd"], "cap_accounted_cost_usd")
        if abs(expected_cost - observed_cost) > 1e-12:
            raise ValueError(f"checkpoint row {index} cost drifted")
        validated.append(
            (
                raw,
                response_from_mapping(raw["response"], request_id=request_id),
            )
        )
    return tuple(validated)


def execute_provider(
    protocol: ExecutionProtocol,
    *,
    binding: ProviderBinding,
    unlock_path: Path,
    dotenv: Path,
    runtime_dir: Path,
) -> Mapping[str, object]:
    """Resume the exact request order and freeze permanently on the first failure."""
    validate_unlock(protocol, unlock_path)
    fixture = _validate_fixture(protocol, binding, unlock_path, runtime_dir)
    failure_path = _execution_failure_path(runtime_dir, binding)
    if failure_path.exists():
        raise RuntimeError("frozen execution failure exists; selective rerun is prohibited")
    checkpoint_path = _checkpoint_path(runtime_dir, binding)
    completion_path = _completion_path(runtime_dir, binding)
    validated = _validate_checkpoint(protocol, binding, checkpoint_path)
    expected_order = _ordered_requests(protocol, binding)
    if completion_path.is_file():
        completion = _read_json(completion_path)
        if (
            len(validated) == len(expected_order)
            and completion.get("checkpoint_sha256") == _sha256_file(checkpoint_path)
            and completion.get("completed_requests") == len(expected_order)
        ):
            return completion
        raise ValueError("completion receipt binding drifted")

    spent = _number(fixture["cap_accounted_cost_usd"], "fixture cost") + sum(
        _number(row["cap_accounted_cost_usd"], "checkpoint cost") for row, _response in validated
    )
    api_key = _credential(binding, dotenv) if len(validated) < len(expected_order) else None
    for index in range(len(validated), len(expected_order)):
        unit, exposure = expected_order[index]
        request_id = unit.request_id(exposure)
        bound = _call_bound(protocol, binding, unit, exposure)
        if spent + bound > binding.hard_cap_usd:
            raise RuntimeError("next request could exceed provider hard cap")
        try:
            if api_key is None:
                raise RuntimeError("credential invariant failed")
            response, usage, attempts, latency_ms, provider_request_hash = _provider_request(
                protocol,
                binding,
                unit=unit,
                exposure=exposure,
                api_key=api_key,
            )
            parsed = response_from_mapping(response, request_id=request_id)
            cost = _cap_cost(binding, usage)
            if spent + cost > binding.hard_cap_usd:
                raise RuntimeError("provider hard cap exceeded")
            row = {
                "request_id": request_id,
                "model": binding.model,
                "execution_protocol_sha256": _sha256_file(protocol.path),
                "base_protocol_sha256": _sha256_file(protocol.base.path),
                "request_payload_sha256": _request_hash(unit, exposure),
                "response": {"action": parsed.action, "answer": parsed.answer},
                "usage": {
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                },
                "attempts": attempts,
                "latency_ms": latency_ms,
                "provider_request_sha256": provider_request_hash,
                "cap_accounted_cost_usd": cost,
            }
            _append_jsonl(checkpoint_path, row)
            spent += cost
        except Exception as error:
            _write_json(
                failure_path,
                {
                    "status": "frozen_execution_failure",
                    "provider": binding.provider,
                    "model": binding.model,
                    "failed_request_id": request_id,
                    "completed_requests": index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "automatic_rerun_allowed": False,
                    "selective_rerun_allowed": False,
                    "output_repair_attempted": False,
                },
            )
            raise
    validated = _validate_checkpoint(protocol, binding, checkpoint_path)
    if len(validated) != len(expected_order):
        raise RuntimeError("complete checkpoint invariant failed")
    completion = {
        "status": "complete_unscored",
        "provider": binding.provider,
        "model": binding.model,
        "completed_requests": len(validated),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "execution_protocol_sha256": _sha256_file(protocol.path),
        "base_protocol_sha256": _sha256_file(protocol.base.path),
        "unlock_sha256": _sha256_file(unlock_path),
        "cap_accounted_cost_usd": spent,
        "prefix_scoring": False,
    }
    _write_json(completion_path, completion)
    return completion


def score_provider(
    protocol: ExecutionProtocol,
    *,
    binding: ProviderBinding,
    unlock_path: Path,
    runtime_dir: Path,
) -> Mapping[str, object]:
    """Score a complete checkpoint locally without another provider call."""
    validate_unlock(protocol, unlock_path)
    checkpoint_path = _checkpoint_path(runtime_dir, binding)
    completion = _read_json(_completion_path(runtime_dir, binding))
    validated = _validate_checkpoint(protocol, binding, checkpoint_path)
    expected_count = len(protocol.base.units) * len(EXPOSURE_STATES)
    if len(validated) != expected_count:
        raise ValueError("complete response bundle required for scoring")
    if completion.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise ValueError("checkpoint changed after completion")
    normalized_path = _normalized_response_path(runtime_dir, binding)
    rows = [
        {
            "request_id": response.request_id,
            "response": {"action": response.action, "answer": response.answer},
        }
        for _raw, response in validated
    ]
    if normalized_path.exists():
        existing = _read_jsonl(normalized_path)
        if tuple(existing) != tuple(rows):
            raise ValueError("normalized response derivative drifted")
    else:
        _write_jsonl_once(normalized_path, rows)
    score_dir = _score_dir(runtime_dir, binding)
    result = zero_runtime.score_response_bundle(
        protocol.base,
        model=binding.model,
        responses_path=normalized_path,
        output_dir=score_dir,
    )
    execution_manifest = {
        "schema_version": 1,
        "status": "complete_local_score_not_official_result",
        "provider": binding.provider,
        "model": binding.model,
        "execution_protocol_sha256": _sha256_file(protocol.path),
        "base_protocol_sha256": _sha256_file(protocol.base.path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "score_manifest_sha256": _sha256_file(score_dir / "manifest.json"),
        "provider_call_made_during_scoring": False,
    }
    _write_json(score_dir / "execution_manifest.json", execution_manifest)
    return {**result, **execution_manifest}


def _print(value: object) -> None:
    print(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--unlock", type=Path, default=DEFAULT_UNLOCK)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    template = subparsers.add_parser("unlock-template")
    template.add_argument("--owner", required=True)
    subparsers.add_parser("preflight")
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--provider", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--provider", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--provider", required=True)
    args = parser.parse_args(argv)

    protocol = load_execution_protocol(args.protocol)
    if args.command == "validate":
        result = validate_execution_protocol(protocol)
    elif args.command == "unlock-template":
        result = unlock_template(protocol, owner=args.owner)
    elif args.command == "preflight":
        result = preflight(protocol, unlock_path=args.unlock)
    elif args.command == "fixture":
        result = fixture_provider(
            protocol,
            binding=_binding(protocol, args.provider),
            unlock_path=args.unlock,
            dotenv=args.dotenv,
            runtime_dir=args.runtime_dir,
        )
    elif args.command == "execute":
        result = execute_provider(
            protocol,
            binding=_binding(protocol, args.provider),
            unlock_path=args.unlock,
            dotenv=args.dotenv,
            runtime_dir=args.runtime_dir,
        )
    else:
        result = score_provider(
            protocol,
            binding=_binding(protocol, args.provider),
            unlock_path=args.unlock,
            runtime_dir=args.runtime_dir,
        )
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
