"""Validate, execute, and score the dev-only text-inferred admissibility diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verify_agent_memory.inferred_admissibility import (
    CasePrediction,
    FilteredRouteScore,
    FilterSetting,
    InferenceCandidate,
    InferenceCase,
    aggregate_route_scores,
    binary_classification_metrics,
    filter_decision_metrics,
    intent_classification_metrics,
    load_cases,
    prediction_from_mapping,
    prompt_payload,
    response_json_schema,
    route_keys,
    score_filtered_route,
    select_filter_setting,
    sha256_file,
    stratified_group_paired_bootstrap,
    validate_sample_contract,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments" / "inferred_admissibility_protocol.json"
DEFAULT_DOTENV = ROOT.parent / "bomi-codex-starter" / ".env"
RETRIABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    if not result >= 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"CSV rows for {path} have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"dotenv file not found: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class ProviderBinding:
    provider: str
    model: str
    api_surface: str
    input_usd_per_million: float
    output_usd_per_million: float
    hard_cap_usd: float
    controls: Mapping[str, str]


@dataclass(frozen=True)
class Protocol:
    path: Path
    raw: Mapping[str, Any]
    prompt_path: Path
    prompt_text: str
    prompt_sha256: str
    providers: tuple[ProviderBinding, ...]


def load_protocol(path: Path) -> Protocol:
    raw = _read_json(path)
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != (
        "natural-dev-text-inferred-admissibility-v1"
    ):
        raise ValueError("inferred-admissibility protocol identity drifted")
    if raw.get("status") != "public_dev_diagnostic_not_official_benchmark_result":
        raise ValueError("inferred-admissibility status drifted")
    prompt = _mapping(raw.get("prompt"), "protocol.prompt")
    prompt_path = ROOT / _string(prompt.get("path"), "protocol.prompt.path")
    expected_prompt_hash = _string(prompt.get("sha256"), "protocol.prompt.sha256")
    if not prompt_path.is_file() or sha256_file(prompt_path) != expected_prompt_hash:
        raise ValueError("inferred-admissibility prompt hash drifted")

    bindings = []
    for index, raw_provider in enumerate(_sequence(raw.get("providers"), "protocol.providers")):
        row = _mapping(raw_provider, f"protocol.providers[{index}]")
        known = {
            "provider",
            "model",
            "api_surface",
            "input_usd_per_million",
            "output_usd_per_million",
            "hard_cap_usd",
        }
        controls = {
            key: _string(value, f"protocol.providers[{index}].{key}")
            for key, value in row.items()
            if key not in known
        }
        bindings.append(
            ProviderBinding(
                provider=_string(row.get("provider"), f"protocol.providers[{index}].provider"),
                model=_string(row.get("model"), f"protocol.providers[{index}].model"),
                api_surface=_string(
                    row.get("api_surface"),
                    f"protocol.providers[{index}].api_surface",
                ),
                input_usd_per_million=_number(
                    row.get("input_usd_per_million"),
                    f"protocol.providers[{index}].input_usd_per_million",
                ),
                output_usd_per_million=_number(
                    row.get("output_usd_per_million"),
                    f"protocol.providers[{index}].output_usd_per_million",
                ),
                hard_cap_usd=_number(
                    row.get("hard_cap_usd"),
                    f"protocol.providers[{index}].hard_cap_usd",
                ),
                controls=controls,
            )
        )
    if len(bindings) != 4 or len({binding.provider for binding in bindings}) != 4:
        raise ValueError("protocol must contain four distinct provider bindings")
    expected_surfaces = {
        "OpenAI": "responses",
        "DeepSeek": "chat_completions",
        "Gemini": "generate_content",
        "Anthropic": "messages",
    }
    if {binding.provider: binding.api_surface for binding in bindings} != expected_surfaces:
        raise ValueError("provider API surfaces drifted")
    execution = _mapping(raw.get("execution"), "protocol.execution")
    total_hard_cap = _number(
        execution.get("total_hard_cap_usd"),
        "protocol.execution.total_hard_cap_usd",
    )
    fixture_cap = _number(
        execution.get("fixture_hard_cap_usd_per_provider"),
        "protocol.execution.fixture_hard_cap_usd_per_provider",
    )
    fixture_total = _number(
        execution.get("fixture_total_hard_cap_usd"),
        "protocol.execution.fixture_total_hard_cap_usd",
    )
    if fixture_cap * len(bindings) > fixture_total:
        raise ValueError("fixture provider caps exceed the fixture total hard cap")
    if sum(binding.hard_cap_usd for binding in bindings) + fixture_total > total_hard_cap:
        raise ValueError("provider and fixture hard caps exceed the protocol total hard cap")
    return Protocol(
        path=path,
        raw=raw,
        prompt_path=prompt_path,
        prompt_text=prompt_path.read_text(encoding="utf-8"),
        prompt_sha256=expected_prompt_hash,
        providers=tuple(bindings),
    )


def validate_cases(protocol: Protocol, cases: Sequence[InferenceCase]) -> dict[str, object]:
    sample = _mapping(protocol.raw.get("sample"), "protocol.sample")
    summary = validate_sample_contract(
        cases,
        strata=tuple(_string(value, "protocol.sample.strata[]") for value in sample["strata"]),
        queries_per_stratum=_integer(
            sample.get("queries_per_stratum"),
            "protocol.sample.queries_per_stratum",
            minimum=1,
        ),
        calibration_per_stratum=_integer(
            sample.get("calibration_queries_per_stratum"),
            "protocol.sample.calibration_queries_per_stratum",
            minimum=1,
        ),
        candidate_depth=_integer(
            sample.get("candidate_depth"),
            "protocol.sample.candidate_depth",
            minimum=1,
        ),
        max_visible_utf8_bytes=_integer(
            sample.get("visible_text_max_utf8_bytes"),
            "protocol.sample.visible_text_max_utf8_bytes",
            minimum=1,
        ),
    )
    return {
        **summary,
        "protocol_sha256": sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "case_bundle_sha256": _sha256_object(
            [
                {
                    "case_id": case.case_id,
                    "source": case.source,
                    "group_id": case.group_id,
                    "role": case.role,
                    "released_query_intent": case.released_query_intent,
                    "anchor_total": case.anchor_total,
                    "candidate_keys": [candidate.candidate_key for candidate in case.candidates],
                }
                for case in cases
            ]
        ),
        "raw_text_reported": False,
    }


def _credential_key(provider: str) -> str:
    return {
        "OpenAI": "OPENAI_API_KEY",
        "DeepSeek": "DEEPSEEK_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
    }[provider]


def _http_json(
    *,
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, object],
    timeout_seconds: int,
    max_retries: int,
) -> tuple[Mapping[str, Any], int, float]:
    payload = _canonical_bytes(body)
    last_error: Exception | None = None
    request_started = time.perf_counter()
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                parsed = _mapping(json.loads(response.read().decode("utf-8")), "provider response")
            return parsed, attempt + 1, (time.perf_counter() - request_started) * 1000
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in RETRIABLE_HTTP or attempt >= max_retries:
                raise RuntimeError(f"provider HTTP failure: {error.code}") from error
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt >= max_retries:
                raise RuntimeError("provider transport failure") from error
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("provider request failed") from last_error


def _openai_request(
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
    body: dict[str, object] = {
        "model": binding.model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": binding.controls["effort"]},
        "store": False,
        "text": {
            "verbosity": "low",
            "format": {
                "name": "inferred_admissibility",
                "type": "json_schema",
                "strict": True,
                "schema": schema,
            },
        },
    }
    response, attempts, latency_ms = _http_json(
        url="https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        body=body,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    if response.get("status") != "completed":
        raise ValueError("OpenAI response did not complete")
    output = _sequence(response.get("output"), "OpenAI output")
    text_blocks = []
    for raw_item in output:
        item = _mapping(raw_item, "OpenAI output item")
        if item.get("type") != "message":
            continue
        for raw_block in _sequence(item.get("content"), "OpenAI message content"):
            block = _mapping(raw_block, "OpenAI content block")
            if block.get("type") == "output_text":
                text_blocks.append(_string(block.get("text"), "OpenAI output text"))
    if len(text_blocks) != 1:
        raise ValueError("OpenAI response must contain exactly one output-text block")
    usage = _mapping(response.get("usage"), "OpenAI usage")
    return (
        _mapping(json.loads(text_blocks[0]), "OpenAI structured output"),
        {
            "input_tokens": _integer(usage.get("input_tokens"), "OpenAI input tokens"),
            "output_tokens": _integer(
                usage.get("output_tokens"),
                "OpenAI output tokens",
            ),
        },
        attempts,
        latency_ms,
        _sha256_object(body),
    )


def _deepseek_request(
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
    del schema
    body: dict[str, object] = {
        "model": binding.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_output_tokens,
        "thinking": {"type": binding.controls["thinking"]},
        "response_format": {"type": "json_object"},
    }
    response, attempts, latency_ms = _http_json(
        url="https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        body=body,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    choices = _sequence(response.get("choices"), "DeepSeek choices")
    if len(choices) != 1:
        raise ValueError("DeepSeek response must contain exactly one choice")
    choice = _mapping(choices[0], "DeepSeek choice")
    if choice.get("finish_reason") != "stop":
        raise ValueError("DeepSeek response did not finish normally")
    message = _mapping(choice.get("message"), "DeepSeek message")
    content = _string(message.get("content"), "DeepSeek message content")
    usage = _mapping(response.get("usage"), "DeepSeek usage")
    return (
        _mapping(json.loads(content), "DeepSeek JSON output"),
        {
            "input_tokens": _integer(usage.get("prompt_tokens"), "DeepSeek prompt tokens"),
            "output_tokens": _integer(
                usage.get("completion_tokens"),
                "DeepSeek completion tokens",
            ),
        },
        attempts,
        latency_ms,
        _sha256_object(body),
    )


def _gemini_request(
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
    body: dict[str, object] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
            "thinkingConfig": {"thinkingLevel": binding.controls["thinking_level"].upper()},
        },
    }
    response, attempts, latency_ms = _http_json(
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
    return (
        _mapping(json.loads(text_blocks[0]), "Gemini JSON output"),
        {
            "input_tokens": _integer(
                usage.get("promptTokenCount"),
                "Gemini prompt tokens",
            ),
            "output_tokens": _integer(
                usage.get("candidatesTokenCount"),
                "Gemini output tokens",
            ),
        },
        attempts,
        latency_ms,
        _sha256_object(body),
    )


def _anthropic_schema(value: object) -> object:
    """Mirror Anthropic SDK constraint stripping; strict parsing uses the original schema."""
    if isinstance(value, Mapping):
        cleaned = {
            key: _anthropic_schema(item)
            for key, item in value.items()
            if key not in {"minimum", "maximum", "minItems", "maxItems"}
        }
        if value.get("type") == "array" and int(value.get("minItems", 0)) > 0:
            cleaned["minItems"] = 1
        return cleaned
    if isinstance(value, list):
        return [_anthropic_schema(item) for item in value]
    return value


def _anthropic_request(
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
    body: dict[str, object] = {
        "model": binding.model,
        "max_tokens": max_output_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "thinking": {"type": "disabled"},
        "output_config": {
            "effort": binding.controls["effort"],
            "format": {"type": "json_schema", "schema": _anthropic_schema(schema)},
        },
    }
    response, attempts, latency_ms = _http_json(
        url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        body=body,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    content = _sequence(response.get("content"), "Anthropic content")
    if response.get("stop_reason") != "end_turn":
        raise ValueError("Anthropic response did not finish normally")
    text_blocks = [
        _string(block.get("text"), "Anthropic text")
        for raw_block in content
        if (block := _mapping(raw_block, "Anthropic content block")).get("type") == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("Anthropic response must contain exactly one text block")
    usage = _mapping(response.get("usage"), "Anthropic usage")
    return (
        _mapping(json.loads(text_blocks[0]), "Anthropic JSON output"),
        {
            "input_tokens": _integer(usage.get("input_tokens"), "Anthropic input tokens"),
            "output_tokens": _integer(usage.get("output_tokens"), "Anthropic output tokens"),
        },
        attempts,
        latency_ms,
        _sha256_object(body),
    )


PROVIDER_CALLS = {
    "OpenAI": _openai_request,
    "DeepSeek": _deepseek_request,
    "Gemini": _gemini_request,
    "Anthropic": _anthropic_request,
}


def _provider_adapter_sha256(provider: str) -> str:
    functions = [_http_json, PROVIDER_CALLS[provider]]
    if provider == "Anthropic":
        functions.append(_anthropic_schema)
    return _sha256_text("\n\n".join(inspect.getsource(function) for function in functions))


def _provider_fixture_case() -> InferenceCase:
    return InferenceCase(
        case_id="synthetic-provider-fixture",
        source="synthetic",
        group_id="synthetic-group",
        role="analysis",
        query_text="Where should the replacement card be sent now?",
        query_visible_time="2026-08-02",
        released_query_intent="current_state",
        anchor_total=1,
        candidates=(
            InferenceCandidate(
                candidate_key="c01",
                rank=1,
                text="The former shipping address was 456 Old Street.",
                visible_order="record-00001",
                required_evidence=False,
                released_policy_allowed=True,
                released_lifecycle_compatible=False,
            ),
            InferenceCandidate(
                candidate_key="c02",
                rank=2,
                text="The current shipping address is 123 Market Street.",
                visible_order="record-00002",
                required_evidence=True,
                released_policy_allowed=True,
                released_lifecycle_compatible=True,
            ),
        ),
    )


def run_provider_fixture(
    protocol: Protocol,
    *,
    binding: ProviderBinding,
    dotenv: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Exercise one provider adapter on synthetic text and save a content-free receipt."""
    case = _provider_fixture_case()
    fixture_hash = _sha256_text(prompt_payload(case))
    adapter_hash = _provider_adapter_sha256(binding.provider)
    output_path = output_dir / f"fixture-{binding.provider.lower()}.json"
    if output_path.is_file():
        receipt = validate_provider_fixture_receipt(
            protocol,
            binding=binding,
            output_dir=output_dir,
        )
        return {**receipt, "checkpoint_reused": True}

    credentials = _load_dotenv(dotenv)
    credential_key = _credential_key(binding.provider)
    api_key = credentials.get(credential_key)
    if not api_key:
        raise RuntimeError(f"{credential_key} is absent or empty")
    execution = _mapping(protocol.raw.get("execution"), "protocol.execution")
    fixture_cap = _number(
        execution.get("fixture_hard_cap_usd_per_provider"),
        "protocol.execution.fixture_hard_cap_usd_per_provider",
    )
    provider_call = PROVIDER_CALLS[binding.provider]
    raw_prediction, usage, attempts, latency_ms, request_sha256 = provider_call(
        binding,
        api_key=api_key,
        system_prompt=protocol.prompt_text,
        user_prompt=prompt_payload(case),
        schema=response_json_schema(len(case.candidates)),
        max_output_tokens=800,
        timeout_seconds=_integer(
            execution.get("timeout_seconds"),
            "protocol.execution.timeout_seconds",
            minimum=1,
        ),
        max_retries=_integer(
            execution.get("max_transport_retries"),
            "protocol.execution.max_transport_retries",
        ),
    )
    prediction_from_mapping(raw_prediction, case)
    cost = (
        usage["input_tokens"] * binding.input_usd_per_million
        + usage["output_tokens"] * binding.output_usd_per_million
    ) / 1_000_000
    receipt = {
        "schema_version": 1,
        "provider": binding.provider,
        "model": binding.model,
        "protocol_sha256": sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "fixture_payload_sha256": fixture_hash,
        "adapter_sha256": adapter_hash,
        "prediction_shape_valid": True,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": cost,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "request_sha256": request_sha256,
        "raw_text_or_response_saved": False,
        "checkpoint_reused": False,
    }
    _write_json(output_path, receipt)
    if cost > fixture_cap:
        raise RuntimeError(f"{binding.provider} fixture exceeded its hard cap")
    return receipt


def validate_provider_fixture_receipt(
    protocol: Protocol,
    *,
    binding: ProviderBinding,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Require a current, successful, content-free provider fixture receipt."""
    case = _provider_fixture_case()
    output_path = output_dir / f"fixture-{binding.provider.lower()}.json"
    if not output_path.is_file():
        raise FileNotFoundError(f"provider fixture receipt is absent: {output_path}")
    receipt = _read_json(output_path)
    expected = {
        "provider": binding.provider,
        "model": binding.model,
        "protocol_sha256": sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "fixture_payload_sha256": _sha256_text(prompt_payload(case)),
        "adapter_sha256": _provider_adapter_sha256(binding.provider),
        "prediction_shape_valid": True,
        "raw_text_or_response_saved": False,
    }
    if not all(receipt.get(key) == value for key, value in expected.items()):
        raise ValueError(f"stale or invalid provider fixture receipt: {output_path}")
    return receipt


def _response_path(output_dir: Path, binding: ProviderBinding) -> Path:
    slug = binding.provider.lower().replace(" ", "-")
    return output_dir / f"responses-{slug}.jsonl"


def _failure_path(output_dir: Path, binding: ProviderBinding) -> Path:
    slug = binding.provider.lower().replace(" ", "-")
    return output_dir / f"failures-{slug}.jsonl"


def _load_response_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        return ()
    rows = tuple(
        _mapping(json.loads(line), str(path))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    case_ids = [_string(row.get("case_id"), "response.case_id") for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"response checkpoint has duplicate case IDs: {path}")
    return rows


def _validate_response_record(
    row: Mapping[str, Any],
    case: InferenceCase,
    binding: ProviderBinding,
    protocol: Protocol,
) -> CasePrediction:
    if row.get("provider") != binding.provider or row.get("model") != binding.model:
        raise ValueError("response provider/model binding drifted")
    if row.get("protocol_sha256") != sha256_file(protocol.path):
        raise ValueError("response protocol binding drifted")
    if row.get("prompt_sha256") != protocol.prompt_sha256:
        raise ValueError("response prompt binding drifted")
    if row.get("case_payload_sha256") != _sha256_text(prompt_payload(case)):
        raise ValueError("response case payload binding drifted")
    return prediction_from_mapping(row.get("prediction"), case)


def _estimated_request_reserve(
    binding: ProviderBinding,
    *,
    prompt_chars: int,
    max_output_tokens: int,
) -> float:
    conservative_input_tokens = max(1, (prompt_chars + 2) // 3)
    return (
        conservative_input_tokens * binding.input_usd_per_million
        + max_output_tokens * binding.output_usd_per_million
    ) / 1_000_000


def execute_provider(
    protocol: Protocol,
    cases: Sequence[InferenceCase],
    *,
    binding: ProviderBinding,
    dotenv: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run one checkpointed provider panel with per-model hard-cap enforcement."""
    credentials = _load_dotenv(dotenv)
    credential_key = _credential_key(binding.provider)
    api_key = credentials.get(credential_key)
    if not api_key:
        raise RuntimeError(f"{credential_key} is absent or empty")
    execution = _mapping(protocol.raw.get("execution"), "protocol.execution")
    max_output_tokens = _integer(
        execution.get("max_output_tokens_per_case"),
        "protocol.execution.max_output_tokens_per_case",
        minimum=1,
    )
    timeout_seconds = _integer(
        execution.get("timeout_seconds"),
        "protocol.execution.timeout_seconds",
        minimum=1,
    )
    max_retries = _integer(
        execution.get("max_transport_retries"),
        "protocol.execution.max_transport_retries",
    )
    response_path = _response_path(output_dir, binding)
    existing = _load_response_records(response_path)
    by_case = {case.case_id: case for case in cases}
    for row in existing:
        case_id = _string(row.get("case_id"), "response.case_id")
        case = by_case.get(case_id)
        if case is None:
            raise ValueError(f"response references unknown case {case_id!r}")
        _validate_response_record(row, case, binding, protocol)
    completed = {_string(row.get("case_id"), "response.case_id") for row in existing}
    spent = sum(_number(row.get("cost_usd"), "response.cost_usd") for row in existing)
    if spent > binding.hard_cap_usd:
        raise RuntimeError(f"existing {binding.provider} checkpoint exceeds its hard cap")
    provider_call = PROVIDER_CALLS[binding.provider]

    for case in cases:
        if case.case_id in completed:
            continue
        user_prompt = prompt_payload(case)
        reserve = _estimated_request_reserve(
            binding,
            prompt_chars=len(protocol.prompt_text) + len(user_prompt),
            max_output_tokens=max_output_tokens,
        )
        if spent + reserve > binding.hard_cap_usd:
            raise RuntimeError(
                f"{binding.provider} hard cap would be exceeded before {case.case_id!r}"
            )
        try:
            raw_prediction, usage, attempts, latency_ms, request_sha256 = provider_call(
                binding,
                api_key=api_key,
                system_prompt=protocol.prompt_text,
                user_prompt=user_prompt,
                schema=response_json_schema(len(case.candidates)),
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            prediction_from_mapping(raw_prediction, case)
        except Exception as error:
            _append_jsonl(
                _failure_path(output_dir, binding),
                {
                    "case_id": case.case_id,
                    "provider": binding.provider,
                    "model": binding.model,
                    "error_type": type(error).__name__,
                    "semantic_or_output_repair_attempted": False,
                },
            )
            raise
        cost = (
            usage["input_tokens"] * binding.input_usd_per_million
            + usage["output_tokens"] * binding.output_usd_per_million
        ) / 1_000_000
        spent += cost
        _append_jsonl(
            response_path,
            {
                "case_id": case.case_id,
                "provider": binding.provider,
                "model": binding.model,
                "prediction": raw_prediction,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cost_usd": cost,
                "attempts": attempts,
                "latency_ms": latency_ms,
                "request_sha256": request_sha256,
                "protocol_sha256": sha256_file(protocol.path),
                "prompt_sha256": protocol.prompt_sha256,
                "case_payload_sha256": _sha256_text(user_prompt),
                "raw_query_or_memory_text_saved": False,
            },
        )
        completed.add(case.case_id)
        if spent > binding.hard_cap_usd:
            raise RuntimeError(f"{binding.provider} hard cap was exceeded")
    return {
        "provider": binding.provider,
        "model": binding.model,
        "completed_cases": len(completed),
        "cost_usd": spent,
        "response_checkpoint": response_path.as_posix(),
        "response_sha256": sha256_file(response_path),
    }


def _predictions_from_records(
    records: Sequence[Mapping[str, Any]],
    cases: Sequence[InferenceCase],
    binding: ProviderBinding,
    protocol: Protocol,
) -> dict[str, CasePrediction]:
    by_case = {case.case_id: case for case in cases}
    predictions: dict[str, CasePrediction] = {}
    for row in records:
        case_id = _string(row.get("case_id"), "response.case_id")
        case = by_case.get(case_id)
        if case is None:
            raise ValueError(f"response references unknown case {case_id!r}")
        predictions[case_id] = _validate_response_record(row, case, binding, protocol)
    if set(predictions) != set(by_case):
        raise ValueError(f"{binding.provider} responses do not cover every frozen case")
    return predictions


def _stratum(case: InferenceCase) -> str:
    return f"{case.source}/{case.released_query_intent}"


def _per_case_route_scores(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
    *,
    text_setting: FilterSetting,
    abstain_setting: FilterSetting,
    target_recall: float,
) -> dict[str, dict[str, FilteredRouteScore]]:
    arm_scores: dict[str, dict[str, FilteredRouteScore]] = {}
    for arm in (
        "namespace_dense",
        "released_oracle",
        "text_inferred",
        "abstaining_verifier",
    ):
        scores = {}
        for case in cases:
            setting = text_setting if arm == "text_inferred" else abstain_setting
            keys = route_keys(
                case,
                arm=arm,
                prediction=predictions[case.case_id],
                setting=setting,
            )
            scores[case.case_id] = score_filtered_route(
                case,
                keys,
                target_recall=target_recall,
            )
        arm_scores[arm] = scores
    return arm_scores


def _route_metric_rows(
    cases: Sequence[InferenceCase],
    arm_scores: Mapping[str, Mapping[str, FilteredRouteScore]],
    *,
    provider: str,
    model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    strata = sorted({_stratum(case) for case in cases})
    for arm in ("namespace_dense", "released_oracle", "text_inferred", "abstaining_verifier"):
        by_stratum: list[dict[str, float | int | None]] = []
        for stratum in strata:
            selected = [case for case in cases if _stratum(case) == stratum]
            scores = [arm_scores[arm][case.case_id] for case in selected]
            aggregate = aggregate_route_scores(scores)
            by_stratum.append(aggregate)
            rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "arm": arm,
                    "stratum": stratum,
                    **aggregate,
                }
            )
        metric_names = (
            "evidence_recall",
            "feasible_rate",
            "penalized_admissibility_upper_risk",
            "conditional_admissibility_upper_risk",
            "known_admissibility_violation_rate",
            "mean_route_width",
        )
        rows.append(
            {
                "provider": provider,
                "model": model,
                "arm": arm,
                "stratum": "equal_stratum_macro",
                "query_count": sum(int(row["query_count"]) for row in by_stratum),
                **{
                    name: sum(float(row[name]) for row in by_stratum if row[name] is not None)
                    / sum(row[name] is not None for row in by_stratum)
                    if any(row[name] is not None for row in by_stratum)
                    else None
                    for name in metric_names
                },
            }
        )
    macro = {row["arm"]: row for row in rows if row["stratum"] == "equal_stratum_macro"}
    baseline = float(macro["namespace_dense"]["penalized_admissibility_upper_risk"])
    oracle = float(macro["released_oracle"]["penalized_admissibility_upper_risk"])
    denominator = baseline - oracle
    for row in rows:
        if row["stratum"] != "equal_stratum_macro" or row["arm"] not in {
            "text_inferred",
            "abstaining_verifier",
        }:
            row["oracle_risk_gap_closed"] = None
        else:
            risk = float(row["penalized_admissibility_upper_risk"])
            row["oracle_risk_gap_closed"] = (
                (baseline - risk) / denominator if denominator > 0 else None
            )
    return rows


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _classification_metric_rows(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
    *,
    provider: str,
    model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    strata = sorted({_stratum(case) for case in cases})
    for axis in ("query_intent", "policy", "lifecycle", "admissibility"):
        stratum_rows = []
        for stratum in strata:
            selected = tuple(case for case in cases if _stratum(case) == stratum)
            if axis == "query_intent":
                metrics = intent_classification_metrics(selected, predictions)
                row = {
                    "provider": provider,
                    "model": model,
                    "stratum": stratum,
                    "axis": axis,
                    "known_count": metrics["known_count"],
                    "positive_count": None,
                    "positive_rate": None,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": None,
                    "brier": metrics["multiclass_brier"],
                    "ece_10": metrics["confidence_ece_10"],
                    "roc_auc": None,
                    "pr_auc": None,
                    "aggregation_note": "query_micro_within_stratum",
                }
            else:
                metrics = binary_classification_metrics(selected, predictions, axis=axis)
                row = {
                    "provider": provider,
                    "model": model,
                    "stratum": stratum,
                    "axis": axis,
                    **metrics,
                    "aggregation_note": "known_candidate_micro_within_stratum",
                }
            rows.append(row)
            stratum_rows.append(row)
        rows.append(
            {
                "provider": provider,
                "model": model,
                "stratum": "equal_stratum_macro",
                "axis": axis,
                "known_count": sum(int(row["known_count"]) for row in stratum_rows),
                "positive_count": (
                    None
                    if axis == "query_intent"
                    else sum(int(row["positive_count"]) for row in stratum_rows)
                ),
                "positive_rate": _mean_field(stratum_rows, "positive_rate"),
                "accuracy": _mean_field(stratum_rows, "accuracy"),
                "balanced_accuracy": _mean_field(stratum_rows, "balanced_accuracy"),
                "brier": _mean_field(stratum_rows, "brier"),
                "ece_10": _mean_field(stratum_rows, "ece_10"),
                "roc_auc": _mean_field(stratum_rows, "roc_auc"),
                "pr_auc": _mean_field(stratum_rows, "pr_auc"),
                "aggregation_note": "equal_stratum_macro_available_metrics",
            }
        )
    return rows


def _filter_metric_rows(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
    *,
    provider: str,
    model: str,
    settings: Mapping[str, FilterSetting],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    strata = sorted({_stratum(case) for case in cases})
    for arm, setting in settings.items():
        stratum_rows = []
        for stratum in strata:
            selected = tuple(case for case in cases if _stratum(case) == stratum)
            metrics = filter_decision_metrics(selected, predictions, setting)
            row = {
                "provider": provider,
                "model": model,
                "arm": arm,
                "stratum": stratum,
                "violation_threshold": setting.violation_threshold,
                "unknown_threshold": setting.unknown_threshold,
                "retains_all": setting.retains_all,
                **metrics,
                "aggregation_note": "candidate_micro_within_stratum",
            }
            rows.append(row)
            stratum_rows.append(row)
        rows.append(
            {
                "provider": provider,
                "model": model,
                "arm": arm,
                "stratum": "equal_stratum_macro",
                "violation_threshold": setting.violation_threshold,
                "unknown_threshold": setting.unknown_threshold,
                "retains_all": setting.retains_all,
                "candidate_count": sum(int(row["candidate_count"]) for row in stratum_rows),
                "known_count": sum(int(row["known_count"]) for row in stratum_rows),
                "known_violation_count": sum(
                    int(row["known_violation_count"]) for row in stratum_rows
                ),
                "dropped_count": sum(int(row["dropped_count"]) for row in stratum_rows),
                "dropped_known_count": sum(int(row["dropped_known_count"]) for row in stratum_rows),
                "dropped_unknown_gold_count": sum(
                    int(row["dropped_unknown_gold_count"]) for row in stratum_rows
                ),
                "violation_precision": _mean_field(stratum_rows, "violation_precision"),
                "violation_recall": _mean_field(stratum_rows, "violation_recall"),
                "required_anchor_count": sum(
                    int(row["required_anchor_count"]) for row in stratum_rows
                ),
                "required_anchor_false_denies": sum(
                    int(row["required_anchor_false_denies"]) for row in stratum_rows
                ),
                "required_anchor_false_deny_rate": _mean_field(
                    stratum_rows, "required_anchor_false_deny_rate"
                ),
                "abstention_coverage": _mean_field(stratum_rows, "abstention_coverage"),
                "aggregation_note": "equal_stratum_macro",
            }
        )
    return rows


def _paired_bootstrap_rows(
    protocol: Protocol,
    cases: Sequence[InferenceCase],
    arm_scores: Mapping[str, Mapping[str, FilteredRouteScore]],
    *,
    provider: str,
    model: str,
) -> list[dict[str, object]]:
    evaluation = _mapping(protocol.raw.get("evaluation"), "protocol.evaluation")
    replicates = _integer(
        evaluation.get("bootstrap_replicates"),
        "protocol.evaluation.bootstrap_replicates",
        minimum=1,
    )
    seed = _integer(
        evaluation.get("bootstrap_seed"),
        "protocol.evaluation.bootstrap_seed",
    )
    metrics = (
        ("evidence_recall", "evidence_recall"),
        ("feasible", "feasible_rate"),
        ("penalized_admissibility_upper_risk", "penalized_admissibility_upper_risk"),
        ("route_width", "mean_route_width"),
    )
    rows = []
    for arm in ("text_inferred", "abstaining_verifier"):
        for comparator in ("namespace_dense", "released_oracle"):
            for internal_metric, output_metric in metrics:
                result = stratified_group_paired_bootstrap(
                    cases,
                    arm_scores[arm],
                    arm_scores[comparator],
                    metric=internal_metric,
                    replicates=replicates,
                    seed=seed,
                )
                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "arm": arm,
                        "comparator": comparator,
                        "metric": output_metric,
                        "delta_direction": "arm_minus_comparator",
                        "estimate": result["estimate"],
                        "ci_lower": result["ci_lower"],
                        "ci_upper": result["ci_upper"],
                        "bootstrap_replicates": result["bootstrap_replicates"],
                        "bootstrap_unit": result["bootstrap_unit"],
                        "bootstrap_stratification": result["bootstrap_stratification"],
                        "analysis_queries": result["analysis_queries"],
                        "stratum_group_count": result["stratum_group_count"],
                    }
                )
    return rows


def score_responses(
    protocol: Protocol,
    cases: Sequence[InferenceCase],
    *,
    response_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Select on calibration and publish content-free analysis aggregates."""
    calibration = tuple(case for case in cases if case.role == "calibration")
    analysis = tuple(case for case in cases if case.role == "analysis")
    selection = _mapping(protocol.raw.get("selection"), "protocol.selection")
    evaluation = _mapping(protocol.raw.get("evaluation"), "protocol.evaluation")
    violation_grid = tuple(
        _number(value, "selection.violation_threshold_grid[]")
        for value in _sequence(
            selection.get("violation_threshold_grid"),
            "selection.violation_threshold_grid",
        )
    )
    unknown_grid = tuple(
        _number(value, "selection.unknown_threshold_grid[]")
        for value in _sequence(
            selection.get("unknown_threshold_grid"),
            "selection.unknown_threshold_grid",
        )
    )
    maximum_false_deny = _number(
        selection.get("maximum_required_anchor_false_deny_rate"),
        "selection.maximum_required_anchor_false_deny_rate",
    )
    target_recall = _number(evaluation.get("target_recall"), "evaluation.target_recall")

    selected_rows = []
    classification_rows = []
    filter_rows = []
    route_rows = []
    paired_rows = []
    usage_rows = []
    response_receipts = {}
    for binding in protocol.providers:
        response_path = _response_path(response_dir, binding)
        records = _load_response_records(response_path)
        predictions = _predictions_from_records(records, cases, binding, protocol)
        text_setting = select_filter_setting(
            calibration,
            predictions,
            violation_thresholds=violation_grid,
            unknown_thresholds=None,
            maximum_required_anchor_false_deny_rate=maximum_false_deny,
        )
        abstain_setting = select_filter_setting(
            calibration,
            predictions,
            violation_thresholds=violation_grid,
            unknown_thresholds=unknown_grid,
            maximum_required_anchor_false_deny_rate=maximum_false_deny,
        )
        for arm, setting in (
            ("text_inferred", text_setting),
            ("abstaining_verifier", abstain_setting),
        ):
            selected_rows.append(
                {
                    "provider": binding.provider,
                    "model": binding.model,
                    "arm": arm,
                    "violation_threshold": setting.violation_threshold,
                    "unknown_threshold": setting.unknown_threshold,
                    "calibration_violation_precision": setting.known_violation_precision,
                    "calibration_violation_recall": setting.known_violation_recall,
                    "calibration_required_anchor_false_deny_rate": (
                        setting.required_anchor_false_deny_rate
                    ),
                    "calibration_dropped_unknown_gold": setting.dropped_unknown_gold,
                    "retains_all": setting.retains_all,
                }
            )
        classification_rows.extend(
            _classification_metric_rows(
                analysis,
                predictions,
                provider=binding.provider,
                model=binding.model,
            )
        )
        settings = {
            "text_inferred": text_setting,
            "abstaining_verifier": abstain_setting,
        }
        filter_rows.extend(
            _filter_metric_rows(
                analysis,
                predictions,
                provider=binding.provider,
                model=binding.model,
                settings=settings,
            )
        )
        arm_scores = _per_case_route_scores(
            analysis,
            predictions,
            text_setting=text_setting,
            abstain_setting=abstain_setting,
            target_recall=target_recall,
        )
        route_rows.extend(
            _route_metric_rows(
                analysis,
                arm_scores,
                provider=binding.provider,
                model=binding.model,
            )
        )
        paired_rows.extend(
            _paired_bootstrap_rows(
                protocol,
                analysis,
                arm_scores,
                provider=binding.provider,
                model=binding.model,
            )
        )
        input_tokens = sum(_integer(row.get("input_tokens"), "input tokens") for row in records)
        output_tokens = sum(_integer(row.get("output_tokens"), "output tokens") for row in records)
        cost_usd = sum(_number(row.get("cost_usd"), "cost") for row in records)
        provider_calls = sum(
            _integer(row.get("attempts"), "attempts", minimum=1) for row in records
        )
        latencies = sorted(_number(row.get("latency_ms"), "latency") for row in records)
        p50_index = round((len(latencies) - 1) * 0.50)
        p95_index = round((len(latencies) - 1) * 0.95)
        usage_rows.append(
            {
                "provider": binding.provider,
                "model": binding.model,
                "cases": len(records),
                "provider_calls": provider_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "mean_cost_usd_per_case": cost_usd / len(records),
                "mean_latency_ms": sum(latencies) / len(latencies),
                "p50_latency_ms": latencies[p50_index],
                "p95_latency_ms": latencies[p95_index],
            }
        )
        response_receipts[binding.provider] = {
            "model": binding.model,
            "sha256": sha256_file(response_path),
            "cases": len(records),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "provider_calls": provider_calls,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "selected_thresholds.csv", selected_rows)
    _write_csv(output_dir / "classification_metrics.csv", classification_rows)
    _write_csv(output_dir / "filter_metrics.csv", filter_rows)
    _write_csv(output_dir / "route_metrics.csv", route_rows)
    _write_csv(output_dir / "paired_route_deltas.csv", paired_rows)
    _write_csv(output_dir / "provider_usage.csv", usage_rows)
    manifest = {
        "schema_version": 1,
        "status": "public_dev_text_inferred_admissibility_diagnostic",
        "protocol_sha256": sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "case_count": len(cases),
        "calibration_cases": len(calibration),
        "analysis_cases": len(analysis),
        "response_receipts": response_receipts,
        "outputs": {
            name: sha256_file(output_dir / name)
            for name in (
                "selected_thresholds.csv",
                "classification_metrics.csv",
                "filter_metrics.csv",
                "route_metrics.csv",
                "paired_route_deltas.csv",
                "provider_usage.csv",
            )
        },
        "raw_text_or_responses_published": False,
        "evaluation_or_heldout_access": False,
        "official_result": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "provider-fixture", "execute", "score"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    parser.add_argument(
        "--provider", choices=("all", "OpenAI", "DeepSeek", "Gemini", "Anthropic"), default="all"
    )
    parser.add_argument(
        "--response-dir", type=Path, default=ROOT / "tmp" / "inferred_admissibility"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "tmp" / "inferred_admissibility_scores"
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=ROOT / "tmp" / "inferred_admissibility_provider_fixture",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol.resolve())
    selected = [
        binding
        for binding in protocol.providers
        if args.provider == "all" or binding.provider == args.provider
    ]
    if args.command == "provider-fixture":
        results = [
            run_provider_fixture(
                protocol,
                binding=binding,
                dotenv=args.dotenv.resolve(),
                output_dir=args.fixture_dir.resolve(),
            )
            for binding in selected
        ]
        print(json.dumps({"status": "complete", "fixtures": results}, sort_keys=True))
        return 0
    if args.cases is None:
        raise ValueError(f"--cases is required for {args.command}")
    cases = load_cases(args.cases.resolve())
    validation = validate_cases(protocol, cases)
    if args.command == "validate":
        print(json.dumps({"status": "valid", **validation}, sort_keys=True))
        return 0
    if args.command == "execute":
        for binding in selected:
            validate_provider_fixture_receipt(
                protocol,
                binding=binding,
                output_dir=args.fixture_dir.resolve(),
            )
        results = [
            execute_provider(
                protocol,
                cases,
                binding=binding,
                dotenv=args.dotenv.resolve(),
                output_dir=args.response_dir.resolve(),
            )
            for binding in selected
        ]
        print(json.dumps({"status": "complete", "providers": results}, sort_keys=True))
        return 0
    manifest = score_responses(
        protocol,
        cases,
        response_dir=args.response_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": "scored", "manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
