"""Validate, execute, score, and publish controlled admissibility pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts import run_inferred_admissibility_experiment as provider_runtime
from verify_agent_memory.counterfactual_admissibility import (
    CONDITIONS,
    CounterfactualPair,
    CounterfactualPrediction,
    aggregate_pair_metrics,
    expand_pairs,
    load_scenarios,
    no_verifier_predictions,
    oracle_predictions,
    pair_score_rows,
    prediction_from_mapping,
    prompt_payload,
    response_json_schema,
    scenario_stratified_bootstrap,
    validate_dataset_contract,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments" / "counterfactual_admissibility_protocol.json"
DEFAULT_RUNTIME_DIR = ROOT / "tmp" / "counterfactual_admissibility"
DEFAULT_SCORE_DIR = DEFAULT_RUNTIME_DIR / "scores"
DEFAULT_PUBLISH_DIR = ROOT / "results" / "counterfactual_admissibility"
DEFAULT_DOTENV = provider_runtime.DEFAULT_DOTENV
ProviderBinding = provider_runtime.ProviderBinding
PROVIDER_CALLS = provider_runtime.PROVIDER_CALLS


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
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        return ()
    return tuple(
        _mapping(json.loads(line), f"{path}:{index}")
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    )


@dataclass(frozen=True)
class Protocol:
    path: Path
    raw: Mapping[str, Any]
    prompt_path: Path
    prompt_text: str
    prompt_sha256: str
    dataset_path: Path
    dataset_sha256: str
    pairs: tuple[CounterfactualPair, ...]
    providers: tuple[ProviderBinding, ...]


def load_protocol(path: Path) -> Protocol:
    """Load the hash-bound protocol and validate the complete controlled dataset."""
    raw = _read_json(path)
    if raw.get("schema_version") != 1:
        raise ValueError("counterfactual protocol schema version drifted")
    if raw.get("protocol_id") != "controlled-counterfactual-admissibility-v1":
        raise ValueError("counterfactual protocol identity drifted")
    if raw.get("status") != "controlled_public_dev_diagnostic_not_official_benchmark_result":
        raise ValueError("counterfactual protocol status drifted")

    prompt = _mapping(raw.get("prompt"), "protocol.prompt")
    prompt_path = ROOT / _string(prompt.get("path"), "protocol.prompt.path")
    prompt_hash = _string(prompt.get("sha256"), "protocol.prompt.sha256")
    if not prompt_path.is_file() or _sha256_file(prompt_path) != prompt_hash:
        raise ValueError("counterfactual prompt hash drifted")

    dataset = _mapping(raw.get("dataset"), "protocol.dataset")
    dataset_path = ROOT / _string(dataset.get("path"), "protocol.dataset.path")
    dataset_hash = _string(dataset.get("sha256"), "protocol.dataset.sha256")
    if not dataset_path.is_file() or _sha256_file(dataset_path) != dataset_hash:
        raise ValueError("counterfactual dataset hash drifted")
    scenarios = load_scenarios(dataset_path)
    sample = _mapping(raw.get("sample"), "protocol.sample")
    axes = tuple(_string(axis, "protocol.sample.axes[]") for axis in sample.get("axes", []))
    contract = validate_dataset_contract(
        scenarios,
        axes=axes,
        scenarios_per_axis=_integer(
            sample.get("scenarios_per_axis"),
            "protocol.sample.scenarios_per_axis",
            minimum=1,
        ),
        query_pairs_per_scenario=_integer(
            sample.get("query_pairs_per_scenario"),
            "protocol.sample.query_pairs_per_scenario",
            minimum=1,
        ),
        candidates_per_pair=_integer(
            sample.get("candidates_per_case"),
            "protocol.sample.candidates_per_case",
            minimum=1,
        ),
        minimum_query_token_jaccard=_number(
            sample.get("minimum_query_token_jaccard"),
            "protocol.sample.minimum_query_token_jaccard",
        ),
    )
    expected_counts = {
        "scenario_count": _integer(sample.get("scenario_count"), "sample.scenario_count"),
        "pair_count": _integer(sample.get("pair_count"), "sample.pair_count"),
        "case_count": _integer(sample.get("condition_case_count"), "sample.condition_case_count"),
    }
    if any(contract[key] != value for key, value in expected_counts.items()):
        raise ValueError("counterfactual protocol sample counts drifted")

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
    expected_surfaces = {
        "OpenAI": "responses",
        "DeepSeek": "chat_completions",
        "Gemini": "generate_content",
        "Anthropic": "messages",
    }
    if {binding.provider: binding.api_surface for binding in bindings} != expected_surfaces:
        raise ValueError("counterfactual provider panel drifted")

    execution = _mapping(raw.get("execution"), "protocol.execution")
    fixture_total = _number(
        execution.get("fixture_total_hard_cap_usd"),
        "protocol.execution.fixture_total_hard_cap_usd",
    )
    total = _number(
        execution.get("total_hard_cap_usd"),
        "protocol.execution.total_hard_cap_usd",
    )
    if sum(binding.hard_cap_usd for binding in bindings) + fixture_total > total:
        raise ValueError("counterfactual provider caps exceed total hard cap")
    return Protocol(
        path=path,
        raw=raw,
        prompt_path=prompt_path,
        prompt_text=prompt_path.read_text(encoding="utf-8"),
        prompt_sha256=prompt_hash,
        dataset_path=dataset_path,
        dataset_sha256=dataset_hash,
        pairs=expand_pairs(scenarios),
        providers=tuple(bindings),
    )


def validate_protocol(protocol: Protocol) -> dict[str, object]:
    """Return a content-free validation receipt."""
    sample = _mapping(protocol.raw["sample"], "protocol.sample")
    summary = validate_dataset_contract(
        load_scenarios(protocol.dataset_path),
        axes=tuple(_string(axis, "protocol.sample.axes[]") for axis in sample["axes"]),
        scenarios_per_axis=_integer(sample["scenarios_per_axis"], "scenarios_per_axis"),
        query_pairs_per_scenario=_integer(
            sample["query_pairs_per_scenario"],
            "query_pairs_per_scenario",
        ),
        candidates_per_pair=_integer(sample["candidates_per_case"], "candidates_per_case"),
        minimum_query_token_jaccard=_number(
            sample["minimum_query_token_jaccard"],
            "minimum_query_token_jaccard",
        ),
    )
    return {
        "status": "valid",
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": _sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "dataset_sha256": protocol.dataset_sha256,
        **summary,
        "providers": [
            {"provider": binding.provider, "model": binding.model} for binding in protocol.providers
        ],
        "heldout_access": False,
        "reader_or_judge": False,
        "official_result": False,
    }


def _provider_slug(binding: ProviderBinding) -> str:
    return f"{binding.provider}-{binding.model}".casefold().replace(" ", "-").replace("/", "-")


def _fixture_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "fixtures" / f"{_provider_slug(binding)}.json"


def _response_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "raw" / f"{_provider_slug(binding)}.jsonl"


def _completion_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "completion" / f"{_provider_slug(binding)}.json"


def _failure_path(runtime_dir: Path, binding: ProviderBinding) -> Path:
    return runtime_dir / "failures" / f"{_provider_slug(binding)}.json"


def _binding(protocol: Protocol, provider: str) -> ProviderBinding:
    matches = [binding for binding in protocol.providers if binding.provider == provider]
    if len(matches) != 1:
        raise ValueError(f"unknown provider {provider!r}")
    return matches[0]


def _execution_settings(protocol: Protocol) -> Mapping[str, Any]:
    return _mapping(protocol.raw["execution"], "protocol.execution")


def _call_cost(binding: ProviderBinding, usage: Mapping[str, int]) -> float:
    return (
        usage["input_tokens"] * binding.input_usd_per_million
        + usage["output_tokens"] * binding.output_usd_per_million
    ) / 1_000_000


def _conservative_call_bound(
    protocol: Protocol,
    binding: ProviderBinding,
    payload: str,
) -> float:
    execution = _execution_settings(protocol)
    max_output = _integer(
        execution["max_output_tokens_per_case"],
        "max_output_tokens_per_case",
        minimum=1,
    )
    input_byte_bound = len(protocol.prompt_text.encode("utf-8")) + len(payload.encode("utf-8"))
    return (
        input_byte_bound * binding.input_usd_per_million
        + max_output * binding.output_usd_per_million
    ) / 1_000_000


def _adapter_sha256(provider: str) -> str:
    return provider_runtime._provider_adapter_sha256(provider)


def _credential(binding: ProviderBinding, dotenv: Path) -> str:
    values = provider_runtime._load_dotenv(dotenv)
    key = provider_runtime._credential_key(binding.provider)
    value = values.get(key)
    if not value:
        raise ValueError(f"credential {key} is missing")
    return value


def _request(
    protocol: Protocol,
    pair: CounterfactualPair,
    *,
    condition: str,
    binding: ProviderBinding,
    api_key: str,
) -> tuple[Mapping[str, object], Mapping[str, int], int, float, str]:
    execution = _execution_settings(protocol)
    adapter = PROVIDER_CALLS[binding.provider]
    return adapter(
        binding,
        api_key=api_key,
        system_prompt=protocol.prompt_text,
        user_prompt=prompt_payload(pair, condition=condition),
        schema=response_json_schema(len(pair.candidates)),
        max_output_tokens=_integer(
            execution["max_output_tokens_per_case"],
            "max_output_tokens_per_case",
            minimum=1,
        ),
        timeout_seconds=_integer(
            execution["timeout_seconds"],
            "timeout_seconds",
            minimum=1,
        ),
        max_retries=_integer(execution["max_transport_retries"], "max_transport_retries"),
    )


def fixture_provider(
    protocol: Protocol,
    *,
    binding: ProviderBinding,
    dotenv: Path,
    runtime_dir: Path,
) -> Mapping[str, object]:
    """Run one public controlled adapter fixture and write a hash-bound receipt."""
    receipt_path = _fixture_path(runtime_dir, binding)
    pair = protocol.pairs[0]
    condition = "allow"
    payload = prompt_payload(pair, condition=condition)
    expected = {
        "status": "passed",
        "provider": binding.provider,
        "model": binding.model,
        "protocol_sha256": _sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "dataset_sha256": protocol.dataset_sha256,
        "adapter_sha256": _adapter_sha256(binding.provider),
        "fixture_case_id": pair.case_id(condition),
        "fixture_payload_sha256": _sha256_text(payload),
    }
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        if all(receipt.get(key) == value for key, value in expected.items()):
            return receipt
        raise ValueError("stale fixture receipt exists; refusing to overwrite it")

    execution = _execution_settings(protocol)
    fixture_cap = _number(
        execution["fixture_hard_cap_usd_per_provider"],
        "fixture_hard_cap_usd_per_provider",
    )
    if _conservative_call_bound(protocol, binding, payload) > fixture_cap:
        raise RuntimeError("conservative fixture call bound exceeds provider fixture cap")
    api_key = _credential(binding, dotenv)
    response, usage, attempts, latency_ms, request_hash = _request(
        protocol,
        pair,
        condition=condition,
        binding=binding,
        api_key=api_key,
    )
    prediction_from_mapping(response, pair, condition=condition)
    cost = _call_cost(binding, usage)
    if cost > fixture_cap:
        raise RuntimeError("provider fixture exceeded its hard cap")
    receipt = {
        **expected,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": cost,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "request_sha256": request_hash,
        "prediction_saved": False,
        "query_or_candidate_text_saved": False,
    }
    _write_json(receipt_path, receipt)
    return receipt


def _validate_fixture(
    protocol: Protocol,
    binding: ProviderBinding,
    runtime_dir: Path,
) -> Mapping[str, Any]:
    path = _fixture_path(runtime_dir, binding)
    if not path.is_file():
        raise FileNotFoundError(f"provider fixture receipt not found: {path}")
    receipt = _read_json(path)
    pair = protocol.pairs[0]
    expected = {
        "status": "passed",
        "provider": binding.provider,
        "model": binding.model,
        "protocol_sha256": _sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "dataset_sha256": protocol.dataset_sha256,
        "adapter_sha256": _adapter_sha256(binding.provider),
        "fixture_case_id": pair.case_id("allow"),
        "fixture_payload_sha256": _sha256_text(prompt_payload(pair, condition="allow")),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("provider fixture receipt binding drifted")
    return receipt


def _case_lookup(protocol: Protocol) -> dict[str, tuple[CounterfactualPair, str]]:
    return {
        pair.case_id(condition): (pair, condition)
        for pair in protocol.pairs
        for condition in CONDITIONS
    }


def _load_provider_predictions(
    protocol: Protocol,
    binding: ProviderBinding,
    runtime_dir: Path,
) -> tuple[dict[str, CounterfactualPrediction], tuple[Mapping[str, Any], ...]]:
    records = _read_jsonl(_response_path(runtime_dir, binding))
    lookup = _case_lookup(protocol)
    predictions = {}
    for record in records:
        case_id = _string(record.get("case_id"), "response.case_id")
        if case_id in predictions:
            raise ValueError(f"duplicate response case {case_id!r}")
        if case_id not in lookup:
            raise ValueError(f"unexpected response case {case_id!r}")
        pair, condition = lookup[case_id]
        expected = {
            "provider": binding.provider,
            "model": binding.model,
            "pair_id": pair.pair_id,
            "condition": condition,
            "protocol_sha256": _sha256_file(protocol.path),
            "prompt_sha256": protocol.prompt_sha256,
            "dataset_sha256": protocol.dataset_sha256,
            "adapter_sha256": _adapter_sha256(binding.provider),
            "case_payload_sha256": _sha256_text(prompt_payload(pair, condition=condition)),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"response binding drifted for {case_id!r}")
        prediction = prediction_from_mapping(
            record.get("prediction"),
            pair,
            condition=condition,
        )
        predictions[case_id] = prediction
    return predictions, records


def execute_provider(
    protocol: Protocol,
    *,
    binding: ProviderBinding,
    dotenv: Path,
    runtime_dir: Path,
) -> Mapping[str, object]:
    """Execute one provider with append-only per-case checkpoints and fail-closed recovery."""
    _validate_fixture(protocol, binding, runtime_dir)
    failure_path = _failure_path(runtime_dir, binding)
    if failure_path.is_file():
        raise RuntimeError("provider has a frozen failure marker; refusing automatic recovery")
    predictions, records = _load_provider_predictions(protocol, binding, runtime_dir)
    if len(predictions) == len(protocol.pairs) * len(CONDITIONS):
        completion = _read_json(_completion_path(runtime_dir, binding))
        return completion

    spent = sum(_number(record.get("cost_usd"), "response.cost_usd") for record in records)
    api_key = _credential(binding, dotenv)
    execution = _execution_settings(protocol)
    total_cap = _number(execution["total_hard_cap_usd"], "total_hard_cap_usd")
    fixture_cost = sum(
        _number(_read_json(_fixture_path(runtime_dir, item)).get("cost_usd"), "fixture.cost_usd")
        for item in protocol.providers
        if _fixture_path(runtime_dir, item).is_file()
    )
    all_execution_cost = sum(
        _number(record.get("cost_usd"), "response.cost_usd")
        for item in protocol.providers
        for record in _read_jsonl(_response_path(runtime_dir, item))
    )

    for pair in protocol.pairs:
        for condition in CONDITIONS:
            case_id = pair.case_id(condition)
            if case_id in predictions:
                continue
            payload = prompt_payload(pair, condition=condition)
            call_bound = _conservative_call_bound(protocol, binding, payload)
            if spent + call_bound > binding.hard_cap_usd:
                raise RuntimeError("next call could exceed provider hard cap")
            if fixture_cost + all_execution_cost + call_bound > total_cap:
                raise RuntimeError("next call could exceed total hard cap")
            try:
                response, usage, attempts, latency_ms, request_hash = _request(
                    protocol,
                    pair,
                    condition=condition,
                    binding=binding,
                    api_key=api_key,
                )
                prediction = prediction_from_mapping(response, pair, condition=condition)
                cost = _call_cost(binding, usage)
                if spent + cost > binding.hard_cap_usd:
                    raise RuntimeError("provider hard cap exceeded")
                if fixture_cost + all_execution_cost + cost > total_cap:
                    raise RuntimeError("total hard cap exceeded")
                record = {
                    "case_id": case_id,
                    "pair_id": pair.pair_id,
                    "condition": condition,
                    "provider": binding.provider,
                    "model": binding.model,
                    "prediction": response,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cost_usd": cost,
                    "attempts": attempts,
                    "latency_ms": latency_ms,
                    "request_sha256": request_hash,
                    "protocol_sha256": _sha256_file(protocol.path),
                    "prompt_sha256": protocol.prompt_sha256,
                    "dataset_sha256": protocol.dataset_sha256,
                    "adapter_sha256": _adapter_sha256(binding.provider),
                    "case_payload_sha256": _sha256_text(payload),
                    "query_or_candidate_text_saved": False,
                }
                _append_jsonl(_response_path(runtime_dir, binding), record)
                predictions[case_id] = prediction
                records = (*records, record)
                spent += cost
                all_execution_cost += cost
            except Exception as error:
                marker = {
                    "status": "frozen_failure",
                    "provider": binding.provider,
                    "model": binding.model,
                    "case_id": case_id,
                    "completed_cases": len(predictions),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:300],
                    "semantic_or_output_repair_attempted": False,
                    "automatic_rerun_allowed": False,
                    "provider_charge_may_be_unaccounted": True,
                    "protocol_sha256": _sha256_file(protocol.path),
                    "checkpoint_sha256": (
                        _sha256_file(_response_path(runtime_dir, binding))
                        if _response_path(runtime_dir, binding).is_file()
                        else None
                    ),
                }
                _write_json(failure_path, marker)
                raise

    response_path = _response_path(runtime_dir, binding)
    completion = {
        "status": "complete",
        "provider": binding.provider,
        "model": binding.model,
        "completed_cases": len(predictions),
        "expected_cases": len(protocol.pairs) * len(CONDITIONS),
        "cost_usd": spent,
        "protocol_sha256": _sha256_file(protocol.path),
        "checkpoint_sha256": _sha256_file(response_path),
        "provider_comparison_eligible": True,
    }
    _write_json(_completion_path(runtime_dir, binding), completion)
    return completion


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


BOOTSTRAP_METRICS = (
    "strict_focal_pair_consistency",
    "focal_direction_accuracy",
    "focal_directional_margin",
    "stable_control_accuracy",
    "stable_control_overflip_rate",
    "candidate_accuracy",
)


def _scope_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, str, tuple[Mapping[str, object], ...]], ...]:
    axes = sorted({_string(row["axis"], "axis") for row in rows})
    return (
        ("overall", "all", tuple(rows)),
        *(("axis", axis, tuple(row for row in rows if row["axis"] == axis)) for axis in axes),
    )


def score_responses(
    protocol: Protocol,
    *,
    runtime_dir: Path,
    score_dir: Path,
) -> Mapping[str, object]:
    """Score only complete provider bundles and write content-free aggregate artifacts."""
    pairs = protocol.pairs
    model_rows = {
        "released_oracle": pair_score_rows(
            pairs,
            oracle_predictions(pairs),
            model="released_oracle",
        ),
        "no_verifier_keep_all": pair_score_rows(
            pairs,
            no_verifier_predictions(pairs),
            model="no_verifier_keep_all",
        ),
    }
    eligible = []
    excluded = []
    usage_rows = []
    expected_cases = len(pairs) * len(CONDITIONS)
    for binding in protocol.providers:
        failure = _failure_path(runtime_dir, binding).is_file()
        try:
            predictions, records = _load_provider_predictions(protocol, binding, runtime_dir)
        except (TypeError, ValueError) as error:
            predictions, records = {}, _read_jsonl(_response_path(runtime_dir, binding))
            failure = True
            excluded.append(
                {
                    "provider": binding.provider,
                    "model": binding.model,
                    "reason": f"invalid_contract:{type(error).__name__}",
                }
            )
        if not any(row["provider"] == binding.provider for row in excluded):
            if failure:
                reason = "frozen_failure"
            elif len(predictions) != expected_cases:
                reason = f"incomplete:{len(predictions)}/{expected_cases}"
            else:
                reason = "complete"
            if reason == "complete":
                model_key = f"{binding.provider}/{binding.model}"
                model_rows[model_key] = pair_score_rows(pairs, predictions, model=model_key)
                eligible.append({"provider": binding.provider, "model": binding.model})
            else:
                excluded.append(
                    {"provider": binding.provider, "model": binding.model, "reason": reason}
                )
        latencies = [_number(record.get("latency_ms"), "latency_ms") for record in records]
        fixture_cost = (
            _number(_read_json(_fixture_path(runtime_dir, binding)).get("cost_usd"), "fixture cost")
            if _fixture_path(runtime_dir, binding).is_file()
            else 0.0
        )
        usage_rows.append(
            {
                "provider": binding.provider,
                "model": binding.model,
                "comparison_eligible": any(
                    item["provider"] == binding.provider for item in eligible
                ),
                "completed_cases": len(records),
                "expected_cases": expected_cases,
                "input_tokens": sum(
                    _integer(record.get("input_tokens"), "input_tokens") for record in records
                ),
                "output_tokens": sum(
                    _integer(record.get("output_tokens"), "output_tokens") for record in records
                ),
                "execution_cost_usd": sum(
                    _number(record.get("cost_usd"), "cost_usd") for record in records
                ),
                "fixture_cost_usd": fixture_cost,
                "total_known_cost_usd": fixture_cost
                + sum(_number(record.get("cost_usd"), "cost_usd") for record in records),
                "latency_p50_ms": _quantile(latencies, 0.5),
                "latency_p95_ms": _quantile(latencies, 0.95),
                "failure_marker": failure,
            }
        )

    pair_rows = [row for rows in model_rows.values() for row in rows]
    _write_csv(score_dir / "pair_scores.csv", pair_rows)

    aggregate_rows = []
    bootstrap_rows = []
    evaluation = _mapping(protocol.raw["evaluation"], "protocol.evaluation")
    replicates = _integer(evaluation["bootstrap_replicates"], "bootstrap_replicates", minimum=1)
    seed = _integer(evaluation["bootstrap_seed"], "bootstrap_seed")
    for model, rows in model_rows.items():
        for scope, axis, selected in _scope_rows(rows):
            aggregate_rows.append(
                {"model": model, "scope": scope, "axis": axis, **aggregate_pair_metrics(selected)}
            )
            for metric in BOOTSTRAP_METRICS:
                result = scenario_stratified_bootstrap(
                    selected,
                    metric=metric,
                    replicates=replicates,
                    seed=seed,
                )
                bootstrap_rows.append(
                    {
                        "model": model,
                        "scope": scope,
                        "axis": axis,
                        **result,
                    }
                )
    _write_csv(score_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(score_dir / "bootstrap_ci.csv", bootstrap_rows)

    paired_delta_rows = []
    comparator_names = ("no_verifier_keep_all", "released_oracle")
    for model, rows in model_rows.items():
        if model in comparator_names:
            continue
        treatment = {row["pair_id"]: row for row in rows}
        for comparator_name in comparator_names:
            comparator = {row["pair_id"]: row for row in model_rows[comparator_name]}
            for metric in BOOTSTRAP_METRICS:
                deltas = tuple(
                    {
                        **row,
                        "delta": float(row[metric]) - float(comparator[pair_id][metric]),
                    }
                    for pair_id, row in treatment.items()
                )
                for scope, axis, selected in _scope_rows(deltas):
                    result = scenario_stratified_bootstrap(
                        selected,
                        metric="delta",
                        replicates=replicates,
                        seed=seed,
                    )
                    paired_delta_rows.append(
                        {
                            "model": model,
                            "comparator": comparator_name,
                            "scope": scope,
                            "axis": axis,
                            "metric": metric,
                            "estimate": result["estimate"],
                            "ci_lower": result["ci_lower"],
                            "ci_upper": result["ci_upper"],
                            "bootstrap_replicates": result["bootstrap_replicates"],
                            "bootstrap_unit": result["bootstrap_unit"],
                            "bootstrap_stratification": result["bootstrap_stratification"],
                        }
                    )
    if paired_delta_rows:
        _write_csv(score_dir / "paired_deltas.csv", paired_delta_rows)
    _write_csv(score_dir / "provider_usage.csv", usage_rows)

    output_names = [
        "pair_scores.csv",
        "aggregate_metrics.csv",
        "bootstrap_ci.csv",
        "provider_usage.csv",
    ]
    if paired_delta_rows:
        output_names.append("paired_deltas.csv")
    manifest = {
        "schema_version": 1,
        "status": "controlled_public_dev_diagnostic_not_official_benchmark_result",
        "protocol_sha256": _sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "dataset_sha256": protocol.dataset_sha256,
        "comparison_eligible_providers": eligible,
        "excluded_providers": excluded,
        "scored_models": list(model_rows),
        "expected_cases_per_provider": expected_cases,
        "prefix_scoring": False,
        "selection_or_tuning": False,
        "reader_or_judge": False,
        "official_result": False,
        "outputs": [
            {"path": name, "sha256": _sha256_file(score_dir / name)} for name in output_names
        ],
    }
    _write_json(score_dir / "manifest.json", manifest)
    return manifest


def _float(value: str) -> float:
    return float(value)


def publish_scores(
    protocol: Protocol,
    *,
    score_dir: Path,
    publish_dir: Path,
) -> Mapping[str, object]:
    """Publish only content-free scored artifacts and a deterministic diagnostic summary."""
    manifest = _read_json(score_dir / "manifest.json")
    if manifest.get("protocol_sha256") != _sha256_file(protocol.path):
        raise ValueError("score manifest protocol binding drifted")
    eligible = _sequence(
        manifest.get("comparison_eligible_providers"),
        "manifest.comparison_eligible_providers",
    )
    if not eligible:
        raise ValueError("refusing to publish without a complete provider bundle")
    for output in _sequence(manifest.get("outputs"), "manifest.outputs"):
        item = _mapping(output, "manifest.outputs[]")
        name = _string(item.get("path"), "manifest.outputs[].path")
        source = score_dir / name
        if _sha256_file(source) != item.get("sha256"):
            raise ValueError(f"score output hash drifted: {name}")
        publish_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, publish_dir / name)
    shutil.copyfile(score_dir / "manifest.json", publish_dir / "manifest.json")

    with (score_dir / "aggregate_metrics.csv").open(encoding="utf-8", newline="") as handle:
        aggregate = [row for row in csv.DictReader(handle) if row["scope"] == "overall"]
    lines = [
        "# Counterfactual Admissibility Diagnostic",
        "",
        "This is a controlled public-development diagnostic, not an official benchmark result.",
        "Candidate pools are fixed within each pair; only the query condition changes.",
        "",
        "| Model | Strict focal pair | Direction | Stable overflip | Candidate accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {model} | {pair:.4f} | {direction:.4f} | {overflip:.4f} | {accuracy:.4f} |".format(
                model=row["model"],
                pair=_float(row["strict_focal_pair_consistency"]),
                direction=_float(row["focal_direction_accuracy"]),
                overflip=_float(row["stable_control_overflip_rate"]),
                accuracy=_float(row["candidate_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "The frozen supportive pattern requires at least 0.80 strict focal-pair consistency,",
            "at least 0.75 on every axis, stable-control overflip at most 0.05, and a",
            "positive mean directional margin. Passing would establish explicit-condition rule",
            "application only; it would not establish latent metadata inference or an E6/M2 gate.",
            "",
        ]
    )
    readme = publish_dir / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "status": "published",
        "publish_dir": str(publish_dir),
        "comparison_eligible_provider_count": len(eligible),
        "readme_sha256": _sha256_file(readme),
    }


def _print(value: object) -> None:
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--provider", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--provider", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    publish.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH_DIR)
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    if args.command == "validate":
        result = validate_protocol(protocol)
    elif args.command == "fixture":
        result = fixture_provider(
            protocol,
            binding=_binding(protocol, args.provider),
            dotenv=args.dotenv,
            runtime_dir=args.runtime_dir,
        )
    elif args.command == "execute":
        result = execute_provider(
            protocol,
            binding=_binding(protocol, args.provider),
            dotenv=args.dotenv,
            runtime_dir=args.runtime_dir,
        )
    elif args.command == "score":
        result = score_responses(
            protocol,
            runtime_dir=args.runtime_dir,
            score_dir=args.score_dir,
        )
    else:
        result = publish_scores(
            protocol,
            score_dir=args.score_dir,
            publish_dir=args.publish_dir,
        )
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
