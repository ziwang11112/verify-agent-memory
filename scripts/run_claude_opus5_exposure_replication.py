"""Run the separately authorized Claude Opus 5 exposure replication."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from scripts import run_counterfactual_exposure_execution as exposure_runtime
from scripts import run_counterfactual_exposure_intervention as zero_runtime
from scripts import run_inferred_admissibility_experiment as provider_runtime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments" / "claude_opus5_exposure_replication_protocol.json"
DEFAULT_RUNTIME_DIR = ROOT / "tmp" / "counterfactual_exposure" / "claude_opus5_replication"
DEFAULT_UNLOCK = DEFAULT_RUNTIME_DIR / "paid_execution_unlock.json"
DEFAULT_DOTENV = provider_runtime.DEFAULT_DOTENV


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _anthropic_opus5_request(
    binding: provider_runtime.ProviderBinding,
    *,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, object],
    max_output_tokens: int,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[Mapping[str, object], dict[str, int], int, float, str]:
    """Call Opus 5 without extended thinking and with strict JSON output."""
    if dict(binding.controls) != {"effort": "medium", "thinking": "disabled"}:
        raise ValueError("Claude Opus 5 controls drifted")
    body: dict[str, object] = {
        "model": binding.model,
        "max_tokens": max_output_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "thinking": {"type": "disabled"},
        "output_config": {
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": provider_runtime._anthropic_schema(schema),
            },
        },
    }
    response, attempts, latency_ms = provider_runtime._http_json(
        url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        body=body,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    content = provider_runtime._sequence(response.get("content"), "Anthropic content")
    if response.get("stop_reason") != "end_turn":
        raise ValueError("Anthropic response did not finish normally")
    text_blocks = [
        provider_runtime._string(block.get("text"), "Anthropic text")
        for raw_block in content
        if (block := provider_runtime._mapping(raw_block, "Anthropic content block")).get("type")
        == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("Anthropic response must contain exactly one text block")
    usage = provider_runtime._mapping(response.get("usage"), "Anthropic usage")
    return (
        provider_runtime._mapping(json.loads(text_blocks[0]), "Anthropic JSON output"),
        {
            "input_tokens": provider_runtime._integer(
                usage.get("input_tokens"), "Anthropic input tokens"
            ),
            "output_tokens": provider_runtime._integer(
                usage.get("output_tokens"), "Anthropic output tokens"
            ),
        },
        attempts,
        latency_ms,
        provider_runtime._sha256_object(body),
    )


def _opus_adapter_sha256() -> str:
    source = "\n\n".join(
        inspect.getsource(function)
        for function in (
            provider_runtime._http_json,
            provider_runtime._anthropic_schema,
            _anthropic_opus5_request,
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@contextmanager
def _installed_opus_adapter() -> Iterator[None]:
    previous_call = exposure_runtime.PROVIDER_CALLS.get("Anthropic")
    previous_hash = exposure_runtime._adapter_sha256
    exposure_runtime.PROVIDER_CALLS["Anthropic"] = _anthropic_opus5_request
    exposure_runtime._adapter_sha256 = lambda provider: (
        _opus_adapter_sha256() if provider == "Anthropic" else previous_hash(provider)
    )
    try:
        yield
    finally:
        if previous_call is None:
            exposure_runtime.PROVIDER_CALLS.pop("Anthropic", None)
        else:
            exposure_runtime.PROVIDER_CALLS["Anthropic"] = previous_call
        exposure_runtime._adapter_sha256 = previous_hash


def _bound_file(value: object, label: str) -> Path:
    _reference, path, _expected_hash = exposure_runtime._bound_file(value, label)
    return path


def load_protocol(path: Path) -> exposure_runtime.ExecutionProtocol:
    """Load the Opus-only replication without touching credentials or providers."""
    raw = exposure_runtime._read_json(path)
    expected_fields = {
        "schema_version",
        "protocol_id",
        "status",
        "base_protocol",
        "implementation",
        "provider",
        "execution",
        "authorization",
        "interpretation",
    }
    if set(raw) != expected_fields:
        raise ValueError("replication protocol has missing or unknown fields")
    if raw["schema_version"] != 1:
        raise ValueError("replication protocol schema version drifted")
    if raw["protocol_id"] != "claude-opus5-paired-exposure-replication-v1":
        raise ValueError("replication protocol identity drifted")
    if raw["status"] != "implementation_only_paid_execution_not_authorized":
        raise ValueError("replication protocol status drifted")

    base_path = _bound_file(raw["base_protocol"], "protocol.base_protocol")
    base_protocol = zero_runtime.load_protocol(base_path)
    implementation = exposure_runtime._mapping(raw["implementation"], "implementation")
    if set(implementation) != {
        "replication_runner",
        "inherited_execution_runtime",
        "provider_adapter_runtime",
    }:
        raise ValueError("replication implementation identity drifted")
    _bound_file(implementation["replication_runner"], "implementation.replication_runner")
    _bound_file(
        implementation["inherited_execution_runtime"],
        "implementation.inherited_execution_runtime",
    )
    _bound_file(
        implementation["provider_adapter_runtime"],
        "implementation.provider_adapter_runtime",
    )

    binding = exposure_runtime._binding_from_mapping(raw["provider"], "protocol.provider")
    expected_identity = (
        "Anthropic",
        "claude-opus-5",
        "messages",
        {"effort": "medium", "thinking": "disabled"},
    )
    if (
        binding.provider,
        binding.model,
        binding.api_surface,
        dict(binding.controls),
    ) != expected_identity:
        raise ValueError("Claude Opus 5 provider contract drifted")

    execution = exposure_runtime._mapping(raw["execution"], "protocol.execution")
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
        raise ValueError("replication execution settings drifted")
    input_bytes = exposure_runtime._all_request_input_bytes(base_protocol)
    request_count = len(input_bytes)
    fixture_calls = exposure_runtime._integer(
        execution["fixture_calls_per_provider"], "fixture_calls_per_provider"
    )
    max_calls = exposure_runtime._integer(
        execution["max_calls_per_provider"], "max_calls_per_provider"
    )
    max_output_tokens = exposure_runtime._integer(
        execution["max_output_tokens_per_request"],
        "max_output_tokens_per_request",
        minimum=1,
    )
    timeout_seconds = exposure_runtime._integer(
        execution["timeout_seconds"], "timeout_seconds", minimum=1
    )
    max_retries = exposure_runtime._integer(
        execution["max_transport_retries"], "max_transport_retries"
    )
    if (
        execution["request_count_per_provider"] != request_count
        or fixture_calls != 1
        or max_calls != request_count + fixture_calls
        or execution["input_byte_bound_total"] != sum(input_bytes)
        or execution["max_input_byte_bound_per_request"] != max(input_bytes)
    ):
        raise ValueError("replication request or byte bounds drifted")
    if (
        execution["checkpoint_after_each_request"] is not True
        or execution["complete_bundle_required"] is not True
        or execution["semantic_or_output_repair"] is not False
        or execution["selective_rerun"] is not False
        or execution["input_token_bound"] != "one_utf8_byte_per_token_conservative_upper_bound"
        or max_retries != 0
    ):
        raise ValueError("replication safety controls drifted")
    total_cap = exposure_runtime._number(execution["total_hard_cap_usd"], "total_hard_cap_usd")
    if binding.hard_cap_usd > total_cap:
        raise ValueError("provider hard cap exceeds total hard cap")
    conservative_bound = (
        (sum(input_bytes) + max(input_bytes)) * binding.input_usd_per_million
        + max_calls * max_output_tokens * binding.output_usd_per_million
    ) / 1_000_000
    if conservative_bound > binding.hard_cap_usd:
        raise ValueError("conservative cost bound exceeds Anthropic hard cap")

    authorization = exposure_runtime._mapping(raw["authorization"], "authorization")
    if authorization != {
        "unlock_required": True,
        "exact_commit_required": True,
        "exact_protocol_hash_required": True,
        "explicit_owner_required": True,
        "credential_read_before_unlock": False,
        "provider_call_before_unlock": False,
    }:
        raise ValueError("replication authorization contract drifted")
    interpretation = exposure_runtime._mapping(raw["interpretation"], "interpretation")
    if interpretation != {
        "role": "cross_provider_reader_replication",
        "changes_existing_results": False,
        "model_pooling": False,
        "official_result": False,
    }:
        raise ValueError("replication interpretation contract drifted")

    return exposure_runtime.ExecutionProtocol(
        path=path,
        raw=raw,
        base=base_protocol,
        providers=(binding,),
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        max_transport_retries=max_retries,
        total_hard_cap_usd=total_cap,
        max_calls_per_provider=max_calls,
    )


def validate_protocol(protocol: exposure_runtime.ExecutionProtocol) -> Mapping[str, object]:
    """Return a credential-free and provider-free replication receipt."""
    receipt = exposure_runtime.validate_execution_protocol(protocol)
    return {
        **receipt,
        "replication_role": "cross_provider_reader_replication",
        "thinking": "disabled",
        "effort": "medium",
        "changes_existing_results": False,
        "model_pooling": False,
    }


def _run_command(
    protocol: exposure_runtime.ExecutionProtocol,
    *,
    command: str,
    unlock_path: Path,
    dotenv: Path,
    runtime_dir: Path,
    owner: str | None = None,
) -> Mapping[str, object]:
    binding = exposure_runtime._binding(protocol, "Anthropic")
    with _installed_opus_adapter():
        if command == "validate":
            return validate_protocol(protocol)
        if command == "unlock-template":
            if owner is None:
                raise ValueError("owner is required")
            return exposure_runtime.unlock_template(protocol, owner=owner)
        if command == "preflight":
            return exposure_runtime.preflight(protocol, unlock_path=unlock_path)
        if command == "fixture":
            return exposure_runtime.fixture_provider(
                protocol,
                binding=binding,
                unlock_path=unlock_path,
                dotenv=dotenv,
                runtime_dir=runtime_dir,
            )
        if command == "execute":
            return exposure_runtime.execute_provider(
                protocol,
                binding=binding,
                unlock_path=unlock_path,
                dotenv=dotenv,
                runtime_dir=runtime_dir,
            )
        if command == "score":
            return exposure_runtime.score_provider(
                protocol,
                binding=binding,
                unlock_path=unlock_path,
                runtime_dir=runtime_dir,
            )
    raise ValueError(f"unknown command {command!r}")


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
    subparsers.add_parser("fixture")
    subparsers.add_parser("execute")
    subparsers.add_parser("score")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    result = _run_command(
        protocol,
        command=args.command,
        unlock_path=args.unlock,
        dotenv=args.dotenv,
        runtime_dir=args.runtime_dir,
        owner=getattr(args, "owner", None),
    )
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
