"""Import compatible v2 natural checkpoints into the cost-aware two-reader runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import import_natural_verifier_bundle as verifier_import  # noqa: E402
from scripts import run_inferred_admissibility_experiment as provider_runtime  # noqa: E402
from scripts import run_natural_end_to_end_experiment as run  # noqa: E402
from verify_agent_memory.natural_end_to_end import (  # noqa: E402
    ReaderResponse,
    build_reader_requests,
    load_cases,
    reader_response_schema,
)

SOURCE_PROTOCOL_PATH = "experiments/natural_end_to_end_protocol.json"
DEFAULT_SOURCE_RUNTIME = ROOT / "tmp" / "natural_end_to_end" / "provider_runtime"
DEFAULT_TARGET_RUNTIME = ROOT / "tmp" / "natural_end_to_end" / "two_reader_runtime"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_set_sha256(path: Path) -> str:
    entries = "".join(
        f"{item.name}:{run._sha256_file(item)}\n"
        for item in sorted(path.iterdir(), key=lambda item: item.name)
        if item.is_file()
    )
    return _sha256_bytes(entries.encode("utf-8"))


def _checkpoint_reuse(protocol: run.Protocol) -> Mapping[str, Any]:
    execution = run._mapping(protocol.raw.get("execution"), "protocol.execution")
    return run._mapping(execution.get("checkpoint_reuse"), "execution.checkpoint_reuse")


def assert_protocol_compatibility(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    if source.get("protocol_id") != "natural-heldout-route-to-reader-v2":
        raise ValueError("checkpoint source is not natural end-to-end v2")
    if target.get("protocol_id") != "natural-heldout-route-to-reader-two-reader-v3":
        raise ValueError("checkpoint target is not the cost-aware two-reader v3")
    for key in ("source_contract", "retrieval", "text_verifier"):
        if source.get(key) != target.get(key):
            raise ValueError(f"two-reader amendment changed frozen {key}")

    source_reader = run._mapping(source.get("reader"), "source.reader")
    target_reader = run._mapping(target.get("reader"), "target.reader")
    for key in (
        "prompt",
        "response_schema",
        "exact_visible_prompt_deduplication",
        "models_reported_separately",
    ):
        if source_reader.get(key) != target_reader.get(key):
            raise ValueError(f"two-reader amendment changed reader {key}")
    source_providers = {
        run._string(run._mapping(row, "source provider").get("provider"), "provider"): row
        for row in run._sequence(source_reader.get("providers"), "source.reader.providers")
    }
    target_providers = tuple(
        run._mapping(row, "target provider")
        for row in run._sequence(target_reader.get("providers"), "target.reader.providers")
    )
    if tuple(row.get("provider") for row in target_providers) != ("Gemini", "DeepSeek"):
        raise ValueError("two-reader amendment provider order drifted")
    for row in target_providers:
        if source_providers.get(row.get("provider")) != row:
            raise ValueError(f"reader binding drifted for {row.get('provider')}")

    source_execution = run._mapping(source.get("execution"), "source.execution")
    target_execution = run._mapping(target.get("execution"), "target.execution")
    for key in (
        "maximum_reader_calls_per_provider_before_exact_prompt_deduplication",
        "maximum_output_tokens_reader",
        "maximum_output_tokens_verifier",
        "timeout_seconds",
        "maximum_transport_retries",
        "checkpoint_after_each_successful_call",
        "semantic_or_output_repair",
        "selective_rerun",
        "complete_bundle_required_for_comparison",
        "official_rhelm_or_memops_claim",
    ):
        if source_execution.get(key) != target_execution.get(key):
            raise ValueError(f"two-reader amendment changed execution field {key}")


def reader_request_body(
    protocol: run.Protocol,
    binding: run.ProviderBinding,
    spec: run.CallSpec,
) -> dict[str, object]:
    if binding.provider == "Gemini":
        return {
            "systemInstruction": {"parts": [{"text": protocol.reader_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": spec.payload}]}],
            "generationConfig": {
                "maxOutputTokens": protocol.maximum_output_tokens_reader,
                "responseMimeType": "application/json",
                "responseJsonSchema": provider_runtime._limited_provider_schema(spec.schema),
                "thinkingConfig": {"thinkingLevel": binding.controls["thinking_level"].upper()},
            },
        }
    if binding.provider == "DeepSeek":
        return {
            "model": binding.model,
            "messages": [
                {"role": "system", "content": protocol.reader_prompt},
                {"role": "user", "content": spec.payload},
            ],
            "max_tokens": protocol.maximum_output_tokens_reader,
            "thinking": {"type": binding.controls["thinking"]},
            "response_format": {"type": "json_object"},
        }
    raise ValueError(f"unsupported two-reader provider {binding.provider!r}")


def migrated_record(
    source: Mapping[str, Any],
    *,
    target_protocol_sha256: str,
    target_commit: str,
    source_record_sha256: str,
) -> dict[str, Any]:
    output = dict(source)
    prior_compatibility = output.pop("compatibility_source", None)
    output["protocol_sha256"] = target_protocol_sha256
    output["implementation_commit"] = target_commit
    output["compatibility_source"] = {
        "protocol_sha256": source["protocol_sha256"],
        "implementation_commit": source["implementation_commit"],
        "record_sha256": source_record_sha256,
        "response_unchanged": True,
        "prior_compatibility_source": prior_compatibility,
    }
    return output


def _validate_source_record(
    source_path: Path,
    *,
    source_protocol_sha256: str,
    source_commit: str,
    stage: str,
    binding: run.ProviderBinding,
    spec: run.CallSpec,
    parser: Callable[[object, str], object],
    request_body_sha256: str,
) -> Mapping[str, Any]:
    source_record = run._mapping(json.loads(source_path.read_bytes()), str(source_path))
    expected = {
        "schema_version": 1,
        "protocol_sha256": source_protocol_sha256,
        "implementation_commit": source_commit,
        "stage": stage,
        "provider": binding.provider,
        "model": binding.model,
        "request_id": spec.request_id,
        "payload_sha256": hashlib.sha256(spec.payload.encode("utf-8")).hexdigest(),
        "request_body_sha256": request_body_sha256,
    }
    for key, value in expected.items():
        if source_record.get(key) != value:
            raise ValueError(f"source checkpoint drifted: {source_path}/{key}")
    parser(source_record.get("response"), spec.request_id)
    run._record_cost(source_record)
    return source_record


def _copy_audit_failures(source: Path, target: Path) -> list[tuple[str, str]]:
    target.mkdir(parents=True)
    entries = []
    for source_path in sorted(source.iterdir(), key=lambda item: item.name):
        if not source_path.is_file():
            continue
        source_hash = run._sha256_file(source_path)
        shutil.copyfile(source_path, target / source_path.name)
        entries.append((source_path.name, source_hash))
    return entries


def import_checkpoints(
    *,
    source_runtime: Path,
    target_runtime: Path,
    cases_path: Path,
    protocol_path: Path,
) -> dict[str, object]:
    target = run.load_protocol(protocol_path)
    if target.protocol_id != "natural-heldout-route-to-reader-two-reader-v3":
        raise ValueError("checkpoint import requires the two-reader v3 protocol")
    reuse = _checkpoint_reuse(target)
    source_commit = run._string(
        reuse.get("source_implementation_commit"),
        "checkpoint_reuse.source_implementation_commit",
    )
    source_protocol_bytes = verifier_import._git_file_bytes(
        source_commit,
        SOURCE_PROTOCOL_PATH,
    )
    source_protocol_sha256 = _sha256_bytes(source_protocol_bytes)
    if source_protocol_sha256 != reuse.get("source_protocol_sha256"):
        raise ValueError("source protocol SHA-256 differs from checkpoint declaration")
    source_protocol_raw = run._mapping(json.loads(source_protocol_bytes), "source protocol")
    assert_protocol_compatibility(source_protocol_raw, target.raw)
    target_commit = run._require_clean_contract()

    if target_runtime.exists():
        raise RuntimeError("two-reader checkpoint import destination is not empty")
    staging = target_runtime.with_name(f".{target_runtime.name}.importing")
    if staging.exists():
        raise RuntimeError("stale two-reader import staging directory exists")
    staging.mkdir(parents=True)

    cases = load_cases(cases_path)
    verifier_specs, inference_cases = run._verifier_specs(cases, target)
    verifier_parser = run._verifier_parser_factory(inference_cases)
    source_verifier = run._stage_root(source_runtime, "verifier", target.verifier)
    if run._sha256_file(source_verifier / "completion.json") != reuse.get(
        "verifier_completion_sha256"
    ):
        raise ValueError("source verifier completion receipt drifted")
    if run._sha256_file(source_verifier / "compatibility_import.json") != reuse.get(
        "verifier_compatibility_import_sha256"
    ):
        raise ValueError("source verifier compatibility receipt drifted")

    target_verifier = run._stage_root(staging, "verifier", target.verifier)
    target_verifier_responses = target_verifier / "responses"
    target_verifier_responses.mkdir(parents=True)
    verifier_records: dict[str, Mapping[str, Any]] = {}
    verifier_scores: dict[str, dict[str, float]] = {}
    for case in cases:
        request_id = case.case_id
        spec = verifier_specs[request_id]
        source_path = source_verifier / "responses" / f"{request_id}.json"
        source_record = _validate_source_record(
            source_path,
            source_protocol_sha256=source_protocol_sha256,
            source_commit=source_commit,
            stage="verifier",
            binding=target.verifier,
            spec=spec,
            parser=verifier_parser,
            request_body_sha256=provider_runtime._sha256_object(
                verifier_import.verifier_request_body(target, spec)
            ),
        )
        migrated = migrated_record(
            source_record,
            target_protocol_sha256=target.protocol_sha256,
            target_commit=target_commit,
            source_record_sha256=run._sha256_file(source_path),
        )
        (target_verifier_responses / source_path.name).write_bytes(_json_bytes(migrated))
        verifier_records[request_id] = migrated
        prediction = run.verifier_contract.prediction_from_mapping(
            source_record["response"],
            inference_cases[request_id],
        )
        verifier_scores[request_id] = {
            candidate.memory_key: candidate_prediction.violation_probability
            for candidate, candidate_prediction in zip(
                case.namespace_candidates,
                prediction.candidates,
                strict=True,
            )
        }

    source_verifier_completion = run._read_json(source_verifier / "completion.json")
    response_set_sha256 = run._sha256_object(
        [
            {
                "request_id": request_id,
                "response_sha256": run._sha256_object(verifier_records[request_id]["response"]),
            }
            for request_id in sorted(verifier_records)
        ]
    )
    if response_set_sha256 != source_verifier_completion.get("response_set_sha256"):
        raise ValueError("imported verifier response content differs from source bundle")
    verifier_compatibility = {
        "schema_version": 1,
        "status": "compatible_two_reader_verifier_import",
        "source_protocol_sha256": source_protocol_sha256,
        "source_implementation_commit": source_commit,
        "source_completion_sha256": reuse["verifier_completion_sha256"],
        "source_compatibility_import_sha256": reuse["verifier_compatibility_import_sha256"],
        "target_protocol_sha256": target.protocol_sha256,
        "target_implementation_commit": target_commit,
        "request_count": len(verifier_records),
        "response_set_sha256": response_set_sha256,
        "response_content_changed": False,
        "provider_calls_made": 0,
    }
    verifier_compatibility_path = target_verifier / "compatibility_import.json"
    verifier_compatibility_path.write_bytes(_json_bytes(verifier_compatibility))
    verifier_completion = {
        "schema_version": 1,
        "protocol_sha256": target.protocol_sha256,
        "implementation_commit": target_commit,
        "stage": "verifier",
        "provider": target.verifier.provider,
        "model": target.verifier.model,
        "request_count": len(verifier_records),
        "input_tokens": sum(
            int(record["usage"]["input_tokens"]) for record in verifier_records.values()
        ),
        "output_tokens": sum(
            int(record["usage"]["output_tokens"]) for record in verifier_records.values()
        ),
        "cost_usd": sum(run._record_cost(record) for record in verifier_records.values()),
        "response_set_sha256": response_set_sha256,
        "complete_bundle": True,
        "provider_calls_made": 0,
        "compatibility_import_sha256": run._sha256_file(verifier_compatibility_path),
        "compatibility_source": {
            "protocol_sha256": source_protocol_sha256,
            "implementation_commit": source_commit,
            "completion_sha256": reuse["verifier_completion_sha256"],
            "response_unchanged": True,
        },
    }
    (target_verifier / "completion.json").write_bytes(_json_bytes(verifier_completion))

    requests, assignments = build_reader_requests(
        cases,
        verifier_scores=verifier_scores,
        verifier_threshold=target.verifier_threshold,
    )
    reader_specs = {
        request.prompt_sha256: run.CallSpec(
            request_id=request.prompt_sha256,
            payload=request.payload,
            schema=reader_response_schema(),
        )
        for request in requests
    }
    reader_manifest: dict[str, object] = {}
    declared_readers = run._mapping(reuse.get("readers"), "checkpoint_reuse.readers")
    for binding in target.readers:
        declaration = run._mapping(
            declared_readers.get(binding.provider),
            f"checkpoint_reuse.readers.{binding.provider}",
        )
        source_reader = run._stage_root(source_runtime, "reader", binding)
        source_responses = source_reader / "responses"
        source_failures = source_reader / "failures"
        if file_set_sha256(source_responses) != declaration.get("response_file_set_sha256"):
            raise ValueError(f"{binding.provider} source response file set drifted")
        if file_set_sha256(source_failures) != declaration.get("failure_file_set_sha256"):
            raise ValueError(f"{binding.provider} source failure file set drifted")

        target_reader = run._stage_root(staging, "reader", binding)
        target_responses = target_reader / "responses"
        target_responses.mkdir(parents=True)
        imported_count = 0
        imported_cost = 0.0
        for source_path in sorted(source_responses.iterdir(), key=lambda item: item.name):
            if not source_path.is_file():
                continue
            request_id = source_path.stem
            spec = reader_specs.get(request_id)
            if spec is None:
                raise ValueError(f"unknown source reader request {request_id}")
            source_record = _validate_source_record(
                source_path,
                source_protocol_sha256=source_protocol_sha256,
                source_commit=source_commit,
                stage="reader",
                binding=binding,
                spec=spec,
                parser=lambda value, _request_id: ReaderResponse.from_mapping(value),
                request_body_sha256=provider_runtime._sha256_object(
                    reader_request_body(target, binding, spec)
                ),
            )
            migrated = migrated_record(
                source_record,
                target_protocol_sha256=target.protocol_sha256,
                target_commit=target_commit,
                source_record_sha256=run._sha256_file(source_path),
            )
            (target_responses / source_path.name).write_bytes(_json_bytes(migrated))
            imported_count += 1
            imported_cost += run._record_cost(source_record)

        expected_count = run._integer(
            declaration.get("response_count"),
            f"checkpoint_reuse.readers.{binding.provider}.response_count",
        )
        if imported_count != expected_count:
            raise ValueError(f"{binding.provider} imported response count drifted")
        failure_entries = _copy_audit_failures(
            source_failures,
            target_reader / "source_failures",
        )
        expected_failure_count = run._integer(
            declaration.get("failure_count"),
            f"checkpoint_reuse.readers.{binding.provider}.failure_count",
        )
        if len(failure_entries) != expected_failure_count:
            raise ValueError(f"{binding.provider} source failure count drifted")
        reader_manifest[binding.provider] = {
            "model": binding.model,
            "imported_response_count": imported_count,
            "remaining_request_count": len(reader_specs) - imported_count,
            "imported_cost_usd": imported_cost,
            "source_response_file_set_sha256": declaration["response_file_set_sha256"],
            "source_failure_file_set_sha256": declaration["failure_file_set_sha256"],
            "source_failure_count": len(failure_entries),
        }

    manifest = {
        "schema_version": 1,
        "status": "compatible_two_reader_checkpoint_import",
        "source_protocol_sha256": source_protocol_sha256,
        "source_implementation_commit": source_commit,
        "target_protocol_sha256": target.protocol_sha256,
        "target_implementation_commit": target_commit,
        "case_count": len(cases),
        "verifier_response_count": len(verifier_records),
        "verifier_response_set_sha256": response_set_sha256,
        "verifier_compatibility_import_sha256": run._sha256_file(verifier_compatibility_path),
        "reader_unique_request_count_per_provider": len(reader_specs),
        "reader_assignment_count_per_provider": len(assignments),
        "readers": reader_manifest,
        "response_content_changed": False,
        "provider_calls_made": 0,
    }
    (staging / "checkpoint_import.json").write_bytes(_json_bytes(manifest))
    target_runtime.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target_runtime)
    return {
        **manifest,
        "checkpoint_import_sha256": run._sha256_file(target_runtime / "checkpoint_import.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", type=Path, default=DEFAULT_SOURCE_RUNTIME)
    parser.add_argument("--target-runtime", type=Path, default=DEFAULT_TARGET_RUNTIME)
    parser.add_argument("--cases", type=Path, default=run.DEFAULT_CASES)
    parser.add_argument("--protocol", type=Path, default=run.TWO_READER_PROTOCOL)
    args = parser.parse_args()
    print(
        json.dumps(
            import_checkpoints(
                source_runtime=args.source_runtime,
                target_runtime=args.target_runtime,
                cases_path=args.cases,
                protocol_path=args.protocol,
            ),
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
