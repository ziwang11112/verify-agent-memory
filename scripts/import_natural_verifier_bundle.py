"""Import an identical v1 verifier bundle into the v2 natural evaluation contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_inferred_admissibility_experiment as provider_runtime  # noqa: E402
from scripts import run_natural_end_to_end_experiment as run  # noqa: E402
from verify_agent_memory.natural_end_to_end import load_cases  # noqa: E402

PROTOCOL_PATH = "experiments/natural_end_to_end_protocol.json"
DEFAULT_SOURCE_RUNTIME = (
    ROOT / "tmp" / "natural_end_to_end" / "provider_runtime_invalid_v1_reader512_20260808T2125Z"
)


def _git_file_bytes(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _protocol_reuse(target: run.Protocol) -> Mapping[str, Any]:
    execution = run._mapping(target.raw.get("execution"), "protocol.execution")
    return run._mapping(
        execution.get("verifier_bundle_reuse"),
        "protocol.execution.verifier_bundle_reuse",
    )


def assert_protocol_compatibility(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    """Require v1 and v2 to differ only in reader budget and reuse declaration."""
    if source.get("protocol_id") != "natural-heldout-route-to-reader-v1":
        raise ValueError("source protocol is not natural end-to-end v1")
    if target.get("protocol_id") != "natural-heldout-route-to-reader-v2":
        raise ValueError("target protocol is not natural end-to-end v2")
    normalized = copy.deepcopy(source)
    normalized["protocol_id"] = target["protocol_id"]
    source_execution = run._mapping(normalized.get("execution"), "source.execution")
    target_execution = run._mapping(target.get("execution"), "target.execution")
    if source_execution.get("maximum_output_tokens_reader") != 512:
        raise ValueError("source reader output cap is not 512")
    if target_execution.get("maximum_output_tokens_reader") != 2048:
        raise ValueError("target reader output cap is not 2048")
    source_execution["maximum_output_tokens_reader"] = 2048
    source_execution["verifier_bundle_reuse"] = target_execution.get("verifier_bundle_reuse")
    if normalized != target:
        raise ValueError("protocols differ outside the allowed reader-only amendment")


def verifier_request_body(
    protocol: run.Protocol,
    spec: run.CallSpec,
) -> dict[str, object]:
    return {
        "model": protocol.verifier.model,
        "instructions": protocol.verifier_prompt,
        "input": spec.payload,
        "max_output_tokens": protocol.maximum_output_tokens_verifier,
        "reasoning": {"effort": protocol.verifier.controls["effort"]},
        "store": False,
        "text": {
            "verbosity": "low",
            "format": {
                "name": "inferred_admissibility",
                "type": "json_schema",
                "strict": True,
                "schema": spec.schema,
            },
        },
    }


def migrated_record(
    source: Mapping[str, Any],
    *,
    target_protocol_sha256: str,
    target_commit: str,
    source_record_sha256: str,
) -> dict[str, Any]:
    output = dict(source)
    output["protocol_sha256"] = target_protocol_sha256
    output["implementation_commit"] = target_commit
    output["compatibility_source"] = {
        "protocol_sha256": source["protocol_sha256"],
        "implementation_commit": source["implementation_commit"],
        "record_sha256": source_record_sha256,
        "response_unchanged": True,
    }
    return output


def import_bundle(
    *,
    source_runtime: Path,
    target_runtime: Path,
    cases_path: Path,
    protocol_path: Path,
) -> dict[str, object]:
    target = run.load_protocol(protocol_path)
    reuse = _protocol_reuse(target)
    source_commit = run._string(
        reuse.get("source_implementation_commit"),
        "verifier_bundle_reuse.source_implementation_commit",
    )
    source_protocol_bytes = _git_file_bytes(source_commit, PROTOCOL_PATH)
    source_protocol_sha256 = _sha256_bytes(source_protocol_bytes)
    if source_protocol_sha256 != reuse.get("source_protocol_sha256"):
        raise ValueError("source protocol SHA-256 differs from reuse declaration")
    source_protocol = run._mapping(
        json.loads(source_protocol_bytes),
        "source_protocol",
    )
    assert_protocol_compatibility(source_protocol, target.raw)
    target_commit = run._require_clean_contract()

    cases = load_cases(cases_path)
    specs, inference_cases = run._verifier_specs(cases, target)
    parser = run._verifier_parser_factory(inference_cases)
    source_stage = run._stage_root(source_runtime, "verifier", target.verifier)
    target_stage = run._stage_root(target_runtime, "verifier", target.verifier)
    staging_stage = target_runtime / ".imports" / "verifier-openai-gpt-5.6-sol"
    if target_stage.exists() or staging_stage.exists():
        raise RuntimeError("verifier import destination is not empty")
    source_completion_path = source_stage / "completion.json"
    source_completion = run._read_json(source_completion_path)
    expected_completion = {
        "schema_version": 1,
        "protocol_sha256": source_protocol_sha256,
        "implementation_commit": source_commit,
        "stage": "verifier",
        "provider": target.verifier.provider,
        "model": target.verifier.model,
        "request_count": len(specs),
        "complete_bundle": True,
    }
    for key, expected in expected_completion.items():
        if source_completion.get(key) != expected:
            raise ValueError(f"source verifier completion drifted: {key}")
    if source_completion.get("response_set_sha256") != reuse.get("source_response_set_sha256"):
        raise ValueError("source verifier response-set SHA-256 drifted")

    responses_path = staging_stage / "responses"
    responses_path.mkdir(parents=True)
    source_file_entries = []
    target_file_entries = []
    request_body_entries = []
    records: dict[str, Mapping[str, Any]] = {}
    for request_id in sorted(specs):
        spec = specs[request_id]
        source_path = source_stage / "responses" / f"{request_id}.json"
        source_bytes = source_path.read_bytes()
        source_sha256 = _sha256_bytes(source_bytes)
        source_record = run._mapping(json.loads(source_bytes), f"source[{request_id}]")
        expected = {
            "schema_version": 1,
            "protocol_sha256": source_protocol_sha256,
            "implementation_commit": source_commit,
            "stage": "verifier",
            "provider": target.verifier.provider,
            "model": target.verifier.model,
            "request_id": request_id,
            "payload_sha256": hashlib.sha256(spec.payload.encode("utf-8")).hexdigest(),
        }
        for key, value in expected.items():
            if source_record.get(key) != value:
                raise ValueError(f"source verifier record drifted: {request_id}/{key}")
        body_sha256 = provider_runtime._sha256_object(verifier_request_body(target, spec))
        if source_record.get("request_body_sha256") != body_sha256:
            raise ValueError(f"verifier request body drifted: {request_id}")
        parser(source_record.get("response"), request_id)
        run._record_cost(source_record)
        target_record = migrated_record(
            source_record,
            target_protocol_sha256=target.protocol_sha256,
            target_commit=target_commit,
            source_record_sha256=source_sha256,
        )
        target_path = responses_path / f"{request_id}.json"
        target_bytes = _json_bytes(target_record)
        target_path.write_bytes(target_bytes)
        source_file_entries.append((request_id, source_sha256))
        target_file_entries.append((request_id, _sha256_bytes(target_bytes)))
        request_body_entries.append((request_id, body_sha256))
        records[request_id] = target_record

    response_set_sha256 = run._sha256_object(
        [
            {
                "request_id": request_id,
                "response_sha256": run._sha256_object(records[request_id]["response"]),
            }
            for request_id in sorted(records)
        ]
    )
    if response_set_sha256 != source_completion.get("response_set_sha256"):
        raise ValueError("imported response content differs from source bundle")
    manifest = {
        "schema_version": 1,
        "status": "compatible_verifier_bundle_import",
        "source_protocol_sha256": source_protocol_sha256,
        "source_implementation_commit": source_commit,
        "source_completion_sha256": run._sha256_file(source_completion_path),
        "source_record_set_sha256": run._sha256_object(source_file_entries),
        "target_protocol_sha256": target.protocol_sha256,
        "target_implementation_commit": target_commit,
        "target_record_set_sha256": run._sha256_object(target_file_entries),
        "verifier_request_body_set_sha256": run._sha256_object(request_body_entries),
        "response_set_sha256": response_set_sha256,
        "request_count": len(records),
        "response_content_changed": False,
        "allowed_protocol_changes": [
            "protocol_id v1 to v2",
            "reader maximum output tokens 512 to 2048",
            "verifier compatibility reuse declaration",
        ],
        "provider_calls_made": 0,
    }
    manifest_path = staging_stage / "compatibility_import.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    completion = {
        "schema_version": 1,
        "protocol_sha256": target.protocol_sha256,
        "implementation_commit": target_commit,
        "stage": "verifier",
        "provider": target.verifier.provider,
        "model": target.verifier.model,
        "request_count": len(records),
        "input_tokens": sum(int(record["usage"]["input_tokens"]) for record in records.values()),
        "output_tokens": sum(int(record["usage"]["output_tokens"]) for record in records.values()),
        "cost_usd": sum(run._record_cost(record) for record in records.values()),
        "response_set_sha256": response_set_sha256,
        "complete_bundle": True,
        "provider_calls_made": 0,
        "compatibility_import_sha256": run._sha256_file(manifest_path),
    }
    (staging_stage / "completion.json").write_bytes(_json_bytes(completion))
    target_stage.parent.mkdir(parents=True, exist_ok=True)
    staging_stage.rename(target_stage)
    return {
        "status": manifest["status"],
        "request_count": len(records),
        "provider_calls_made": 0,
        "source_protocol_sha256": source_protocol_sha256,
        "target_protocol_sha256": target.protocol_sha256,
        "response_set_sha256": response_set_sha256,
        "compatibility_import_sha256": completion["compatibility_import_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", type=Path, default=DEFAULT_SOURCE_RUNTIME)
    parser.add_argument("--target-runtime", type=Path, default=run.DEFAULT_RUNTIME)
    parser.add_argument("--cases", type=Path, default=run.DEFAULT_CASES)
    parser.add_argument("--protocol", type=Path, default=run.DEFAULT_PROTOCOL)
    args = parser.parse_args()
    print(
        json.dumps(
            import_bundle(
                source_runtime=args.source_runtime,
                target_runtime=args.target_runtime,
                cases_path=args.cases,
                protocol_path=args.protocol,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
