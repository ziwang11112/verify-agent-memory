"""Publish or verify the content-free inferred-admissibility dev diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from verify_agent_memory.provenance import sha256_file

SCORE_FILES = (
    "selected_thresholds.csv",
    "classification_metrics.csv",
    "filter_metrics.csv",
    "route_metrics.csv",
    "paired_route_deltas.csv",
    "provider_usage.csv",
)
EXPECTED_ROWS = {
    "selected_thresholds.csv": 4,
    "classification_metrics.csv": 32,
    "filter_metrics.csv": 16,
    "route_metrics.csv": 32,
    "paired_route_deltas.csv": 40,
    "provider_usage.csv": 2,
    "provider_execution_status.csv": 4,
    "provider_fixtures.csv": 4,
}
FORBIDDEN_COLUMNS = {
    "query_id",
    "case_id",
    "memory_id",
    "group_id",
    "candidate_key",
    "text",
    "query_text",
    "memory_text",
    "prompt",
    "response",
    "prediction",
    "answer",
}
PROTOCOL_SHA256 = "9ddd7575321ae8f15d832249dc94262799b39477c8a330b116feee533fc584cc"
PROMPT_SHA256 = "933d2f15971f59e4fd1dcdee87539248b26f47018eaca9eb0aa13ca548fa43b1"
EXECUTION_COMMIT = "9c665b43afbac3954fda92e70f8940b3efd98d90"
RECOVERY_COMMIT = "0ede9a8e415cf9d553fba122e629f48ee6e3c66d"
SCORING_COMMIT = "0af6ecfebadcac92a78e4c11f2e28e4ab9f187e8"
PROVIDERS = (
    {
        "provider": "OpenAI",
        "slug": "openai",
        "model": "gpt-5.6-sol",
        "comparison_status": "complete_scored",
        "raw_rows": 96,
        "unique_cases": 96,
        "failure_markers": 0,
        "formal_failure_markers": 0,
        "execution_commit": EXECUTION_COMMIT,
        "note": "complete frozen panel",
    },
    {
        "provider": "DeepSeek",
        "slug": "deepseek",
        "model": "deepseek-v4-pro",
        "comparison_status": "contract_incomplete_not_scored",
        "raw_rows": 60,
        "unique_cases": 60,
        "failure_markers": 1,
        "formal_failure_markers": 1,
        "execution_commit": EXECUTION_COMMIT,
        "note": "strict parser ValueError on case 61; no rerun or prefix scoring",
    },
    {
        "provider": "Gemini",
        "slug": "gemini",
        "model": "gemini-3.6-flash",
        "comparison_status": "complete_scored",
        "raw_rows": 103,
        "unique_cases": 96,
        "failure_markers": 1,
        "formal_failure_markers": 0,
        "execution_commit": EXECUTION_COMMIT,
        "note": "seven concurrent-resume duplicates; first committed is primary",
    },
    {
        "provider": "Anthropic",
        "slug": "anthropic",
        "model": "claude-sonnet-5",
        "comparison_status": "contract_incomplete_not_scored",
        "raw_rows": 0,
        "unique_cases": 0,
        "failure_markers": 2,
        "formal_failure_markers": 1,
        "execution_commit": RECOVERY_COMMIT,
        "note": "fixture passed; first public-dev response had fewer than 20 candidates",
    },
)


def _object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} contains a non-object row")
        rows.append(value)
    return rows


def _write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_json(path: Path, value: object) -> None:
    _write_text_lf(path, json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"inconsistent CSV fields for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_csv(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not fields or fields.intersection(FORBIDDEN_COLUMNS):
            raise ValueError(f"{path.name} contains forbidden or absent columns")
        rows = list(reader)
    expected = EXPECTED_ROWS.get(path.name)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{path.name} has {len(rows)} rows; expected {expected}")
    return len(rows)


def _validate_score_bundle(path: Path) -> Mapping[str, object]:
    manifest = _object(path / "manifest.json")
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("scoring protocol hash drifted")
    if manifest.get("prompt_sha256") != PROMPT_SHA256:
        raise ValueError("scoring prompt hash drifted")
    if manifest.get("case_count") != 96 or manifest.get("analysis_cases") != 72:
        raise ValueError("scoring population drifted")
    if manifest.get("scored_complete_providers") != ["OpenAI", "Gemini"]:
        raise ValueError("complete-provider set drifted")
    if manifest.get("excluded_providers") != ["DeepSeek", "Anthropic"]:
        raise ValueError("excluded-provider set drifted")
    if manifest.get("raw_text_or_responses_published") is not False:
        raise ValueError("score bundle is not marked content-free")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(SCORE_FILES):
        raise ValueError("score output set drifted")
    for name in SCORE_FILES:
        file_path = path / name
        if not file_path.is_file() or outputs.get(name) != sha256_file(file_path):
            raise ValueError(f"score output failed its receipt: {name}")
        _validate_csv(file_path)
    return manifest


def _sanitized_checkpoint_audit(path: Path) -> dict[str, object]:
    audit = _object(path)
    if audit.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("checkpoint audit protocol drifted")
    if audit.get("raw_checkpoints_modified") is not False:
        raise ValueError("checkpoint audit does not preserve raw files")
    providers = audit.get("providers")
    if not isinstance(providers, Sequence) or isinstance(providers, (str, bytes)):
        raise TypeError("checkpoint provider audit must be an array")
    safe_fields = (
        "provider",
        "model",
        "source_sha256",
        "raw_rows",
        "unique_cases",
        "duplicate_case_groups",
        "duplicate_calls",
        "duplicate_groups_with_different_predictions",
        "raw_cost_usd",
        "primary_selected_cost_usd",
        "duplicate_call_cost_usd",
        "primary_sha256",
        "sensitivity_sha256",
    )
    sanitized = []
    for raw in providers:
        if not isinstance(raw, Mapping):
            raise TypeError("checkpoint provider audit row must be an object")
        sanitized.append({field: raw[field] for field in safe_fields})
    if [row["provider"] for row in sanitized] != ["OpenAI", "Gemini"]:
        raise ValueError("checkpoint audit provider set drifted")
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "case_count": 96,
        "raw_checkpoints_modified": False,
        "primary_rule": audit.get("primary_rule"),
        "sensitivity_rule": audit.get("sensitivity_rule"),
        "providers": sanitized,
    }


def _execution_rows(response_dir: Path) -> list[dict[str, object]]:
    rows = []
    for expected in PROVIDERS:
        slug = str(expected["slug"])
        response_path = response_dir / f"responses-{slug}.jsonl"
        failure_path = response_dir / f"failures-{slug}.jsonl"
        responses = _jsonl(response_path)
        failures = _jsonl(failure_path)
        unique_cases = {str(row.get("case_id")) for row in responses}
        if len(responses) != expected["raw_rows"] or len(unique_cases) != expected["unique_cases"]:
            raise ValueError(f"{expected['provider']} response checkpoint count drifted")
        if len(failures) != expected["failure_markers"]:
            raise ValueError(f"{expected['provider']} failure-marker count drifted")
        if any(row.get("semantic_or_output_repair_attempted") is not False for row in failures):
            raise ValueError(f"{expected['provider']} failure marker records output repair")
        recorded_cost = sum(float(row.get("cost_usd", 0)) for row in responses)
        rows.append(
            {
                "provider": expected["provider"],
                "model": expected["model"],
                "comparison_status": expected["comparison_status"],
                "raw_response_rows": len(responses),
                "unique_completed_cases": len(unique_cases),
                "duplicate_calls": len(responses) - len(unique_cases),
                "failure_markers": len(failures),
                "formal_failure_markers": expected["formal_failure_markers"],
                "checkpoint_sha256": sha256_file(response_path) if response_path.is_file() else "",
                "failure_log_sha256": sha256_file(failure_path) if failure_path.is_file() else "",
                "recorded_response_cost_usd": recorded_cost,
                "execution_commit": expected["execution_commit"],
                "note": expected["note"],
            }
        )
    return rows


def _fixture_rows(fixture_v4: Path, fixture_v5: Path) -> list[dict[str, object]]:
    rows = []
    for expected in PROVIDERS:
        slug = str(expected["slug"])
        directory = fixture_v5 if expected["provider"] == "Anthropic" else fixture_v4
        path = directory / f"fixture-{slug}.json"
        receipt = _object(path)
        if (
            receipt.get("provider") != expected["provider"]
            or receipt.get("model") != expected["model"]
        ):
            raise ValueError(f"{expected['provider']} fixture binding drifted")
        if receipt.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"{expected['provider']} fixture protocol drifted")
        if receipt.get("prediction_shape_valid") is not True:
            raise ValueError(f"{expected['provider']} fixture shape is invalid")
        rows.append(
            {
                "provider": expected["provider"],
                "model": expected["model"],
                "prediction_shape_valid": True,
                "input_tokens": receipt["input_tokens"],
                "output_tokens": receipt["output_tokens"],
                "cost_usd": receipt["cost_usd"],
                "latency_ms": receipt["latency_ms"],
                "attempts": receipt["attempts"],
                "adapter_sha256": receipt["adapter_sha256"],
                "receipt_sha256": sha256_file(path),
            }
        )
    return rows


def publish(
    primary_dir: Path,
    sensitivity_dir: Path,
    checkpoint_audit: Path,
    response_dir: Path,
    fixture_v4: Path,
    fixture_v5: Path,
    output_dir: Path,
) -> Path:
    """Publish verified aggregates without copying provider-derived content."""
    primary = _validate_score_bundle(primary_dir)
    sensitivity = _validate_score_bundle(sensitivity_dir)
    primary_outputs = primary["outputs"]
    sensitivity_outputs = sensitivity["outputs"]
    assert isinstance(primary_outputs, Mapping)
    assert isinstance(sensitivity_outputs, Mapping)
    invariant_files = (
        "selected_thresholds.csv",
        "filter_metrics.csv",
        "route_metrics.csv",
        "paired_route_deltas.csv",
    )
    invariance = {
        name: primary_outputs[name] == sensitivity_outputs[name] for name in invariant_files
    }
    if not all(invariance.values()):
        raise ValueError("duplicate-response sensitivity changed routing conclusions")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SCORE_FILES:
        _write_text_lf(output_dir / name, (primary_dir / name).read_text(encoding="utf-8"))
    _write_json(output_dir / "scoring_manifest.json", primary)
    _write_json(output_dir / "sensitivity_scoring_manifest.json", sensitivity)
    checkpoint = _sanitized_checkpoint_audit(checkpoint_audit)
    _write_json(output_dir / "checkpoint_recovery.json", checkpoint)
    _write_json(
        output_dir / "sensitivity_summary.json",
        {
            "schema_version": 1,
            "primary_rule": checkpoint["primary_rule"],
            "sensitivity_rule": checkpoint["sensitivity_rule"],
            "invariant_output_hashes": invariance,
            "classification_metrics_identical": (
                primary_outputs["classification_metrics.csv"]
                == sensitivity_outputs["classification_metrics.csv"]
            ),
            "provider_usage_identical": (
                primary_outputs["provider_usage.csv"] == sensitivity_outputs["provider_usage.csv"]
            ),
            "downstream_conclusion_changed": False,
        },
    )
    _write_csv(output_dir / "provider_execution_status.csv", _execution_rows(response_dir))
    _write_csv(output_dir / "provider_fixtures.csv", _fixture_rows(fixture_v4, fixture_v5))

    published_files = (
        *SCORE_FILES,
        "scoring_manifest.json",
        "sensitivity_scoring_manifest.json",
        "checkpoint_recovery.json",
        "sensitivity_summary.json",
        "provider_execution_status.csv",
        "provider_fixtures.csv",
        "README.md",
    )
    files: dict[str, dict[str, object]] = {}
    for name in published_files:
        record: dict[str, object] = {"sha256": sha256_file(output_dir / name)}
        if name.endswith(".csv"):
            record["row_count"] = _validate_csv(output_dir / name)
        files[name] = record
    manifest = {
        "schema_version": 1,
        "status": "public_dev_text_inferred_admissibility_diagnostic_not_official_benchmark",
        "population": {
            "cases": 96,
            "calibration_cases": 24,
            "analysis_cases": 72,
            "candidate_depth": 20,
            "sources": ["RHELM", "MemOps"],
        },
        "protocol_sha256": PROTOCOL_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "provider_execution_commit": EXECUTION_COMMIT,
        "checkpoint_recovery_commit": RECOVERY_COMMIT,
        "scoring_commit": SCORING_COMMIT,
        "comparison_eligible_providers": ["OpenAI", "Gemini"],
        "contract_incomplete_providers": ["DeepSeek", "Anthropic"],
        "evaluation_or_heldout_access": False,
        "evaluation_retuning": False,
        "contains_query_memory_or_case_ids": False,
        "contains_raw_text_prompts_or_responses": False,
        "reader_calls": 0,
        "judge_calls": 0,
        "official_result": False,
        "publisher": "scripts/publish_inferred_admissibility_results.py",
        "publisher_sha256": sha256_file(Path(__file__).resolve()),
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def verify(output_dir: Path) -> None:
    """Verify the public package without reading any local response checkpoint."""
    manifest = _object(output_dir / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("inferred-admissibility manifest schema is unsupported")
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("published protocol hash drifted")
    if manifest.get("comparison_eligible_providers") != ["OpenAI", "Gemini"]:
        raise ValueError("published complete-provider set drifted")
    if manifest.get("contract_incomplete_providers") != ["DeepSeek", "Anthropic"]:
        raise ValueError("published incomplete-provider set drifted")
    if manifest.get("contains_query_memory_or_case_ids") is not False:
        raise ValueError("published package is not identifier-free")
    if manifest.get("contains_raw_text_prompts_or_responses") is not False:
        raise ValueError("published package is not content-free")
    if manifest.get("publisher_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("inferred-admissibility publisher hash drifted")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("published file map is malformed")
    expected_files = {
        *SCORE_FILES,
        "scoring_manifest.json",
        "sensitivity_scoring_manifest.json",
        "checkpoint_recovery.json",
        "sensitivity_summary.json",
        "provider_execution_status.csv",
        "provider_fixtures.csv",
        "README.md",
    }
    if set(files) != expected_files:
        raise ValueError("published file set drifted")
    for name, raw_record in files.items():
        if not isinstance(name, str) or not isinstance(raw_record, Mapping):
            raise TypeError("published file record is malformed")
        path = output_dir / name
        if not path.is_file() or raw_record.get("sha256") != sha256_file(path):
            raise ValueError(f"published file hash drifted: {name}")
        if path.suffix == ".csv":
            rows = _validate_csv(path)
            if raw_record.get("row_count") != rows:
                raise ValueError(f"published row count drifted: {name}")
    sensitivity = _object(output_dir / "sensitivity_summary.json")
    if sensitivity.get("downstream_conclusion_changed") is not False:
        raise ValueError("duplicate-response sensitivity changed the conclusion")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "verify"))
    parser.add_argument(
        "--primary-dir", type=Path, default=root / "tmp" / "inferred_admissibility_scores_primary"
    )
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=root / "tmp" / "inferred_admissibility_scores_sensitivity",
    )
    parser.add_argument(
        "--checkpoint-audit",
        type=Path,
        default=(
            root
            / "tmp"
            / "inferred_admissibility_canonical"
            / "checkpoint_canonicalization_audit.json"
        ),
    )
    parser.add_argument(
        "--response-dir", type=Path, default=root / "tmp" / "inferred_admissibility"
    )
    parser.add_argument(
        "--fixture-v4",
        type=Path,
        default=root / "tmp" / "inferred_admissibility_provider_fixture_v4",
    )
    parser.add_argument(
        "--fixture-v5",
        type=Path,
        default=root / "tmp" / "inferred_admissibility_provider_fixture_v5",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "inferred_admissibility"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "publish":
        path = publish(
            args.primary_dir.resolve(),
            args.sensitivity_dir.resolve(),
            args.checkpoint_audit.resolve(),
            args.response_dir.resolve(),
            args.fixture_v4.resolve(),
            args.fixture_v5.resolve(),
            args.output_dir.resolve(),
        )
        print(json.dumps({"manifest": path.as_posix(), "status": "published"}, sort_keys=True))
    else:
        verify(args.output_dir.resolve())
        print(json.dumps({"status": "valid"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
