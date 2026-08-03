"""Validate, materialize, and score the zero-call paired exposure experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from verify_agent_memory.counterfactual_admissibility import load_scenarios
from verify_agent_memory.exposure_intervention import (
    CELLS,
    EXPOSURE_STATES,
    ExposurePairScore,
    ExposureUnit,
    ReaderResponse,
    aggregate_cell_scores,
    build_exposure_units,
    exposure_contract_summary,
    load_target_overlay,
    ordered_requests,
    request_payload,
    response_from_mapping,
    response_json_schema,
    scenario_stratified_cell_bootstrap,
    scenario_stratified_selectivity_bootstrap,
    score_exposure_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments" / "counterfactual_exposure_protocol.json"
DEFAULT_REQUESTS = ROOT / "tmp" / "counterfactual_exposure" / "requests.jsonl"
DEFAULT_SCORE_DIR = ROOT / "tmp" / "counterfactual_exposure" / "scores"


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


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer at least {minimum}")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing request bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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
class Protocol:
    path: Path
    raw: Mapping[str, Any]
    prompt_text: str
    prompt_sha256: str
    source_reference: str
    source_sha256: str
    overlay_sha256: str
    units: tuple[ExposureUnit, ...]
    request_order_seed: int
    bootstrap_replicates: int
    bootstrap_seed: int


def load_protocol(path: Path) -> Protocol:
    """Load the complete hash-bound design without constructing a provider client."""
    raw = _read_json(path)
    expected_fields = {
        "schema_version",
        "protocol_id",
        "status",
        "prompt",
        "source_dataset",
        "target_overlay",
        "implementation",
        "sample",
        "execution",
        "evaluation",
    }
    if set(raw) != expected_fields:
        raise ValueError("exposure protocol has missing or unknown fields")
    if raw["schema_version"] != 1:
        raise ValueError("exposure protocol schema version drifted")
    if raw["protocol_id"] != "paired-counterfactual-exposure-v1":
        raise ValueError("exposure protocol identity drifted")
    if raw["status"] != "zero_call_design_only_paid_execution_not_authorized":
        raise ValueError("exposure protocol status drifted")

    _prompt_reference, prompt_path, prompt_hash = _bound_file(raw["prompt"], "protocol.prompt")
    source_reference, source_path, source_hash = _bound_file(
        raw["source_dataset"], "protocol.source_dataset"
    )
    _overlay_reference, overlay_path, overlay_hash = _bound_file(
        raw["target_overlay"], "protocol.target_overlay"
    )
    implementation = _mapping(raw["implementation"], "protocol.implementation")
    if set(implementation) != {"runner", "scoring_module"}:
        raise ValueError("protocol implementation identity drifted")
    _bound_file(implementation["runner"], "protocol.implementation.runner")
    _bound_file(implementation["scoring_module"], "protocol.implementation.scoring_module")

    scenarios = load_scenarios(source_path)
    overlay = load_target_overlay(
        overlay_path,
        source_path=source_path,
        source_reference=source_reference,
        scenarios=scenarios,
    )
    units = build_exposure_units(scenarios, overlay)
    summary = exposure_contract_summary(units)
    sample = _mapping(raw["sample"], "protocol.sample")
    if sample != summary:
        raise ValueError("exposure protocol sample contract drifted")

    execution = _mapping(raw["execution"], "protocol.execution")
    exact_execution = {
        "paid_execution_authorized": False,
        "provider_client_present": False,
        "stateless": True,
        "temperature": 0,
        "output_repair": False,
        "prefix_scoring": False,
        "request_order_seed": execution.get("request_order_seed"),
        "max_output_tokens": 192,
    }
    if execution != exact_execution:
        raise ValueError("exposure execution contract drifted")
    request_order_seed = _integer(
        execution["request_order_seed"], "protocol.execution.request_order_seed"
    )

    evaluation = _mapping(raw["evaluation"], "protocol.evaluation")
    expected_evaluation_fields = {
        "primary_outcome",
        "primary_contrast",
        "bootstrap_unit",
        "bootstrap_stratification",
        "bootstrap_replicates",
        "bootstrap_seed",
        "llm_judge",
    }
    if set(evaluation) != expected_evaluation_fields:
        raise ValueError("exposure evaluation contract drifted")
    expected_evaluation = {
        "primary_outcome": "literal_target_disclosure",
        "primary_contrast": ("relevant_admissible_minus_relevant_inadmissible_exposure_effect"),
        "bootstrap_unit": "scenario",
        "bootstrap_stratification": "axis",
        "llm_judge": False,
    }
    if any(evaluation[key] != value for key, value in expected_evaluation.items()):
        raise ValueError("exposure evaluation semantics drifted")
    return Protocol(
        path=path,
        raw=raw,
        prompt_text=prompt_path.read_text(encoding="utf-8"),
        prompt_sha256=prompt_hash,
        source_reference=source_reference,
        source_sha256=source_hash,
        overlay_sha256=overlay_hash,
        units=units,
        request_order_seed=request_order_seed,
        bootstrap_replicates=_integer(
            evaluation["bootstrap_replicates"],
            "protocol.evaluation.bootstrap_replicates",
            minimum=1,
        ),
        bootstrap_seed=_integer(evaluation["bootstrap_seed"], "protocol.evaluation.bootstrap_seed"),
    )


def validate_protocol(protocol: Protocol) -> dict[str, object]:
    """Return a content-free validation receipt."""
    return {
        "status": "valid_zero_call_design",
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": _sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "source_dataset_sha256": protocol.source_sha256,
        "target_overlay_sha256": protocol.overlay_sha256,
        **exposure_contract_summary(protocol.units),
        "provider_client_present": False,
        "paid_execution_authorized": False,
        "model_call_made": False,
    }


def materialize_requests(
    protocol: Protocol,
    *,
    model: str,
    output: Path,
) -> dict[str, object]:
    """Write a provider-neutral request bundle without making a model call."""
    model = _string(model, "model")
    rows = []
    for unit, exposure in ordered_requests(
        protocol.units,
        model=model,
        seed=protocol.request_order_seed,
    ):
        rows.append(
            {
                "request_id": unit.request_id(exposure),
                "model": model,
                "system_prompt": protocol.prompt_text,
                "user_prompt": request_payload(unit, exposure=exposure),
                "response_schema": response_json_schema(),
            }
        )
    _write_jsonl(output, rows)
    return {
        "status": "requests_materialized_no_calls",
        "model": model,
        "request_count": len(rows),
        "request_bundle": str(output),
        "request_bundle_sha256": _sha256_file(output),
        "provider_client_present": False,
        "model_call_made": False,
    }


def load_responses(path: Path) -> dict[str, ReaderResponse]:
    """Load strict response rows and reject duplicates or unbound fields."""
    responses = {}
    for line_number, raw in enumerate(_read_jsonl(path), start=1):
        if set(raw) != {"request_id", "response"}:
            raise ValueError(f"response row {line_number} has missing or unknown fields")
        request_id = _string(raw["request_id"], f"response row {line_number}.request_id")
        if request_id in responses:
            raise ValueError(f"duplicate response request ID {request_id!r}")
        responses[request_id] = response_from_mapping(raw["response"], request_id=request_id)
    return responses


def _aggregate_rows(scores: Sequence[ExposurePairScore]) -> list[dict[str, object]]:
    rows = []
    scopes: list[tuple[str, Sequence[ExposurePairScore]]] = [("overall", scores)]
    scopes.extend(
        (axis, [score for score in scores if score.axis == axis])
        for axis in sorted({score.axis for score in scores})
    )
    for scope, scoped_scores in scopes:
        for cell in CELLS:
            rows.append({"scope": scope, **aggregate_cell_scores(scoped_scores, cell=cell)})
    return rows


def _bootstrap_rows(
    protocol: Protocol,
    scores: Sequence[ExposurePairScore],
) -> list[dict[str, object]]:
    rows = []
    scopes: list[tuple[str, Sequence[ExposurePairScore]]] = [("overall", scores)]
    scopes.extend(
        (axis, [score for score in scores if score.axis == axis])
        for axis in sorted({score.axis for score in scores})
    )
    for scope, scoped_scores in scopes:
        for cell in CELLS:
            result = scenario_stratified_cell_bootstrap(
                scoped_scores,
                cell=cell,
                replicates=protocol.bootstrap_replicates,
                seed=protocol.bootstrap_seed,
            )
            rows.append(
                {
                    "scope": scope,
                    "contrast": cell,
                    "metric": result["metric"],
                    "estimate": result["estimate"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "bootstrap_replicates": result["bootstrap_replicates"],
                    "scenario_count": result["scenario_count"],
                    "unit_count": result["unit_count"],
                }
            )
        selectivity = scenario_stratified_selectivity_bootstrap(
            scoped_scores,
            replicates=protocol.bootstrap_replicates,
            seed=protocol.bootstrap_seed,
        )
        rows.append(
            {
                "scope": scope,
                "contrast": selectivity["contrast"],
                "metric": "exposure_effect",
                "estimate": selectivity["estimate"],
                "ci_lower": selectivity["ci_lower"],
                "ci_upper": selectivity["ci_upper"],
                "bootstrap_replicates": selectivity["bootstrap_replicates"],
                "scenario_count": selectivity["scenario_count"],
                "unit_count": len(
                    [score for score in scoped_scores if score.cell.startswith("relevant_")]
                ),
            }
        )
    return rows


def score_response_bundle(
    protocol: Protocol,
    *,
    model: str,
    responses_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Score one complete model bundle and publish content-free local derivatives."""
    model = _string(model, "model")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty score directory: {output_dir}")
    responses = load_responses(responses_path)
    scores = score_exposure_pairs(protocol.units, responses, model=model)
    pair_rows = [asdict(score) for score in scores]
    aggregate_rows = _aggregate_rows(scores)
    bootstrap_rows = _bootstrap_rows(protocol, scores)
    outputs = {
        "pair_scores.csv": pair_rows,
        "cell_metrics.csv": aggregate_rows,
        "bootstrap_ci.csv": bootstrap_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / name, rows)
    manifest = {
        "schema_version": 1,
        "status": "complete_local_score_not_official_result",
        "model": model,
        "protocol_sha256": _sha256_file(protocol.path),
        "response_bundle_sha256": _sha256_file(responses_path),
        "expected_request_count": len(protocol.units) * len(EXPOSURE_STATES),
        "scored_pair_count": len(scores),
        "prefix_scoring": False,
        "llm_judge": False,
        "answer_text_published": False,
        "outputs": [{"path": name, "sha256": _sha256_file(output_dir / name)} for name in outputs],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _print(value: object) -> None:
    print(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    requests = subparsers.add_parser("requests")
    requests.add_argument("--model", required=True)
    requests.add_argument("--output", type=Path, default=DEFAULT_REQUESTS)
    score = subparsers.add_parser("score")
    score.add_argument("--model", required=True)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, default=DEFAULT_SCORE_DIR)
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    if args.command == "validate":
        result = validate_protocol(protocol)
    elif args.command == "requests":
        result = materialize_requests(protocol, model=args.model, output=args.output)
    else:
        result = score_response_bundle(
            protocol,
            model=args.model,
            responses_path=args.responses,
            output_dir=args.output_dir,
        )
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
