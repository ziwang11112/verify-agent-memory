from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_counterfactual_exposure_intervention as runtime
from verify_agent_memory.counterfactual_admissibility import load_scenarios
from verify_agent_memory.exposure_intervention import (
    CELLS,
    ReaderResponse,
    aggregate_cell_scores,
    build_exposure_units,
    exposure_contract_summary,
    load_target_overlay,
    ordered_requests,
    request_payload,
    response_from_mapping,
    scenario_stratified_cell_bootstrap,
    scenario_stratified_selectivity_bootstrap,
    score_exposure_pairs,
    target_disclosed,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "counterfactual_admissibility_cases.json"
OVERLAY = ROOT / "experiments" / "counterfactual_exposure_targets.json"
SOURCE_REFERENCE = "experiments/counterfactual_admissibility_cases.json"


def _units():
    scenarios = load_scenarios(SOURCE)
    overlay = load_target_overlay(
        OVERLAY,
        source_path=SOURCE,
        source_reference=SOURCE_REFERENCE,
        scenarios=scenarios,
    )
    return build_exposure_units(scenarios, overlay)


def _synthetic_responses(units):
    responses = {}
    for unit in units:
        for exposure in ("withheld", "exposed"):
            disclose = exposure == "exposed" and unit.cell == "relevant_admissible"
            action = (
                "refuse"
                if exposure == "exposed" and unit.cell == "relevant_inadmissible"
                else "answer"
            )
            request_id = unit.request_id(exposure)
            responses[request_id] = ReaderResponse(
                request_id=request_id,
                action=action,
                answer=unit.disclosure_markers[0] if disclose else "No target fact disclosed.",
            )
    return responses


def test_protocol_is_hash_bound_balanced_and_zero_call() -> None:
    protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    receipt = runtime.validate_protocol(protocol)

    assert receipt["scenario_count"] == 16
    assert receipt["pair_count"] == 32
    assert receipt["condition_count"] == 64
    assert receipt["unit_count"] == 192
    assert receipt["request_count_per_model"] == 384
    assert receipt["cell_counts"] == {
        "relevant_admissible": 32,
        "relevant_inadmissible": 32,
        "irrelevant_admissible": 64,
        "irrelevant_inadmissible": 64,
    }
    assert set(receipt["axis_counts"].values()) == {48}
    assert receipt["provider_client_present"] is False
    assert receipt["paid_execution_authorized"] is False
    assert receipt["model_call_made"] is False


def test_overlay_has_explicit_two_by_two_labels() -> None:
    units = _units()
    summary = exposure_contract_summary(units)

    assert summary["cell_counts"] == {
        "relevant_admissible": 32,
        "relevant_inadmissible": 32,
        "irrelevant_admissible": 64,
        "irrelevant_inadmissible": 64,
    }
    assert all(
        unit.cell.startswith("relevant_") == (unit.candidate_role == "focal") for unit in units
    )


def test_overlay_rejects_marker_absent_or_present_in_query(tmp_path: Path) -> None:
    scenarios = load_scenarios(SOURCE)
    raw = json.loads(OVERLAY.read_text(encoding="utf-8"))
    raw["scenarios"][0]["candidates"][0]["disclosure_markers"] = ["not in candidate"]
    absent = tmp_path / "absent.json"
    absent.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="absent from candidate"):
        load_target_overlay(
            absent,
            source_path=SOURCE,
            source_reference=SOURCE_REFERENCE,
            scenarios=scenarios,
        )

    raw = json.loads(OVERLAY.read_text(encoding="utf-8"))
    calendar = next(row for row in raw["scenarios"] if row["scenario_id"] == "scope-calendar")
    calendar["candidates"][0]["disclosure_markers"] = ["Priya"]
    leaked = tmp_path / "leaked.json"
    leaked.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="appears in a query"):
        load_target_overlay(
            leaked,
            source_path=SOURCE,
            source_reference=SOURCE_REFERENCE,
            scenarios=scenarios,
        )


def test_overlay_rejects_source_hash_drift(tmp_path: Path) -> None:
    changed_source = tmp_path / "counterfactual_admissibility_cases.json"
    changed_source.write_bytes(SOURCE.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source hash mismatch"):
        load_target_overlay(
            OVERLAY,
            source_path=changed_source,
            source_reference=SOURCE_REFERENCE,
            scenarios=load_scenarios(changed_source),
        )


def test_request_pair_changes_only_manipulated_candidate() -> None:
    unit = _units()[0]
    exposed = json.loads(request_payload(unit, exposure="exposed"))
    withheld = json.loads(request_payload(unit, exposure="withheld"))

    assert exposed["query"] == withheld["query"] == unit.query
    assert len(exposed["candidates"]) == 3
    assert len(withheld["candidates"]) == 2
    assert [
        row for row in exposed["candidates"] if row["candidate_key"] != unit.candidate_key
    ] == withheld["candidates"]
    visible = json.dumps(exposed)
    for hidden_field in ("role", "relevant", "admissible", "cell", "disclosure_markers"):
        assert hidden_field not in visible


def test_request_order_is_deterministic_and_model_mixed() -> None:
    units = _units()
    first = ordered_requests(units, model="reader-a", seed=17)
    repeated = ordered_requests(units, model="reader-a", seed=17)
    other_model = ordered_requests(units, model="reader-b", seed=17)

    first_ids = [unit.request_id(exposure) for unit, exposure in first]
    repeated_ids = [unit.request_id(exposure) for unit, exposure in repeated]
    other_ids = [unit.request_id(exposure) for unit, exposure in other_model]
    assert first_ids == repeated_ids
    assert first_ids != other_ids
    assert len(first_ids) == len(set(first_ids)) == 384


def test_response_parser_and_literal_detector_are_strict() -> None:
    parsed = response_from_mapping(
        {"action": "answer", "answer": "It was OCTOBER-3."},
        request_id="request",
    )
    assert parsed.action == "answer"
    assert target_disclosed(parsed.answer, ("October 3",)) is True
    assert target_disclosed("The date was October 31.", ("October 3",)) is False
    assert target_disclosed("I refuse to reveal 123 Market Street.", ("123 Market Street",))

    with pytest.raises(ValueError, match="unknown fields"):
        response_from_mapping(
            {"action": "answer", "answer": "ok", "candidate_key": "m1"},
            request_id="request",
        )
    with pytest.raises(ValueError, match="reader action"):
        response_from_mapping(
            {"action": "maybe", "answer": "ok"},
            request_id="request",
        )


def test_scorer_rejects_partial_and_extra_bundles() -> None:
    unit = _units()[0]
    withheld_id = unit.request_id("withheld")
    exposed_id = unit.request_id("exposed")
    partial = {
        withheld_id: ReaderResponse(
            request_id=withheld_id,
            action="answer",
            answer="No target fact disclosed.",
        )
    }
    with pytest.raises(ValueError, match="incomplete or unbound"):
        score_exposure_pairs((unit,), partial, model="reader")

    complete = {
        **partial,
        exposed_id: ReaderResponse(
            request_id=exposed_id,
            action="answer",
            answer=unit.disclosure_markers[0],
        ),
    }
    extra = {
        **complete,
        "extra": ReaderResponse(request_id="extra", action="answer", answer=""),
    }
    with pytest.raises(ValueError, match="incomplete or unbound"):
        score_exposure_pairs((unit,), extra, model="reader")


def test_scoring_and_scenario_bootstrap_recover_synthetic_effects() -> None:
    units = _units()
    scores = score_exposure_pairs(
        units,
        _synthetic_responses(units),
        model="synthetic-reader",
    )

    assert aggregate_cell_scores(scores, cell="relevant_admissible")["exposure_effect"] == 1
    assert aggregate_cell_scores(scores, cell="relevant_inadmissible")["exposure_effect"] == 0
    assert aggregate_cell_scores(scores, cell="irrelevant_admissible")["exposure_effect"] == 0
    assert aggregate_cell_scores(scores, cell="irrelevant_inadmissible")["exposure_effect"] == 0
    interval = scenario_stratified_cell_bootstrap(
        scores,
        cell="relevant_admissible",
        replicates=50,
        seed=23,
    )
    selectivity = scenario_stratified_selectivity_bootstrap(
        scores,
        replicates=50,
        seed=23,
    )
    assert interval["estimate"] == interval["ci_lower"] == interval["ci_upper"] == 1
    assert selectivity["estimate"] == selectivity["ci_lower"] == selectivity["ci_upper"] == 1


def test_zero_call_cli_materializes_and_scores_complete_bundle(tmp_path: Path) -> None:
    protocol = replace(
        runtime.load_protocol(runtime.DEFAULT_PROTOCOL),
        bootstrap_replicates=20,
    )
    request_path = tmp_path / "requests.jsonl"
    receipt = runtime.materialize_requests(
        protocol,
        model="synthetic-reader",
        output=request_path,
    )
    request_rows = [json.loads(line) for line in request_path.read_text().splitlines()]
    assert receipt["request_count"] == len(request_rows) == 384
    assert receipt["provider_client_present"] is False
    assert receipt["model_call_made"] is False
    assert all(
        set(row) == {"request_id", "model", "system_prompt", "user_prompt", "response_schema"}
        for row in request_rows
    )

    units_by_request = {
        unit.request_id(exposure): unit
        for unit in protocol.units
        for exposure in ("withheld", "exposed")
    }
    responses_path = tmp_path / "responses.jsonl"
    with responses_path.open("w", encoding="utf-8", newline="\n") as handle:
        synthetic = _synthetic_responses(protocol.units)
        for request_id in sorted(units_by_request):
            response = synthetic[request_id]
            handle.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "response": {"action": response.action, "answer": response.answer},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    output_dir = tmp_path / "scores"
    manifest = runtime.score_response_bundle(
        protocol,
        model="synthetic-reader",
        responses_path=responses_path,
        output_dir=output_dir,
    )
    assert manifest["scored_pair_count"] == 192
    assert manifest["expected_request_count"] == 384
    assert manifest["answer_text_published"] is False
    assert {path.name for path in output_dir.iterdir()} == {
        "pair_scores.csv",
        "cell_metrics.csv",
        "bootstrap_ci.csv",
        "manifest.json",
    }
    cell_metrics = (output_dir / "cell_metrics.csv").read_text(encoding="utf-8")
    assert all(cell in cell_metrics for cell in CELLS)
    with pytest.raises(FileExistsError, match="nonempty score directory"):
        runtime.score_response_bundle(
            protocol,
            model="synthetic-reader",
            responses_path=responses_path,
            output_dir=output_dir,
        )
