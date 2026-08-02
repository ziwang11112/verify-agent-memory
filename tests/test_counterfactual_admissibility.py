from __future__ import annotations

from pathlib import Path

import pytest

from verify_agent_memory.counterfactual_admissibility import (
    ADMISSIBILITY_LABELS,
    CounterfactualPrediction,
    ProbabilityVector,
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
CASES = ROOT / "experiments" / "counterfactual_admissibility_cases.json"


def _prediction_mapping(candidate_keys: tuple[str, ...]) -> dict[str, object]:
    rows = []
    for index, key in enumerate(candidate_keys):
        probabilities = (
            {"admissible": 0.9, "inadmissible": 0.05, "unknown": 0.05}
            if index < 2
            else {"admissible": 0.05, "inadmissible": 0.9, "unknown": 0.05}
        )
        rows.append({"candidate_key": key, "admissibility": probabilities})
    return {"candidates": rows}


def test_frozen_dataset_is_balanced_and_counterfactual() -> None:
    scenarios = load_scenarios(CASES)
    summary = validate_dataset_contract(
        scenarios,
        axes=("principal_scope", "lifecycle_intent", "policy_purpose", "as_of_time"),
        scenarios_per_axis=4,
        query_pairs_per_scenario=2,
        candidates_per_pair=3,
        minimum_query_token_jaccard=0.5,
    )

    assert summary["scenario_count"] == 16
    assert summary["pair_count"] == 32
    assert summary["case_count"] == 64
    assert summary["candidate_judgment_count"] == 192
    assert set(summary["axes"]) == {
        "principal_scope",
        "lifecycle_intent",
        "policy_purpose",
        "as_of_time",
    }


def test_prompt_payload_hides_gold_and_scenario_metadata() -> None:
    pair = expand_pairs(load_scenarios(CASES))[0]
    payload = prompt_payload(pair, condition="allow")

    assert pair.scenario_id not in payload
    assert pair.axis not in payload
    assert pair.source_basis not in payload
    assert "focal" not in payload
    assert "stable_admissible" not in payload
    assert "stable_inadmissible" not in payload
    assert pair.allow_query in payload
    assert pair.candidates[0].text in payload


def test_response_schema_and_parser_bind_exact_candidate_order() -> None:
    pair = expand_pairs(load_scenarios(CASES))[0]
    schema = response_json_schema(len(pair.candidates))
    candidate_schema = schema["properties"]["candidates"]  # type: ignore[index]
    assert candidate_schema["minItems"] == 3  # type: ignore[index]
    assert candidate_schema["maxItems"] == 3  # type: ignore[index]

    keys = tuple(candidate.candidate_key for candidate in pair.candidates)
    parsed = prediction_from_mapping(
        _prediction_mapping(keys),
        pair,
        condition="allow",
    )
    assert parsed.candidates[0].probabilities.argmax() == "admissible"

    bad_order = _prediction_mapping(keys)
    candidates = bad_order["candidates"]
    assert isinstance(candidates, list)
    candidates.reverse()
    with pytest.raises(ValueError, match="request order"):
        prediction_from_mapping(bad_order, pair, condition="allow")

    bad_sum = _prediction_mapping(keys)
    candidates = bad_sum["candidates"]
    assert isinstance(candidates, list)
    first = candidates[0]
    assert isinstance(first, dict)
    first["admissibility"] = {"admissible": 0.8, "inadmissible": 0.2, "unknown": 0.2}
    with pytest.raises(ValueError, match="sum to 1"):
        prediction_from_mapping(bad_sum, pair, condition="allow")


def test_oracle_has_perfect_pair_consistency_without_control_flips() -> None:
    pairs = expand_pairs(load_scenarios(CASES))
    rows = pair_score_rows(pairs, oracle_predictions(pairs), model="released_oracle")
    metrics = aggregate_pair_metrics(rows)

    assert metrics["strict_focal_pair_consistency"] == 1
    assert metrics["focal_direction_accuracy"] == 1
    assert metrics["stable_control_accuracy"] == 1
    assert metrics["stable_control_overflip_rate"] == 0
    assert metrics["candidate_accuracy"] == 1


def test_query_blind_no_verifier_cannot_solve_focal_flip() -> None:
    pairs = expand_pairs(load_scenarios(CASES))
    rows = pair_score_rows(pairs, no_verifier_predictions(pairs), model="no_verifier")
    metrics = aggregate_pair_metrics(rows)

    assert metrics["strict_focal_pair_consistency"] == 0
    assert metrics["focal_direction_accuracy"] == 0
    assert metrics["stable_control_overflip_rate"] == 0
    assert metrics["candidate_accuracy"] == pytest.approx(0.5)


def test_scoring_uses_symmetric_directional_margin() -> None:
    pair = expand_pairs(load_scenarios(CASES))[0]
    oracle = oracle_predictions((pair,))
    row = pair_score_rows((pair,), oracle, model="oracle")[0]

    assert row["focal_directional_margin"] == 1
    assert row["focal_allow_correct"] is True
    assert row["focal_block_correct"] is True
    assert row["stable_control_overflip_count"] == 0


def test_bootstrap_resamples_scenarios_within_axes() -> None:
    pairs = expand_pairs(load_scenarios(CASES))
    rows = pair_score_rows(pairs, oracle_predictions(pairs), model="oracle")
    result = scenario_stratified_bootstrap(
        rows,
        metric="strict_focal_pair_consistency",
        replicates=100,
        seed=17,
    )

    assert result["estimate"] == 1
    assert result["ci_lower"] == 1
    assert result["ci_upper"] == 1
    assert result["bootstrap_unit"] == "scenario"
    assert result["bootstrap_stratification"] == "axis"


def test_prediction_object_rejects_wrong_condition() -> None:
    vector = ProbabilityVector(
        labels=ADMISSIBILITY_LABELS,
        values=(0.8, 0.1, 0.1),
    )
    with pytest.raises(ValueError, match="condition"):
        CounterfactualPrediction(
            case_id="pair::other",
            pair_id="pair",
            condition="other",
            candidates=(),
        )
    assert vector.argmax() == "admissible"
