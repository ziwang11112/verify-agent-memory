from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from scripts import analyze_counterfactual_error_taxonomy as analysis
from verify_agent_memory.counterfactual_admissibility import (
    CandidatePrediction,
    ProbabilityVector,
    expand_pairs,
    load_scenarios,
    no_verifier_predictions,
    oracle_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments" / "counterfactual_admissibility_cases.json"


def test_oracle_taxonomy_has_no_error_or_overflip() -> None:
    pairs = expand_pairs(load_scenarios(CASES))
    observations = analysis.candidate_observations(
        pairs,
        oracle_predictions(pairs),
        model="oracle",
    )
    taxonomy = analysis.aggregate_judgments(observations)
    stable = analysis.aggregate_stable_pairs(analysis.stable_pair_observations(observations))

    overall_stable = next(
        row for row in stable if row["axis"] == "all" and row["role"] == "all_stable"
    )
    assert overall_stable["both_correct_rate"] == 1
    assert overall_stable["overflip_rate"] == 0
    assert all(row["accuracy"] == 1 for row in taxonomy)


def test_keep_all_separates_false_admission_from_query_overflip() -> None:
    pairs = expand_pairs(load_scenarios(CASES))
    observations = analysis.candidate_observations(
        pairs,
        no_verifier_predictions(pairs),
        model="keep-all",
    )
    taxonomy = analysis.aggregate_judgments(observations)
    stable = analysis.aggregate_stable_pairs(analysis.stable_pair_observations(observations))

    inadmissible = next(
        row
        for row in taxonomy
        if row["axis"] == "all"
        and row["role"] == "stable_inadmissible"
        and row["condition"] == "all"
    )
    overall = next(row for row in stable if row["axis"] == "all" and row["role"] == "all_stable")
    assert inadmissible["false_admit_rate"] == 1
    assert overall["overflip_rate"] == 0


def test_stable_transition_detects_one_condition_only_change() -> None:
    pair = expand_pairs(load_scenarios(CASES))[0]
    predictions = oracle_predictions((pair,))
    block = predictions[pair.case_id("block")]
    candidates = list(block.candidates)
    stable_index = next(
        index
        for index, candidate in enumerate(pair.candidates)
        if candidate.role == "stable_admissible"
    )
    candidates[stable_index] = CandidatePrediction(
        candidate_key=candidates[stable_index].candidate_key,
        probabilities=ProbabilityVector(
            labels=("admissible", "inadmissible", "unknown"),
            values=(0.0, 1.0, 0.0),
        ),
    )
    predictions[pair.case_id("block")] = replace(block, candidates=tuple(candidates))

    observations = analysis.candidate_observations((pair,), predictions, model="m")
    stable_rows = analysis.stable_pair_observations(observations)
    changed = next(row for row in stable_rows if row["role"] == "stable_admissible")
    assert changed["allow_prediction"] == "admissible"
    assert changed["block_prediction"] == "inadmissible"
    assert changed["overflip"] == 1
    assert changed["both_correct"] == 0


def test_aggregate_outputs_drop_private_pair_and_scenario_ids() -> None:
    pairs = expand_pairs(load_scenarios(CASES))
    observations = analysis.candidate_observations(
        pairs,
        oracle_predictions(pairs),
        model="oracle",
    )
    taxonomy = analysis.aggregate_judgments(observations)
    stable_observations = analysis.stable_pair_observations(observations)
    stable = analysis.aggregate_stable_pairs(stable_observations)
    transitions = analysis.aggregate_transitions(stable_observations)

    for row in (*taxonomy, *stable, *transitions):
        assert "pair_id" not in row
        assert "scenario_id" not in row
        assert "query" not in row
        assert "text" not in row
