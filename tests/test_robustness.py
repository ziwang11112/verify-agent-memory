from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_metadata_robustness import main
from scripts.run_retrieval_experiment import load_cases
from verify_agent_memory.experiment import SettingSummary
from verify_agent_memory.retrieval import RetrievalArm
from verify_agent_memory.robustness import (
    CorruptionChannel,
    MetadataCorruption,
    RobustnessPoint,
    break_even_interval,
    corrupt_case,
)
from verify_agent_memory.serialization import case_from_mapping

ROOT = Path(__file__).resolve().parents[1]


def summary(
    arm: RetrievalArm,
    *,
    feasible: float,
    admissibility_risk: float,
) -> SettingSummary:
    return SettingSummary(
        setting_id=arm.value,
        arm=arm,
        source_count=2,
        query_count=10,
        feasible_rate=feasible,
        penalized_non_usable_upper_risk=admissibility_risk,
        penalized_non_usable_known_risk=admissibility_risk,
        penalized_admissibility_upper_risk=admissibility_risk,
        penalized_admissibility_known_risk=admissibility_risk,
        infeasibility_risk_component=0.0,
        non_usable_conditional_risk_component=admissibility_risk,
        admissibility_conditional_risk_component=admissibility_risk,
        conditional_non_usable_upper_risk=admissibility_risk,
        conditional_admissibility_upper_risk=admissibility_risk,
        evidence_recall=feasible,
        candidates_scored=10.0,
    )


def test_corruption_changes_visible_metadata_but_not_gold_assessments() -> None:
    case = load_cases(ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl")[0]
    corrupted = corrupt_case(
        case,
        MetadataCorruption(
            "false-allow-all",
            CorruptionChannel.NAMESPACE_FALSE_ALLOW,
            1.0,
        ),
    )

    original = {memory.memory_id: memory.namespace for memory in case.memories}
    observed = {memory.memory_id: memory.namespace for memory in corrupted.memories}
    assert original["b-current"] == "person-b"
    assert observed["b-current"] == "person-a"
    assert corrupted.assessments == case.assessments
    assert case.memories != corrupted.memories


def test_corruption_samples_are_nested_across_rates() -> None:
    case = load_cases(ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl")[0]
    low = corrupt_case(
        case,
        MetadataCorruption("low", CorruptionChannel.NAMESPACE_MISSING, 0.25, seed=7),
    )
    high = corrupt_case(
        case,
        MetadataCorruption("high", CorruptionChannel.NAMESPACE_MISSING, 0.75, seed=7),
    )
    original = {memory.memory_id: memory.namespace for memory in case.memories}

    def changed_ids(corrupted: object) -> set[str]:
        return {
            memory.memory_id
            for memory in corrupted.memories
            if memory.namespace != original[memory.memory_id]
        }

    assert changed_ids(low) <= changed_ids(high)


def test_policy_corruption_is_separate_from_gold_policy_label() -> None:
    case = load_cases(ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl")[0]
    corrupted = corrupt_case(
        case,
        MetadataCorruption("allow", CorruptionChannel.POLICY_FALSE_ALLOW, 1.0),
    )
    observed = {decision.memory_id: decision for decision in corrupted.policy_decisions}
    gold = {assessment.memory_id: assessment for assessment in corrupted.assessments}

    assert observed["a-blocked"].content_disclosure_allowed is True
    assert gold["a-blocked"].policy_allowed is False


def test_legacy_memory_level_policy_is_rejected_instead_of_silently_reused() -> None:
    row = json.loads(
        (ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    row["memories"][0]["policy_allowed"] = True

    with pytest.raises(ValueError, match="policy_allowed is deprecated"):
        case_from_mapping(row)


def test_break_even_brackets_first_non_dominating_rate() -> None:
    channel = CorruptionChannel.NAMESPACE_FALSE_ALLOW
    points = (
        RobustnessPoint(
            "zero-treatment",
            channel,
            0.0,
            0,
            summary(RetrievalArm.NAMESPACE_DENSE, feasible=0.9, admissibility_risk=0.2),
        ),
        RobustnessPoint(
            "zero-reference",
            channel,
            0.0,
            0,
            summary(RetrievalArm.GLOBAL_DENSE, feasible=0.8, admissibility_risk=0.3),
        ),
        RobustnessPoint(
            "noisy-treatment",
            channel,
            0.1,
            0,
            summary(RetrievalArm.NAMESPACE_DENSE, feasible=0.7, admissibility_risk=0.4),
        ),
        RobustnessPoint(
            "noisy-reference",
            channel,
            0.1,
            0,
            summary(RetrievalArm.GLOBAL_DENSE, feasible=0.8, admissibility_risk=0.3),
        ),
    )

    interval = break_even_interval(points, channel=channel)

    assert interval.last_dominating_rate == 0.0
    assert interval.first_non_dominating_rate == 0.1


def test_metadata_robustness_cli_smoke(tmp_path: Path) -> None:
    protocol = {
        "schema_version": 1,
        "target_recall": 0.8,
        "selection_risk": "admissibility_upper_bound",
        "fixed_arms": [
            "global_dense",
            "namespace_dense",
            "query_agnostic_current_only",
            "released_intent_lifecycle_upper_bound",
        ],
        "channels": ["namespace_false_allow"],
        "rates": [0.0, 1.0],
        "seeds": [0],
    }
    robustness_protocol = tmp_path / "robustness.json"
    robustness_protocol.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "curve.jsonl"
    break_even = tmp_path / "break-even.jsonl"

    assert (
        main(
            [
                "--cases",
                str(ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl"),
                "--retrieval-protocol",
                str(ROOT / "experiments" / "frozen_natural_protocol.json"),
                "--robustness-protocol",
                str(robustness_protocol),
                "--output",
                str(output),
                "--break-even-output",
                str(break_even),
            ]
        )
        == 0
    )
    assert len(output.read_text(encoding="utf-8").splitlines()) == 8
    assert len(break_even.read_text(encoding="utf-8").splitlines()) == 1
    first_row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first_row["metric_schema_version"] == 2
