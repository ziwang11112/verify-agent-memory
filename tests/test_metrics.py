from __future__ import annotations

import math

import pytest

from verify_agent_memory.metrics import aggregate_scores, score_route
from verify_agent_memory.schema import (
    LifecycleState,
    MemoryAssessment,
    QueryIntent,
    Relevance,
    Scope,
)


def item(
    memory_id: str,
    *,
    relevance: Relevance,
    scope: Scope = Scope.ALLOWED,
    lifecycle_state: LifecycleState = LifecycleState.CURRENT,
    policy_allowed: bool | None = True,
    query_intent: QueryIntent = QueryIntent.CURRENT_STATE,
) -> MemoryAssessment:
    return MemoryAssessment(
        memory_id=memory_id,
        relevance=relevance,
        scope=scope,
        lifecycle_state=lifecycle_state,
        policy_allowed=policy_allowed,
        query_intent=query_intent,
    )


def test_matched_recall_uses_smallest_feasible_prefix() -> None:
    assessments = [
        item("a", relevance=Relevance.REQUIRED),
        item("b", relevance=Relevance.SUPPORTIVE),
        item("x", relevance=Relevance.NOT_USEFUL),
        item("y", relevance=Relevance.NOT_USEFUL, scope=Scope.DISALLOWED),
    ]
    score = score_route(
        ["x", "a", "y", "b"],
        assessments,
        target_recall=0.5,
        candidates_scored=9,
        latency_ms=2.5,
    )
    assert score.anchor_total == 2
    assert score.anchor_retrieved == 2
    assert score.evidence_recall == 1.0
    assert score.feasible is True
    assert score.matched_prefix_count == 2
    assert score.contamination_known_rate == 0.5
    assert score.wrong_scope_exposure_rate == 0.0
    assert score.candidates_scored == 9
    assert score.latency_ms == 2.5


def test_incomplete_labels_produce_coverage_and_contamination_bounds() -> None:
    assessments = [
        item("unknown", relevance=Relevance.UNKNOWN, scope=Scope.UNKNOWN),
        item("bad", relevance=Relevance.NOT_USEFUL),
        item("anchor", relevance=Relevance.REQUIRED),
    ]
    score = score_route(
        ["unknown", "bad", "anchor"],
        assessments,
        target_recall=1.0,
    )
    assert score.matched_prefix_count == 3
    assert score.contamination_known_rate == 0.5
    assert score.contamination_label_coverage == pytest.approx(2 / 3)
    assert score.contamination_lower_bound == pytest.approx(1 / 3)
    assert score.contamination_upper_bound == pytest.approx(2 / 3)
    assert score.non_usable_known_rate == 0.5
    assert score.admissibility_violation_known_rate == 0.0
    assert score.admissibility_label_coverage == pytest.approx(2 / 3)
    assert score.admissibility_violation_lower_bound == 0.0
    assert score.admissibility_violation_upper_bound == pytest.approx(1 / 3)
    assert score.unresolved_admissibility_rate == pytest.approx(1 / 3)


def test_relevance_noise_and_admissibility_violations_are_separate() -> None:
    assessments = [
        item("relevant-banned", relevance=Relevance.SUPPORTIVE, policy_allowed=False),
        item("irrelevant-allowed", relevance=Relevance.NOT_USEFUL),
        item(
            "irrelevant-banned",
            relevance=Relevance.NOT_USEFUL,
            scope=Scope.DISALLOWED,
        ),
        item("anchor", relevance=Relevance.REQUIRED),
    ]
    score = score_route(
        ["relevant-banned", "irrelevant-allowed", "irrelevant-banned", "anchor"],
        assessments,
        target_recall=1.0,
    )

    assert score.non_usable_known_rate == pytest.approx(3 / 4)
    assert score.admissibility_violation_known_rate == pytest.approx(2 / 4)
    assert score.known_relevant_admissible_rate == pytest.approx(1 / 4)
    assert score.known_relevant_inadmissible_rate == pytest.approx(1 / 4)
    assert score.known_irrelevant_admissible_rate == pytest.approx(1 / 4)
    assert score.known_irrelevant_inadmissible_rate == pytest.approx(1 / 4)
    assert score.joint_label_coverage == 1.0


def test_no_usable_anchors_keeps_recall_and_matched_metrics_undefined() -> None:
    assessments = [item("x", relevance=Relevance.NOT_USEFUL)]
    score = score_route(["x"], assessments)
    assert score.anchor_total == 0
    assert score.evidence_recall is None
    assert score.feasible is None
    assert score.matched_prefix_count is None
    assert score.contamination_known_rate is None


def test_infeasible_route_has_no_matched_recall_score() -> None:
    assessments = [
        item("a", relevance=Relevance.REQUIRED),
        item("b", relevance=Relevance.REQUIRED),
    ]
    score = score_route(["a"], assessments, target_recall=0.8)
    assert score.evidence_recall == 0.5
    assert score.feasible is False
    assert score.matched_prefix_count is None
    assert score.contamination_known_rate is None


@pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf])
def test_nonfinite_or_negative_runtime_measurements_are_rejected(
    bad_value: float,
) -> None:
    assessments = [item("a", relevance=Relevance.REQUIRED)]
    with pytest.raises(ValueError):
        score_route(["a"], assessments, latency_ms=bad_value)


def test_duplicate_or_unassessed_ranked_ids_are_rejected() -> None:
    assessments = [item("a", relevance=Relevance.REQUIRED)]
    with pytest.raises(ValueError, match="duplicate ranked"):
        score_route(["a", "a"], assessments)
    with pytest.raises(ValueError, match="lack assessments"):
        score_route(["missing"], assessments)


def test_aggregate_scores_skip_missing_values_without_imputation() -> None:
    with_anchor = score_route(
        ["a"],
        [item("a", relevance=Relevance.REQUIRED)],
        latency_ms=4.0,
    )
    without_anchor = score_route(
        ["x"],
        [item("x", relevance=Relevance.NOT_USEFUL)],
    )
    aggregate = aggregate_scores([with_anchor, without_anchor])
    assert aggregate.query_count == 2
    assert aggregate.recall_evaluable_rate == 0.5
    assert aggregate.mean_evidence_recall == 1.0
    assert aggregate.feasible_rate == 1.0
    assert aggregate.matched_recall_evaluable_rate == 0.5
    assert aggregate.mean_latency_ms == 4.0


def test_aggregate_requires_at_least_one_query() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_scores([])
