"""Matched-recall metrics with explicit incomplete-label bounds."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from verify_agent_memory.admissibility import (
    admissible_status,
    lifecycle_compatible,
    relevant_status,
    usable_status,
)
from verify_agent_memory.schema import MemoryAssessment, Scope


@dataclass(frozen=True)
class RouteScore:
    target_recall: float
    anchor_total: int
    anchor_retrieved: int
    evidence_recall: float | None
    feasible: bool | None
    route_width: int
    matched_prefix_count: int | None
    non_usable_known_rate: float | None
    non_usable_label_coverage: float | None
    non_usable_lower_bound: float | None
    non_usable_upper_bound: float | None
    admissibility_violation_known_rate: float | None
    admissibility_label_coverage: float | None
    admissibility_violation_lower_bound: float | None
    admissibility_violation_upper_bound: float | None
    known_relevant_admissible_rate: float | None
    known_relevant_inadmissible_rate: float | None
    known_irrelevant_admissible_rate: float | None
    known_irrelevant_inadmissible_rate: float | None
    joint_label_coverage: float | None
    wrong_scope_exposure_rate: float | None
    policy_disallowed_exposure_rate: float | None
    lifecycle_incompatible_exposure_rate: float | None
    unresolved_admissibility_rate: float | None
    candidates_scored: int | None
    latency_ms: float | None

    @property
    def contamination_known_rate(self) -> float | None:
        """Legacy alias; the historical metric is non-usable, not inadmissible."""
        return self.non_usable_known_rate

    @property
    def contamination_label_coverage(self) -> float | None:
        return self.non_usable_label_coverage

    @property
    def contamination_lower_bound(self) -> float | None:
        return self.non_usable_lower_bound

    @property
    def contamination_upper_bound(self) -> float | None:
        return self.non_usable_upper_bound


@dataclass(frozen=True)
class AggregateScore:
    query_count: int
    recall_evaluable_rate: float
    mean_evidence_recall: float | None
    feasible_rate: float | None
    matched_recall_evaluable_rate: float
    non_usable_known_rate: float | None
    non_usable_label_coverage: float | None
    non_usable_lower_bound: float | None
    non_usable_upper_bound: float | None
    admissibility_violation_known_rate: float | None
    admissibility_label_coverage: float | None
    admissibility_violation_lower_bound: float | None
    admissibility_violation_upper_bound: float | None
    known_relevant_admissible_rate: float | None
    known_relevant_inadmissible_rate: float | None
    known_irrelevant_admissible_rate: float | None
    known_irrelevant_inadmissible_rate: float | None
    joint_label_coverage: float | None
    wrong_scope_exposure_rate: float | None
    policy_disallowed_exposure_rate: float | None
    lifecycle_incompatible_exposure_rate: float | None
    unresolved_admissibility_rate: float | None
    mean_route_width: float
    mean_matched_prefix_count: float | None
    mean_candidates_scored: float | None
    mean_latency_ms: float | None

    @property
    def contamination_known_rate(self) -> float | None:
        """Legacy alias; the historical metric is non-usable, not inadmissible."""
        return self.non_usable_known_rate

    @property
    def contamination_label_coverage(self) -> float | None:
        return self.non_usable_label_coverage

    @property
    def contamination_lower_bound(self) -> float | None:
        return self.non_usable_lower_bound

    @property
    def contamination_upper_bound(self) -> float | None:
        return self.non_usable_upper_bound


def _optional_nonnegative_finite(value: int | float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric or null")
    if value < 0 or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and non-negative")


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[int | float | bool | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _violation_bounds(
    statuses: Sequence[bool | None], *, prefix_exists: bool
) -> tuple[float | None, float | None, float | None, float | None]:
    if not prefix_exists:
        return None, None, None, None
    known_count = sum(value is not None for value in statuses)
    violation_count = sum(value is False for value in statuses)
    unresolved_count = len(statuses) - known_count
    return (
        _rate(violation_count, known_count),
        _rate(known_count, len(statuses)),
        _rate(violation_count, len(statuses)),
        _rate(violation_count + unresolved_count, len(statuses)),
    )


def score_route(
    ranked_memory_ids: Sequence[str],
    assessments: Sequence[MemoryAssessment],
    *,
    target_recall: float = 0.8,
    route_width: int | None = None,
    candidates_scored: int | None = None,
    latency_ms: float | None = None,
) -> RouteScore:
    """Score the smallest ranking prefix that reaches target anchor recall."""
    if isinstance(target_recall, bool) or not isinstance(target_recall, (int, float)):
        raise TypeError("target_recall must be numeric")
    if not math.isfinite(float(target_recall)) or not 0 <= target_recall <= 1:
        raise ValueError("target_recall must be finite and in [0, 1]")
    _optional_nonnegative_finite(route_width, "route_width")
    _optional_nonnegative_finite(candidates_scored, "candidates_scored")
    _optional_nonnegative_finite(latency_ms, "latency_ms")

    if any(not isinstance(memory_id, str) or not memory_id for memory_id in ranked_memory_ids):
        raise ValueError("ranked memory IDs must be nonempty strings")
    if len(set(ranked_memory_ids)) != len(ranked_memory_ids):
        raise ValueError("duplicate ranked memory IDs are not allowed")

    by_id: dict[str, MemoryAssessment] = {}
    for assessment in assessments:
        if assessment.memory_id in by_id:
            raise ValueError(f"duplicate assessment for {assessment.memory_id!r}")
        by_id[assessment.memory_id] = assessment
    unknown_ids = set(ranked_memory_ids) - by_id.keys()
    if unknown_ids:
        raise ValueError(f"ranked memories lack assessments: {sorted(unknown_ids)!r}")

    anchors = {
        assessment.memory_id for assessment in assessments if usable_status(assessment) is True
    }
    anchor_retrieved = len(anchors.intersection(ranked_memory_ids))
    evidence_recall = _rate(anchor_retrieved, len(anchors))
    feasible = evidence_recall >= target_recall if evidence_recall is not None else None

    matched_prefix: tuple[str, ...] | None = None
    if anchors and feasible:
        if target_recall == 0:
            matched_prefix = ()
        else:
            hits = 0
            for index, memory_id in enumerate(ranked_memory_ids, start=1):
                hits += int(memory_id in anchors)
                if hits / len(anchors) >= target_recall:
                    matched_prefix = tuple(ranked_memory_ids[:index])
                    break
        if matched_prefix is None:
            raise RuntimeError("feasible route has no matched-recall prefix")

    matched = tuple(by_id[memory_id] for memory_id in (matched_prefix or ()))
    usability = tuple(usable_status(assessment) for assessment in matched)
    admissibility = tuple(admissible_status(assessment) for assessment in matched)

    def matched_rate(predicate: Callable[[MemoryAssessment], bool]) -> float | None:
        if matched_prefix is None:
            return None
        return _rate(sum(predicate(assessment) for assessment in matched), len(matched))

    non_usable = _violation_bounds(usability, prefix_exists=matched_prefix is not None)
    inadmissible = _violation_bounds(admissibility, prefix_exists=matched_prefix is not None)

    return RouteScore(
        target_recall=float(target_recall),
        anchor_total=len(anchors),
        anchor_retrieved=anchor_retrieved,
        evidence_recall=evidence_recall,
        feasible=feasible,
        route_width=len(ranked_memory_ids) if route_width is None else int(route_width),
        matched_prefix_count=len(matched_prefix) if matched_prefix is not None else None,
        non_usable_known_rate=non_usable[0],
        non_usable_label_coverage=non_usable[1],
        non_usable_lower_bound=non_usable[2],
        non_usable_upper_bound=non_usable[3],
        admissibility_violation_known_rate=inadmissible[0],
        admissibility_label_coverage=inadmissible[1],
        admissibility_violation_lower_bound=inadmissible[2],
        admissibility_violation_upper_bound=inadmissible[3],
        known_relevant_admissible_rate=matched_rate(
            lambda assessment: (
                relevant_status(assessment) is True and admissible_status(assessment) is True
            )
        ),
        known_relevant_inadmissible_rate=matched_rate(
            lambda assessment: (
                relevant_status(assessment) is True and admissible_status(assessment) is False
            )
        ),
        known_irrelevant_admissible_rate=matched_rate(
            lambda assessment: (
                relevant_status(assessment) is False and admissible_status(assessment) is True
            )
        ),
        known_irrelevant_inadmissible_rate=matched_rate(
            lambda assessment: (
                relevant_status(assessment) is False and admissible_status(assessment) is False
            )
        ),
        joint_label_coverage=matched_rate(
            lambda assessment: (
                relevant_status(assessment) is not None
                and admissible_status(assessment) is not None
            )
        ),
        wrong_scope_exposure_rate=matched_rate(
            lambda assessment: assessment.scope is Scope.DISALLOWED
        ),
        policy_disallowed_exposure_rate=matched_rate(
            lambda assessment: assessment.policy_allowed is False
        ),
        lifecycle_incompatible_exposure_rate=matched_rate(
            lambda assessment: (
                lifecycle_compatible(
                    assessment.lifecycle_state,
                    assessment.query_intent,
                )
                is False
            )
        ),
        unresolved_admissibility_rate=matched_rate(
            lambda assessment: admissible_status(assessment) is None
        ),
        candidates_scored=(None if candidates_scored is None else int(candidates_scored)),
        latency_ms=None if latency_ms is None else float(latency_ms),
    )


def aggregate_scores(scores: Sequence[RouteScore]) -> AggregateScore:
    """Macro-average available query metrics without missing-value imputation."""
    if not scores:
        raise ValueError("at least one route score is required")
    recall_evaluable = sum(score.evidence_recall is not None for score in scores)
    feasible_evaluable = [score.feasible for score in scores if score.feasible is not None]
    matched_evaluable = sum(score.matched_prefix_count is not None for score in scores)
    return AggregateScore(
        query_count=len(scores),
        recall_evaluable_rate=recall_evaluable / len(scores),
        mean_evidence_recall=_mean(score.evidence_recall for score in scores),
        feasible_rate=_mean(feasible_evaluable),
        matched_recall_evaluable_rate=matched_evaluable / len(scores),
        non_usable_known_rate=_mean(score.non_usable_known_rate for score in scores),
        non_usable_label_coverage=_mean(score.non_usable_label_coverage for score in scores),
        non_usable_lower_bound=_mean(score.non_usable_lower_bound for score in scores),
        non_usable_upper_bound=_mean(score.non_usable_upper_bound for score in scores),
        admissibility_violation_known_rate=_mean(
            score.admissibility_violation_known_rate for score in scores
        ),
        admissibility_label_coverage=_mean(score.admissibility_label_coverage for score in scores),
        admissibility_violation_lower_bound=_mean(
            score.admissibility_violation_lower_bound for score in scores
        ),
        admissibility_violation_upper_bound=_mean(
            score.admissibility_violation_upper_bound for score in scores
        ),
        known_relevant_admissible_rate=_mean(
            score.known_relevant_admissible_rate for score in scores
        ),
        known_relevant_inadmissible_rate=_mean(
            score.known_relevant_inadmissible_rate for score in scores
        ),
        known_irrelevant_admissible_rate=_mean(
            score.known_irrelevant_admissible_rate for score in scores
        ),
        known_irrelevant_inadmissible_rate=_mean(
            score.known_irrelevant_inadmissible_rate for score in scores
        ),
        joint_label_coverage=_mean(score.joint_label_coverage for score in scores),
        wrong_scope_exposure_rate=_mean(score.wrong_scope_exposure_rate for score in scores),
        policy_disallowed_exposure_rate=_mean(
            score.policy_disallowed_exposure_rate for score in scores
        ),
        lifecycle_incompatible_exposure_rate=_mean(
            score.lifecycle_incompatible_exposure_rate for score in scores
        ),
        unresolved_admissibility_rate=_mean(
            score.unresolved_admissibility_rate for score in scores
        ),
        mean_route_width=float(sum(score.route_width for score in scores) / len(scores)),
        mean_matched_prefix_count=_mean(score.matched_prefix_count for score in scores),
        mean_candidates_scored=_mean(score.candidates_scored for score in scores),
        mean_latency_ms=_mean(score.latency_ms for score in scores),
    )
