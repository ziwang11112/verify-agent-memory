"""Execution and development-only selection for retrieval experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verify_agent_memory.metrics import RouteScore, score_route
from verify_agent_memory.retrieval import (
    MemoryRecord,
    QueryRecord,
    RetrievalArm,
    RetrievalConfig,
    RouteResult,
    route,
)
from verify_agent_memory.schema import MemoryAssessment


@dataclass(frozen=True)
class ExperimentCase:
    source: str
    group_id: str
    query: QueryRecord
    memories: tuple[MemoryRecord, ...]
    assessments: tuple[MemoryAssessment, ...]

    def __post_init__(self) -> None:
        if not self.source or not self.group_id:
            raise ValueError("source and group_id must be nonempty")
        memory_ids = {memory.memory_id for memory in self.memories}
        if len(memory_ids) != len(self.memories):
            raise ValueError("case memory IDs must be unique")
        assessment_ids = {assessment.memory_id for assessment in self.assessments}
        if len(assessment_ids) != len(self.assessments):
            raise ValueError("case assessment IDs must be unique")
        if assessment_ids != memory_ids:
            raise ValueError("assessments must cover every case memory exactly once")


@dataclass(frozen=True)
class QueryRun:
    source: str
    group_id: str
    query_id: str
    setting_id: str
    arm: RetrievalArm
    route: RouteResult
    score: RouteScore


@dataclass(frozen=True)
class SettingSummary:
    setting_id: str
    arm: RetrievalArm
    source_count: int
    query_count: int
    feasible_rate: float
    penalized_conservative_risk: float
    penalized_resolved_contamination: float
    evidence_recall: float
    candidates_scored: float

    @property
    def selection_key(self) -> tuple[float | str, ...]:
        """Lexicographic key; smaller is better."""
        return (
            -self.feasible_rate,
            self.penalized_conservative_risk,
            self.penalized_resolved_contamination,
            -self.evidence_recall,
            self.candidates_scored,
            self.setting_id,
        )


def run_case(
    case: ExperimentCase,
    config: RetrievalConfig,
    *,
    target_recall: float = 0.8,
) -> QueryRun:
    """Route and score one query without mutating the case or configuration."""
    routed = route(case.memories, case.query, config)
    score = score_route(
        routed.ranked_memory_ids,
        case.assessments,
        target_recall=target_recall,
        route_width=routed.route_width,
        candidates_scored=routed.candidates_scored,
    )
    return QueryRun(
        source=case.source,
        group_id=case.group_id,
        query_id=case.query.query_id,
        setting_id=config.setting_id,
        arm=config.arm,
        route=routed,
        score=score,
    )


def run_experiment(
    cases: Sequence[ExperimentCase],
    configs: Sequence[RetrievalConfig],
    *,
    target_recall: float = 0.8,
) -> tuple[QueryRun, ...]:
    """Run the complete Cartesian product of cases and frozen settings."""
    if not cases or not configs:
        raise ValueError("cases and configs must be nonempty")
    case_ids = [(case.source, case.query.query_id) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("source/query identities must be unique")
    setting_ids = [config.setting_id for config in configs]
    if len(set(setting_ids)) != len(setting_ids):
        raise ValueError("setting IDs must be unique")
    return tuple(
        run_case(case, config, target_recall=target_recall) for config in configs for case in cases
    )


def _mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("macro means require finite, nonempty values")
    return sum(values) / len(values)


def _penalized_upper(score: RouteScore) -> float | None:
    if score.feasible is None:
        return None
    if score.feasible and score.contamination_upper_bound is not None:
        return score.contamination_upper_bound
    return 1.0


def _penalized_resolved(score: RouteScore) -> float | None:
    if score.feasible is None:
        return None
    if score.feasible and score.contamination_known_rate is not None:
        return score.contamination_known_rate
    return 1.0


def _available(values: Sequence[float | bool | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def summarize_setting(runs: Sequence[QueryRun]) -> SettingSummary:
    """Equal-weight sources after query-macro aggregation within each source."""
    if not runs:
        raise ValueError("at least one query run is required")
    setting_ids = {run.setting_id for run in runs}
    arms = {run.arm for run in runs}
    if len(setting_ids) != 1 or len(arms) != 1:
        raise ValueError("one setting summary cannot mix settings or arms")
    identities = [(run.source, run.query_id) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("setting rows contain duplicate source/query identities")

    by_source: defaultdict[str, list[QueryRun]] = defaultdict(list)
    for run in runs:
        by_source[run.source].append(run)

    source_metrics: list[tuple[float, float, float, float, float]] = []
    for source_runs in by_source.values():
        feasible = _available([run.score.feasible for run in source_runs])
        upper = _available([_penalized_upper(run.score) for run in source_runs])
        resolved = _available([_penalized_resolved(run.score) for run in source_runs])
        recall = _available([run.score.evidence_recall for run in source_runs])
        candidates = [float(run.route.candidates_scored) for run in source_runs]
        if not feasible or not upper or not resolved or not recall:
            raise ValueError("every source needs recall-evaluable development queries")
        source_metrics.append(
            (
                _mean(feasible),
                _mean(upper),
                _mean(resolved),
                _mean(recall),
                _mean(candidates),
            )
        )

    return SettingSummary(
        setting_id=next(iter(setting_ids)),
        arm=next(iter(arms)),
        source_count=len(by_source),
        query_count=len(runs),
        feasible_rate=_mean([row[0] for row in source_metrics]),
        penalized_conservative_risk=_mean([row[1] for row in source_metrics]),
        penalized_resolved_contamination=_mean([row[2] for row in source_metrics]),
        evidence_recall=_mean([row[3] for row in source_metrics]),
        candidates_scored=_mean([row[4] for row in source_metrics]),
    )


def select_dev_settings(runs: Sequence[QueryRun]) -> Mapping[RetrievalArm, SettingSummary]:
    """Select one setting per arm after checking complete dev-grid coverage."""
    if not runs:
        raise ValueError("development rows are required")
    by_setting: defaultdict[tuple[RetrievalArm, str], list[QueryRun]] = defaultdict(list)
    cases_by_setting: dict[tuple[RetrievalArm, str], set[tuple[str, str]]] = {}
    for run in runs:
        key = (run.arm, run.setting_id)
        by_setting[key].append(run)
        cases_by_setting.setdefault(key, set()).add((run.source, run.query_id))

    reference_cases = next(iter(cases_by_setting.values()))
    if any(cases != reference_cases for cases in cases_by_setting.values()):
        raise ValueError("every development setting must cover the same source/query rows")

    summaries: defaultdict[RetrievalArm, list[SettingSummary]] = defaultdict(list)
    for (arm, _), setting_runs in by_setting.items():
        summaries[arm].append(summarize_setting(setting_runs))
    return {
        arm: min(arm_summaries, key=lambda summary: summary.selection_key)
        for arm, arm_summaries in sorted(summaries.items(), key=lambda row: row[0].value)
    }
