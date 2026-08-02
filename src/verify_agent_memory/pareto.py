"""Fixed-ranking top-k recall, risk, and route-cost frontiers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from verify_agent_memory.experiment import (
    ExperimentCase,
    QueryRun,
    SettingSummary,
    run_case,
    summarize_setting,
)
from verify_agent_memory.metrics import score_route
from verify_agent_memory.retrieval import RetrievalArm, RetrievalConfig


@dataclass(frozen=True)
class ParetoPoint:
    base_setting_id: str
    setting_id: str
    arm: RetrievalArm
    top_k: int
    summary: SettingSummary
    mean_returned_count: float
    mean_candidates_scored: float
    mean_route_width: float
    mean_latency_ms: float | None


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _source_macro_costs(
    runs: Sequence[QueryRun],
) -> tuple[float, float, float, float | None]:
    by_source: defaultdict[str, list[QueryRun]] = defaultdict(list)
    for run in runs:
        by_source[run.source].append(run)
    returned = []
    candidates = []
    route_width = []
    latency = []
    for source_runs in by_source.values():
        returned.append(_mean([float(len(run.route.ranked_memory_ids)) for run in source_runs]))
        candidates.append(_mean([float(run.route.candidates_scored) for run in source_runs]))
        route_width.append(_mean([float(run.route.route_width) for run in source_runs]))
        available_latency = [
            float(run.score.latency_ms) for run in source_runs if run.score.latency_ms is not None
        ]
        if available_latency:
            latency.append(_mean(available_latency))
    return (
        _mean(returned),
        _mean(candidates),
        _mean(route_width),
        _mean(latency) if latency else None,
    )


def run_top_k_pareto(
    cases: Sequence[ExperimentCase],
    configs: Sequence[RetrievalConfig],
    top_k_values: Sequence[int],
    *,
    target_recall: float = 0.8,
) -> tuple[ParetoPoint, ...]:
    """Run each ranking once at max depth, then score stable prefixes."""
    if not cases or not configs:
        raise ValueError("cases and configs must be nonempty")
    if (
        not top_k_values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in top_k_values
        )
        or len(set(top_k_values)) != len(top_k_values)
    ):
        raise ValueError("top_k values must be unique positive integers")
    ordered_depths = tuple(sorted(top_k_values))
    maximum_depth = ordered_depths[-1]

    points: list[ParetoPoint] = []
    for config in configs:
        maximum_config = replace(config, top_k=maximum_depth)
        maximum_runs = tuple(
            run_case(case, maximum_config, target_recall=target_recall) for case in cases
        )
        for depth in ordered_depths:
            setting_id = f"{config.setting_id}:top_k={depth}"
            prefix_runs = []
            for case, full_run in zip(cases, maximum_runs, strict=True):
                prefix_route = replace(
                    full_run.route,
                    setting_id=setting_id,
                    ranked_memory_ids=full_run.route.ranked_memory_ids[:depth],
                )
                prefix_runs.append(
                    QueryRun(
                        source=full_run.source,
                        group_id=full_run.group_id,
                        query_id=full_run.query_id,
                        setting_id=setting_id,
                        arm=full_run.arm,
                        route=prefix_route,
                        score=score_route(
                            prefix_route.ranked_memory_ids,
                            case.assessments,
                            target_recall=target_recall,
                            route_width=prefix_route.route_width,
                            candidates_scored=prefix_route.candidates_scored,
                        ),
                    )
                )
            summary = summarize_setting(prefix_runs)
            returned, candidates_scored, route_width, latency_ms = _source_macro_costs(prefix_runs)
            points.append(
                ParetoPoint(
                    base_setting_id=config.setting_id,
                    setting_id=setting_id,
                    arm=config.arm,
                    top_k=depth,
                    summary=summary,
                    mean_returned_count=returned,
                    mean_candidates_scored=candidates_scored,
                    mean_route_width=route_width,
                    mean_latency_ms=latency_ms,
                )
            )
    return tuple(points)
