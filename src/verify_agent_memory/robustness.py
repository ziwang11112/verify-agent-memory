"""Deterministic corruption curves for method-visible governance metadata."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from verify_agent_memory.experiment import (
    ExperimentCase,
    QueryRun,
    SettingSummary,
    run_experiment,
    summarize_setting,
)
from verify_agent_memory.retrieval import PolicyDecision, RetrievalArm, RetrievalConfig
from verify_agent_memory.schema import LifecycleState, QueryIntent


class CorruptionChannel(StrEnum):
    NAMESPACE_FALSE_ALLOW = "namespace_false_allow"
    NAMESPACE_FALSE_DENY = "namespace_false_deny"
    NAMESPACE_MISSING = "namespace_missing"
    NAMESPACE_SWAP = "namespace_swap"
    LIFECYCLE_FALSE_CURRENT = "lifecycle_false_current"
    LIFECYCLE_FALSE_STALE = "lifecycle_false_stale"
    LIFECYCLE_MISSING = "lifecycle_missing"
    POLICY_FALSE_ALLOW = "policy_false_allow"
    POLICY_FALSE_DENY = "policy_false_deny"
    POLICY_MISSING = "policy_missing"
    QUERY_INTENT_FLIP = "query_intent_flip"


@dataclass(frozen=True)
class MetadataCorruption:
    corruption_id: str
    channel: CorruptionChannel
    rate: float
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.corruption_id:
            raise ValueError("corruption_id must be nonempty")
        if not isinstance(self.channel, CorruptionChannel):
            raise TypeError("channel must be CorruptionChannel")
        if isinstance(self.rate, bool) or not math.isfinite(self.rate) or not 0 <= self.rate <= 1:
            raise ValueError("rate must be finite and in [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class RobustnessPoint:
    corruption_id: str
    channel: CorruptionChannel
    rate: float
    seed: int
    setting: SettingSummary


@dataclass(frozen=True)
class BreakEvenInterval:
    channel: CorruptionChannel
    treatment_arm: RetrievalArm
    reference_arm: RetrievalArm
    last_dominating_rate: float | None
    first_non_dominating_rate: float | None


def _uniform(config: MetadataCorruption, *identity: str) -> float:
    payload = "\0".join((str(config.seed), config.channel.value, *identity)).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def _selected(config: MetadataCorruption, *identity: str) -> bool:
    return _uniform(config, *identity) < config.rate


def _alternate_namespace(
    current: str,
    namespaces: Sequence[str],
    config: MetadataCorruption,
    *identity: str,
) -> str:
    alternatives = sorted(namespace for namespace in set(namespaces) if namespace != current)
    if not alternatives:
        return f"__corrupt_namespace__:{current}"
    index = int(_uniform(config, *identity, "alternate") * len(alternatives))
    return alternatives[min(index, len(alternatives) - 1)]


def _corrupt_policy(
    decision: PolicyDecision,
    config: MetadataCorruption,
    *,
    source: str,
    query_id: str,
) -> PolicyDecision:
    if not _selected(config, source, query_id, decision.memory_id, "policy"):
        return decision

    def corrupt(value: bool | None) -> bool | None:
        if config.channel is CorruptionChannel.POLICY_FALSE_ALLOW and value is False:
            return True
        if config.channel is CorruptionChannel.POLICY_FALSE_DENY and value is True:
            return False
        if config.channel is CorruptionChannel.POLICY_MISSING:
            return None
        return value

    return replace(
        decision,
        content_disclosure_allowed=corrupt(decision.content_disclosure_allowed),
        operation_trace_allowed=corrupt(decision.operation_trace_allowed),
    )


def corrupt_case(
    case: ExperimentCase,
    config: MetadataCorruption,
    *,
    namespace_vocabulary: Sequence[str] | None = None,
) -> ExperimentCase:
    """Corrupt route-visible metadata while preserving scorer assessments exactly.

    Namespace swap/missing/false-deny and lifecycle channels are record-level.
    Namespace false-allow is a query-memory gate error, policy channels are
    query-memory-level, and intent channels are query-level.
    """
    if config.rate == 0:
        return case

    namespaces = tuple(
        namespace_vocabulary
        or (case.query.namespace, *(memory.namespace for memory in case.memories))
    )
    memories = []
    for memory in case.memories:
        namespace = memory.namespace
        state = memory.lifecycle_state
        record_selected = _selected(
            config,
            case.source,
            memory.memory_id,
            "memory",
        )
        false_allow_selected = _selected(
            config,
            case.source,
            case.query.query_id,
            memory.memory_id,
            "namespace_gate",
        )
        if config.channel is CorruptionChannel.NAMESPACE_FALSE_ALLOW:
            if false_allow_selected and namespace != case.query.namespace:
                namespace = case.query.namespace
        elif record_selected:
            if (
                config.channel is CorruptionChannel.NAMESPACE_FALSE_DENY
                and namespace == case.query.namespace
            ):
                namespace = f"__false_deny__:{memory.memory_id}"
            elif config.channel is CorruptionChannel.NAMESPACE_MISSING:
                namespace = f"__missing_namespace__:{memory.memory_id}"
            elif config.channel is CorruptionChannel.NAMESPACE_SWAP:
                namespace = _alternate_namespace(
                    namespace,
                    namespaces,
                    config,
                    case.source,
                    memory.memory_id,
                )
            elif config.channel is CorruptionChannel.LIFECYCLE_FALSE_CURRENT and state in {
                LifecycleState.STALE,
                LifecycleState.SUPERSEDED,
            }:
                state = LifecycleState.CURRENT
            elif (
                config.channel is CorruptionChannel.LIFECYCLE_FALSE_STALE
                and state is LifecycleState.CURRENT
            ):
                state = LifecycleState.STALE
            elif config.channel is CorruptionChannel.LIFECYCLE_MISSING:
                state = LifecycleState.UNKNOWN
        memories.append(replace(memory, namespace=namespace, lifecycle_state=state))

    query = case.query
    if config.channel is CorruptionChannel.QUERY_INTENT_FLIP and _selected(
        config, case.source, query.query_id, "query_intent"
    ):
        if query.intent is QueryIntent.CURRENT_STATE:
            query = replace(query, intent=QueryIntent.HISTORY)
        elif query.intent is QueryIntent.HISTORY:
            query = replace(query, intent=QueryIntent.CURRENT_STATE)

    policy_decisions = tuple(
        _corrupt_policy(
            decision,
            config,
            source=case.source,
            query_id=case.query.query_id,
        )
        for decision in case.policy_decisions
    )
    return replace(
        case,
        query=query,
        memories=tuple(memories),
        policy_decisions=policy_decisions,
    )


def run_corruption_curve(
    cases: Sequence[ExperimentCase],
    configs: Sequence[RetrievalConfig],
    corruptions: Sequence[MetadataCorruption],
    *,
    target_recall: float = 0.8,
) -> tuple[RobustnessPoint, ...]:
    """Run fixed retrieval settings over an immutable metadata corruption grid."""
    if not corruptions:
        raise ValueError("at least one corruption setting is required")
    identities = [corruption.corruption_id for corruption in corruptions]
    if len(set(identities)) != len(identities):
        raise ValueError("corruption IDs must be unique")

    namespaces_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for case in cases:
        namespaces_by_source[case.source].add(case.query.namespace)
        namespaces_by_source[case.source].update(memory.namespace for memory in case.memories)

    points: list[RobustnessPoint] = []
    for corruption in corruptions:
        corrupted_cases = tuple(
            corrupt_case(
                case,
                corruption,
                namespace_vocabulary=tuple(sorted(namespaces_by_source[case.source])),
            )
            for case in cases
        )
        runs = run_experiment(corrupted_cases, configs, target_recall=target_recall)
        by_setting: defaultdict[str, list[QueryRun]] = defaultdict(list)
        for run in runs:
            by_setting[run.setting_id].append(run)
        for setting_id in sorted(by_setting):
            summary = summarize_setting(by_setting[setting_id])
            points.append(
                RobustnessPoint(
                    corruption_id=corruption.corruption_id,
                    channel=corruption.channel,
                    rate=corruption.rate,
                    seed=corruption.seed,
                    setting=summary,
                )
            )
    return tuple(points)


def break_even_interval(
    points: Sequence[RobustnessPoint],
    *,
    channel: CorruptionChannel,
    treatment_arm: RetrievalArm = RetrievalArm.NAMESPACE_DENSE,
    reference_arm: RetrievalArm = RetrievalArm.GLOBAL_DENSE,
    tolerance: float = 1e-12,
) -> BreakEvenInterval:
    """Bracket the first observed rate where treatment stops weakly dominating."""
    by_rate: defaultdict[float, dict[RetrievalArm, SettingSummary]] = defaultdict(dict)
    for point in points:
        if point.channel is channel:
            if point.setting.arm in by_rate[point.rate]:
                raise ValueError("break-even input has duplicate arm/rate rows")
            by_rate[point.rate][point.setting.arm] = point.setting
    if not by_rate:
        raise ValueError(f"no robustness points for {channel.value}")

    last_dominating: float | None = None
    first_non_dominating: float | None = None
    for rate, summaries in sorted(by_rate.items()):
        if treatment_arm not in summaries or reference_arm not in summaries:
            raise ValueError("every rate needs treatment and reference arms")
        treatment = summaries[treatment_arm]
        reference = summaries[reference_arm]
        dominates = (
            treatment.feasible_rate + tolerance >= reference.feasible_rate
            and treatment.penalized_admissibility_upper_risk
            <= reference.penalized_admissibility_upper_risk + tolerance
        )
        if dominates and first_non_dominating is None:
            last_dominating = rate
        elif not dominates and first_non_dominating is None:
            first_non_dominating = rate
    return BreakEvenInterval(
        channel=channel,
        treatment_arm=treatment_arm,
        reference_arm=reference_arm,
        last_dominating_rate=last_dominating,
        first_non_dominating_rate=first_non_dominating,
    )
