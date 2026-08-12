"""Pure query-conditioned admissibility rules."""

from __future__ import annotations

from verify_agent_memory.schema import (
    LifecycleState,
    MemoryAssessment,
    QueryIntent,
    Relevance,
    Scope,
)

TriState = bool | None


def _conjunction(values: tuple[TriState, ...]) -> TriState:
    if any(value is False for value in values):
        return False
    if all(value is True for value in values):
        return True
    return None


def lifecycle_compatible(state: LifecycleState, intent: QueryIntent) -> TriState:
    """Return whether a lifecycle state can support the query intent."""
    if state is LifecycleState.UNKNOWN or intent is QueryIntent.UNKNOWN:
        return None
    if state is LifecycleState.CURRENT:
        return True
    return intent is QueryIntent.HISTORY


def admissible_status(assessment: MemoryAssessment) -> TriState:
    """Evaluate scope, policy, and lifecycle without using relevance."""
    scope_allowed: TriState
    if assessment.scope is Scope.ALLOWED:
        scope_allowed = True
    elif assessment.scope is Scope.DISALLOWED:
        scope_allowed = False
    else:
        scope_allowed = None
    return _conjunction(
        (
            scope_allowed,
            assessment.policy_allowed,
            lifecycle_compatible(
                assessment.lifecycle_state,
                assessment.query_intent,
            ),
        )
    )


def relevant_status(assessment: MemoryAssessment) -> TriState:
    """Evaluate relevance alone without imputing unknown labels."""
    if assessment.relevance in {Relevance.REQUIRED, Relevance.SUPPORTIVE}:
        return True
    if assessment.relevance is Relevance.NOT_USEFUL:
        return False
    return None


def usable_status(assessment: MemoryAssessment) -> TriState:
    """Evaluate relevance and admissibility without imputing unknown labels."""
    return _conjunction((relevant_status(assessment), admissible_status(assessment)))
