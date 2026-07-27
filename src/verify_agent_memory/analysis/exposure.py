"""Descriptive exposure-to-use association statistics."""

from __future__ import annotations

from collections.abc import Sequence

from verify_agent_memory.schema import ExposureUseRecord


def exposure_risk_difference(records: Sequence[ExposureUseRecord]) -> float:
    """Return exposed-minus-unexposed answer-use risk without a causal interpretation."""
    exposed = [record.used_in_answer for record in records if record.exposed]
    unexposed = [record.used_in_answer for record in records if not record.exposed]
    if not exposed or not unexposed:
        raise ValueError("both exposed and unexposed observations are required")
    return sum(exposed) / len(exposed) - sum(unexposed) / len(unexposed)
