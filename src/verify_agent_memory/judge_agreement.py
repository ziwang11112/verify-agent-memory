"""Pure sampling and agreement utilities for blinded cross-judge audits."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class AuditAssignment:
    """One reader/source/route assignment to an exact-deduplicated judge payload."""

    request_id: str
    case_id: str
    reader: str
    source: str
    arm: str

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.reader, self.source, self.arm)


@dataclass(frozen=True)
class SelectedAuditUnit:
    """A unique payload selected through one prespecified sampling stratum."""

    request_id: str
    case_id: str
    source: str
    anchor_reader: str
    anchor_arm: str
    associated_readers: tuple[str, ...]
    associated_arms: tuple[str, ...]
    assignment_count: int

    @property
    def anchor_stratum(self) -> tuple[str, str, str]:
        return (self.anchor_reader, self.source, self.anchor_arm)


def _stable_rank(seed: str, stratum: tuple[str, str, str], request_id: str) -> str:
    material = "\x1f".join((seed, *stratum, request_id)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def select_unique_stratified(
    assignments: Sequence[AuditAssignment],
    quotas: Mapping[tuple[str, str, str], int],
    *,
    seed: str,
) -> tuple[SelectedAuditUnit, ...]:
    """Select unique payloads without inspecting responses or judge outcomes."""
    if not seed:
        raise ValueError("sampling seed must be nonempty")
    if not assignments:
        raise ValueError("assignments must be nonempty")
    if not quotas or any(value < 1 for value in quotas.values()):
        raise ValueError("every sampling quota must be positive")

    by_request: dict[str, list[AuditAssignment]] = defaultdict(list)
    by_stratum: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    seen_assignments = set()
    for assignment in assignments:
        if not all(
            (
                assignment.request_id,
                assignment.case_id,
                assignment.reader,
                assignment.source,
                assignment.arm,
            )
        ):
            raise ValueError("audit assignment fields must be nonempty")
        if assignment in seen_assignments:
            raise ValueError("duplicate audit assignment")
        seen_assignments.add(assignment)
        by_request[assignment.request_id].append(assignment)
        by_stratum[assignment.stratum].add(assignment.request_id)

    for request_id, rows in by_request.items():
        if len({row.case_id for row in rows}) != 1 or len({row.source for row in rows}) != 1:
            raise ValueError(f"request {request_id} crosses case or source boundaries")

    if set(by_stratum) != set(quotas):
        missing = sorted(set(quotas) - set(by_stratum))
        extra = sorted(set(by_stratum) - set(quotas))
        raise ValueError(f"sampling strata differ from quotas: missing={missing}, extra={extra}")

    selected: list[SelectedAuditUnit] = []
    used_request_ids = set()
    for stratum in sorted(quotas):
        ranked = sorted(
            by_stratum[stratum],
            key=lambda request_id: (_stable_rank(seed, stratum, request_id), request_id),
        )
        chosen = [request_id for request_id in ranked if request_id not in used_request_ids][
            : quotas[stratum]
        ]
        if len(chosen) != quotas[stratum]:
            raise ValueError(f"stratum {stratum} cannot satisfy its unique-payload quota")
        reader, source, arm = stratum
        for request_id in chosen:
            rows = sorted(by_request[request_id])
            selected.append(
                SelectedAuditUnit(
                    request_id=request_id,
                    case_id=rows[0].case_id,
                    source=source,
                    anchor_reader=reader,
                    anchor_arm=arm,
                    associated_readers=tuple(sorted({row.reader for row in rows})),
                    associated_arms=tuple(sorted({row.arm for row in rows})),
                    assignment_count=len(rows),
                )
            )
            used_request_ids.add(request_id)

    return tuple(selected)


def binary_agreement(first: Sequence[bool], second: Sequence[bool]) -> dict[str, float | int]:
    """Return exact agreement, Cohen's kappa, and Gwet's AC1 for binary labels."""
    if len(first) != len(second) or not first:
        raise ValueError("binary label sequences must have equal nonzero length")
    n = len(first)
    both_true = sum(left and right for left, right in zip(first, second, strict=True))
    both_false = sum(not left and not right for left, right in zip(first, second, strict=True))
    first_true_second_false = sum(
        left and not right for left, right in zip(first, second, strict=True)
    )
    first_false_second_true = n - both_true - both_false - first_true_second_false
    observed = (both_true + both_false) / n

    first_positive = (both_true + first_true_second_false) / n
    second_positive = (both_true + first_false_second_true) / n
    kappa_chance = first_positive * second_positive + ((1 - first_positive) * (1 - second_positive))
    kappa = (observed - kappa_chance) / (1 - kappa_chance) if kappa_chance < 1 else 1.0

    mean_positive = (first_positive + second_positive) / 2
    ac1_chance = 2 * mean_positive * (1 - mean_positive)
    ac1 = (observed - ac1_chance) / (1 - ac1_chance) if ac1_chance < 1 else 1.0
    return {
        "n": n,
        "exact_agreement": observed,
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
        "both_true": both_true,
        "both_false": both_false,
        "first_true_second_false": first_true_second_false,
        "first_false_second_true": first_false_second_true,
    }


def quality_agreement(first: Sequence[int], second: Sequence[int]) -> dict[str, float | int]:
    """Return agreement diagnostics for bounded integer quality scores in [0, 10]."""
    if len(first) != len(second) or not first:
        raise ValueError("quality sequences must have equal nonzero length")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10
        for value in (*first, *second)
    ):
        raise ValueError("quality values must be integers in [0, 10]")
    n = len(first)
    differences = [right - left for left, right in zip(first, second, strict=True)]
    observed_disagreement = sum(difference**2 for difference in differences) / (n * 100)

    first_counts = [first.count(value) for value in range(11)]
    second_counts = [second.count(value) for value in range(11)]
    expected_disagreement = sum(
        first_counts[left] * second_counts[right] * ((left - right) ** 2 / 100)
        for left in range(11)
        for right in range(11)
    ) / (n * n)
    weighted_kappa = (
        1 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else 1.0
    )
    return {
        "n": n,
        "exact_agreement": sum(difference == 0 for difference in differences) / n,
        "within_one_agreement": sum(abs(difference) <= 1 for difference in differences) / n,
        "mean_absolute_error": sum(abs(difference) for difference in differences) / n,
        "mean_strong_minus_cheap": sum(differences) / n,
        "quadratic_weighted_kappa": weighted_kappa,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_binary_agreement(
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Return item-bootstrap percentile intervals for the three agreement measures."""
    if len(first) != len(second) or not first:
        raise ValueError("binary label sequences must have equal nonzero length")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        "exact_agreement": [],
        "cohen_kappa": [],
        "gwet_ac1": [],
    }
    for _ in range(iterations):
        indices = [rng.randrange(len(first)) for _ in first]
        metrics = binary_agreement(
            [first[index] for index in indices],
            [second[index] for index in indices],
        )
        for key in samples:
            samples[key].append(float(metrics[key]))
    return {
        key: (_percentile(values, 0.025), _percentile(values, 0.975))
        for key, values in samples.items()
    }
