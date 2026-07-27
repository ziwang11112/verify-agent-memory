"""Agreement statistics that never expose reviewer identity."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisAgreement:
    axis: str
    records: int
    exact_agreement: float
    uncertain_labels: int
    uncertainty_rate: float
    alpha_evaluable_records: int
    krippendorff_alpha_nominal: float | None


def nominal_alpha(
    pairs: Sequence[tuple[object, object]],
    *,
    uncertain_value: object = "uncertain",
) -> float | None:
    """Compute two-rater nominal Krippendorff alpha after excluding uncertainty."""
    complete = [
        (left, right)
        for left, right in pairs
        if left != uncertain_value and right != uncertain_value
    ]
    if not complete:
        return None
    observed_disagreement = sum(left != right for left, right in complete) / len(complete)
    counts = Counter(value for pair in complete for value in pair)
    total = sum(counts.values())
    if total < 2:
        return None
    expected_agreement = sum(count * (count - 1) for count in counts.values()) / (
        total * (total - 1)
    )
    expected_disagreement = 1 - expected_agreement
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def agreement_by_axis(
    pairs_by_axis: Mapping[str, Sequence[tuple[object, object]]],
    *,
    uncertain_value: object = "uncertain",
) -> tuple[AxisAgreement, ...]:
    """Summarize paired labels without retaining rater identifiers."""
    if not pairs_by_axis:
        raise ValueError("at least one label axis is required")
    summaries: list[AxisAgreement] = []
    expected_records: int | None = None
    for axis, pairs in sorted(pairs_by_axis.items()):
        if not axis or not pairs:
            raise ValueError("axis names and paired labels must be nonempty")
        if expected_records is None:
            expected_records = len(pairs)
        elif len(pairs) != expected_records:
            raise ValueError("all axes must have identical record coverage")
        uncertain = sum(value == uncertain_value for pair in pairs for value in pair)
        complete = [
            pair for pair in pairs if pair[0] != uncertain_value and pair[1] != uncertain_value
        ]
        summaries.append(
            AxisAgreement(
                axis=axis,
                records=len(pairs),
                exact_agreement=sum(left == right for left, right in pairs) / len(pairs),
                uncertain_labels=uncertain,
                uncertainty_rate=uncertain / (2 * len(pairs)),
                alpha_evaluable_records=len(complete),
                krippendorff_alpha_nominal=nominal_alpha(
                    pairs,
                    uncertain_value=uncertain_value,
                ),
            )
        )
    return tuple(summaries)
