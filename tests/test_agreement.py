from __future__ import annotations

import pytest

from verify_agent_memory.agreement import agreement_by_axis, nominal_alpha


def test_nominal_alpha_excludes_uncertain_pairs() -> None:
    pairs = [("yes", "yes"), ("yes", "no"), ("uncertain", "yes")]
    assert nominal_alpha(pairs) == pytest.approx(0.0)


def test_nominal_alpha_handles_perfect_and_unevaluable_labels() -> None:
    assert nominal_alpha([("yes", "yes"), ("yes", "yes")]) == 1.0
    assert nominal_alpha([("uncertain", "yes")]) is None


def test_axis_summary_reports_uncertainty_without_reviewer_identity() -> None:
    summaries = agreement_by_axis(
        {
            "scope": [("allowed", "allowed"), ("uncertain", "disallowed")],
            "usable": [("yes", "yes"), ("no", "yes")],
        }
    )
    scope = next(summary for summary in summaries if summary.axis == "scope")
    assert scope.records == 2
    assert scope.exact_agreement == 0.5
    assert scope.uncertain_labels == 1
    assert scope.uncertainty_rate == 0.25
    assert scope.alpha_evaluable_records == 1


def test_axes_must_have_identical_record_coverage() -> None:
    with pytest.raises(ValueError, match="identical record coverage"):
        agreement_by_axis({"scope": [("a", "a")], "usable": [("a", "a"), ("b", "b")]})
