from __future__ import annotations

import numpy as np
import pytest

from scripts.run_frozen_natural_support_expansion import (
    Accumulator,
    _alternate_namespace_indices,
    _false_allow_support_at,
    _swap_support_at,
    _uniform_vector,
)


def test_false_allow_curve_is_nested_and_preserves_own_namespace() -> None:
    positions = np.arange(6)
    own_namespace = np.array([True, False, False, True, False, False])
    gates = _uniform_vector(3, "namespace_false_allow", "source", "query", count=6)

    support_10 = _false_allow_support_at(
        positions,
        own_namespace=own_namespace,
        gate_values=gates,
        rate=0.1,
    )
    support_50 = _false_allow_support_at(
        positions,
        own_namespace=own_namespace,
        gate_values=gates,
        rate=0.5,
    )

    assert np.all(support_10 <= support_50)
    assert np.all(support_50[own_namespace])


def test_namespace_swap_never_selects_original_label() -> None:
    original = np.array([0, 1, 2, 3, 0, 1])
    random_values = _uniform_vector(4, "namespace_swap", "source", count=len(original))
    alternate = _alternate_namespace_indices(original, 4, random_values)

    assert np.all(alternate != original)
    assert np.all((0 <= alternate) & (alternate < 4))

    positions = np.arange(len(original))
    selected = np.zeros(len(original))
    swapped_support = _swap_support_at(
        positions,
        selected=selected,
        alternate=alternate,
        candidate_original=original,
        query_namespace=0,
        rate=1.0,
    )
    assert np.array_equal(swapped_support, alternate == 0)


def test_accumulator_preserves_penalized_risk_decomposition() -> None:
    accumulator = Accumulator()
    accumulator.update(
        {
            "feasible": True,
            "admissibility_upper": 0.25,
            "evidence_recall": 1.0,
            "returned_wrong_scope_rate": 0.1,
            "matched_prefix_wrong_scope_exposure_rate": 0.2,
            "known_relevant_inadmissible_rate": 0.0,
            "joint_label_coverage": 0.5,
            "candidates_scored": 10,
        }
    )
    accumulator.update(
        {
            "feasible": False,
            "admissibility_upper": None,
            "evidence_recall": 0.5,
            "returned_wrong_scope_rate": 0.2,
            "matched_prefix_wrong_scope_exposure_rate": None,
            "known_relevant_inadmissible_rate": None,
            "joint_label_coverage": None,
            "candidates_scored": 20,
        }
    )

    means = accumulator.means()
    assert means["penalized_admissibility_upper_risk"] == pytest.approx(0.625)
    assert means["infeasibility_risk_component"] == pytest.approx(0.5)
    assert means["admissibility_conditional_risk_component"] == pytest.approx(0.125)
    assert means["penalized_admissibility_upper_risk"] == pytest.approx(
        means["infeasibility_risk_component"] + means["admissibility_conditional_risk_component"]
    )
