from __future__ import annotations

import math

import pytest

from verify_agent_memory.analysis.bootstrap import cluster_macro_bootstrap
from verify_agent_memory.analysis.exposure import exposure_risk_difference
from verify_agent_memory.schema import ExposureUseRecord


def test_exposure_risk_difference_is_exposed_minus_unexposed() -> None:
    records = [
        ExposureUseRecord("a", exposed=True, used_in_answer=True),
        ExposureUseRecord("b", exposed=True, used_in_answer=False),
        ExposureUseRecord("c", exposed=False, used_in_answer=False),
        ExposureUseRecord("d", exposed=False, used_in_answer=False),
    ]
    assert exposure_risk_difference(records) == 0.5


def test_exposure_risk_difference_requires_both_groups() -> None:
    with pytest.raises(ValueError, match="both exposed and unexposed"):
        exposure_risk_difference([ExposureUseRecord("a", exposed=True, used_in_answer=True)])


def test_cluster_macro_bootstrap_is_deterministic_and_cluster_weighted() -> None:
    values = {"large": [1.0, 1.0, 1.0, 1.0], "small": [-1.0]}
    first = cluster_macro_bootstrap(values, replicates=200, seed=17)
    second = cluster_macro_bootstrap(values, replicates=200, seed=17)
    assert first == second
    assert first.estimate == 0.0
    assert first.lower <= first.estimate <= first.upper


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_cluster_macro_bootstrap_rejects_nonfinite_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        cluster_macro_bootstrap({"source": [bad_value]}, replicates=10)
