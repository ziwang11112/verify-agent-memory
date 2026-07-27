"""Deterministic cluster-macro bootstrap utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    weight = position - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def cluster_macro_bootstrap(
    deltas_by_cluster: Mapping[str, Sequence[float]],
    *,
    replicates: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap the unweighted macro mean of finite cluster-level means."""
    if not deltas_by_cluster:
        raise ValueError("at least one cluster is required")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    cluster_means: list[float] = []
    for cluster, values in sorted(deltas_by_cluster.items()):
        if not cluster or not values:
            raise ValueError("cluster names and values must be nonempty")
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("bootstrap values must be finite")
        cluster_means.append(_mean(numeric))

    rng = random.Random(seed)
    samples = sorted(
        _mean([rng.choice(cluster_means) for _ in cluster_means]) for _ in range(replicates)
    )
    tail = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=_mean(cluster_means),
        lower=_quantile(samples, tail),
        upper=_quantile(samples, 1 - tail),
        replicates=replicates,
        seed=seed,
    )
