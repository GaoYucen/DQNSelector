from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import networkx as nx
import numpy as np

from .influence import mc_activation_probabilities


@dataclass
class ECMInstance:
    """Paper-level Effective Coverage Maximization instance.

    Arrays are indexed by the order of `nodes`. `participation[v,i]` is p_v^i,
    `quality[v,i]` is q_v^i, and `demand[i]` is d_i in Eqs. (2)--(6).
    """

    graph: nx.DiGraph
    nodes: list[int]
    participation: np.ndarray
    quality: np.ndarray
    demand: np.ndarray
    worker_pool: set[int] | None = None

    def __post_init__(self) -> None:
        self.nodes = list(self.nodes)
        n = len(self.nodes)
        self.participation = np.asarray(self.participation, dtype=np.float64)
        self.quality = np.asarray(self.quality, dtype=np.float64)
        self.demand = np.asarray(self.demand, dtype=np.float64)
        if self.participation.shape != self.quality.shape:
            raise ValueError("participation and quality must have the same shape")
        if self.participation.ndim != 2 or self.participation.shape[0] != n:
            raise ValueError("participation/quality must be [n_nodes, n_subareas]")
        if self.demand.shape != (self.participation.shape[1],):
            raise ValueError("demand must have one value per subarea")
        if np.any(self.demand <= 0):
            raise ValueError("all demands must be positive")
        if np.any(self.participation < 0) or np.any(self.participation > 1):
            raise ValueError("participation probabilities must lie in [0,1]")
        missing = set(self.nodes) - set(self.graph.nodes())
        if missing:
            raise ValueError(f"nodes absent from graph: {sorted(missing)[:5]}")
        if self.worker_pool is None:
            self.worker_pool = set(self.nodes)
        else:
            self.worker_pool = set(self.worker_pool)
            if not self.worker_pool.issubset(set(self.nodes)):
                raise ValueError("worker_pool must be a subset of nodes")

    @property
    def n_subareas(self) -> int:
        return int(self.participation.shape[1])


def expected_subarea_coverage(
    instance: ECMInstance,
    activation_probabilities: dict[int, float],
) -> np.ndarray:
    """Compute C(a_i,S) in Eq. (3)."""
    idx = {v: i for i, v in enumerate(instance.nodes)}
    p_active = np.array([activation_probabilities.get(v, 0.0) for v in instance.nodes])
    contribution = instance.participation * instance.quality
    return (p_active[:, None] * contribution).sum(axis=0)


def effective_coverage_from_activation(
    instance: ECMInstance,
    activation_probabilities: dict[int, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute EC(S) and per-subarea values from Eqs. (4)--(5)."""
    coverage = expected_subarea_coverage(instance, activation_probabilities)
    effective = np.minimum(coverage / instance.demand, 1.0)
    return float(effective.mean()), effective, coverage


def effective_coverage(
    instance: ECMInstance,
    seeds: Iterable[int],
    mc_times: int = 100,
    random_seed: int | None = None,
) -> float:
    """Monte-Carlo estimate of the paper's EC(S)."""
    seeds = set(int(v) for v in seeds)
    if not seeds.issubset(instance.worker_pool or set()):
        raise ValueError("all seeds must belong to the worker pool")
    probs = mc_activation_probabilities(
        instance.graph, seeds, mc_times=mc_times, random_seed=random_seed
    )
    ec, _, _ = effective_coverage_from_activation(instance, probs)
    return ec


def marginal_reward(
    instance: ECMInstance,
    seed_set: Iterable[int],
    candidate: int,
    mc_times: int = 100,
    random_seed: int | None = None,
) -> float:
    """Eq. (15): EC(S U {v}) - EC(S).

    Common random numbers are used for the two estimates when a random_seed is
    supplied. This reduces avoidable reward noise without changing the objective.
    """
    current = set(int(v) for v in seed_set)
    if candidate in current:
        return 0.0
    before = effective_coverage(instance, current, mc_times, random_seed)
    after = effective_coverage(instance, current | {int(candidate)}, mc_times, random_seed)
    return float(after - before)
