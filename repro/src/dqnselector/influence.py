from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np


def _edge_probability(graph: nx.DiGraph, u: int, v: int) -> float:
    p = float(graph[u][v].get("weight", 1.0))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"edge probability must lie in [0,1], got {p} for ({u},{v})")
    return p


def simulate_ic_once(
    graph: nx.DiGraph,
    seeds: Iterable[int],
    rng: np.random.Generator | None = None,
) -> set[int]:
    """Run one Independent Cascade realization.

    Each newly activated node gets exactly one chance to activate each currently
    inactive out-neighbor. This is the stochastic process described in Definition 2
    of the paper.
    """
    if rng is None:
        rng = np.random.default_rng()
    active = set(int(v) for v in seeds)
    frontier = list(active)
    while frontier:
        next_frontier: list[int] = []
        for u in frontier:
            for v in graph.successors(u):
                if v in active:
                    continue
                if rng.random() < _edge_probability(graph, u, v):
                    active.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
    return active


def mc_activation_probabilities(
    graph: nx.DiGraph,
    seeds: Iterable[int],
    mc_times: int = 100,
    random_seed: int | None = None,
) -> dict[int, float]:
    """Estimate p_v(S) by Monte-Carlo simulation under the IC model."""
    if mc_times <= 0:
        raise ValueError("mc_times must be positive")
    nodes = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    counts = np.zeros(len(nodes), dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    seeds = set(int(v) for v in seeds)
    for v in seeds:
        if v not in index:
            raise KeyError(f"seed {v} is not a graph node")
    for _ in range(mc_times):
        active = simulate_ic_once(graph, seeds, rng)
        for v in active:
            counts[index[v]] += 1
    probs = counts.astype(np.float64) / float(mc_times)
    return {v: float(probs[i]) for i, v in enumerate(nodes)}


def single_seed_probability_pairs(
    graph: nx.DiGraph,
    seed: int,
    mc_times: int = 100,
    random_seed: int | None = None,
    include_seed: bool = False,
) -> list[tuple[int, int, float]]:
    """Return (seed, node, estimated probability) pairs for embedding learning."""
    probs = mc_activation_probabilities(graph, [seed], mc_times, random_seed)
    out: list[tuple[int, int, float]] = []
    for node, prob in probs.items():
        if node == seed and not include_seed:
            continue
        out.append((seed, node, prob))
    return out


def layered_product_activation_probabilities(
    graph: nx.DiGraph,
    seeds: Iterable[int],
) -> dict[int, float]:
    """Compatibility evaluator close to the released legacy script.

    This is *not* a general exact IC probability computation. It processes nodes in
    first-discovery frontiers and combines parent activation probabilities using
    1-prod(1-p_u*w_uv). The public legacy code uses essentially this recurrence for
    its reward/evaluation path, so it is kept only for forensic comparison.
    """
    seeds = set(int(v) for v in seeds)
    prob = {v: (1.0 if v in seeds else 0.0) for v in graph.nodes()}
    active = set(seeds)
    frontier = set(seeds)
    while frontier:
        next_frontier: set[int] = set()
        for u in frontier:
            for v in graph.successors(u):
                if v not in active:
                    next_frontier.add(v)
        if not next_frontier:
            break
        for v in next_frontier:
            failure = 1.0
            for u in graph.predecessors(v):
                if u in frontier:
                    failure *= 1.0 - prob[u] * _edge_probability(graph, u, v)
            prob[v] = 1.0 - failure
        active.update(next_frontier)
        frontier = next_frontier
    return prob
