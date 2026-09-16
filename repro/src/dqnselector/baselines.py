from __future__ import annotations

from collections.abc import Callable, Iterable
import heapq

import networkx as nx


def degree_greedy(graph: nx.DiGraph, k: int, worker_pool: Iterable[int] | None = None) -> list[int]:
    pool = list(graph.nodes() if worker_pool is None else worker_pool)
    pool.sort(key=lambda v: (graph.out_degree(v), -int(v)), reverse=True)
    return pool[:k]


def one_step_coverage_greedy(
    graph: nx.DiGraph,
    node_values,
    k: int,
    nodes: list[int],
    worker_pool: Iterable[int] | None = None,
) -> list[int]:
    """Paper CovGreedy idea: rank nodes by directly influenced coverage.

    `node_values[u]` should contain the per-subarea p_u^i*q_u^i term. Edge
    probability w(v,u) is applied here, matching the paper's CovGreedy formula.
    """
    index = {v: i for i, v in enumerate(nodes)}
    pool = list(nodes if worker_pool is None else worker_pool)
    scores: list[tuple[float, int]] = []
    for v in pool:
        total = 0.0
        for u in graph.successors(v):
            if u in index:
                total += float(graph[v][u].get("weight", 1.0)) * float(node_values[index[u]].sum())
        scores.append((total, int(v)))
    scores.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [v for _, v in scores[:k]]


def greedy_by_marginal_gain(
    candidates: Iterable[int],
    k: int,
    marginal_gain: Callable[[set[int], int], float],
) -> list[int]:
    selected_set: set[int] = set()
    selected_order: list[int] = []
    candidates = sorted(int(v) for v in candidates)
    for _ in range(min(k, len(candidates))):
        remaining = [v for v in candidates if v not in selected_set]
        best = max(remaining, key=lambda v: (float(marginal_gain(selected_set, v)), -v))
        selected_set.add(best)
        selected_order.append(best)
    return selected_order


def celf(
    candidates: Iterable[int],
    k: int,
    marginal_gain: Callable[[set[int], int], float],
) -> list[int]:
    """Cost-Effective Lazy Forward for a submodular marginal-gain oracle.

    This implementation keeps the actual selection order so one CELF run up to the
    maximum budget can be evaluated at all smaller seed budgets. If the supplied
    objective is not submodular, CELF's lazy guarantee does not apply.
    """
    candidates = sorted(int(v) for v in candidates)
    selected_set: set[int] = set()
    selected_order: list[int] = []
    heap: list[tuple[float, int, int]] = []
    for v in candidates:
        heapq.heappush(heap, (-float(marginal_gain(set(), v)), v, 0))
    limit = min(k, len(candidates))
    while heap and len(selected_order) < limit:
        _neg_gain, v, last_updated = heapq.heappop(heap)
        if v in selected_set:
            continue
        if last_updated == len(selected_order):
            selected_set.add(v)
            selected_order.append(v)
            continue
        gain = float(marginal_gain(selected_set, v))
        heapq.heappush(heap, (-gain, v, len(selected_order)))
    return selected_order
