from __future__ import annotations

from collections.abc import Callable, Iterable
import heapq

import networkx as nx


def degree_greedy(graph: nx.DiGraph, k: int, worker_pool: Iterable[int] | None = None) -> list[int]:
    pool = list(graph.nodes() if worker_pool is None else worker_pool)
    pool.sort(key=lambda v: graph.out_degree(v), reverse=True)
    return pool[:k]


def one_step_coverage_greedy(
    graph: nx.DiGraph,
    node_values,
    k: int,
    nodes: list[int],
    worker_pool: Iterable[int] | None = None,
) -> list[int]:
    """Released-paper CovGreedy idea: rank nodes by directly influenced coverage."""
    index = {v: i for i, v in enumerate(nodes)}
    pool = list(nodes if worker_pool is None else worker_pool)
    scores: list[tuple[float, int]] = []
    for v in pool:
        total = 0.0
        for u in graph.successors(v):
            if u in index:
                total += float(graph[v][u].get("weight", 1.0)) * float(node_values[index[u]].sum())
        scores.append((total, v))
    scores.sort(reverse=True)
    return [v for _, v in scores[:k]]


def greedy_by_marginal_gain(
    candidates: Iterable[int],
    k: int,
    marginal_gain: Callable[[set[int], int], float],
) -> list[int]:
    selected: set[int] = set()
    candidates = list(candidates)
    for _ in range(min(k, len(candidates))):
        best = max((v for v in candidates if v not in selected), key=lambda v: marginal_gain(selected, v))
        selected.add(best)
    return list(selected)


def celf(
    candidates: Iterable[int],
    k: int,
    marginal_gain: Callable[[set[int], int], float],
) -> list[int]:
    """Cost-Effective Lazy Forward for a submodular marginal-gain oracle.

    This implementation is intentionally explicit and is used as a correctness
    reference. If the supplied objective is not submodular, CELF's lazy guarantee
    does not apply.
    """
    candidates = list(candidates)
    selected: set[int] = set()
    heap: list[tuple[float, int, int]] = []
    for v in candidates:
        heapq.heappush(heap, (-float(marginal_gain(set(), v)), v, 0))
    while heap and len(selected) < min(k, len(candidates)):
        neg_gain, v, last_updated = heapq.heappop(heap)
        if v in selected:
            continue
        if last_updated == len(selected):
            selected.add(v)
            continue
        gain = float(marginal_gain(selected, v))
        heapq.heappush(heap, (-gain, v, len(selected)))
    return list(selected)
