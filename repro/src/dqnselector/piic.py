from __future__ import annotations

import networkx as nx
import numpy as np


def weighted_adjacency(graph: nx.DiGraph, nodes: list[int]) -> np.ndarray:
    """A[u,v] = w(u,v) in the node order supplied."""
    idx = {v: i for i, v in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for u, v, data in graph.edges(data=True):
        if u not in idx or v not in idx:
            continue
        w = float(data.get("weight", 1.0))
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"edge probability outside [0,1]: ({u},{v})={w}")
        a[idx[u], idx[v]] = w
    return a


def piic(
    graph: nx.DiGraph,
    quality: np.ndarray,
    influence_range: int,
    nodes: list[int] | None = None,
) -> np.ndarray:
    """Path Increment Iterative Calculation from Eq. (11) / Algorithm 2.

    `quality[v]` is the h-dimensional q_v vector (Delta r_v^0). Each round
    performs Delta^L = A @ Delta^(L-1), then accumulates Delta^L into r.

    This follows the published recurrence literally. On cyclic graphs it counts
    weighted walks, which is why the paper itself describes PIIC as an estimate.
    The test suite validates exact path equivalence on DAGs, where no cycle-return
    ambiguity exists.
    """
    if influence_range < 0:
        raise ValueError("influence_range must be non-negative")
    if nodes is None:
        nodes = list(graph.nodes())
    quality = np.asarray(quality, dtype=np.float64)
    if quality.ndim != 2 or quality.shape[0] != len(nodes):
        raise ValueError("quality must have shape [n_nodes, n_subareas]")
    a = weighted_adjacency(graph, nodes)
    delta = quality.copy()
    result = np.zeros_like(quality, dtype=np.float64)
    for _ in range(influence_range):
        delta = a @ delta
        result += delta
    return result


def enumerate_path_embedding_dag(
    graph: nx.DiGraph,
    quality: np.ndarray,
    influence_range: int,
    nodes: list[int] | None = None,
) -> np.ndarray:
    """Slow reference implementation of Eq. (10) for DAG tests."""
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("reference path enumerator is intentionally DAG-only")
    if nodes is None:
        nodes = list(graph.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    quality = np.asarray(quality, dtype=np.float64)
    out = np.zeros_like(quality)

    def dfs(source: int, current: int, depth: int, path_prob: float) -> None:
        if depth >= influence_range:
            return
        for nxt in graph.successors(current):
            w = float(graph[current][nxt].get("weight", 1.0))
            next_prob = path_prob * w
            out[idx[source]] += next_prob * quality[idx[nxt]]
            dfs(source, nxt, depth + 1, next_prob)

    for source in nodes:
        dfs(source, source, 0, 1.0)
    return out
