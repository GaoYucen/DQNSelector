from __future__ import annotations

import networkx as nx
import numpy as np

from .journal import JournalInstance


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return ((x - mu) / np.where(sd > 1e-12, sd, 1.0)).astype(np.float32)


def structural_social_embedding(graph: nx.DiGraph) -> np.ndarray:
    """Fast transparent social embedding for journal-v1 validation.

    This is intentionally not claimed as the final representation learner. It
    supplies topology/influence features while the new problem formulation and
    benchmark are validated first.
    """
    nodes = list(range(graph.number_of_nodes()))
    indeg = np.asarray([graph.in_degree(u) for u in nodes], dtype=float)
    outdeg = np.asarray([graph.out_degree(u) for u in nodes], dtype=float)
    win = np.asarray([sum(float(d.get('weight', 0.0)) for _, _, d in graph.in_edges(u, data=True)) for u in nodes])
    wout = np.asarray([sum(float(d.get('weight', 0.0)) for _, _, d in graph.out_edges(u, data=True)) for u in nodes])
    pr_d = nx.pagerank(graph, weight='weight', max_iter=200)
    pr = np.asarray([pr_d.get(u, 0.0) for u in nodes], dtype=float)
    ug = graph.to_undirected()
    cl_d = nx.clustering(ug)
    cl = np.asarray([cl_d.get(u, 0.0) for u in nodes], dtype=float)
    feat = np.stack([
        np.log1p(indeg), np.log1p(outdeg), np.log1p(win), np.log1p(wout), pr, cl,
    ], axis=1)
    return _zscore(feat)


def task_need_embedding(instance: JournalInstance) -> np.ndarray:
    """Worker-task service potential normalized by task demand."""
    demand = np.maximum(instance.demand.astype(np.float64), 1e-12)
    x = instance.suitability.astype(np.float64) / demand[None, :]
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def build_journal_embeddings(instance: JournalInstance) -> tuple[np.ndarray, np.ndarray]:
    return structural_social_embedding(instance.graph), task_need_embedding(instance)
