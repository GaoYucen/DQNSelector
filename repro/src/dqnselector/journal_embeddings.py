from __future__ import annotations

import networkx as nx
import numpy as np

from .assignment import greedy_capacity_assignment
from .journal import JournalInstance
from .journal_oracle import JournalLiveEdgeOracle


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
    """Direct worker-task service potential normalized by task demand."""
    demand = np.maximum(instance.demand.astype(np.float64), 1e-12)
    x = instance.suitability.astype(np.float64) / demand[None, :]
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def singleton_service_embedding(
    instance: JournalInstance,
    mc_times: int = 20,
    random_seed: int = 0,
    worker_capacity: int = 1,
) -> np.ndarray:
    """Expected per-task satisfaction profile caused by recruiting one seed.

    For each candidate worker, fixed live-edge worlds simulate the global social
    recruitment cascade. Activated workers are then allocated under the same
    finite-capacity dispatcher used by the journal objective. The resulting
    T-dimensional vector therefore summarizes *downstream usable task service*,
    not merely direct geographic suitability of the seed itself.

    Non-candidate rows are kept at zero because the selector only chooses from
    `worker_pool`; this makes the precomputation proportional to the candidate
    pool rather than all graph nodes.
    """
    oracle = JournalLiveEdgeOracle(
        instance,
        mc_times=mc_times,
        random_seed=random_seed,
        worker_capacity=worker_capacity,
        precompute_candidates=True,
    )
    n = instance.graph.number_of_nodes()
    t = instance.suitability.shape[1]
    out = np.zeros((n, t), dtype=np.float32)
    for u in sorted(instance.worker_pool):
        profiles = []
        for w in range(oracle.mc_times):
            active = oracle._active_for_world((int(u),), w)
            r = greedy_capacity_assignment(
                active,
                oracle.suitability,
                instance.demand,
                capacity=worker_capacity,
            )
            profiles.append(r.per_task_satisfaction)
        out[int(u)] = np.mean(np.asarray(profiles, dtype=np.float64), axis=0).astype(np.float32)
    return out


def build_journal_embeddings(
    instance: JournalInstance,
    task_mode: str = 'direct',
    mc_times: int = 20,
    random_seed: int = 0,
    worker_capacity: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    social = structural_social_embedding(instance.graph)
    if task_mode == 'direct':
        task = task_need_embedding(instance)
    elif task_mode == 'singleton_mc':
        task = singleton_service_embedding(
            instance,
            mc_times=mc_times,
            random_seed=random_seed,
            worker_capacity=worker_capacity,
        )
    else:
        raise ValueError(f'unknown task_mode={task_mode!r}')
    return social, task
