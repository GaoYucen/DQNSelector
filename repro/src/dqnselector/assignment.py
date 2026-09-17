from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np


@dataclass
class AssignmentResult:
    mean_satisfaction: float
    p10_satisfaction: float
    unsatisfied_ratio: float
    saturated_ratio: float
    per_task_satisfaction: np.ndarray
    assignments: list[tuple[int, int, float]]


def greedy_capacity_assignment(
    active_workers,
    suitability: np.ndarray,
    demand: np.ndarray,
    capacity: int = 1,
) -> AssignmentResult:
    """Allocate activated workers to tasks under finite per-worker capacity.

    Each worker-task pair contributes `suitability[u,t]` units of effective
    sensing capacity. The objective is mean capped task satisfaction. The
    dispatcher greedily applies the currently largest exact marginal gain and
    lazily refreshes heap entries when task saturation changes.
    """
    workers = sorted({int(u) for u in active_workers})
    suitability = np.asarray(suitability, dtype=float)
    demand = np.asarray(demand, dtype=float)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if suitability.ndim != 2 or suitability.shape[1] != len(demand):
        raise ValueError("suitability/demand shape mismatch")
    if any(u < 0 or u >= suitability.shape[0] for u in workers):
        raise ValueError("active worker out of range")
    if np.any(demand <= 0):
        raise ValueError("demand must be positive")

    t = len(demand)
    achieved = np.zeros(t, dtype=float)
    used = {u: 0 for u in workers}
    heap: list[tuple[float, int, int, float]] = []

    for u in workers:
        row = suitability[u]
        for j in np.flatnonzero(row > 0):
            gain = min(float(row[j]) / float(demand[j]), 1.0)
            if gain > 0:
                heapq.heappush(heap, (-gain, u, int(j), gain))

    assignments: list[tuple[int, int, float]] = []
    while heap:
        neg_gain, u, j, cached_gain = heapq.heappop(heap)
        if used[u] >= capacity or achieved[j] >= demand[j] - 1e-12:
            continue
        contribution = float(suitability[u, j])
        before = min(achieved[j] / demand[j], 1.0)
        after = min((achieved[j] + contribution) / demand[j], 1.0)
        gain = after - before
        if gain <= 1e-15:
            continue
        # Lazy refresh: if the task filled since this entry was inserted, put
        # the exact smaller marginal gain back into the heap.
        if gain + 1e-12 < cached_gain:
            heapq.heappush(heap, (-gain, u, j, gain))
            continue
        used[u] += 1
        achieved[j] += contribution
        assignments.append((u, j, contribution))

    sat = np.minimum(achieved / demand, 1.0)
    return AssignmentResult(
        mean_satisfaction=float(np.mean(sat)) if len(sat) else 0.0,
        p10_satisfaction=float(np.quantile(sat, 0.10)) if len(sat) else 0.0,
        unsatisfied_ratio=float(np.mean(sat < 1.0 - 1e-9)) if len(sat) else 0.0,
        saturated_ratio=float(np.mean(sat >= 1.0 - 1e-9)) if len(sat) else 0.0,
        per_task_satisfaction=sat,
        assignments=assignments,
    )


def lp_relaxation_upper_bound(
    active_workers,
    suitability: np.ndarray,
    demand: np.ndarray,
    capacity: int = 1,
) -> float:
    """LP relaxation upper bound for the finite-capacity assignment objective.

    Binary assignment variables x[u,t] are relaxed to [0,1]. Task variables y[t]
    represent capped satisfaction and obey y[t] <= sum_u s[u,t]/d[t] * x[u,t]
    and y[t] <= 1. Because the integer feasible region is contained in this LP,
    the returned mean satisfaction is a rigorous upper bound on the optimal
    integral assignment. SciPy is imported lazily so the core package does not
    require it for ordinary experiments.
    """
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception as exc:  # pragma: no cover - optional validation path
        raise RuntimeError("SciPy is required for LP assignment validation") from exc

    workers = sorted({int(u) for u in active_workers})
    s = np.asarray(suitability, dtype=float)
    d = np.asarray(demand, dtype=float)
    if not workers:
        return 0.0
    if capacity <= 0 or np.any(d <= 0):
        raise ValueError("positive capacity and demand are required")
    if s.ndim != 2 or s.shape[1] != len(d):
        raise ValueError("suitability/demand shape mismatch")

    m, t = len(workers), len(d)
    n_x = m * t
    n_var = n_x + t
    # Minimize -mean(y).
    c = np.zeros(n_var, dtype=float)
    c[n_x:] = -1.0 / max(t, 1)

    rows = []
    cols = []
    vals = []
    b = []
    r = 0
    # Worker capacity constraints: sum_t x[u,t] <= capacity.
    for i in range(m):
        for j in range(t):
            rows.append(r); cols.append(i * t + j); vals.append(1.0)
        b.append(float(capacity)); r += 1
    # Task satisfaction constraints: y_t - sum_u s[u,t]/d_t x[u,t] <= 0.
    for j in range(t):
        for i, u in enumerate(workers):
            coeff = float(s[u, j] / d[j])
            if coeff != 0.0:
                rows.append(r); cols.append(i * t + j); vals.append(-coeff)
        rows.append(r); cols.append(n_x + j); vals.append(1.0)
        b.append(0.0); r += 1

    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    bounds = [(0.0, 1.0)] * n_var
    res = linprog(c, A_ub=A_ub, b_ub=np.asarray(b), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP assignment validation failed: {res.message}")
    return float(-res.fun)


def unconstrained_parallel_satisfaction(active_workers, suitability: np.ndarray, demand: np.ndarray) -> float:
    """Diagnostic corresponding to the unrealistic 'every active worker serves every task' relaxation."""
    workers = sorted({int(u) for u in active_workers})
    if not workers:
        return 0.0
    total = np.asarray(suitability, dtype=float)[workers].sum(axis=0)
    return float(np.mean(np.minimum(total / np.asarray(demand, dtype=float), 1.0)))
