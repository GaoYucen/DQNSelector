"""SIGIR baseline adapters with explicit pool and cardinality constraints.

FastSelector follows SIGIR Sec. 5.1 and Wang et al. Eq. (11), updating
average cosine similarity to all selected workers. KTVoting follows the released
SIGIR voting recurrence (beta=.9, T=4), with k groups, complete area assignment,
and no synthetic node-zero insertion. See docs/SIGIR_BASELINE_PROTOCOL.md.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def mobility_profiles(checkins: str | Path, manifest: dict, pool: list[int]) -> np.ndarray:
    cfg = manifest['config']
    nx, ny = int(cfg['lon_cells']), int(cfg['lat_cells'])
    original = manifest['sampled_original_user_ids']
    positions = {int(original[v]): i for i, v in enumerate(pool)}
    counts = np.zeros((len(pool), nx * ny), dtype=np.float32)
    opener = gzip.open if str(checkins).endswith('.gz') else open
    with opener(checkins, 'rt') as f:
        for line in f:
            fields = line.split()
            if len(fields) < 4 or fields[0].startswith('#'):
                continue
            row = positions.get(int(fields[0]))
            if row is None:
                continue
            lat, lon = float(fields[2]), float(fields[3])
            if not (cfg['lon_min'] <= lon <= cfg['lon_max'] and cfg['lat_min'] <= lat <= cfg['lat_max']):
                continue
            x = min(nx - 1, int((lon - cfg['lon_min']) / (cfg['lon_max'] - cfg['lon_min']) * nx))
            y = min(ny - 1, int((lat - cfg['lat_min']) / (cfg['lat_max'] - cfg['lat_min']) * ny))
            counts[row, x * ny + y] += 1
    return counts


def _rank(values: np.ndarray, ids: np.ndarray, descending: bool) -> np.ndarray:
    order = np.lexsort((ids, -values if descending else values))
    rank = np.empty(len(ids), dtype=float)
    rank[order] = np.arange(len(ids))
    return rank


def fast_selector(graph, profiles: np.ndarray, pool: list[int], k: int, alpha: float) -> list[int]:
    pool = np.asarray(pool, dtype=int)
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.shape[0] != len(pool) or not 0 <= alpha <= 1:
        raise ValueError('invalid profiles or alpha')
    normalized = profiles / np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    degrees = np.asarray([graph.out_degree(int(v)) for v in pool], dtype=float)
    available = np.ones(len(pool), dtype=bool)
    similarity_sum = np.zeros(len(pool))
    order = []
    for step in range(min(k, len(pool))):
        pos = np.flatnonzero(available)
        degree_rank = _rank(degrees[pos], pool[pos], True)
        # Empty-set trajectory ranks tie: the first action is maximum degree.
        diversity_rank = _rank(similarity_sum[pos] / step, pool[pos], False) if step else np.zeros(len(pos))
        score = alpha * degree_rank + (1 - alpha) * diversity_rank
        chosen = pos[np.lexsort((pool[pos], score))[0]]
        order.append(int(pool[chosen]))
        available[chosen] = False
        similarity_sum += similarity[:, chosen]
    return order


def kt_voting(graph, node_values: np.ndarray, pool: list[int], k: int, beta: float = .9, steps: int = 4) -> list[int]:
    values = np.asarray(node_values, dtype=np.float64)
    k = min(k, len(pool))
    if k <= 0:
        return []
    groups = np.array_split(np.arange(values.shape[1]), min(k, values.shape[1]))
    current = np.stack([values[:, g].sum(axis=1) for g in groups], axis=1)
    votes = np.zeros_like(current)
    edges = list(graph.edges(data=True))
    src = np.asarray([u for u, _, _ in edges], dtype=int)
    dst = np.asarray([v for _, v, _ in edges], dtype=int)
    weights = np.asarray([d.get('weight', 1.) for _, _, d in edges])
    for _ in range(steps):
        nxt = np.zeros_like(current)
        np.add.at(nxt, src, weights[:, None] * current[dst])
        votes += nxt
        current = nxt
    pool_arr = np.asarray(sorted(pool), dtype=int)
    candidate_votes = votes[pool_arr]
    preferred = candidate_votes == candidate_votes.max(axis=1, keepdims=True)
    scores = np.where(preferred, beta * candidate_votes, (1 - beta) * candidate_votes).sum(axis=1)
    selected = []
    remaining = np.ones(len(pool_arr), dtype=bool)
    for group in range(len(groups)):
        candidates = np.flatnonzero(remaining & preferred[:, group] & (scores > 0))
        if not candidates.size:
            continue
        best = candidates[np.lexsort((pool_arr[candidates], -scores[candidates]))[0]]
        selected.append(int(pool_arr[best]))
        remaining[best] = False
        if len(selected) == k:
            return selected
    candidates = np.flatnonzero(remaining)
    candidates = candidates[np.lexsort((pool_arr[candidates], -scores[candidates]))]
    return selected + [int(pool_arr[p]) for p in candidates[:k-len(selected)]]
