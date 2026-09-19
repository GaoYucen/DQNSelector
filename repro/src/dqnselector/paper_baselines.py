"""SIGIR baseline adapters with explicit pool and cardinality constraints.

FastSelector follows SIGIR Sec. 5.1 and Wang et al. Eq. (11), updating
average cosine similarity to all selected workers. KTVoting follows the released
SIGIR voting recurrence (beta=.9, T=4), with k groups, complete area assignment,
and no synthetic node-zero insertion. See docs/SIGIR_BASELINE_PROTOCOL.md.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SparseMobilityProfiles:
    """CSR-like mobility profiles without materialising 7200 x 168 columns."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    norms: np.ndarray

    def cosine_to(self, row: int) -> np.ndarray:
        """Cosine similarity of every row to ``row``; zero for an empty profile."""
        start, end = int(self.indptr[row]), int(self.indptr[row + 1])
        keys = self.indices[start:end]
        values = self.data[start:end]
        out = np.zeros(len(self.norms), dtype=np.float64)
        if not len(keys) or self.norms[row] <= 0:
            return out
        for other in range(len(self.norms)):
            if self.norms[other] <= 0:
                continue
            a, b = int(self.indptr[other]), int(self.indptr[other + 1])
            # Both index sequences are sorted, so intersect without allocating a dense vector.
            left = right = 0
            dot = 0.0
            other_keys, other_values = self.indices[a:b], self.data[a:b]
            while left < len(keys) and right < len(other_keys):
                if keys[left] == other_keys[right]:
                    dot += float(values[left]) * float(other_values[right])
                    left += 1; right += 1
                elif keys[left] < other_keys[right]:
                    left += 1
                else:
                    right += 1
            out[other] = dot / (self.norms[row] * self.norms[other])
        return out


def save_mobility_profiles(path: str | Path, profiles: SparseMobilityProfiles, pool: list[int]) -> None:
    np.savez_compressed(path, indptr=profiles.indptr, indices=profiles.indices, data=profiles.data,
                        norms=profiles.norms, pool=np.asarray(pool, dtype=np.int64), format_version=2)


def load_mobility_profiles(path: str | Path, pool: list[int]) -> SparseMobilityProfiles:
    data = np.load(path)
    if int(data.get("format_version", 0)) != 2 or data["pool"].tolist() != list(pool):
        raise ValueError("incompatible mobility profile cache")
    return SparseMobilityProfiles(data["indptr"], data["indices"], data["data"], data["norms"])


def _hour_of_week(raw_timestamp: str) -> int:
    """Return a stable 0..167 local-free temporal bin for SNAP ISO timestamps."""
    value = raw_timestamp.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return 0
    return int(dt.weekday() * 24 + dt.hour)


def mobility_profiles(checkins: str | Path, manifest: dict, pool: list[int]) -> SparseMobilityProfiles:
    """Build the FastSelector spatial-temporal 7200-cell x 168-cycle profiles."""
    cfg = manifest['config']
    nx, ny = int(cfg['lon_cells']), int(cfg['lat_cells'])
    original = manifest['sampled_original_user_ids']
    positions = {int(original[v]): i for i, v in enumerate(pool)}
    rows: list[dict[int, float]] = [dict() for _ in pool]
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
            key = (x * ny + y) * 168 + _hour_of_week(fields[1])
            rows[row][key] = rows[row].get(key, 0.0) + 1.0
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    norms = []
    for values in rows:
        ordered = sorted(values.items())
        indices.extend(key for key, _ in ordered)
        row_data = [value for _, value in ordered]
        data.extend(row_data)
        norms.append(float(np.linalg.norm(row_data)))
        indptr.append(len(indices))
    return SparseMobilityProfiles(
        np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64),
        np.asarray(data, dtype=np.float32), np.asarray(norms, dtype=np.float64),
    )


def _average_rank(values: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """One-based average ranks; caller supplies deterministic ID tie-breaking later."""
    ordered = np.argsort(values if lower_is_better else -values, kind="mergesort")
    rank = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(ordered):
        j = i + 1
        value = values[ordered[i]]
        while j < len(ordered) and np.isclose(values[ordered[j]], value, rtol=0.0, atol=1e-12):
            j += 1
        rank[ordered[i:j]] = (i + 1 + j) / 2.0
        i = j
    return rank


def fast_selector(graph, profiles: np.ndarray | SparseMobilityProfiles, pool: list[int], k: int, alpha: float) -> list[int]:
    pool = np.asarray(pool, dtype=int)
    if not 0 <= alpha <= 1:
        raise ValueError('invalid profiles or alpha')
    sparse = isinstance(profiles, SparseMobilityProfiles)
    if sparse:
        if len(profiles.norms) != len(pool):
            raise ValueError('profile rows do not match pool')
    else:
        profiles = np.asarray(profiles, dtype=np.float64)
        if profiles.shape[0] != len(pool):
            raise ValueError('profile rows do not match pool')
        normalized = profiles / np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
        similarity = normalized @ normalized.T
    degrees = np.asarray([graph.out_degree(int(v)) for v in pool], dtype=float)
    available = np.ones(len(pool), dtype=bool)
    similarity_sum = np.zeros(len(pool))
    order = []
    for step in range(min(k, len(pool))):
        pos = np.flatnonzero(available)
        degree_rank = _average_rank(degrees[pos], lower_is_better=False)
        # Empty-set trajectory ranks tie: the first action is maximum degree.
        diversity_rank = _average_rank(similarity_sum[pos] / step, lower_is_better=True) if step else np.zeros(len(pos))
        score = alpha * degree_rank + (1 - alpha) * diversity_rank
        chosen = pos[np.lexsort((pool[pos], score))[0]]
        order.append(int(pool[chosen]))
        available[chosen] = False
        similarity_sum += profiles.cosine_to(chosen) if sparse else similarity[:, chosen]
    return order


def _voting_scores(graph, values: np.ndarray, group_indices: list[np.ndarray], steps: int) -> np.ndarray:
    current = np.stack([values[:, group].sum(axis=1) for group in group_indices], axis=1)
    votes = np.zeros_like(current)
    edges = list(graph.edges(data=True))
    if not edges:
        return votes
    src = np.asarray([u for u, _, _ in edges], dtype=int)
    dst = np.asarray([v for _, v, _ in edges], dtype=int)
    weights = np.asarray([d.get('weight', 1.) for _, _, d in edges])
    for _ in range(steps):
        nxt = np.zeros_like(current)
        np.add.at(nxt, src, weights[:, None] * current[dst])
        votes += nxt
        current = nxt
    return votes


def kt_voting_grouped(graph, node_values: np.ndarray, pool: list[int], k: int, beta: float = .9, steps: int = 4) -> list[int]:
    """Former k-group repair, retained only as an appendix sensitivity method."""
    values = np.asarray(node_values, dtype=np.float64)
    k = min(k, len(pool))
    if k <= 0:
        return []
    groups = np.array_split(np.arange(values.shape[1]), min(k, values.shape[1]))
    votes = _voting_scores(graph, values, list(groups), steps)
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


def kt_voting(graph, node_values: np.ndarray, pool: list[int], k: int, beta: float = .9, steps: int = 4) -> list[int]:
    """Feasible adapter of the released driver's per-subarea ``KTVoting2`` path."""
    values = np.asarray(node_values, dtype=np.float64)
    pool_arr = np.asarray(sorted(set(pool)), dtype=int)
    k = min(int(k), len(pool_arr))
    if k <= 0:
        return []
    groups = [np.asarray([area], dtype=int) for area in range(values.shape[1])]
    votes = _voting_scores(graph, values, groups, steps)
    candidate_votes = votes[pool_arr]
    preferred = candidate_votes == candidate_votes.max(axis=1, keepdims=True)
    scores = np.where(preferred, beta * candidate_votes, (1.0 - beta) * candidate_votes).sum(axis=1)
    remaining = np.ones(len(pool_arr), dtype=bool)
    selected: list[int] = []
    # Preserve KTVoting2's per-area election order, while skipping infeasible elections.
    for area in range(values.shape[1]):
        choices = np.flatnonzero(remaining & preferred[:, area] & (scores > 0))
        if not choices.size:
            continue
        best = choices[np.lexsort((pool_arr[choices], -scores[choices]))[0]]
        selected.append(int(pool_arr[best]))
        remaining[best] = False
        if len(selected) == k:
            return selected
    fill = np.flatnonzero(remaining)
    fill = fill[np.lexsort((pool_arr[fill], -scores[fill]))]
    return selected + [int(pool_arr[p]) for p in fill[: k - len(selected)]]
