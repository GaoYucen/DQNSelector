from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .ecm import ECMInstance


@dataclass
class OracleStats:
    mc_worlds: int
    precomputed_seeds: int
    mean_reachability: float
    max_reachability: int


class LiveEdgeECOracle:
    """Reusable fixed-world Monte-Carlo oracle for canonical and realized EC.

    ``score`` implements the paper-defined objective: average activation first,
    then form expected coverage and apply the demand cap once. ``realized_score``
    is a robustness metric that applies the cap inside each live-edge world and
    averages the realized task fulfillment afterwards. The two are intentionally
    kept separate because clipping is nonlinear.
    """

    def __init__(
        self,
        instance: ECMInstance,
        mc_times: int = 100,
        random_seed: int = 0,
        seed_candidates: Iterable[int] | None = None,
    ) -> None:
        if mc_times <= 0:
            raise ValueError("mc_times must be positive")
        self.instance = instance
        self.mc_times = int(mc_times)
        self.random_seed = int(random_seed)
        self.nodes = list(instance.nodes)
        self.node_index = {v: i for i, v in enumerate(self.nodes)}
        if self.nodes != list(range(len(self.nodes))):
            raise ValueError("LiveEdgeECOracle currently requires dense node ids 0..n-1")
        if seed_candidates is None:
            seed_candidates = sorted(instance.worker_pool or set(instance.nodes))
        self.seed_candidates = [int(v) for v in seed_candidates]
        invalid = set(self.seed_candidates) - set(instance.worker_pool or set())
        if invalid:
            raise ValueError(f"seed candidates outside worker pool: {sorted(invalid)[:5]}")
        self.candidate_position = {v: i for i, v in enumerate(self.seed_candidates)}
        self.contribution = np.asarray(
            instance.participation * instance.quality, dtype=np.float64
        )
        self._reachability: list[list[np.ndarray]] = []
        self._score_cache: dict[frozenset[int], float] = {}
        self._realized_score_cache: dict[frozenset[int], float] = {}
        self._activation_cache: dict[frozenset[int], np.ndarray] = {}
        self._build_worlds()

    def _build_worlds(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        n = len(self.nodes)
        edges = [
            (int(u), int(v), float(data.get("weight", 1.0)))
            for u, v, data in self.instance.graph.edges(data=True)
        ]
        reach_sizes: list[int] = []
        for _ in range(self.mc_times):
            adjacency: list[list[int]] = [[] for _ in range(n)]
            for u, v, p in edges:
                if rng.random() < p:
                    adjacency[u].append(v)
            world_reach: list[np.ndarray] = []
            for seed in self.seed_candidates:
                seen = np.zeros(n, dtype=np.bool_)
                seen[seed] = True
                stack = [seed]
                while stack:
                    u = stack.pop()
                    for v in adjacency[u]:
                        if not seen[v]:
                            seen[v] = True
                            stack.append(v)
                idx = np.flatnonzero(seen).astype(np.int32, copy=False)
                world_reach.append(idx)
                reach_sizes.append(int(idx.size))
            self._reachability.append(world_reach)
        self.stats = OracleStats(
            mc_worlds=self.mc_times,
            precomputed_seeds=len(self.seed_candidates),
            mean_reachability=float(np.mean(reach_sizes)) if reach_sizes else 0.0,
            max_reachability=max(reach_sizes) if reach_sizes else 0,
        )

    def clear_score_cache(self) -> None:
        self._score_cache.clear()
        self._realized_score_cache.clear()
        self._activation_cache.clear()

    def _validate_seed_set(self, seeds: Iterable[int]) -> frozenset[int]:
        key = frozenset(int(v) for v in seeds)
        unknown = key - set(self.seed_candidates)
        if unknown:
            raise ValueError(f"oracle has no precomputed reachability for seeds {sorted(unknown)[:5]}")
        return key

    def activation_probability(self, seeds: Iterable[int]) -> np.ndarray:
        """Return fixed-world Monte-Carlo activation probabilities p_v(S)."""
        key = self._validate_seed_set(seeds)
        cached = self._activation_cache.get(key)
        if cached is not None:
            return cached.copy()
        n = len(self.nodes)
        if not key:
            probs = np.zeros(n, dtype=np.float64)
            self._activation_cache[key] = probs
            return probs.copy()
        counts = np.zeros(n, dtype=np.int64)
        positions = [self.candidate_position[v] for v in key]
        for world in self._reachability:
            active = np.zeros(n, dtype=np.bool_)
            for pos in positions:
                active[world[pos]] = True
            counts += active
        probs = counts.astype(np.float64) / self.mc_times
        self._activation_cache[key] = probs
        return probs.copy()

    def score(self, seeds: Iterable[int]) -> float:
        """Paper-defined EC: clip expected coverage after MC averaging."""
        key = self._validate_seed_set(seeds)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        if not key:
            self._score_cache[key] = 0.0
            return 0.0
        activation = self.activation_probability(key)
        coverage = (activation[:, None] * self.contribution).sum(axis=0)
        score = float(np.minimum(coverage / self.instance.demand, 1.0).mean())
        self._score_cache[key] = score
        return score

    def realized_score(self, seeds: Iterable[int]) -> float:
        """Expected realized EC: clip per live-edge world, then average.

        This is not substituted for the conference objective. It is reported as a
        construct-validity/robustness metric for stochastic task fulfillment.
        """
        key = self._validate_seed_set(seeds)
        cached = self._realized_score_cache.get(key)
        if cached is not None:
            return cached
        if not key:
            self._realized_score_cache[key] = 0.0
            return 0.0
        n = len(self.nodes)
        positions = [self.candidate_position[v] for v in key]
        total = 0.0
        for world in self._reachability:
            active = np.zeros(n, dtype=np.bool_)
            for pos in positions:
                active[world[pos]] = True
            coverage = self.contribution[active].sum(axis=0)
            total += float(np.minimum(coverage / self.instance.demand, 1.0).mean())
        score = total / self.mc_times
        self._realized_score_cache[key] = score
        return score

    def marginal_gain(self, selected: Iterable[int], candidate: int) -> float:
        selected_key = self._validate_seed_set(selected)
        candidate = int(candidate)
        if candidate in selected_key:
            return 0.0
        if candidate not in self.candidate_position:
            raise ValueError(f"candidate {candidate} is not precomputed")
        return self.score(selected_key | {candidate}) - self.score(selected_key)
