from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .assignment import greedy_capacity_assignment, AssignmentResult, unconstrained_parallel_satisfaction
from .journal import JournalInstance


@dataclass
class JournalOracleStats:
    worlds: int
    candidate_singleton_bfs: int


class JournalLiveEdgeOracle:
    """Fixed-world objective oracle for social recruitment + finite-capacity assignment."""

    def __init__(
        self,
        instance: JournalInstance,
        mc_times: int = 30,
        random_seed: int = 0,
        worker_capacity: int = 1,
        precompute_candidates: bool = True,
    ) -> None:
        if mc_times <= 0:
            raise ValueError("mc_times must be positive")
        self.instance = instance
        self.mc_times = int(mc_times)
        self.random_seed = int(random_seed)
        self.worker_capacity = int(worker_capacity)
        self.suitability = instance.suitability.astype(np.float64, copy=False)
        self._worlds = self._sample_worlds()
        self._reach = {}
        bfs = 0
        if precompute_candidates:
            for wi in sorted(instance.worker_pool):
                self._reach[int(wi)] = []
                for world in self._worlds:
                    self._reach[int(wi)].append(self._spread_from(int(wi), world))
                    bfs += 1
        self.stats = JournalOracleStats(self.mc_times, bfs)

    def _sample_worlds(self):
        rng = np.random.default_rng(self.random_seed)
        worlds = []
        graph = self.instance.graph
        for _ in range(self.mc_times):
            live = {u: [] for u in graph.nodes()}
            for u, v, data in graph.edges(data=True):
                if rng.random() < float(np.clip(data.get("weight", 0.0), 0.0, 1.0)):
                    live[int(u)].append(int(v))
            worlds.append(live)
        return worlds

    @staticmethod
    def _spread_from(seed: int, live):
        active = {int(seed)}
        stack = [int(seed)]
        while stack:
            u = stack.pop()
            for v in live.get(u, ()):
                if v not in active:
                    active.add(v); stack.append(v)
        return active

    def _active_for_world(self, seeds, world_idx: int):
        active: set[int] = set()
        for s in seeds:
            s = int(s)
            if s in self._reach:
                active.update(self._reach[s][world_idx])
            else:
                active.update(self._spread_from(s, self._worlds[world_idx]))
        return active

    def evaluate(self, seeds) -> dict:
        seeds = tuple(sorted({int(x) for x in seeds}))
        if not seeds:
            t = self.suitability.shape[1]
            return {
                "mean": 0.0,
                "p10": 0.0,
                "unsatisfied_ratio": 1.0 if t else 0.0,
                "saturated_ratio": 0.0,
                "mean_active": 0.0,
                "mean_assignments": 0.0,
                "parallel_relaxation": 0.0,
            }
        vals=[]; p10=[]; unsat=[]; sat=[]; nactive=[]; nass=[]; parallel=[]
        for w in range(self.mc_times):
            active = self._active_for_world(seeds, w)
            r: AssignmentResult = greedy_capacity_assignment(
                active,
                self.suitability,
                self.instance.demand,
                capacity=self.worker_capacity,
            )
            vals.append(r.mean_satisfaction); p10.append(r.p10_satisfaction)
            unsat.append(r.unsatisfied_ratio); sat.append(r.saturated_ratio)
            nactive.append(len(active)); nass.append(len(r.assignments))
            parallel.append(unconstrained_parallel_satisfaction(active, self.suitability, self.instance.demand))
        return {
            "mean": float(np.mean(vals)),
            "p10": float(np.mean(p10)),
            "unsatisfied_ratio": float(np.mean(unsat)),
            "saturated_ratio": float(np.mean(sat)),
            "mean_active": float(np.mean(nactive)),
            "mean_assignments": float(np.mean(nass)),
            "parallel_relaxation": float(np.mean(parallel)),
        }

    def score(self, seeds) -> float:
        return float(self.evaluate(seeds)["mean"])

    def marginal_gain(self, selected, candidate: int) -> float:
        base = tuple(selected)
        return self.score((*base, int(candidate))) - self.score(base)
