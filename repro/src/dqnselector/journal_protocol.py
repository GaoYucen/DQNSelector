"""Frozen scenario family and aggregation rules for the journal study."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np


BUDGETS = (50, 60, 70, 80, 90, 100)
NON_CELF_BASELINES = (
    "DegGreedy", "CovGreedy", "FastSelector-SIGIR-adapted", "KTVoting2-feasible", "PIANO",
)


@dataclass(frozen=True)
class Scenario:
    graph_sampling: str
    trivalency_values: tuple[float, float, float]
    quality_mode: str
    load_factor: float

    @property
    def scenario_id(self) -> str:
        tri = "low" if self.trivalency_values == (0.001, 0.01, 0.1) else "medium"
        return f"graph-{self.graph_sampling}__tri-{tri}__quality-{self.quality_mode}__load-{self.load_factor:.1f}"

    def as_manifest(self) -> dict:
        result = asdict(self)
        result["scenario_id"] = self.scenario_id
        return result


def scenario_grid() -> list[Scenario]:
    return [
        Scenario(graph, tri, quality, load)
        for graph, tri, quality, load in product(
            ("uniform_induced", "community_bfs"),
            ((0.001, 0.01, 0.1), (0.01, 0.05, 0.1)),
            ("uniform", "activity_reliability"),
            (1.5, 2.0),
        )
    ]


def nondegenerate(rows: list[dict]) -> bool:
    """Reject only floor/ceiling experiments, never a configuration for its ranking."""
    degenerate = 0
    for budget in BUDGETS:
        values = [float(row["ec"]) for row in rows if int(row["k"]) == budget]
        if values and (min(values) > .95 or max(values) < .05):
            degenerate += 1
    return degenerate < 4


def instance_relative_gaps(rows: list[dict], learned_method: str = "DQNSelector-J") -> dict[int, float]:
    """Compare to the strongest non-CELF method on the same instance and budget."""
    result: dict[int, float] = {}
    for budget in BUDGETS:
        dqn = [float(row["ec"]) for row in rows if row["method"] == learned_method and int(row["k"]) == budget]
        baselines = [float(row["ec"]) for row in rows if row["method"] in NON_CELF_BASELINES and int(row["k"]) == budget]
        if dqn and baselines and max(baselines) > 0:
            result[budget] = (float(np.mean(dqn)) - max(baselines)) / max(baselines)
    return result


def scenario_rank_key(dataset_rows: dict[str, list[list[dict]]]) -> tuple:
    """Higher is better; use the weaker dataset first to avoid one-dataset wins."""
    per_dataset = []
    for instances in dataset_rows.values():
        if not instances or any(not nondegenerate(rows) for rows in instances):
            return (-1, -np.inf)
        gap_matrix = np.asarray([
            [instance_relative_gaps(rows).get(budget, np.nan) for budget in BUDGETS]
            for rows in instances
        ])
        if np.isnan(gap_matrix).any():
            return (-1, -np.inf)
        mean_gaps = gap_matrix.mean(axis=0)
        per_dataset.append((int(np.sum(mean_gaps >= .05)), float(np.mean(mean_gaps))))
    return (min(item[0] for item in per_dataset), min(item[1] for item in per_dataset))


def simultaneous_bootstrap(gaps: np.ndarray, draws: int = 10000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Max-t simultaneous 95% intervals across budgets for paired instance gaps."""
    gaps = np.asarray(gaps, dtype=np.float64)
    if gaps.ndim != 2 or gaps.shape[0] < 2:
        raise ValueError("gaps must be [instances,budgets] with at least two instances")
    rng = np.random.default_rng(seed)
    observed = gaps.mean(axis=0)
    samples = np.empty((draws, gaps.shape[1]), dtype=np.float64)
    for draw in range(draws):
        indices = rng.integers(0, gaps.shape[0], size=gaps.shape[0])
        samples[draw] = gaps[indices].mean(axis=0)
    centered = samples - observed
    scale = np.maximum(gaps.std(axis=0, ddof=1) / np.sqrt(gaps.shape[0]), 1e-12)
    critical = float(np.quantile(np.max(np.abs(centered / scale), axis=1), .95))
    return observed - critical * scale, observed + critical * scale
