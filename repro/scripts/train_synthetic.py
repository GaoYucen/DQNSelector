#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "repro"
sys.path.insert(0, str(REPRO / "src"))

from dqnselector.ecm import ECMInstance, effective_coverage, marginal_reward
from dqnselector.embedding import build_balanced_influence_pairs, fit_social_influence_embedding
from dqnselector.piic import piic
from dqnselector.selector import RainbowSelector, greedy_select, train_rainbow_selector


def make_instance() -> ECMInstance:
    rng = np.random.default_rng(4)
    n, h = 10, 3
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < 0.18:
                g.add_edge(u, v, weight=float(rng.uniform(0.1, 0.5)))
    participation = rng.uniform(0.1, 1.0, size=(n, h))
    quality = rng.uniform(0.1, 1.0, size=(n, h))
    demand = np.full(h, 2.5)
    return ECMInstance(g, list(range(n)), participation, quality, demand, set(range(n)))


def main() -> None:
    inst = make_instance()
    pairs = build_balanced_influence_pairs(inst.graph, mc_times=30, graph_iterations=1, random_seed=2)
    social, _, hist = fit_social_influence_embedding(
        pairs, dim=8, epochs=20, learning_rate=0.05, batch_size=64, random_seed=2
    )
    coverage = piic(inst.graph, inst.quality, influence_range=2, nodes=inst.nodes)
    model = RainbowSelector(social, coverage, hidden_dim=32, atoms=31, v_min=0.0, v_max=1.0)

    def reward_fn(selected: set[int], candidate: int) -> float:
        return marginal_reward(inst, selected, candidate, mc_times=80, random_seed=123)

    model, stats = train_rainbow_selector(
        model,
        reward_fn,
        seed_budget=3,
        episodes=8,
        batch_size=4,
        warmup=4,
        replay_capacity=128,
        target_update_interval=4,
        random_seed=3,
    )
    chosen = greedy_select(model, 3)
    ec = effective_coverage(inst, chosen, mc_times=500, random_seed=99)
    print(f"embedding_loss: {hist[0]:.6f} -> {hist[-1]:.6f}")
    print(f"episode_returns={np.round(stats.episode_returns, 4).tolist()}")
    print(f"chosen={chosen} EC={ec:.6f}")


if __name__ == "__main__":
    main()
