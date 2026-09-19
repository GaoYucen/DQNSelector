#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.reconstruction import save_reconstruction
from dqnselector.scientific_v2 import (
    ScientificV2ReconstructionConfig,
    reconstruct_scientific_v2_instance,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build journal scientific-v2 DQNSelector instance")
    p.add_argument("--dataset", required=True, choices=["gowalla", "brightkite"])
    p.add_argument("--edges", required=True)
    p.add_argument("--checkins", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--users", type=int, default=3000)
    p.add_argument("--worker-pool", type=int, default=300)
    p.add_argument("--targets", type=int, default=100)
    p.add_argument("--structure-seed", type=int, default=2024)
    p.add_argument("--task-seed", type=int, default=2024)
    p.add_argument("--load-factor", type=float, default=1.50)
    p.add_argument("--demand-activity-power", type=float, default=0.50)
    p.add_argument("--demand-floor-ratio", type=float, default=0.05)
    p.add_argument("--participation-quantile", type=float, default=0.50)
    p.add_argument(
        "--quality-mode",
        choices=["uniform", "activity_reliability"],
        default="uniform",
    )
    p.add_argument("--quality-floor", type=float, default=0.25)
    p.add_argument("--quality-lower-quantile", type=float, default=0.10)
    p.add_argument("--quality-upper-quantile", type=float, default=0.90)
    p.add_argument(
        "--target-sampling",
        choices=["active_uniform", "active_weighted", "all_uniform"],
        default="active_uniform",
    )
    p.add_argument(
        "--trivalency-values", type=float, nargs="+", default=[0.001, 0.01, 0.10]
    )
    p.add_argument("--directed-source", action="store_true")
    p.add_argument("--graph-sampling", choices=["uniform_induced", "community_bfs"], default="uniform_induced")
    p.add_argument("--worker-pool-policy", choices=["all_uniform", "nonisolated_uniform"], default="nonisolated_uniform")
    return p.parse_args()


def main():
    a = parse_args()
    tri = tuple(float(x) for x in a.trivalency_values)
    cfg = ScientificV2ReconstructionConfig(
        dataset_name=a.dataset,
        n_users=a.users,
        worker_pool_size=a.worker_pool,
        n_target_subareas=a.targets,
        structure_seed=a.structure_seed,
        task_seed=a.task_seed,
        bidirectional_edges=not a.directed_source,
        graph_sampling=a.graph_sampling,
        worker_pool_policy=a.worker_pool_policy,
        participation_distance_quantile=a.participation_quantile,
        quality_mode=a.quality_mode,
        quality_floor=a.quality_floor,
        quality_lower_quantile=a.quality_lower_quantile,
        quality_upper_quantile=a.quality_upper_quantile,
        load_factor=a.load_factor,
        demand_activity_power=a.demand_activity_power,
        demand_floor_ratio=a.demand_floor_ratio,
        target_sampling=a.target_sampling,
        trivalency_values=tri,
    )
    inst, manifest = reconstruct_scientific_v2_instance(a.edges, a.checkins, cfg)
    save_reconstruction(inst, manifest, a.output)
    print(
        f"saved {a.output}: profile={manifest['profile']} nodes={inst.graph.number_of_nodes()} "
        f"edges={inst.graph.number_of_edges()} subareas={inst.n_subareas} "
        f"worker_pool={len(inst.worker_pool or set())}"
    )
    print(
        f"demand min={inst.demand.min():.6g} mean={inst.demand.mean():.6g} "
        f"max={inst.demand.max():.6g} load_factor={a.load_factor:.4g}"
    )
    print(
        f"participation mean={inst.participation.mean():.6g}; "
        f"quality mean={inst.quality.mean():.6g}; "
        f"targets_active={manifest['targets']['all_have_historical_activity']}"
    )


if __name__ == "__main__":
    main()
