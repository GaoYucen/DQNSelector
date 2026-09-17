#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.reconstruction import save_reconstruction
from dqnselector.scientific import ScientificReconstructionConfig, reconstruct_scientific_instance


MAIN_LOAD_FACTOR = 1.25


def parse_args():
    p = argparse.ArgumentParser(description="Build the journal scientific DQNSelector instance")
    p.add_argument("--dataset", required=True, choices=["gowalla", "brightkite"])
    p.add_argument("--edges", required=True)
    p.add_argument("--checkins", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--users", type=int, default=3000)
    p.add_argument("--worker-pool", type=int, default=300)
    p.add_argument("--targets", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument(
        "--load-factor",
        type=float,
        default=MAIN_LOAD_FACTOR,
        help="Scientific-v1 main setting is rho=1.25; sweep other regimes explicitly.",
    )
    p.add_argument("--demand-activity-power", type=float, default=0.50)
    p.add_argument("--demand-floor-ratio", type=float, default=0.05)
    p.add_argument("--participation-quantile", type=float, default=0.50)
    p.add_argument("--quality-floor", type=float, default=0.25)
    p.add_argument("--quality-lower-quantile", type=float, default=0.10)
    p.add_argument("--quality-upper-quantile", type=float, default=0.90)
    p.add_argument(
        "--trivalency-values", type=float, nargs="+", default=[0.01, 0.05, 0.10]
    )
    p.add_argument("--directed-source", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    tri = tuple(float(x) for x in a.trivalency_values)
    if len(tri) == 0:
        raise ValueError("--trivalency-values must contain at least one value")
    cfg = ScientificReconstructionConfig(
        dataset_name=a.dataset,
        n_users=a.users,
        worker_pool_size=a.worker_pool,
        n_target_subareas=a.targets,
        random_seed=a.seed,
        bidirectional_edges=not a.directed_source,
        participation_distance_quantile=a.participation_quantile,
        quality_floor=a.quality_floor,
        quality_lower_quantile=a.quality_lower_quantile,
        quality_upper_quantile=a.quality_upper_quantile,
        load_factor=a.load_factor,
        demand_activity_power=a.demand_activity_power,
        demand_floor_ratio=a.demand_floor_ratio,
        trivalency_values=tri,
    )
    inst, manifest = reconstruct_scientific_instance(a.edges, a.checkins, cfg)
    save_reconstruction(inst, manifest, a.output)
    print(
        f"saved {a.output}: nodes={inst.graph.number_of_nodes()} "
        f"edges={inst.graph.number_of_edges()} subareas={inst.n_subareas} "
        f"worker_pool={len(inst.worker_pool or set())}"
    )
    print(
        f"demand min={inst.demand.min():.6g} mean={inst.demand.mean():.6g} "
        f"max={inst.demand.max():.6g} load_factor={a.load_factor:.4g}"
    )
    print(
        f"participation mean={inst.participation.mean():.6g}; "
        f"quality mean={inst.quality.mean():.6g}"
    )


if __name__ == "__main__":
    main()
