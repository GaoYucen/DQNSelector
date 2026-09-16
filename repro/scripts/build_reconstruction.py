#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.reconstruction import ReconstructionConfig, reconstruct_instance, save_reconstruction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["gowalla", "brightkite"])
    p.add_argument("--edges", required=True)
    p.add_argument("--checkins", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--users", type=int, default=3000)
    p.add_argument("--worker-pool", type=int, default=300)
    p.add_argument("--targets", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--demand-mode",
        choices=["raw_checkin_count", "max_normalized", "sqrt_count"],
        default="raw_checkin_count",
    )
    p.add_argument("--demand-scale", type=float, default=1.0)
    p.add_argument("--directed-source", action="store_true", help="Do not duplicate source friendship edges in both directions")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = ReconstructionConfig(
        dataset_name=a.dataset,
        n_users=a.users,
        worker_pool_size=a.worker_pool,
        n_target_subareas=a.targets,
        random_seed=a.seed,
        demand_mode=a.demand_mode,
        demand_scale=a.demand_scale,
        bidirectional_edges=not a.directed_source,
    )
    inst, manifest = reconstruct_instance(a.edges, a.checkins, cfg)
    save_reconstruction(inst, manifest, a.output)
    print(f"saved {a.output}: nodes={inst.graph.number_of_nodes()} edges={inst.graph.number_of_edges()} subareas={inst.n_subareas}")
    print(f"demand min={inst.demand.min():.6g} max={inst.demand.max():.6g} mean={inst.demand.mean():.6g}")


if __name__ == "__main__":
    main()
