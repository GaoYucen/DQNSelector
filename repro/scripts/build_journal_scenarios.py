#!/usr/bin/env python3
"""Materialise the frozen 16-scenario family from raw SNAP inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.journal_protocol import scenario_grid
from dqnselector.reconstruction import save_reconstruction
from dqnselector.scientific_v2 import ScientificV2ReconstructionConfig, reconstruct_scientific_v2_instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gowalla", "brightkite"], required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--checkins", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--structure-seed", type=int, required=True)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument("--scenario", help="optional frozen scenario id")
    parser.add_argument("--allow-unavailable", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_root)
    built = []
    for scenario in scenario_grid():
        if args.scenario and scenario.scenario_id != args.scenario:
            continue
        output = root / args.dataset / f"n{args.users}" / f"struct{args.structure_seed}-task{args.task_seed}" / scenario.scenario_id
        cfg = ScientificV2ReconstructionConfig(
            dataset_name=args.dataset, n_users=args.users, worker_pool_size=300,
            n_target_subareas=100, structure_seed=args.structure_seed, task_seed=args.task_seed,
            graph_sampling=scenario.graph_sampling, worker_pool_policy="nonisolated_uniform",
            trivalency_values=scenario.trivalency_values, quality_mode=scenario.quality_mode,
            load_factor=scenario.load_factor, target_sampling="active_uniform",
        )
        try:
            instance, manifest = reconstruct_scientific_v2_instance(args.edges, args.checkins, cfg)
        except ValueError as error:
            unavailable = "nonisolated users" in str(error) or "no region social component" in str(error)
            if not args.allow_unavailable or not unavailable:
                raise
            output.mkdir(parents=True, exist_ok=True)
            (output / "unavailable.json").write_text(json.dumps({
                "dataset": args.dataset, "users": args.users,
                "structure_seed": args.structure_seed, "task_seed": args.task_seed,
                "scenario_id": scenario.scenario_id, "reason": str(error),
            }, indent=2), encoding="utf-8")
            built.append({"scenario_id": scenario.scenario_id, "output": str(output), "unavailable": True})
            continue
        manifest["journal_scenario"] = scenario.as_manifest()
        manifest.setdefault("notes", []).append("Frozen journal scenario grid; not selected using final test results.")
        save_reconstruction(instance, manifest, output)
        built.append({"scenario_id": scenario.scenario_id, "output": str(output)})
    print(json.dumps({"dataset": args.dataset, "users": args.users, "built": built}, indent=2))


if __name__ == "__main__":
    main()
