#!/usr/bin/env python3
"""Single entry point for gate, development screening, and frozen-scenario tests."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
from dqnselector.journal_protocol import scenario_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("gate", "screen", "full-development", "test"), required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--primary-scenario")
    parser.add_argument("--development-scenarios", nargs="*")
    parser.add_argument("--timing-repeats", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool):
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def instance_path(data_root: Path, dataset: str, users: int, structure: int, task: int, scenario: str) -> Path:
    return data_root / dataset / f"n{users}" / f"struct{structure}-task{task}" / scenario


def prepare_instance(args, dataset, users, structure, task, scenario):
    raw = Path(args.raw_root)
    target = instance_path(Path(args.data_root), dataset, users, structure, task, scenario)
    if not (target / "instance.npz").exists() or not (target / "manifest.json").exists():
        run([sys.executable, str(ROOT / "repro/scripts/build_journal_scenarios.py"), "--dataset", dataset,
             "--edges", str(raw / f"loc-{dataset}_edges.txt.gz"), "--checkins", str(raw / f"loc-{dataset}_totalCheckins.txt.gz"),
             "--output-root", str(Path(args.data_root)), "--users", str(users), "--structure-seed", str(structure),
             "--task-seed", str(task), "--scenario", scenario, "--allow-unavailable"], args.dry_run)
    if (target / "unavailable.json").exists():
        return None
    if not (target / "embedding/embeddings.npz").exists() or not (target / "embedding/metadata.json").exists():
        run([sys.executable, str(ROOT / "repro/scripts/precompute_embeddings.py"), "--instance", str(target),
             "--output", str(target / "embedding"), "--seed", "2024"], args.dry_run)
    return target


def run_instance(args, target, dataset, label, train_seed, dqn_episodes, journal_episodes, piano_episodes, evaluation_mc, ablations=False, hidden=128, weight=.3, temperature=.1, include_baseline=True):
    raw = Path(args.raw_root)
    output = Path(args.result_root) / label
    common = [sys.executable, str(ROOT / "repro/scripts/run_journal_instance.py"), "--instance", str(target),
              "--embeddings", str(target / "embedding/embeddings.npz"), "--checkins", str(raw / f"loc-{dataset}_totalCheckins.txt.gz"),
              "--device", args.device, "--dqn-episodes", str(dqn_episodes), "--journal-episodes", str(journal_episodes),
              "--piano-episodes", str(piano_episodes), "--evaluation-mc", str(evaluation_mc), "--journal-hidden", str(hidden),
              "--ranking-weight", str(weight), "--ranking-temperature", str(temperature), "--timing-repeats", str(args.timing_repeats)]
    try:
        if include_baseline:
            run(common + ["--role", "baseline", "--output", str(output / "baseline")], args.dry_run)
        run(common + ["--role", "trained", "--training-seed", str(train_seed), "--output", str(output / f"train-{train_seed}")] + (["--run-ablations"] if ablations else []), args.dry_run)
    except subprocess.CalledProcessError as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(json.dumps({"stage": "baseline" if include_baseline else "trained", "returncode": error.returncode, "command": error.cmd}, indent=2), encoding="utf-8")
        raise


def main():
    args = parse_args()
    all_scenarios = {item.scenario_id for item in scenario_grid()}
    gate = "graph-uniform_induced__tri-low__quality-uniform__load-1.5"
    if args.phase == "gate":
        args.timing_repeats = 3
        target = prepare_instance(args, "gowalla", 3000, 2024, 2024, gate)
        if target is None:
            gate = "graph-community_bfs__tri-low__quality-uniform__load-1.5"
            target = prepare_instance(args, "gowalla", 3000, 2024, 2024, gate)
        if target is None:
            raise RuntimeError("both predefined gate scenarios are unavailable")
        run_instance(args, target, "gowalla", "gate/gowalla-n3000-struct2024-task2024", 2024, 10, 20, 10, 500, ablations=True)
        return
    if args.phase == "screen":
        for dataset, structure, task, scenario in product(("gowalla", "brightkite"), (2024, 2025), (2024, 2025), sorted(all_scenarios)):
            target = prepare_instance(args, dataset, 3000, structure, task, scenario)
            if target is not None:
                label = f"screen/{dataset}/n3000/struct{structure}-task{task}/{scenario}"
                run_instance(args, target, dataset, label, 2024, 50, 100, 50, 500)
        return
    if args.phase == "full-development":
        scenarios = args.development_scenarios or []
        if len(scenarios) != 3 or not set(scenarios) <= all_scenarios:
            raise ValueError("full-development requires exactly three scenario ids selected by screen")
        for dataset, structure, task, scenario, hidden, weight, temperature in product(("gowalla", "brightkite"), (2024, 2025), (2024, 2025), scenarios, (128, 256), (.1, .3), (.05, .1)):
            target = prepare_instance(args, dataset, 3000, structure, task, scenario)
            if target is not None:
                label = f"full-development/{dataset}/n3000/struct{structure}-task{task}/{scenario}/h{hidden}-w{weight}-t{temperature}"
                run_instance(args, target, dataset, label, 2024, 200, 400, 200, 500, hidden=hidden, weight=weight, temperature=temperature)
        return
    scenario = args.primary_scenario
    if scenario not in all_scenarios:
        raise ValueError("test requires --primary-scenario frozen from development")
    for dataset, users, structure, task in product(("gowalla", "brightkite"), (3000, 5000), range(4001, 4006), (5001, 5002)):
        target = prepare_instance(args, dataset, users, structure, task, scenario)
        if target is not None:
            label = f"test/{dataset}/n{users}/struct{structure}-task{task}/{scenario}"
            for index, train_seed in enumerate((6001, 6002, 6003)):
                run_instance(args, target, dataset, label, train_seed, 200, 400, 200, 2000, include_baseline=index == 0)


if __name__ == "__main__":
    main()
