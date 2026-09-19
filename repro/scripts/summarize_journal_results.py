#!/usr/bin/env python3
"""Freeze a development scenario and test the journal acceptance criterion."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
from dqnselector.journal_protocol import BUDGETS, instance_relative_gaps, scenario_rank_key, simultaneous_bootstrap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="directory containing per-run results.json files")
    parser.add_argument("--phase", choices=["development", "test"], required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_runs(root: Path):
    runs = []
    for path in root.rglob("results.json"):
        metadata = path.with_name("metadata.json")
        if not metadata.exists():
            continue
        runs.append((json.loads(metadata.read_text()), json.loads(path.read_text()), path.parent))
    return runs


def average_training_seeds(runs: list[tuple[dict, list[dict], Path]]):
    """Average learned-model seeds before treating a reconstruction as an instance.

    Baselines are deterministically repeated in every run; averaging all methods
    makes an accidental seed-dependent baseline implementation visible while
    preserving the planned three-seed estimate for learned methods.
    """
    grouped: dict[tuple[str, str, str], list[tuple[dict, list[dict], Path]]] = defaultdict(list)
    for metadata, rows, path in runs:
        scenario = metadata.get("manifest", {}).get("scenario_id", "missing")
        dataset = f'{metadata["dataset"]}-{metadata["users"]}'
        grouped[(scenario, dataset, metadata["instance_sha256"])].append((metadata, rows, path))
    result = []
    for (scenario, dataset, instance_hash), trials in grouped.items():
        values: dict[tuple[str, int], list[float]] = defaultdict(list)
        for _, rows, _ in trials:
            for row in rows:
                values[(row["method"], int(row["k"]))].append(float(row["ec"]))
        averaged = [
            {"method": method, "k": budget, "ec": float(np.mean(ec))}
            for (method, budget), ec in sorted(values.items())
        ]
        result.append((scenario, dataset, instance_hash, averaged, len(trials)))
    return result


def main():
    args = parse_args(); root = Path(args.root)
    runs = load_runs(root)
    instances = average_training_seeds(runs)
    by_scenario: dict[str, dict[str, list[list[dict]]]] = defaultdict(lambda: defaultdict(list))
    for scenario, dataset, _, rows, _ in instances:
        by_scenario[scenario][dataset].append(rows)
    if args.phase == "development":
        ranked = []
        for scenario, datasets in by_scenario.items():
            ranked.append((scenario_rank_key(datasets), scenario))
        ranked.sort(reverse=True)
        result = {"phase": "development", "ranking": [{"scenario_id": scenario, "rank_key": key} for key, scenario in ranked],
                  "frozen_primary": ranked[0][1] if ranked else None}
    else:
        reports = []
        for scenario, datasets in by_scenario.items():
            for dataset, dataset_instances in datasets.items():
                gaps = np.asarray([[instance_relative_gaps(rows).get(k, np.nan) for k in BUDGETS] for rows in dataset_instances])
                valid = gaps[~np.isnan(gaps).any(axis=1)]
                if valid.shape[0] < 2:
                    raise ValueError(f"{scenario}/{dataset} needs at least two complete paired instances")
                lower, upper = simultaneous_bootstrap(valid, seed=20260919)
                means = valid.mean(0)
                reports.append({"scenario_id": scenario, "dataset": dataset, "instances": int(valid.shape[0]), "budgets": list(BUDGETS),
                                "mean_relative_gap": means.tolist(), "simultaneous_ci_lower": lower.tolist(), "simultaneous_ci_upper": upper.tolist(),
                                "accepted_budget_count": int(np.sum((means >= .05) & (lower > 0))),
                                "passes": bool(np.sum((means >= .05) & (lower > 0)) >= 4)})
        result = {"phase": "test", "reports": reports, "passes_all_dataset_groups": bool(reports) and all(report["passes"] for report in reports)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
