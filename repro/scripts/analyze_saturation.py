#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.baselines import degree_greedy, one_step_coverage_greedy
from dqnselector.ecm import effective_coverage_from_activation
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose EC objective saturation")
    p.add_argument("--instance", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 30, 40, 50])
    p.add_argument("--mc", type=int, default=200)
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def diagnostics(inst, oracle: LiveEdgeECOracle, seeds: list[int]) -> dict[str, float]:
    if not seeds:
        ratio = np.zeros(inst.n_subareas, dtype=np.float64)
        ec = 0.0
        active_mean = 0.0
    else:
        activation = oracle.activation_probability(seeds)
        ec, _, coverage = effective_coverage_from_activation(
            inst,
            {i: float(p) for i, p in enumerate(activation)},
        )
        ratio = coverage / inst.demand
        active_mean = float(activation.sum())
    return {
        "ec": float(ec),
        "saturated_fraction": float(np.mean(ratio >= 1.0)),
        "coverage_demand_mean": float(np.mean(ratio)),
        "coverage_demand_p50": float(np.quantile(ratio, 0.50)),
        "coverage_demand_p90": float(np.quantile(ratio, 0.90)),
        "coverage_demand_max": float(np.max(ratio)) if ratio.size else 0.0,
        "expected_active_nodes": active_mean,
    }


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    inst, manifest = load_reconstruction(a.instance)
    pool = sorted(inst.worker_pool or set())
    budgets = sorted({k for k in a.budgets if 0 <= k <= len(pool)})
    max_k = max(budgets)

    t0 = time.perf_counter()
    oracle = LiveEdgeECOracle(inst, mc_times=a.mc, random_seed=a.seed)
    oracle_seconds = time.perf_counter() - t0

    methods: dict[str, list[int]] = {
        "DegreeGreedy": degree_greedy(inst.graph, max_k, pool),
        "CoverageGreedy": one_step_coverage_greedy(
            inst.graph,
            inst.participation * inst.quality,
            max_k,
            inst.nodes,
            pool,
        ),
    }

    rows: list[dict] = []
    for method, order in methods.items():
        for k in budgets:
            d = diagnostics(inst, oracle, order[:k])
            rows.append({"method": method, "repeat": 0, "k": k, **d})
            print(
                f"{method:14s} k={k:3d} EC={d['ec']:.4f} "
                f"sat={d['saturated_fraction']:.3f} ratio50={d['coverage_demand_p50']:.3f} "
                f"active={d['expected_active_nodes']:.1f}"
            )

    rng = np.random.default_rng(a.seed + 1009)
    for rep in range(a.random_repeats):
        order = rng.permutation(pool).tolist()
        for k in budgets:
            d = diagnostics(inst, oracle, order[:k])
            rows.append({"method": "Random", "repeat": rep, "k": k, **d})

    with (out / "saturation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    random_summary = {}
    for k in budgets:
        vals = [r for r in rows if r["method"] == "Random" and r["k"] == k]
        random_summary[str(k)] = {
            key: float(np.mean([v[key] for v in vals]))
            for key in [
                "ec",
                "saturated_fraction",
                "coverage_demand_mean",
                "coverage_demand_p50",
                "coverage_demand_p90",
                "expected_active_nodes",
            ]
        }

    summary = {
        "instance": str(Path(a.instance).resolve()),
        "profile": manifest.get("profile", "legacy_or_unspecified"),
        "dataset": manifest.get("config", {}).get("dataset_name"),
        "nodes": inst.graph.number_of_nodes(),
        "edges": inst.graph.number_of_edges(),
        "worker_pool": len(pool),
        "subareas": inst.n_subareas,
        "budgets": budgets,
        "mc": a.mc,
        "random_repeats": a.random_repeats,
        "oracle_build_seconds": oracle_seconds,
        "oracle_stats": {
            "mean_reachability": oracle.stats.mean_reachability,
            "max_reachability": oracle.stats.max_reachability,
        },
        "demand": {
            "min": float(inst.demand.min()),
            "mean": float(inst.demand.mean()),
            "max": float(inst.demand.max()),
        },
        "random_mean_by_k": random_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
