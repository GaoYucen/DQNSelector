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
    p.add_argument(
        "--load-multipliers",
        type=float,
        nargs="+",
        default=[1.0],
        help="Scale the stored demand without rebuilding graph/proxies; e.g. 0.5 1 1.5 2.",
    )
    return p.parse_args()


def activation_and_coverage(inst, oracle: LiveEdgeECOracle, seeds: list[int]):
    if not seeds:
        return np.zeros(inst.n_subareas, dtype=np.float64), 0.0
    activation = oracle.activation_probability(seeds)
    contribution = inst.participation * inst.quality
    coverage = (activation[:, None] * contribution).sum(axis=0)
    return coverage, float(activation.sum())


def diagnostics(coverage: np.ndarray, demand: np.ndarray, active_mean: float) -> dict[str, float]:
    ratio = coverage / demand
    return {
        "ec": float(np.minimum(ratio, 1.0).mean()),
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
    multipliers = sorted({float(x) for x in a.load_multipliers if x > 0})
    if not multipliers:
        raise ValueError("at least one positive --load-multipliers value is required")

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

    base_load = manifest.get("demand", {}).get("load_factor")
    rows: list[dict] = []

    def record(method: str, repeat: int, k: int, order: list[int]) -> None:
        coverage, active_mean = activation_and_coverage(inst, oracle, order[:k])
        for mult in multipliers:
            d = diagnostics(coverage, inst.demand * mult, active_mean)
            effective_load = float(base_load) * mult if base_load is not None else None
            rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "k": k,
                    "load_multiplier": mult,
                    "effective_load_factor": effective_load,
                    **d,
                }
            )
            if repeat == 0 and method != "Random":
                load_text = f"rho={effective_load:.3g}" if effective_load is not None else f"x{mult:g}"
                print(
                    f"{method:14s} {load_text:9s} k={k:3d} EC={d['ec']:.4f} "
                    f"sat={d['saturated_fraction']:.3f} ratio50={d['coverage_demand_p50']:.3f} "
                    f"active={d['expected_active_nodes']:.1f}"
                )

    for method, order in methods.items():
        for k in budgets:
            record(method, 0, k, order)

    rng = np.random.default_rng(a.seed + 1009)
    for rep in range(a.random_repeats):
        order = rng.permutation(pool).tolist()
        for k in budgets:
            record("Random", rep, k, order)

    with (out / "saturation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    random_summary = {}
    for mult in multipliers:
        per_load = {}
        for k in budgets:
            vals = [
                r
                for r in rows
                if r["method"] == "Random" and r["k"] == k and r["load_multiplier"] == mult
            ]
            per_load[str(k)] = {
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
        random_summary[str(mult)] = per_load

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
        "load_multipliers": multipliers,
        "base_load_factor": base_load,
        "oracle_build_seconds": oracle_seconds,
        "oracle_stats": {
            "mean_reachability": oracle.stats.mean_reachability,
            "max_reachability": oracle.stats.max_reachability,
        },
        "base_demand": {
            "min": float(inst.demand.min()),
            "mean": float(inst.demand.mean()),
            "max": float(inst.demand.max()),
        },
        "random_mean_by_load_and_k": random_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
