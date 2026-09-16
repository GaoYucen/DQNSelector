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

from dqnselector.baselines import celf, degree_greedy, one_step_coverage_greedy
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True, help="directory containing instance.npz + manifest.json")
    p.add_argument("--output", required=True)
    p.add_argument("--mc", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 60, 70, 80, 90, 100])
    p.add_argument("--skip-celf", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    inst, manifest = load_reconstruction(a.instance)
    budgets = sorted(set(int(k) for k in a.budgets if k > 0))
    max_k = min(max(budgets), len(inst.worker_pool or set()))
    budgets = [k for k in budgets if k <= max_k]

    t0 = time.perf_counter()
    oracle = LiveEdgeECOracle(inst, mc_times=a.mc, random_seed=a.seed)
    oracle_time = time.perf_counter() - t0
    pool = sorted(inst.worker_pool or set())

    methods: dict[str, list[int]] = {}
    t = time.perf_counter()
    methods["DegGreedy"] = degree_greedy(inst.graph, max_k, pool)
    deg_time = time.perf_counter() - t

    t = time.perf_counter()
    node_values = inst.participation * inst.quality
    methods["CovGreedy"] = one_step_coverage_greedy(
        inst.graph, node_values, max_k, inst.nodes, pool
    )
    cov_time = time.perf_counter() - t

    celf_time = None
    if not a.skip_celf:
        t = time.perf_counter()
        methods["CELF"] = celf(pool, max_k, oracle.marginal_gain)
        celf_time = time.perf_counter() - t

    rows: list[dict] = []
    for method, order in methods.items():
        for k in budgets:
            selected = order[:k]
            t = time.perf_counter()
            score = oracle.score(selected)
            eval_time = time.perf_counter() - t
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "ec": score,
                    "evaluation_seconds": eval_time,
                    "selected": " ".join(map(str, selected)),
                }
            )
            print(method, "k=", k, "EC=", f"{score:.6f}")

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "instance": str(Path(a.instance).resolve()),
        "manifest_config": manifest.get("config", {}),
        "graph_nodes": inst.graph.number_of_nodes(),
        "graph_edges": inst.graph.number_of_edges(),
        "worker_pool": len(pool),
        "mc_worlds": a.mc,
        "oracle_random_seed": a.seed,
        "oracle_build_seconds": oracle_time,
        "oracle_stats": oracle.stats.__dict__,
        "selection_seconds": {
            "DegGreedy": deg_time,
            "CovGreedy": cov_time,
            "CELF": celf_time,
        },
        "warning": "These are transparent-reconstruction results, not historical-paper reproduction, unless the original preprocessing details are recovered.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
