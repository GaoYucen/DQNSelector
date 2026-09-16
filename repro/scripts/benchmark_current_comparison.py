#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.baselines import celf, degree_greedy, one_step_coverage_greedy
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction
from dqnselector.selector import RainbowSelector, greedy_select


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    p.add_argument("--selection-mc", type=int, default=100)
    p.add_argument("--evaluation-mc", type=int, default=300)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--atoms", type=int, default=31)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-celf", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    inst, manifest = load_reconstruction(a.instance)
    emb = np.load(a.embeddings)
    social = emb["social_s"]
    coverage = emb["coverage_r"]
    pool = sorted(inst.worker_pool or set())
    budgets = sorted({k for k in a.budgets if 0 < k <= len(pool)})
    max_k = max(budgets)
    device = a.device if (not a.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"

    # Independent worlds for objective-aware selection and final evaluation.
    t0 = time.perf_counter()
    selection_oracle = LiveEdgeECOracle(inst, mc_times=a.selection_mc, random_seed=a.seed + 17)
    selection_oracle_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    eval_oracle = LiveEdgeECOracle(inst, mc_times=a.evaluation_mc, random_seed=a.seed + 100003)
    eval_oracle_seconds = time.perf_counter() - t0

    methods = {}
    selection_seconds = {}

    t0 = time.perf_counter()
    methods["DegGreedy"] = degree_greedy(inst.graph, max_k, pool)
    selection_seconds["DegGreedy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    node_values = inst.participation * inst.quality
    methods["CovGreedy"] = one_step_coverage_greedy(inst.graph, node_values, max_k, inst.nodes, pool)
    selection_seconds["CovGreedy"] = time.perf_counter() - t0

    if not a.skip_celf:
        t0 = time.perf_counter()
        methods["CELF"] = celf(pool, max_k, selection_oracle.marginal_gain)
        selection_seconds["CELF"] = time.perf_counter() - t0

    model = RainbowSelector(
        social,
        coverage,
        worker_pool=inst.worker_pool,
        hidden_dim=a.hidden,
        atoms=a.atoms,
        v_min=0.0,
        v_max=1.0,
    ).to(device)
    state = torch.load(a.model, map_location=device)
    model.load_state_dict(state)
    t0 = time.perf_counter()
    methods["DQNSelector"] = greedy_select(model, max_k, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    selection_seconds["DQNSelector"] = time.perf_counter() - t0

    rows = []
    for method, order in methods.items():
        for k in budgets:
            t0 = time.perf_counter()
            ec = float(eval_oracle.score(order[:k]))
            eval_seconds = time.perf_counter() - t0
            rows.append({
                "dataset": a.label,
                "method": method,
                "k": k,
                "ec": ec,
                "selection_seconds_max_k": selection_seconds[method],
                "evaluation_seconds": eval_seconds,
            })
            print(f"{a.label}\t{method}\tk={k}\tEC={ec:.6f}\tsel50={selection_seconds[method]:.6f}s")

    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {
        "label": a.label,
        "instance": str(Path(a.instance).resolve()),
        "graph_nodes": inst.graph.number_of_nodes(),
        "graph_edges": inst.graph.number_of_edges(),
        "worker_pool": len(pool),
        "demand_mode": manifest.get("config", {}).get("demand_mode"),
        "budgets": budgets,
        "selection_mc": a.selection_mc,
        "evaluation_mc": a.evaluation_mc,
        "selection_oracle_build_seconds": selection_oracle_seconds,
        "evaluation_oracle_build_seconds": eval_oracle_seconds,
        "selection_seconds": selection_seconds,
        "device": device,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
