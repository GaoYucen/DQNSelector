#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.baselines import celf, objective_aware_one_step_greedy
from dqnselector.ecm import ECMInstance
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction


def parse_args():
    p = argparse.ArgumentParser(description="Validate a candidate load with CELF")
    p.add_argument("--instance", required=True)
    p.add_argument("--load-multiplier", type=float, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    p.add_argument("--selection-mc", type=int, default=100)
    p.add_argument("--evaluation-mc", type=int, default=300)
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def scaled_instance(inst: ECMInstance, multiplier: float) -> ECMInstance:
    if multiplier <= 0:
        raise ValueError("load multiplier must be positive")
    return ECMInstance(
        graph=inst.graph,
        nodes=inst.nodes,
        participation=inst.participation,
        quality=inst.quality,
        demand=inst.demand * multiplier,
        worker_pool=inst.worker_pool,
    )


def diagnostics(inst: ECMInstance, oracle: LiveEdgeECOracle, order: list[int], k: int) -> dict[str, float]:
    activation = oracle.activation_probability(order[:k])
    coverage = (activation[:, None] * (inst.participation * inst.quality)).sum(axis=0)
    ratio = coverage / inst.demand
    return {
        "ec": float(np.minimum(ratio, 1.0).mean()),
        "saturated_fraction": float(np.mean(ratio >= 1.0)),
        "ratio_p50": float(np.quantile(ratio, 0.5)),
        "ratio_p90": float(np.quantile(ratio, 0.9)),
    }


def main():
    a = parse_args()
    base, manifest = load_reconstruction(a.instance)
    inst = scaled_instance(base, a.load_multiplier)
    pool = sorted(inst.worker_pool or set())
    budgets = sorted({k for k in a.budgets if 0 < k <= len(pool)})
    max_k = max(budgets)
    direct = inst.participation * inst.quality

    t0 = time.perf_counter()
    selection_oracle = LiveEdgeECOracle(inst, mc_times=a.selection_mc, random_seed=a.seed + 17)
    celf_order = celf(pool, max_k, selection_oracle.marginal_gain)
    celf_seconds = time.perf_counter() - t0

    obj_order = objective_aware_one_step_greedy(
        inst.graph, direct, inst.demand, max_k, inst.nodes, pool
    )
    eval_oracle = LiveEdgeECOracle(inst, mc_times=a.evaluation_mc, random_seed=a.seed + 100003)

    rows = []
    for method, order in [("CELF", celf_order), ("ObjCovGreedy", obj_order)]:
        for k in budgets:
            d = diagnostics(inst, eval_oracle, order, k)
            rows.append({"method": method, "k": k, **d})
            print(method, "k=", k, "EC=", round(d["ec"], 6), "sat=", round(d["saturated_fraction"], 3))

    rng = np.random.default_rng(a.seed + 1009)
    random_summary = {}
    for k in budgets:
        vals = []
        sats = []
        for _ in range(a.random_repeats):
            order = rng.permutation(pool).tolist()
            d = diagnostics(inst, eval_oracle, order, k)
            vals.append(d["ec"])
            sats.append(d["saturated_fraction"])
        random_summary[str(k)] = {"ec": float(np.mean(vals)), "saturated_fraction": float(np.mean(sats))}

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    base_load = manifest.get("demand", {}).get("load_factor")
    result = {
        "dataset": manifest.get("config", {}).get("dataset_name"),
        "base_load_factor": base_load,
        "load_multiplier": a.load_multiplier,
        "effective_load_factor": (float(base_load) * a.load_multiplier if base_load is not None else None),
        "selection_mc": a.selection_mc,
        "evaluation_mc": a.evaluation_mc,
        "celf_selection_seconds": celf_seconds,
        "rows": rows,
        "random_mean": random_summary,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
