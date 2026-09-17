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

from dqnselector.baselines import (
    celf,
    degree_greedy,
    objective_aware_one_step_greedy,
    one_step_coverage_greedy,
)
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction
from dqnselector.selector import RainbowSelector, greedy_select


BUFFER_KEYS = {"social_embedding", "coverage_embedding", "worker_pool_mask"}


def parse_args():
    p = argparse.ArgumentParser(description="Scientific-v2 comparison/transfer benchmark")
    p.add_argument("--instance", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    p.add_argument("--selection-mc", type=int, default=100)
    p.add_argument("--evaluation-mc", type=int, default=300)
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--atoms", type=int, default=51)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-celf", action="store_true")
    p.add_argument(
        "--transfer-weights-only",
        action="store_true",
        help="load encoder/Q-network parameters but preserve target instance buffers",
    )
    return p.parse_args()


def diagnostic(inst, oracle: LiveEdgeECOracle, seeds: list[int]) -> dict[str, float]:
    activation = oracle.activation_probability(seeds)
    coverage = (activation[:, None] * (inst.participation * inst.quality)).sum(axis=0)
    ratio = coverage / inst.demand
    return {
        "ec": float(np.minimum(ratio, 1.0).mean()),
        "realized_ec": float(oracle.realized_score(seeds)),
        "saturated_fraction": float(np.mean(ratio >= 1.0)),
        "coverage_demand_mean": float(np.mean(ratio)),
        "coverage_demand_p50": float(np.quantile(ratio, 0.50)),
        "coverage_demand_p90": float(np.quantile(ratio, 0.90)),
        "coverage_demand_max": float(np.max(ratio)),
        "expected_active_nodes": float(activation.sum()),
    }


def load_model_state(model: RainbowSelector, path: str, transfer_weights_only: bool, device: str):
    state = torch.load(path, map_location=device)
    if not transfer_weights_only:
        model.load_state_dict(state)
        return {"mode": "full_state", "missing": [], "unexpected": []}
    trainable_state = {k: v for k, v in state.items() if k not in BUFFER_KEYS}
    incompatible = model.load_state_dict(trainable_state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if set(missing) != BUFFER_KEYS or unexpected:
        raise RuntimeError(
            f"unexpected transfer state mismatch: missing={missing} unexpected={unexpected}"
        )
    return {"mode": "weights_only", "missing": missing, "unexpected": unexpected}


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    inst, manifest = load_reconstruction(a.instance)
    profile = manifest.get("profile", "legacy_or_unspecified")
    if profile != "journal_scientific_v2":
        raise ValueError(f"expected journal_scientific_v2, got {profile!r}")

    emb = np.load(a.embeddings)
    social = emb["social_s"]
    coverage = emb["coverage_r"]
    pool = sorted(inst.worker_pool or set())
    budgets = sorted({k for k in a.budgets if 0 < k <= len(pool)})
    if not budgets:
        raise ValueError("no valid budgets")
    max_k = max(budgets)
    device = a.device if (not a.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"

    t0 = time.perf_counter()
    selection_oracle = LiveEdgeECOracle(inst, mc_times=a.selection_mc, random_seed=a.seed + 17)
    selection_oracle_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    eval_oracle = LiveEdgeECOracle(inst, mc_times=a.evaluation_mc, random_seed=a.seed + 100003)
    evaluation_oracle_seconds = time.perf_counter() - t0

    methods: dict[str, list[int]] = {}
    selection_seconds: dict[str, float] = {}
    direct_values = inst.participation * inst.quality

    t0 = time.perf_counter()
    methods["DegreeGreedy"] = degree_greedy(inst.graph, max_k, pool)
    selection_seconds["DegreeGreedy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    methods["CoverageGreedy"] = one_step_coverage_greedy(
        inst.graph, direct_values, max_k, inst.nodes, pool
    )
    selection_seconds["CoverageGreedy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    methods["ObjCovGreedy"] = objective_aware_one_step_greedy(
        inst.graph, direct_values, inst.demand, max_k, inst.nodes, pool
    )
    selection_seconds["ObjCovGreedy"] = time.perf_counter() - t0

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
    load_info = load_model_state(model, a.model, a.transfer_weights_only, device)
    t0 = time.perf_counter()
    methods["DQNSelector"] = greedy_select(model, max_k, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    selection_seconds["DQNSelector"] = time.perf_counter() - t0

    rows: list[dict] = []
    for method, order in methods.items():
        for k in budgets:
            t0 = time.perf_counter()
            d = diagnostic(inst, eval_oracle, order[:k])
            eval_seconds = time.perf_counter() - t0
            rows.append({
                "dataset": a.label,
                "method": method,
                "repeat": 0,
                "k": k,
                **d,
                "selection_seconds_max_k": selection_seconds[method],
                "evaluation_seconds": eval_seconds,
            })
            print(
                f"{a.label}\t{method}\tk={k}\tEC={d['ec']:.6f}\t"
                f"realized={d['realized_ec']:.6f}\tsat={d['saturated_fraction']:.3f}\t"
                f"sel={selection_seconds[method]:.6f}s"
            )

    rng = np.random.default_rng(a.seed + 1009)
    random_selection_seconds = []
    for rep in range(a.random_repeats):
        t0 = time.perf_counter()
        order = rng.permutation(pool).tolist()
        random_selection_seconds.append(time.perf_counter() - t0)
        for k in budgets:
            t0 = time.perf_counter()
            d = diagnostic(inst, eval_oracle, order[:k])
            eval_seconds = time.perf_counter() - t0
            rows.append({
                "dataset": a.label,
                "method": "Random",
                "repeat": rep,
                "k": k,
                **d,
                "selection_seconds_max_k": random_selection_seconds[-1],
                "evaluation_seconds": eval_seconds,
            })

    fields = list(rows[0].keys())
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    random_mean = {}
    for k in budgets:
        vals = [r for r in rows if r["method"] == "Random" and r["k"] == k]
        random_mean[str(k)] = {
            key: float(np.mean([v[key] for v in vals]))
            for key in ["ec", "realized_ec", "saturated_fraction", "expected_active_nodes"]
        }

    summary = {
        "label": a.label,
        "profile": profile,
        "instance": str(Path(a.instance).resolve()),
        "model": str(Path(a.model).resolve()),
        "model_load": load_info,
        "transfer_weights_only": a.transfer_weights_only,
        "graph_nodes": inst.graph.number_of_nodes(),
        "graph_edges": inst.graph.number_of_edges(),
        "worker_pool": len(pool),
        "subareas": inst.n_subareas,
        "structure_seed": manifest.get("config", {}).get("structure_seed"),
        "task_seed": manifest.get("config", {}).get("task_seed"),
        "load_factor": manifest.get("demand", {}).get("load_factor"),
        "budgets": budgets,
        "selection_mc": a.selection_mc,
        "evaluation_mc": a.evaluation_mc,
        "random_repeats": a.random_repeats,
        "selection_oracle_build_seconds": selection_oracle_seconds,
        "evaluation_oracle_build_seconds": evaluation_oracle_seconds,
        "selection_seconds": selection_seconds,
        "random_selection_seconds_mean": float(np.mean(random_selection_seconds)),
        "random_mean_by_k": random_mean,
        "device": device,
        "note": (
            "Canonical EC remains the optimization objective. realized_ec clips task fulfillment "
            "inside each live-edge world and is reported only as a robustness metric. "
            "weights_only transfer preserves target-instance embeddings and worker-pool buffers."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
