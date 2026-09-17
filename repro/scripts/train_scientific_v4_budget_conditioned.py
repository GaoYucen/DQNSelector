#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.budget_conditioned import (
    BudgetConditionedRainbowSelector,
    greedy_select_for_budget,
    train_budget_conditioned_selector,
)
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction


def parse_args():
    p = argparse.ArgumentParser(description="Train budget-conditioned DQNSelector scientific-v4")
    p.add_argument("--instance", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--train-budgets", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    p.add_argument("--eval-budgets", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    p.add_argument("--oracle-mc", type=int, default=100)
    p.add_argument("--evaluation-mc", type=int, default=300)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--atoms", type=int, default=51)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=64)
    p.add_argument("--target-update", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    inst, manifest = load_reconstruction(a.instance)
    profile = manifest.get("profile", "legacy_or_unspecified")
    if profile != "journal_scientific_v2":
        raise ValueError(f"expected journal_scientific_v2 instance, got {profile!r}")

    emb = np.load(a.embeddings)
    social = emb["social_s"]
    coverage = emb["coverage_r"]
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    worker_pool_size = len(inst.worker_pool or set())
    train_budgets = sorted({int(k) for k in a.train_budgets})
    eval_budgets = sorted({int(k) for k in a.eval_budgets})
    if not train_budgets or train_budgets[0] <= 0 or train_budgets[-1] > worker_pool_size:
        raise ValueError(f"training budgets must lie in [1,{worker_pool_size}]")
    if not eval_budgets or eval_budgets[0] <= 0 or eval_budgets[-1] > worker_pool_size:
        raise ValueError(f"evaluation budgets must lie in [1,{worker_pool_size}]")

    t0 = time.perf_counter()
    train_oracle = LiveEdgeECOracle(inst, mc_times=a.oracle_mc, random_seed=a.seed)
    oracle_seconds = time.perf_counter() - t0

    model = BudgetConditionedRainbowSelector(
        social,
        coverage,
        worker_pool=inst.worker_pool,
        hidden_dim=a.hidden,
        atoms=a.atoms,
        v_min=0.0,
        v_max=1.0,
    )

    t0 = time.perf_counter()
    model, stats = train_budget_conditioned_selector(
        model,
        train_oracle.marginal_gain,
        budgets=train_budgets,
        episodes=a.episodes,
        learning_rate=a.learning_rate,
        gamma=a.gamma,
        n_step=a.n_step,
        batch_size=a.batch_size,
        warmup=a.warmup,
        target_update_interval=a.target_update,
        random_seed=a.seed,
        device=device,
    )
    training_seconds = time.perf_counter() - t0

    eval_oracle = LiveEdgeECOracle(inst, mc_times=a.evaluation_mc, random_seed=a.seed + 100003)
    evaluation: dict[str, float] = {}
    realized_evaluation: dict[str, float] = {}
    selections: dict[str, list[int]] = {}
    selection_seconds: dict[str, float] = {}
    for k in eval_budgets:
        t0 = time.perf_counter()
        seeds = greedy_select_for_budget(model, k, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        selection_seconds[str(k)] = time.perf_counter() - t0
        selections[str(k)] = seeds
        evaluation[str(k)] = float(eval_oracle.score(seeds))
        realized_evaluation[str(k)] = float(eval_oracle.realized_score(seeds))
        print(
            f"k={k} EC={evaluation[str(k)]:.6f} realized_EC={realized_evaluation[str(k)]:.6f} "
            f"selection_s={selection_seconds[str(k)]:.6f}"
        )

    torch.save(model.state_dict(), out / "model.pt")
    result = {
        "profile": profile,
        "training_mode": "budget_conditioned_balanced_multibudget",
        "instance": str(Path(a.instance).resolve()),
        "embeddings": str(Path(a.embeddings).resolve()),
        "instance_config": manifest.get("config", {}),
        "device": device,
        "episodes": a.episodes,
        "train_budgets": train_budgets,
        "episode_budgets": stats.episode_budgets,
        "training_transitions": int(sum(stats.episode_budgets)),
        "eval_budgets": eval_budgets,
        "oracle_mc": a.oracle_mc,
        "evaluation_mc": a.evaluation_mc,
        "oracle_build_seconds": oracle_seconds,
        "training_seconds": training_seconds,
        "online_selection_seconds": selection_seconds,
        "gamma": a.gamma,
        "n_step": a.n_step,
        "hidden": a.hidden,
        "atoms": a.atoms,
        "learning_rate": a.learning_rate,
        "episode_returns": stats.episode_returns,
        "losses": stats.losses,
        "evaluation": evaluation,
        "realized_evaluation": realized_evaluation,
        "selections": selections,
        "train_oracle_stats": train_oracle.stats.__dict__,
        "eval_oracle_stats": eval_oracle.stats.__dict__,
        "note": (
            "Scientific-v4 retains the scientific-v2 EC objective and instances but conditions "
            "the policy state on target budget and remaining horizon. This removes the state aliasing "
            "introduced by mixed-budget training while preserving independent train/evaluation live-edge worlds."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"saved {out} transitions={result['training_transitions']} "
        f"train_seconds={training_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
