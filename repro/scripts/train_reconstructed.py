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

from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction
from dqnselector.selector import RainbowSelector, greedy_select, train_rainbow_selector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--train-budget", type=int, default=50)
    p.add_argument("--eval-budgets", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    p.add_argument("--oracle-mc", type=int, default=100)
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
    emb = np.load(a.embeddings)
    social = emb["social_s"]
    coverage = emb["coverage_r"]
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    worker_pool_size = len(inst.worker_pool or set())
    if not 0 < a.train_budget <= worker_pool_size:
        raise ValueError(f"train budget must be in [1,{worker_pool_size}]")
    eval_budgets = sorted({k for k in a.eval_budgets if 0 < k <= worker_pool_size})
    if not eval_budgets:
        raise ValueError("no valid evaluation budgets")

    t0 = time.perf_counter()
    train_oracle = LiveEdgeECOracle(inst, mc_times=a.oracle_mc, random_seed=a.seed)
    oracle_seconds = time.perf_counter() - t0

    model = RainbowSelector(
        social,
        coverage,
        worker_pool=inst.worker_pool,
        hidden_dim=a.hidden,
        atoms=a.atoms,
        v_min=0.0,
        v_max=1.0,
    )

    t0 = time.perf_counter()
    model, stats = train_rainbow_selector(
        model,
        train_oracle.marginal_gain,
        seed_budget=a.train_budget,
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

    # Independent live-edge worlds avoid evaluating on the stochastic worlds
    # that generated the training rewards.
    eval_oracle = LiveEdgeECOracle(
        inst,
        mc_times=max(a.oracle_mc, 100),
        random_seed=a.seed + 100003,
    )
    max_budget = max(eval_budgets)
    selection_order = greedy_select(model, max_budget, device=device)
    evaluation = {}
    for k in eval_budgets:
        evaluation[str(k)] = float(eval_oracle.score(selection_order[:k]))
        print(f"k={k} EC={evaluation[str(k)]:.6f}")

    torch.save(model.state_dict(), out / "model.pt")
    np.save(out / "selection_order.npy", np.asarray(selection_order, dtype=np.int64))
    result = {
        "profile": profile,
        "instance": str(Path(a.instance).resolve()),
        "embeddings": str(Path(a.embeddings).resolve()),
        "instance_config": manifest.get("config", {}),
        "demand_metadata": manifest.get("demand", {}),
        "influence_metadata": manifest.get("influence", {}),
        "device": device,
        "episodes": a.episodes,
        "train_budget": a.train_budget,
        "eval_budgets": eval_budgets,
        "oracle_mc": a.oracle_mc,
        "oracle_build_seconds": oracle_seconds,
        "training_seconds": training_seconds,
        "gamma": a.gamma,
        "n_step": a.n_step,
        "hidden": a.hidden,
        "atoms": a.atoms,
        "learning_rate": a.learning_rate,
        "episode_returns": stats.episode_returns,
        "losses": stats.losses,
        "evaluation": evaluation,
        "selection_order": selection_order,
        "train_oracle_stats": train_oracle.stats.__dict__,
        "eval_oracle_stats": eval_oracle.stats.__dict__,
        "note": (
            "journal_scientific_v1 is a calibrated journal experiment profile, not an exact "
            "conference numerical reproduction. Training and evaluation use independent "
            "live-edge worlds."
            if profile == "journal_scientific_v1"
            else "Transparent reconstruction run; preprocessing assumptions are recorded explicitly."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
