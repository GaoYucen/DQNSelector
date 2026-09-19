#!/usr/bin/env python3
"""One auditable journal-instance run: repaired baselines, frozen DQN, and DQN-J."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
from dqnselector.baselines import celf, degree_greedy, one_step_coverage_greedy
from dqnselector.journal_protocol import BUDGETS
from dqnselector.journal_selector import JournalSelector, build_teacher_states, journal_greedy_select, train_journal_selector
from dqnselector.multibudget import train_rainbow_selector_multibudget
from dqnselector.objective_features import objective_aware_coverage_embedding
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.paper_baselines import fast_selector, kt_voting, mobility_profiles
from dqnselector.piano import PianoQNet, piano_select, train_piano_on_subgraphs
from dqnselector.reconstruction import load_reconstruction
from dqnselector.selector import RainbowSelector, greedy_select


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def cloned_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--checkins", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dqn-episodes", type=int, default=200)
    parser.add_argument("--journal-episodes", type=int, default=400)
    parser.add_argument("--journal-hidden", type=int, default=128)
    parser.add_argument("--ranking-weight", type=float, default=.3)
    parser.add_argument("--ranking-temperature", type=float, default=.1)
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--piano-episodes", type=int, default=200)
    parser.add_argument("--skip-piano", action="store_true")
    return parser.parse_args()


def timed_selection(selector, budget: int, device: str, repeats: int = 30, reset=None):
    selector(budget)  # warmup
    elapsed = []
    order = None
    for _ in range(repeats):
        if reset is not None:
            reset()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter(); order = selector(budget)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    return order, float(np.median(elapsed))


def main():
    args = parse_args()
    device = args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    instance_path, embedding_path = Path(args.instance), Path(args.embeddings)
    inst, manifest = load_reconstruction(instance_path)
    emb = np.load(embedding_path)
    pool = sorted(inst.worker_pool or set())
    if max(BUDGETS) > len(pool):
        raise ValueError("worker pool is too small for journal budgets")
    final = LiveEdgeECOracle(inst, mc_times=2000, random_seed=900000 + args.training_seed)
    validation = LiveEdgeECOracle(inst, mc_times=500, random_seed=800000 + args.training_seed)
    selection = LiveEdgeECOracle(inst, mc_times=300, random_seed=700000 + args.training_seed)
    rows: list[dict] = []

    def record(method: str, order: list[int], selection_seconds: float, extra: dict | None = None):
        for budget in BUDGETS:
            chosen = order[:budget]
            if len(chosen) != budget or len(set(chosen)) != budget or not set(chosen) <= set(pool):
                raise RuntimeError(f"{method} returned invalid selection at k={budget}")
            rows.append({"method": method, "k": budget, "ec": final.score(chosen), "realized_ec": final.realized_score(chosen),
                         "selection_seconds_median": selection_seconds, **(extra or {})})

    direct = inst.participation * inst.quality
    preprocess_start = time.perf_counter()
    profiles = mobility_profiles(args.checkins, manifest, pool)
    baseline_preprocess_seconds = time.perf_counter() - preprocess_start
    baseline_specs = {
        "DegGreedy": lambda: degree_greedy(inst.graph, max(BUDGETS), pool),
        "CovGreedy": lambda: one_step_coverage_greedy(inst.graph, direct, max(BUDGETS), inst.nodes, pool),
        "FastSelector-SIGIR-adapted": lambda: fast_selector(inst.graph, profiles, pool, max(BUDGETS), .56 if manifest["config"]["dataset_name"] == "gowalla" else .64),
        "KTVoting2-feasible": lambda: kt_voting(inst.graph, direct, pool, max(BUDGETS)),
        "CELF": lambda: celf(pool, max(BUDGETS), selection.marginal_gain),
    }
    selections = {}
    for method, fn in baseline_specs.items():
        reset = selection.clear_score_cache if method == "CELF" else None
        if reset is not None:
            reset()
        order, seconds = timed_selection(lambda _k: fn(), max(BUDGETS), device, reset=reset)
        selections[method] = order; record(method, order, seconds, {"implementation": "repaired-v1"})

    # Frozen DQNSelector baseline.
    torch.manual_seed(args.training_seed); np.random.seed(args.training_seed)
    old = RainbowSelector(emb["social_s"], emb["coverage_r"], worker_pool=pool)
    train_oracle = LiveEdgeECOracle(inst, mc_times=100, random_seed=args.training_seed)
    start = time.perf_counter()
    old, old_stats = train_rainbow_selector_multibudget(old, train_oracle.marginal_gain, BUDGETS, episodes=args.dqn_episodes,
                                                        random_seed=args.training_seed, device=device)
    old_training_seconds = time.perf_counter() - start
    old_order, old_seconds = timed_selection(lambda k: greedy_select(old, k, device), max(BUDGETS), device)
    record("DQNSelector", old_order, old_seconds, {"training_seconds": old_training_seconds, "checkpoint_version": "frozen-v3"})
    torch.save(old.state_dict(), out / "dqnselector-frozen.pt")

    # DQNSelector-J: footprints and teacher states use only train worlds.
    preprocess_start = time.perf_counter()
    footprint = objective_aware_coverage_embedding(inst, influence_range=2, include_direct=True).astype(np.float32)
    journal_preprocess_seconds = time.perf_counter() - preprocess_start
    teacher_orders = [selections["CELF"], old_order, np.random.default_rng(args.training_seed + 17).permutation(pool).tolist()]
    teacher_start = time.perf_counter()
    teachers = build_teacher_states(train_oracle, pool, BUDGETS, teacher_orders)
    teacher_seconds = time.perf_counter() - teacher_start
    journal = JournalSelector(emb["social_s"], footprint, pool, hidden_dim=args.journal_hidden)
    start = time.perf_counter()
    journal, journal_stats = train_journal_selector(journal, train_oracle.marginal_gain, BUDGETS, teachers,
        episodes=args.journal_episodes, ranking_weight=args.ranking_weight,
        ranking_temperature=args.ranking_temperature, seed=args.training_seed, device=device)
    journal_training_seconds = time.perf_counter() - start
    journal_order, journal_seconds = timed_selection(lambda k: journal_greedy_select(journal, k, device), max(BUDGETS), device)
    record("DQNSelector-J", journal_order, journal_seconds, {"training_seconds": journal_training_seconds, "checkpoint_version": journal.checkpoint_version})
    torch.save(journal.state_dict(), out / "dqnselector-j.pt")

    if args.run_ablations:
        ablations = (
            ("DQNSelector-J-residual-only", True, 0.0),
            ("DQNSelector-J-ranking-only", False, args.ranking_weight),
        )
        for method, residual_state, ranking_weight in ablations:
            torch.manual_seed(args.training_seed); np.random.seed(args.training_seed)
            ablation = JournalSelector(emb["social_s"], footprint, pool, hidden_dim=args.journal_hidden,
                                       use_residual_state=residual_state)
            start = time.perf_counter()
            ablation, stats = train_journal_selector(
                ablation, train_oracle.marginal_gain, BUDGETS, teachers,
                episodes=args.journal_episodes, ranking_weight=ranking_weight,
                ranking_temperature=args.ranking_temperature, seed=args.training_seed, device=device,
            )
            training_seconds = time.perf_counter() - start
            order, seconds = timed_selection(lambda k: journal_greedy_select(ablation, k, device), max(BUDGETS), device)
            record(method, order, seconds, {"training_seconds": training_seconds,
                                             "checkpoint_version": ablation.checkpoint_version,
                                             "ablation": True, "stats": stats})
            torch.save(ablation.state_dict(), out / f"{method.lower()}.pt")

    if not args.skip_piano:
        torch.manual_seed(args.training_seed)
        piano = PianoQNet(inst.graph, pool)
        checkpoints = {"initial": cloned_state(piano)}
        piano_trace = []
        def spread(order):
            return float(validation.activation_probability(order[:max(BUDGETS)]).sum())
        best = [spread(piano_select(piano, max(BUDGETS), device)), "initial"]
        def piano_progress(episode, total, seconds):
            order = piano_select(piano, max(BUDGETS), device); value = spread(order)
            piano_trace.append({"episode": episode, "training_return": total, "validation_influence": value,
                                "validation_ec": float(np.mean([validation.score(order[:k]) for k in BUDGETS]))})
            if value > best[0]:
                best[:] = [value, f"episode-{episode}"]; checkpoints[best[1]] = cloned_state(piano)
        piano, piano_stats = train_piano_on_subgraphs(piano, inst.graph, pool,
            lambda selected, node: float(train_oracle.activation_probability(set(selected) | {node}).sum() - train_oracle.activation_probability(selected).sum()),
            episodes=args.piano_episodes, seed=args.training_seed, device=device, progress=piano_progress)
        checkpoints["final"] = cloned_state(piano)
        torch.save(checkpoints["initial"], out / "piano-initial.pt"); torch.save(checkpoints["final"], out / "piano-final.pt")
        torch.save(checkpoints[best[1]], out / "piano-best-influence.pt")
        piano.load_state_dict(checkpoints[best[1]])
        piano_order, piano_seconds = timed_selection(lambda k: piano_select(piano, k, device), max(BUDGETS), device)
        record("PIANO", piano_order, piano_seconds, {"checkpoint": best[1], "training_updates": piano_stats["updates"], "implementation": "paper-equation-v1"})
        dump(out / "piano-metrics.json", {**piano_stats, "best_validation_influence": best[0], "best_checkpoint": best[1], "trace": piano_trace})

    meta = {"dataset": manifest["config"]["dataset_name"], "users": inst.graph.number_of_nodes(),
            "instance": str(instance_path.resolve()), "instance_sha256": sha256(instance_path / "instance.npz"),
            "embeddings": str(embedding_path.resolve()), "embedding_sha256": sha256(embedding_path), "training_seed": args.training_seed,
            "manifest": manifest.get("journal_scenario", {}), "budgets": BUDGETS, "train_mc": 100, "selection_mc": 300,
            "validation_mc": 500, "final_mc": 2000, "device": device, "dqn_stats": old_stats.__dict__, "journal_stats": journal_stats,
            "timing": {"baseline_preprocess_seconds": baseline_preprocess_seconds,
                       "journal_preprocess_seconds": journal_preprocess_seconds,
                       "teacher_label_seconds": teacher_seconds}}
    dump(out / "metadata.json", meta); dump(out / "results.json", rows)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
