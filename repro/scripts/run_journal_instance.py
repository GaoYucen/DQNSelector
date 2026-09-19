#!/usr/bin/env python3
"""Run one reproducible journal experiment unit.

``baseline`` runs once per instance; ``trained`` runs once per neural seed.
The shared instance hash permits paired aggregation without repeated baselines.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
from dqnselector.baselines import celf, degree_greedy, one_step_coverage_greedy
from dqnselector.journal_protocol import BUDGETS, instance_world_seed
from dqnselector.journal_selector import JournalSelector, build_teacher_states, journal_greedy_select, train_journal_selector
from dqnselector.multibudget import train_rainbow_selector_multibudget
from dqnselector.objective_features import objective_aware_coverage_embedding
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.paper_baselines import fast_selector, kt_voting, mobility_profiles
from dqnselector.piano import PianoQNet, piano_select, train_piano_on_subgraphs
from dqnselector.reconstruction import load_reconstruction
from dqnselector.selector import RainbowSelector, greedy_select


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("baseline", "trained"), required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--checkins", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dqn-episodes", type=int, default=200)
    parser.add_argument("--journal-episodes", type=int, default=400)
    parser.add_argument("--journal-hidden", type=int, default=128)
    parser.add_argument("--ranking-weight", type=float, default=.3)
    parser.add_argument("--ranking-temperature", type=float, default=.1)
    parser.add_argument("--piano-episodes", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--train-mc", type=int, default=100)
    parser.add_argument("--selection-mc", type=int, default=300)
    parser.add_argument("--validation-mc", type=int, default=500)
    parser.add_argument("--evaluation-mc", type=int, default=2000)
    parser.add_argument("--timing-repeats", type=int, default=30)
    parser.add_argument("--teacher-states-per-order", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=(
        "DegGreedy", "CovGreedy", "FastSelector-SIGIR-adapted",
        "KTVoting2-feasible", "CELF", "DQNSelector", "DQNSelector-J", "PIANO",
    ))
    parser.add_argument("--run-ablations", action="store_true")
    return parser.parse_args()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_per_budget(select, budgets: list[int], device: str, repeats: int = 30):
    orders, medians = {}, {}
    for budget in budgets:
        if repeats == 0:
            synchronize(device); start = time.perf_counter(); order = select(budget)
            synchronize(device)
            orders[budget], medians[budget] = order, float(time.perf_counter() - start)
            continue
        select(budget)
        elapsed, order = [], None
        for _ in range(repeats):
            synchronize(device); start = time.perf_counter(); order = select(budget)
            synchronize(device); elapsed.append(time.perf_counter() - start)
        orders[budget], medians[budget] = order, float(np.median(elapsed))
    return orders, medians


def validate_orders(method: str, orders: dict[int, list[int]], pool: set[int], budgets: list[int]) -> None:
    if set(orders) != set(budgets):
        raise RuntimeError(f"{method} returned budgets {sorted(orders)}, expected {budgets}")
    for budget, order in orders.items():
        if len(order) != budget or len(set(order)) != budget or not set(order) <= pool:
            raise RuntimeError(f"{method} returned an invalid selection for k={budget}")


def complete(out: Path, config_hash: str, expected_rows: int) -> bool:
    marker, metadata, result = out / "complete.json", out / "metadata.json", out / "results.json"
    if not all(path.exists() for path in (marker, metadata, result)):
        return False
    try:
        return (json.loads(marker.read_text())["config_hash"] == config_hash
                and json.loads(marker.read_text())["rows"] == expected_rows
                and len(json.loads(result.read_text())) == expected_rows
                and json.loads(metadata.read_text())["config_hash"] == config_hash)
    except (KeyError, json.JSONDecodeError):
        return False


def write_results(out: Path, metadata: dict, rows: list[dict], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise RuntimeError(f"incomplete result rows: expected {expected_rows}, got {len(rows)}")
    atomic_json(out / "metadata.json", metadata)
    atomic_json(out / "results.json", rows)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    atomic_json(out / "complete.json", {"completed": True, "rows": expected_rows, "config_hash": metadata["config_hash"]})


def main():
    args = parse_args()
    if args.role == "trained" and args.training_seed is None:
        raise ValueError("--training-seed is required for role=trained")
    if args.role == "baseline" and args.training_seed is not None:
        raise ValueError("baseline is instance-level and must not have a training seed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    instance_path, embedding_path, checkin_path = Path(args.instance), Path(args.embeddings), Path(args.checkins)
    instance_sha, embedding_sha, checkin_sha = file_hash(instance_path / "instance.npz"), file_hash(embedding_path), file_hash(checkin_path)
    budgets = sorted(set(int(value) for value in args.budgets))
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("--budgets must contain positive integers")
    if args.timing_repeats < 0:
        raise ValueError("--timing-repeats must be nonnegative")
    for name in ("train_mc", "selection_mc", "validation_mc", "evaluation_mc", "teacher_states_per_order"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    baseline_methods = ["DegGreedy", "CovGreedy", "FastSelector-SIGIR-adapted", "KTVoting2-feasible", "CELF"]
    trained_methods = ["DQNSelector", "DQNSelector-J", "PIANO"]
    methods = list(dict.fromkeys(args.methods or (baseline_methods if args.role == "baseline" else trained_methods)))
    allowed = set(baseline_methods if args.role == "baseline" else trained_methods)
    if not set(methods) <= allowed:
        raise ValueError(f"methods {methods} do not belong to role={args.role}")
    if args.role == "trained" and "DQNSelector-J" in methods and "DQNSelector" not in methods:
        raise ValueError("DQNSelector-J quick training requires DQNSelector for its teacher trajectory")
    config = {**vars(args), "instance_sha256": instance_sha, "embedding_sha256": embedding_sha,
              "checkin_sha256": checkin_sha, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "budgets": budgets, "selection_mc": args.selection_mc, "validation_mc": args.validation_mc,
              "selection_timing_repeats": args.timing_repeats, "methods": methods}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    expected_methods = len(methods) + (2 if args.role == "trained" and args.run_ablations else 0)
    expected_rows = expected_methods * len(budgets)
    if complete(out, config_hash, expected_rows):
        print(f"JOURNAL_REUSED role={args.role} output={out}")
        return

    inst, manifest = load_reconstruction(instance_path)
    emb = np.load(embedding_path)
    pool = set(inst.worker_pool or set())
    pool_list = sorted(pool)
    if max(budgets) > len(pool):
        raise ValueError("worker pool is too small for journal budgets")
    final_seed = instance_world_seed(instance_sha, f"evaluation-{args.evaluation_mc}")
    selection_seed = instance_world_seed(instance_sha, f"celf-selection-{args.selection_mc}")
    validation_seed = instance_world_seed(instance_sha, f"validation-{args.validation_mc}")
    final = LiveEdgeECOracle(inst, mc_times=args.evaluation_mc, random_seed=final_seed)
    rows, timing = [], {}

    def record(method: str, orders: dict[int, list[int]], times: dict[int, float], extra: dict | None = None):
        validate_orders(method, orders, pool, budgets)
        for budget in budgets:
            selected = orders[budget]
            rows.append({"method": method, "k": budget, "ec": final.score(selected), "realized_ec": final.realized_score(selected),
                         "selection_seconds_median": times[budget], **(extra or {})})

    if args.role == "baseline":
        print(f"JOURNAL_STAGE baseline_start methods={','.join(methods)} budgets={budgets}", flush=True)
        start = time.perf_counter(); profiles = mobility_profiles(checkin_path, manifest, pool_list); timing["mobility_preprocess_seconds"] = time.perf_counter() - start
        selection = LiveEdgeECOracle(inst, mc_times=args.selection_mc, random_seed=selection_seed)
        direct = inst.participation * inst.quality
        specs = {
            "DegGreedy": lambda k: degree_greedy(inst.graph, k, pool_list),
            "CovGreedy": lambda k: one_step_coverage_greedy(inst.graph, direct, k, inst.nodes, pool_list),
            "FastSelector-SIGIR-adapted": lambda k: fast_selector(inst.graph, profiles, pool_list, k, .56 if manifest["config"]["dataset_name"] == "gowalla" else .64),
            "KTVoting2-feasible": lambda k: kt_voting(inst.graph, direct, pool_list, k),
            "CELF": lambda k: celf(pool_list, k, selection.marginal_gain),
        }
        for method, select in specs.items():
            if method not in methods:
                continue
            print(f"JOURNAL_STAGE method_start method={method}", flush=True)
            if method == "CELF":
                def clear_then_select(k, _select=select):
                    selection.clear_score_cache(); return _select(k)
                orders, times = time_per_budget(clear_then_select, budgets, args.device, args.timing_repeats)
            else:
                orders, times = time_per_budget(select, budgets, args.device, args.timing_repeats)
            record(method, orders, times, {"implementation": "journal-repaired-v1"})
            print(f"JOURNAL_STAGE method_done method={method}", flush=True)
    else:
        seed = int(args.training_seed)
        train = LiveEdgeECOracle(inst, mc_times=args.train_mc, random_seed=seed)
        validation = LiveEdgeECOracle(inst, mc_times=args.validation_mc, random_seed=validation_seed)
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"JOURNAL_STAGE method_start method=DQNSelector episodes={args.dqn_episodes}", flush=True)
        old = RainbowSelector(emb["social_s"], emb["coverage_r"], worker_pool=pool)
        start = time.perf_counter(); old, old_stats = train_rainbow_selector_multibudget(old, train.marginal_gain, budgets, episodes=args.dqn_episodes, random_seed=seed, device=args.device)
        synchronize(args.device); timing["dqn_training_seconds"] = time.perf_counter() - start
        old_path = out / "dqnselector-frozen.pt"; torch.save(old.state_dict(), old_path)
        start = time.perf_counter(); old_loaded = RainbowSelector(emb["social_s"], emb["coverage_r"], worker_pool=pool).to(args.device)
        old_loaded.load_state_dict(torch.load(old_path, map_location=args.device, weights_only=True)); synchronize(args.device); timing["dqn_model_load_seconds"] = time.perf_counter() - start
        old_select = lambda k: greedy_select(old_loaded, k, args.device)
        old_orders, old_times = time_per_budget(old_select, budgets, args.device, args.timing_repeats)
        if "DQNSelector" in methods:
            record("DQNSelector", old_orders, old_times, {"training_updates": len(old_stats.losses), "checkpoint_version": "frozen-v3"})
        print(f"JOURNAL_STAGE method_done method=DQNSelector updates={len(old_stats.losses)}", flush=True)

        start = time.perf_counter(); footprint = objective_aware_coverage_embedding(inst, influence_range=2, include_direct=True).astype(np.float32); timing["journal_preprocess_seconds"] = time.perf_counter() - start
        teacher_celf = celf(pool_list, max(budgets), train.marginal_gain)
        start = time.perf_counter(); teachers = build_teacher_states(
            train, pool_list, budgets,
            [teacher_celf, old_orders[max(budgets)], np.random.default_rng(seed + 17).permutation(pool_list).tolist()],
            states_per_order=args.teacher_states_per_order,
        ); timing["teacher_label_seconds"] = time.perf_counter() - start

        def run_journal(method: str, residual_state: bool, ranking_weight: float):
            print(f"JOURNAL_STAGE method_start method={method} episodes={args.journal_episodes}", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = JournalSelector(emb["social_s"], footprint, pool, hidden_dim=args.journal_hidden, use_residual_state=residual_state)
            start = time.perf_counter(); model, stats = train_journal_selector(model, train.marginal_gain, budgets, teachers, episodes=args.journal_episodes, ranking_weight=ranking_weight, ranking_temperature=args.ranking_temperature, seed=seed, device=args.device)
            synchronize(args.device); train_seconds = time.perf_counter() - start
            checkpoint = out / f"{method.lower()}.pt"; torch.save(model.state_dict(), checkpoint)
            start = time.perf_counter(); loaded = JournalSelector(emb["social_s"], footprint, pool, hidden_dim=args.journal_hidden, use_residual_state=residual_state).to(args.device)
            loaded.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True)); synchronize(args.device); load_seconds = time.perf_counter() - start
            j_select = lambda k: journal_greedy_select(loaded, k, args.device)
            j_orders, j_times = time_per_budget(j_select, budgets, args.device, args.timing_repeats)
            record(method, j_orders, j_times, {"training_seconds": train_seconds, "model_load_seconds": load_seconds, "training_updates": stats["updates"], "checkpoint_version": loaded.checkpoint_version})
            print(f"JOURNAL_STAGE method_done method={method} updates={stats['updates']}", flush=True)

        if "DQNSelector-J" in methods:
            run_journal("DQNSelector-J", True, args.ranking_weight)
        if args.run_ablations:
            run_journal("DQNSelector-J-residual-only", True, 0.0)
            run_journal("DQNSelector-J-ranking-only", False, args.ranking_weight)

        if "PIANO" not in methods:
            metadata = {"config": config, "config_hash": config_hash, "dataset": manifest["config"]["dataset_name"], "users": inst.graph.number_of_nodes(), "instance_sha256": instance_sha, "embedding_sha256": embedding_sha, "checkin_sha256": checkin_sha, "manifest": manifest.get("journal_scenario", {}), "timing": timing, "worlds": {"final": {"mc": args.evaluation_mc, "seed": final_seed}, "selection": {"mc": args.selection_mc, "seed": selection_seed}, "validation": {"mc": args.validation_mc, "seed": validation_seed}, "train": {"mc": args.train_mc, "seed": args.training_seed}}}
            write_results(out, metadata, rows, expected_rows)
            print(f"JOURNAL_COMPLETE role={args.role} output={out} rows={len(rows)}")
            return

        print(f"JOURNAL_STAGE method_start method=PIANO episodes={args.piano_episodes}", flush=True)
        torch.manual_seed(seed); piano = PianoQNet(inst.graph, pool)
        checkpoints = {"initial": {key: value.detach().cpu().clone() for key, value in piano.state_dict().items()}}
        best = [-np.inf, "initial"]
        def piano_progress(episode, total, seconds):
            order = piano_select(piano, max(budgets), args.device)
            value = float(np.mean([validation.score(order[:k]) for k in budgets]))
            if value > best[0]:
                best[:] = [value, f"episode-{episode}"]; checkpoints[best[1]] = {key: value.detach().cpu().clone() for key, value in piano.state_dict().items()}
        start = time.perf_counter(); piano, piano_stats = train_piano_on_subgraphs(piano, inst.graph, pool, lambda selected, node: float(train.activation_probability(set(selected) | {node}).sum() - train.activation_probability(selected).sum()), episodes=args.piano_episodes, budget=max(budgets), seed=seed, device=args.device, progress=piano_progress)
        synchronize(args.device); timing["piano_training_seconds"] = time.perf_counter() - start
        checkpoints["final"] = {key: value.detach().cpu().clone() for key, value in piano.state_dict().items()}
        for name, state in checkpoints.items(): torch.save(state, out / f"piano-{name}.pt")
        piano.load_state_dict(checkpoints[best[1]])
        piano_select_fn = lambda k: piano_select(piano, k, args.device)
        piano_orders, piano_times = time_per_budget(piano_select_fn, budgets, args.device, args.timing_repeats)
        record("PIANO", piano_orders, piano_times, {"training_updates": piano_stats["updates"], "checkpoint": best[1], "validation_macro_ec": best[0], "implementation": "paper-equation-v1"})
        print(f"JOURNAL_STAGE method_done method=PIANO updates={piano_stats['updates']}", flush=True)

    metadata = {"config": config, "config_hash": config_hash, "dataset": manifest["config"]["dataset_name"], "users": inst.graph.number_of_nodes(), "instance_sha256": instance_sha, "embedding_sha256": embedding_sha, "checkin_sha256": checkin_sha, "manifest": manifest.get("journal_scenario", {}), "timing": timing, "worlds": {"final": {"mc": args.evaluation_mc, "seed": final_seed}, "selection": {"mc": args.selection_mc, "seed": selection_seed}, "validation": {"mc": args.validation_mc, "seed": validation_seed}, "train": {"mc": args.train_mc, "seed": args.training_seed}}}
    write_results(out, metadata, rows, expected_rows)
    print(f"JOURNAL_COMPLETE role={args.role} output={out} rows={len(rows)}")


if __name__ == "__main__":
    main()
