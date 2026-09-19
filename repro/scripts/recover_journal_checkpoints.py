#!/usr/bin/env python3
"""Evaluate completed checkpoints left by an interrupted journal run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
from dqnselector.journal_protocol import instance_world_seed
from dqnselector.journal_selector import JournalSelector, journal_greedy_select
from dqnselector.objective_features import objective_aware_coverage_embedding
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.reconstruction import load_reconstruction
from dqnselector.selector import RainbowSelector, greedy_select


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--evaluation-mc", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    instance_path = Path(args.instance)
    checkpoint_dir = Path(args.checkpoint_dir)
    inst, manifest = load_reconstruction(instance_path)
    embeddings = np.load(args.embeddings)
    pool = set(inst.worker_pool or set())
    instance_sha = sha256(instance_path / "instance.npz")
    seed = instance_world_seed(instance_sha, f"evaluation-{args.evaluation_mc}")
    oracle = LiveEdgeECOracle(inst, mc_times=args.evaluation_mc, random_seed=seed)
    rows = []

    old_path = checkpoint_dir / "dqnselector-frozen.pt"
    if old_path.exists():
        model = RainbowSelector(embeddings["social_s"], embeddings["coverage_r"], worker_pool=pool).to(args.device)
        model.load_state_dict(torch.load(old_path, map_location=args.device, weights_only=True))
        selected = greedy_select(model, args.budget, args.device)
        rows.append({"method": "DQNSelector", "k": args.budget, "ec": oracle.score(selected), "checkpoint": str(old_path)})

    journal_path = checkpoint_dir / "dqnselector-j.pt"
    if journal_path.exists():
        footprint = objective_aware_coverage_embedding(inst, influence_range=2, include_direct=True).astype(np.float32)
        model = JournalSelector(embeddings["social_s"], footprint, pool).to(args.device)
        model.load_state_dict(torch.load(journal_path, map_location=args.device, weights_only=True))
        selected = journal_greedy_select(model, args.budget, args.device)
        rows.append({"method": "DQNSelector-J", "k": args.budget, "ec": oracle.score(selected), "checkpoint": str(journal_path)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": manifest["config"]["dataset_name"],
        "users": inst.graph.number_of_nodes(),
        "instance_sha256": instance_sha,
        "evaluation_mc": args.evaluation_mc,
        "evaluation_seed": seed,
        "rows": rows,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
