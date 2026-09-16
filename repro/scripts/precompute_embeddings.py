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

from dqnselector.embedding import fit_social_influence_embedding_algorithm1
from dqnselector.piic import piic
from dqnselector.reconstruction import load_reconstruction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--social-dim", type=int, default=128)
    p.add_argument("--mc", type=int, default=100)
    p.add_argument("--graph-iterations", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--piic-range", type=int, default=2)
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    inst, manifest = load_reconstruction(a.instance)

    t0 = time.perf_counter()
    social_s, social_t, loss_trace = fit_social_influence_embedding_algorithm1(
        inst.graph,
        dim=a.social_dim,
        mc_times=a.mc,
        graph_iterations=a.graph_iterations,
        learning_rate=a.learning_rate,
        random_seed=a.seed,
    )
    social_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    coverage_r = piic(
        inst.graph,
        inst.quality,
        influence_range=a.piic_range,
        nodes=inst.nodes,
    ).astype(np.float32)
    piic_seconds = time.perf_counter() - t0

    np.savez_compressed(
        out / "embeddings.npz",
        social_s=social_s,
        social_t=social_t,
        coverage_r=coverage_r,
        loss_trace=np.asarray(loss_trace, dtype=np.float64),
    )
    metadata = {
        "instance": str(Path(a.instance).resolve()),
        "instance_config": manifest.get("config", {}),
        "social_dim": a.social_dim,
        "mc_times": a.mc,
        "graph_iterations": a.graph_iterations,
        "learning_rate": a.learning_rate,
        "piic_range": a.piic_range,
        "random_seed": a.seed,
        "social_seconds": social_seconds,
        "piic_seconds": piic_seconds,
        "loss_trace": loss_trace,
        "social_s_finite": bool(np.isfinite(social_s).all()),
        "coverage_r_finite": bool(np.isfinite(coverage_r).all()),
        "coverage_r_min": float(np.nanmin(coverage_r)),
        "coverage_r_max": float(np.nanmax(coverage_r)),
        "warning": "PIIC influence range R is not numerically specified in the paper; this run records the chosen value explicitly.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
