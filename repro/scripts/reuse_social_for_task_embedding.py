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

from dqnselector.objective_features import objective_aware_coverage_embedding
from dqnselector.reconstruction import load_reconstruction


def parse_args():
    p = argparse.ArgumentParser(
        description="Reuse a fixed-structure social embedding and recompute task-side coverage"
    )
    p.add_argument("--instance", required=True)
    p.add_argument("--source-embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--piic-range", type=int, default=2)
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    inst, manifest = load_reconstruction(a.instance)
    if manifest.get("profile") != "journal_scientific_v2":
        raise ValueError("this helper is only for journal_scientific_v2")

    source_path = Path(a.source_embeddings)
    source = np.load(source_path)
    social_s = np.asarray(source["social_s"])
    social_t = np.asarray(source["social_t"])
    loss_trace = np.asarray(source["loss_trace"])
    n = len(inst.nodes)
    if social_s.shape[0] != n or social_t.shape[0] != n:
        raise ValueError("source social embedding node count does not match target instance")

    t0 = time.perf_counter()
    coverage_r = objective_aware_coverage_embedding(
        inst,
        influence_range=a.piic_range,
        include_direct=True,
    ).astype(np.float32)
    coverage_seconds = time.perf_counter() - t0

    np.savez_compressed(
        out / "embeddings.npz",
        social_s=social_s,
        social_t=social_t,
        coverage_r=coverage_r,
        loss_trace=loss_trace,
    )
    metadata = {
        "instance": str(Path(a.instance).resolve()),
        "profile": manifest.get("profile"),
        "structure_seed": manifest.get("config", {}).get("structure_seed"),
        "task_seed": manifest.get("config", {}).get("task_seed"),
        "social_embedding_reused_from": str(source_path.resolve()),
        "coverage_feature": "objective-contribution",
        "coverage_includes_direct_term": True,
        "piic_range": a.piic_range,
        "coverage_seconds": coverage_seconds,
        "social_s_finite": bool(np.isfinite(social_s).all()),
        "coverage_r_finite": bool(np.isfinite(coverage_r).all()),
        "coverage_r_min": float(np.min(coverage_r)),
        "coverage_r_mean": float(np.mean(coverage_r)),
        "coverage_r_max": float(np.max(coverage_r)),
        "note": (
            "Social embedding is intentionally reused because structure_seed fixes sampled users, "
            "social topology and edge probabilities. Only task-side objective-aware coverage is recomputed."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
