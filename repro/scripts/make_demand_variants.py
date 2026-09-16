#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.ecm import ECMInstance
from dqnselector.reconstruction import load_reconstruction, save_reconstruction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--modes", nargs="+", default=["raw_checkin_count", "sqrt_count", "max_normalized"])
    p.add_argument("--scale", type=float, default=1.0)
    return p.parse_args()


def demand(counts: np.ndarray, mode: str, scale: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if mode == "raw_checkin_count":
        return np.maximum(counts, 1.0) * scale
    if mode == "sqrt_count":
        return np.maximum(np.sqrt(np.maximum(counts, 1.0)) * scale, 1e-12)
    if mode == "max_normalized":
        denom = max(float(counts.max()), 1.0)
        return 1.0 + counts / denom * scale
    raise ValueError(mode)


def main():
    a = parse_args()
    inst, manifest = load_reconstruction(a.source)
    counts = np.asarray(manifest["target_checkin_counts"], dtype=np.float64)
    root = Path(a.output_root)
    for mode in a.modes:
        new_manifest = deepcopy(manifest)
        new_manifest["config"]["demand_mode"] = mode
        new_manifest["config"]["demand_scale"] = a.scale
        new_manifest.setdefault("notes", []).append(
            f"Demand sensitivity variant derived from the same sampled graph/targets using mode={mode}, scale={a.scale}."
        )
        new_inst = ECMInstance(
            graph=inst.graph.copy(),
            nodes=list(inst.nodes),
            participation=inst.participation.copy(),
            quality=inst.quality.copy(),
            demand=demand(counts, mode, a.scale),
            worker_pool=set(inst.worker_pool or set()),
        )
        out = root / mode
        save_reconstruction(new_inst, new_manifest, out)
        print(mode, "demand", float(new_inst.demand.min()), float(new_inst.demand.mean()), float(new_inst.demand.max()), "->", out)


if __name__ == "__main__":
    main()
