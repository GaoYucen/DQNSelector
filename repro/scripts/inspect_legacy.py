#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "repro"
sys.path.insert(0, str(REPRO / "src"))

from dqnselector.legacy import load_legacy_processed_instance, legacy_effective_coverage
from dqnselector.baselines import degree_greedy
from dqnselector.piic import piic


def main() -> None:
    data_dir = ROOT / "dataset" / "data_1"
    node_file = data_dir / "input_node_3000_3.txt"
    edge_file = data_dir / "input_edge_3000_3.txt"
    if not node_file.exists() or not edge_file.exists():
        raise SystemExit(f"legacy files not found under {data_dir}")
    inst = load_legacy_processed_instance(node_file, edge_file, n_subareas=100)
    print(f"nodes={inst.graph.number_of_nodes()} edges={inst.graph.number_of_edges()} subareas={inst.n_subareas}")
    print(f"node_values min={inst.node_values.min():.6g} max={inst.node_values.max():.6g} mean={inst.node_values.mean():.6g}")
    k = min(50, inst.graph.number_of_nodes())
    seeds = degree_greedy(inst.graph, k)
    uncapped, raw = legacy_effective_coverage(inst, seeds, clip_per_subarea=False)
    capped, _ = legacy_effective_coverage(inst, seeds, clip_per_subarea=True)
    print(f"degree-greedy k={k} legacy_mean_uncapped={uncapped:.6f} diagnostic_capped={capped:.6f}")
    print(f"raw subarea coverage range=[{raw.min():.6f}, {raw.max():.6f}]")
    r = piic(inst.graph, inst.node_values, influence_range=2, nodes=inst.nodes)
    print(f"PIIC(R=2) shape={r.shape} finite={bool(np.isfinite(r).all())} range=[{np.nanmin(r):.6g}, {np.nanmax(r):.6g}]")
    print("NOTE: legacy node vectors cannot be uniquely decomposed into paper p_v^i, q_v^i, d_i from released files alone.")


if __name__ == "__main__":
    main()
