#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.reconstruction import load_reconstruction


def parse_args():
    p = argparse.ArgumentParser(description="Validate two v2 task instances share structure")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    return p.parse_args()


def edge_rows(inst):
    return sorted((int(u), int(v), float(d.get("weight", 1.0))) for u, v, d in inst.graph.edges(data=True))


def main():
    a = parse_args()
    ia, ma = load_reconstruction(a.a)
    ib, mb = load_reconstruction(a.b)
    assert ma.get("profile") == mb.get("profile") == "journal_scientific_v2"
    ca, cb = ma["config"], mb["config"]
    assert ca["structure_seed"] == cb["structure_seed"]
    assert ma["sampled_original_user_ids"] == mb["sampled_original_user_ids"]
    assert sorted(ia.worker_pool or set()) == sorted(ib.worker_pool or set())
    assert edge_rows(ia) == edge_rows(ib)
    assert ma["target_cells_xy"] != mb["target_cells_xy"]
    assert np.allclose(ia.quality, 1.0) and np.allclose(ib.quality, 1.0)
    assert ma["targets"]["all_have_historical_activity"]
    assert mb["targets"]["all_have_historical_activity"]
    result = {
        "profile": "journal_scientific_v2",
        "structure_seed": ca["structure_seed"],
        "task_seed_a": ca["task_seed"],
        "task_seed_b": cb["task_seed"],
        "nodes": ia.graph.number_of_nodes(),
        "edges": ia.graph.number_of_edges(),
        "worker_pool": len(ia.worker_pool or set()),
        "same_structure": True,
        "different_targets": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
