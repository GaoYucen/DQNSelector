from pathlib import Path

import numpy as np

from dqnselector.scientific_v2 import (
    ScientificV2ReconstructionConfig,
    _frequency_aware_participation,
    reconstruct_scientific_v2_instance,
)


def test_frequency_aware_participation_distinguishes_visit_frequency():
    centers = np.array([[0.0, 0.0]], dtype=np.float64)
    frequent = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]], dtype=np.float64)
    rare = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1]], dtype=np.float64)
    p, tau, _ = _frequency_aware_participation([frequent, rare], centers, 0.5)
    assert tau > 0
    # Both users have a historical visit exactly at the target, but the repeated
    # nearby visitor should receive the larger future-availability proxy.
    assert p[0, 0] > p[1, 0]


def test_scientific_v2_main_protocol_and_task_seed_separation(tmp_path: Path):
    edge_file = tmp_path / "edges.txt"
    checkin_file = tmp_path / "checkins.txt"
    edge_file.write_text("0\t1\n1\t2\n2\t3\n0\t3\n", encoding="utf-8")

    coords = [
        (34.0, -122.0),
        (35.0, -121.0),
        (36.0, -120.0),
        (37.0, -119.0),
    ]
    rows = []
    # Give every user multiple active cells so there are enough target candidates.
    for user, (lat, lon) in enumerate(coords):
        for j in range(4):
            rows.append(
                f"{user}\t2010-01-{j + 1:02d}T00:00:00Z\t"
                f"{lat + 0.08 * j}\t{lon + 0.08 * j}\tloc{user}_{j}\n"
            )
    checkin_file.write_text("".join(rows), encoding="utf-8")

    base = dict(
        dataset_name="tiny",
        n_users=4,
        worker_pool_size=3,
        n_target_subareas=3,
        lon_cells=20,
        lat_cells=20,
        structure_seed=7,
        load_factor=0.5,
    )
    inst_a, manifest_a = reconstruct_scientific_v2_instance(
        edge_file, checkin_file, ScientificV2ReconstructionConfig(task_seed=7, **base)
    )
    inst_b, manifest_b = reconstruct_scientific_v2_instance(
        edge_file, checkin_file, ScientificV2ReconstructionConfig(task_seed=11, **base)
    )

    assert manifest_a["profile"] == "journal_scientific_v2"
    assert np.allclose(inst_a.quality, 1.0)
    assert manifest_a["targets"]["all_have_historical_activity"]
    assert manifest_a["quality"]["mode"] == "uniform"
    assert {round(float(d["weight"]), 3) for _, _, d in inst_a.graph.edges(data=True)} <= {
        0.001,
        0.01,
        0.1,
    }
    # Same structure seed must preserve users, graph weights, and worker pool.
    assert manifest_a["sampled_original_user_ids"] == manifest_b["sampled_original_user_ids"]
    assert manifest_a["worker_pool"] == manifest_b["worker_pool"]
    edges_a = sorted((u, v, d["weight"]) for u, v, d in inst_a.graph.edges(data=True))
    edges_b = sorted((u, v, d["weight"]) for u, v, d in inst_b.graph.edges(data=True))
    assert edges_a == edges_b
    # Task seed is allowed to change only task-side quantities.
    assert manifest_a["target_cells_xy"] != manifest_b["target_cells_xy"]
    assert inst_a.participation.shape == inst_b.participation.shape == (4, 3)
    assert np.all(inst_a.demand > 0) and np.all(inst_b.demand > 0)
