from pathlib import Path

import numpy as np

from dqnselector.scientific import ScientificReconstructionConfig, reconstruct_scientific_instance


def test_scientific_reconstruction_decouples_quantities(tmp_path: Path):
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
    for user, (lat, lon) in enumerate(coords):
        # Different history sizes make the activity-quality proxy observable.
        for j in range(user + 1):
            rows.append(
                f"{user}\t2010-01-{j + 1:02d}T00:00:00Z\t"
                f"{lat + 0.005 * j}\t{lon + 0.005 * j}\tloc{user}_{j}\n"
            )
    checkin_file.write_text("".join(rows), encoding="utf-8")

    cfg = ScientificReconstructionConfig(
        dataset_name="tiny",
        n_users=4,
        worker_pool_size=3,
        n_target_subareas=3,
        lon_cells=4,
        lat_cells=4,
        random_seed=7,
        load_factor=0.5,
    )
    inst, manifest = reconstruct_scientific_instance(edge_file, checkin_file, cfg)

    assert manifest["profile"] == "journal_scientific_v1"
    assert inst.participation.shape == (4, 3)
    assert inst.quality.shape == (4, 3)
    assert np.all((inst.participation >= 0.0) & (inst.participation <= 1.0))
    assert np.all((inst.quality >= cfg.quality_floor) & (inst.quality <= 1.0))
    assert not np.allclose(inst.participation, inst.quality)
    assert np.all(inst.demand > 0.0)
    assert len(inst.worker_pool or set()) == 3
    assert {round(float(d["weight"]), 2) for _, _, d in inst.graph.edges(data=True)} <= {
        0.01,
        0.05,
        0.10,
    }
    assert manifest["demand"]["load_factor"] == 0.5
