from pathlib import Path

from dqnselector.reconstruction import ReconstructionConfig, reconstruct_instance


def test_reconstruction_on_tiny_snap_format(tmp_path: Path):
    edge_file = tmp_path / "edges.txt"
    checkin_file = tmp_path / "checkins.txt"
    edge_file.write_text("0\t1\n1\t2\n2\t3\n0\t3\n", encoding="utf-8")
    rows = []
    coords = [
        (34.0, -122.0),
        (35.0, -121.0),
        (36.0, -120.0),
        (37.0, -119.0),
    ]
    for user, (lat, lon) in enumerate(coords):
        rows.append(f"{user}\t2010-01-01T00:00:00Z\t{lat}\t{lon}\tloc{user}\n")
        rows.append(f"{user}\t2010-01-02T00:00:00Z\t{lat + 0.01}\t{lon + 0.01}\tloc{user}b\n")
    checkin_file.write_text("".join(rows), encoding="utf-8")
    cfg = ReconstructionConfig(
        dataset_name="tiny",
        n_users=4,
        worker_pool_size=2,
        n_target_subareas=3,
        lon_cells=4,
        lat_cells=4,
        random_seed=9,
        demand_mode="max_normalized",
    )
    inst, manifest = reconstruct_instance(edge_file, checkin_file, cfg)
    assert inst.participation.shape == (4, 3)
    assert inst.quality.shape == (4, 3)
    assert inst.demand.shape == (3,)
    assert len(inst.worker_pool) == 2
    assert inst.graph.number_of_nodes() == 4
    assert manifest["config"]["random_seed"] == 9
