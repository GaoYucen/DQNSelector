from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path

import networkx as nx
import numpy as np

from .ecm import ECMInstance


@dataclass
class ReconstructionConfig:
    dataset_name: str
    n_users: int = 3000
    worker_pool_size: int = 300
    n_target_subareas: int = 100
    lon_min: float = -122.50
    lon_max: float = -118.00
    lat_min: float = 33.80
    lat_max: float = 37.90
    lon_cells: int = 80
    lat_cells: int = 90
    edge_prob_min: float = 0.1
    edge_prob_max: float = 0.5
    random_seed: int = 0
    demand_mode: str = "raw_checkin_count"
    demand_scale: float = 1.0
    bidirectional_edges: bool = True


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _grid_cell(lon: float, lat: float, cfg: ReconstructionConfig) -> tuple[int, int] | None:
    if not (cfg.lon_min <= lon <= cfg.lon_max and cfg.lat_min <= lat <= cfg.lat_max):
        return None
    x = min(cfg.lon_cells - 1, int((lon - cfg.lon_min) / (cfg.lon_max - cfg.lon_min) * cfg.lon_cells))
    y = min(cfg.lat_cells - 1, int((lat - cfg.lat_min) / (cfg.lat_max - cfg.lat_min) * cfg.lat_cells))
    return x, y


def _cell_center(x: int, y: int, cfg: ReconstructionConfig) -> tuple[float, float]:
    lon_w = (cfg.lon_max - cfg.lon_min) / cfg.lon_cells
    lat_h = (cfg.lat_max - cfg.lat_min) / cfg.lat_cells
    return cfg.lon_min + (x + 0.5) * lon_w, cfg.lat_min + (y + 0.5) * lat_h


def _read_edge_nodes(edge_path: str | Path) -> tuple[list[tuple[int, int]], set[int]]:
    edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    with _open_text(edge_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    return edges, nodes


def _scan_checkins(checkin_path: str | Path, cfg: ReconstructionConfig) -> tuple[set[int], np.ndarray]:
    users: set[int] = set()
    counts = np.zeros((cfg.lon_cells, cfg.lat_cells), dtype=np.int64)
    with _open_text(checkin_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                user = int(parts[0])
                lat = float(parts[2])
                lon = float(parts[3])
            except ValueError:
                continue
            users.add(user)
            cell = _grid_cell(lon, lat, cfg)
            if cell is not None:
                counts[cell] += 1
    return users, counts


def _collect_user_locations(checkin_path: str | Path, selected_users: set[int]) -> dict[int, list[tuple[float, float]]]:
    locs = {u: [] for u in selected_users}
    with _open_text(checkin_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                user = int(parts[0])
            except ValueError:
                continue
            if user not in selected_users:
                continue
            try:
                lat = float(parts[2])
                lon = float(parts[3])
            except ValueError:
                continue
            locs[user].append((lon, lat))
    return locs


def _demand_from_counts(counts: np.ndarray, cfg: ReconstructionConfig) -> np.ndarray:
    counts = counts.astype(np.float64)
    if cfg.demand_mode == "raw_checkin_count":
        return np.maximum(counts, 1.0) * cfg.demand_scale
    if cfg.demand_mode == "max_normalized":
        denom = max(float(counts.max()), 1.0)
        return 1.0 + counts / denom * cfg.demand_scale
    if cfg.demand_mode == "sqrt_count":
        return np.maximum(np.sqrt(np.maximum(counts, 1.0)) * cfg.demand_scale, 1e-12)
    raise ValueError(f"unknown demand_mode={cfg.demand_mode!r}")


def reconstruct_instance(
    edge_path: str | Path,
    checkin_path: str | Path,
    cfg: ReconstructionConfig,
) -> tuple[ECMInstance, dict]:
    """Build a fully manifested reconstruction from SNAP-format data."""
    rng = np.random.default_rng(cfg.random_seed)
    source_edges, edge_nodes = _read_edge_nodes(edge_path)
    checkin_users, grid_counts = _scan_checkins(checkin_path, cfg)
    eligible = np.asarray(sorted(edge_nodes & checkin_users), dtype=np.int64)
    if eligible.size < cfg.n_users:
        raise ValueError(f"only {eligible.size} eligible users, need {cfg.n_users}")
    sampled_original = rng.choice(eligible, size=cfg.n_users, replace=False)
    sampled_set = set(int(x) for x in sampled_original.tolist())
    locs = _collect_user_locations(checkin_path, sampled_set)
    if any(len(locs[u]) == 0 for u in sampled_set):
        raise RuntimeError("sampled user unexpectedly has no check-ins")

    total_cells = cfg.lon_cells * cfg.lat_cells
    if cfg.n_target_subareas > total_cells:
        raise ValueError("n_target_subareas exceeds grid size")
    flat_cells = rng.choice(total_cells, size=cfg.n_target_subareas, replace=False)
    cells = [(int(c // cfg.lat_cells), int(c % cfg.lat_cells)) for c in flat_cells]
    centers = np.asarray([_cell_center(x, y, cfg) for x, y in cells], dtype=np.float64)
    selected_counts = np.asarray([grid_counts[x, y] for x, y in cells], dtype=np.float64)
    demand = _demand_from_counts(selected_counts, cfg)

    original_to_new = {int(u): i for i, u in enumerate(sampled_original.tolist())}
    n, h = cfg.n_users, cfg.n_target_subareas
    min_dist = np.empty((n, h), dtype=np.float64)
    for original, new in original_to_new.items():
        user_xy = np.asarray(locs[original], dtype=np.float64)
        diff = user_xy[:, None, :] - centers[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        min_dist[new] = dist.min(axis=0)
    max_dist = float(min_dist.max())
    if max_dist <= 0:
        raise RuntimeError("degenerate distance normalization")
    p = np.clip(1.0 - min_dist / max_dist, 0.0, 1.0)
    q = p.copy()

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    retained_source_edges = 0
    for u0, v0 in source_edges:
        if u0 not in sampled_set or v0 not in sampled_set:
            continue
        retained_source_edges += 1
        u, v = original_to_new[u0], original_to_new[v0]
        graph.add_edge(u, v, weight=float(rng.uniform(cfg.edge_prob_min, cfg.edge_prob_max)))
        if cfg.bidirectional_edges:
            graph.add_edge(v, u, weight=float(rng.uniform(cfg.edge_prob_min, cfg.edge_prob_max)))

    pool_size = min(cfg.worker_pool_size, n)
    worker_pool = set(int(x) for x in rng.choice(n, size=pool_size, replace=False).tolist())
    instance = ECMInstance(graph, list(range(n)), p, q, demand, worker_pool)
    manifest = {
        "config": asdict(cfg),
        "sampled_original_user_ids": sampled_original.tolist(),
        "target_cells_xy": cells,
        "target_centers_lon_lat": centers.tolist(),
        "target_checkin_counts": selected_counts.tolist(),
        "max_distance_eq23": max_dist,
        "retained_source_edges": retained_source_edges,
        "directed_edges_after_conversion": graph.number_of_edges(),
        "worker_pool": sorted(worker_pool),
        "notes": [
            "Demand transform is explicitly configured because the paper only states that d_i is based on check-in count.",
            "Source SNAP friendship edges are treated as bidirectional by default because the released SNAP friendship networks are undirected; this choice is configurable.",
            "Euclidean distance is applied directly to (longitude, latitude) coordinates following the literal wording of Eq. (24).",
        ],
    }
    return instance, manifest


def save_reconstruction(instance: ECMInstance, manifest: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(
        [(u, v, float(data.get("weight", 1.0))) for u, v, data in instance.graph.edges(data=True)],
        dtype=np.float64,
    )
    np.savez_compressed(
        output_dir / "instance.npz",
        participation=instance.participation,
        quality=instance.quality,
        demand=instance.demand,
        edges=edges,
        worker_pool=np.asarray(sorted(instance.worker_pool or set()), dtype=np.int64),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_reconstruction(output_dir: str | Path) -> tuple[ECMInstance, dict]:
    output_dir = Path(output_dir)
    data = np.load(output_dir / "instance.npz")
    participation = data["participation"]
    quality = data["quality"]
    demand = data["demand"]
    edges = data["edges"]
    worker_pool = set(int(x) for x in data["worker_pool"].tolist())
    graph = nx.DiGraph()
    graph.add_nodes_from(range(participation.shape[0]))
    for row in edges:
        if row.size < 3:
            continue
        graph.add_edge(int(row[0]), int(row[1]), weight=float(row[2]))
    instance = ECMInstance(
        graph=graph,
        nodes=list(range(participation.shape[0])),
        participation=participation,
        quality=quality,
        demand=demand,
        worker_pool=worker_pool,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    return instance, manifest
