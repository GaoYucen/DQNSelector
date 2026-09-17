from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .ecm import ECMInstance
from .reconstruction import (
    _cell_center,
    _collect_user_locations,
    _read_edge_nodes,
    _scan_checkins,
)


@dataclass
class ScientificReconstructionConfig:
    """Scientifically calibrated journal setting.

    The public data do not contain ground-truth worker acceptance probabilities,
    sensing quality labels, task demand, or empirical social diffusion rates.
    Instead of silently equating these quantities, this profile uses explicit,
    independently configurable proxies and records every choice in the manifest.
    """

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
    random_seed: int = 2024
    bidirectional_edges: bool = True

    # Participation: spatial willingness/availability proxy.
    participation_distance_quantile: float = 0.50

    # Quality: worker-level reliability proxy derived from historical activity,
    # deliberately decoupled from target distance / participation.
    quality_floor: float = 0.25
    quality_lower_quantile: float = 0.10
    quality_upper_quantile: float = 0.90

    # Demand: task intensity is proportional to observed area activity, while the
    # overall scale is calibrated against the candidate pool's direct capacity.
    load_factor: float = 0.50
    demand_activity_power: float = 0.50
    demand_floor_ratio: float = 0.05

    # Social influence: standard low-probability trivalency protocol because the
    # SNAP friendship graph does not provide empirical diffusion probabilities.
    trivalency_values: tuple[float, float, float] = (0.01, 0.05, 0.10)


def _validate_config(cfg: ScientificReconstructionConfig) -> None:
    if not (0.0 < cfg.participation_distance_quantile <= 1.0):
        raise ValueError("participation_distance_quantile must lie in (0,1]")
    if not (0.0 <= cfg.quality_floor <= 1.0):
        raise ValueError("quality_floor must lie in [0,1]")
    if not (0.0 <= cfg.quality_lower_quantile < cfg.quality_upper_quantile <= 1.0):
        raise ValueError("quality quantiles must satisfy 0 <= lower < upper <= 1")
    if cfg.load_factor <= 0:
        raise ValueError("load_factor must be positive")
    if cfg.demand_activity_power < 0:
        raise ValueError("demand_activity_power must be non-negative")
    if not (0.0 < cfg.demand_floor_ratio <= 1.0):
        raise ValueError("demand_floor_ratio must lie in (0,1]")
    if not cfg.trivalency_values or any(not (0.0 <= p <= 1.0) for p in cfg.trivalency_values):
        raise ValueError("trivalency probabilities must lie in [0,1]")


def _robust_activity_quality(
    user_checkin_counts: np.ndarray,
    cfg: ScientificReconstructionConfig,
) -> np.ndarray:
    x = np.log1p(np.asarray(user_checkin_counts, dtype=np.float64))
    lo = float(np.quantile(x, cfg.quality_lower_quantile))
    hi = float(np.quantile(x, cfg.quality_upper_quantile))
    if hi <= lo + 1e-12:
        normalized = np.full_like(x, 0.5)
    else:
        normalized = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return cfg.quality_floor + (1.0 - cfg.quality_floor) * normalized


def _load_calibrated_demand(
    target_checkin_counts: np.ndarray,
    contribution: np.ndarray,
    worker_pool: set[int],
    cfg: ScientificReconstructionConfig,
) -> tuple[np.ndarray, dict]:
    pool = np.asarray(sorted(worker_pool), dtype=np.int64)
    direct_capacity = contribution[pool].sum(axis=0)
    positive_capacity = direct_capacity[direct_capacity > 1e-12]
    if positive_capacity.size == 0:
        raise RuntimeError("candidate pool has zero direct capacity")

    # Use a robust global capacity scale rather than setting demand separately to
    # each area's capacity; otherwise supply/demand mismatch would be erased.
    capacity_scale = float(np.median(positive_capacity))

    counts = np.asarray(target_checkin_counts, dtype=np.float64)
    activity_weight = np.power(counts + 1.0, cfg.demand_activity_power)
    activity_weight /= max(float(activity_weight.mean()), 1e-12)

    demand = cfg.load_factor * capacity_scale * activity_weight
    floor = cfg.demand_floor_ratio * cfg.load_factor * capacity_scale
    demand = np.maximum(demand, floor)
    info = {
        "candidate_direct_capacity_min": float(direct_capacity.min()),
        "candidate_direct_capacity_mean": float(direct_capacity.mean()),
        "candidate_direct_capacity_median": float(np.median(direct_capacity)),
        "candidate_direct_capacity_max": float(direct_capacity.max()),
        "capacity_scale_median_positive": capacity_scale,
        "activity_weight_min": float(activity_weight.min()),
        "activity_weight_mean": float(activity_weight.mean()),
        "activity_weight_max": float(activity_weight.max()),
    }
    return demand, info


def reconstruct_scientific_instance(
    edge_path: str | Path,
    checkin_path: str | Path,
    cfg: ScientificReconstructionConfig,
) -> tuple[ECMInstance, dict]:
    """Build a journal instance with explicit, non-saturating scientific proxies."""
    _validate_config(cfg)
    rng = np.random.default_rng(cfg.random_seed)

    # Reuse the same spatial filtering protocol as the clean reproduction so the
    # scientific and legacy settings can be compared on identical source data.
    source_edges, edge_nodes = _read_edge_nodes(edge_path)
    region_users, grid_counts, region_records = _scan_checkins(checkin_path, cfg)
    eligible = np.asarray(sorted(edge_nodes & region_users), dtype=np.int64)
    if eligible.size < cfg.n_users:
        raise ValueError(f"only {eligible.size} region-eligible users, need {cfg.n_users}")

    sampled_original = rng.choice(eligible, size=cfg.n_users, replace=False)
    sampled_set = set(int(x) for x in sampled_original.tolist())
    locs = _collect_user_locations(checkin_path, sampled_set, cfg)
    if any(len(locs[u]) == 0 for u in sampled_set):
        raise RuntimeError("sampled region user unexpectedly has no in-region check-ins")

    total_cells = cfg.lon_cells * cfg.lat_cells
    if cfg.n_target_subareas > total_cells:
        raise ValueError("n_target_subareas exceeds grid size")
    flat_cells = rng.choice(total_cells, size=cfg.n_target_subareas, replace=False)
    cells = [(int(c // cfg.lat_cells), int(c % cfg.lat_cells)) for c in flat_cells]
    centers = np.asarray([_cell_center(x, y, cfg) for x, y in cells], dtype=np.float64)
    selected_counts = np.asarray([grid_counts[x, y] for x, y in cells], dtype=np.float64)

    original_to_new = {int(u): i for i, u in enumerate(sampled_original.tolist())}
    n, h = cfg.n_users, cfg.n_target_subareas
    min_dist = np.empty((n, h), dtype=np.float64)
    user_checkin_counts = np.empty(n, dtype=np.float64)
    for original, new in original_to_new.items():
        user_xy = np.asarray(locs[original], dtype=np.float64)
        user_checkin_counts[new] = float(user_xy.shape[0])
        diff = user_xy[:, None, :] - centers[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        min_dist[new] = dist.min(axis=0)

    positive_dist = min_dist[min_dist > 1e-12]
    if positive_dist.size == 0:
        raise RuntimeError("degenerate spatial distances")
    distance_scale = float(np.quantile(positive_dist, cfg.participation_distance_quantile))
    distance_scale = max(distance_scale, 1e-12)
    participation = np.exp(-min_dist / distance_scale)
    participation = np.clip(participation, 0.0, 1.0)

    worker_quality = _robust_activity_quality(user_checkin_counts, cfg)
    quality = np.repeat(worker_quality[:, None], h, axis=1)
    contribution = participation * quality

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    retained_source_edges = 0
    tri = np.asarray(cfg.trivalency_values, dtype=np.float64)
    for u0, v0 in source_edges:
        if u0 not in sampled_set or v0 not in sampled_set:
            continue
        retained_source_edges += 1
        u, v = original_to_new[u0], original_to_new[v0]
        graph.add_edge(u, v, weight=float(rng.choice(tri)))
        if cfg.bidirectional_edges:
            graph.add_edge(v, u, weight=float(rng.choice(tri)))

    pool_size = min(cfg.worker_pool_size, n)
    worker_pool = set(int(x) for x in rng.choice(n, size=pool_size, replace=False).tolist())
    demand, demand_info = _load_calibrated_demand(
        selected_counts, contribution, worker_pool, cfg
    )

    instance = ECMInstance(
        graph=graph,
        nodes=list(range(n)),
        participation=participation,
        quality=quality,
        demand=demand,
        worker_pool=worker_pool,
    )
    edge_probs = np.asarray([float(d["weight"]) for _, _, d in graph.edges(data=True)])
    manifest = {
        "profile": "journal_scientific_v1",
        "config": asdict(cfg),
        "region_user_count_with_social_edge": int(eligible.size),
        "region_checkin_records": int(region_records),
        "sampled_original_user_ids": sampled_original.tolist(),
        "target_cells_xy": cells,
        "target_centers_lon_lat": centers.tolist(),
        "target_checkin_counts": selected_counts.tolist(),
        "retained_source_edges": retained_source_edges,
        "directed_edges_after_conversion": graph.number_of_edges(),
        "worker_pool": sorted(worker_pool),
        "participation": {
            "proxy": "exp(-minimum historical distance to target / robust distance scale)",
            "distance_scale_quantile": cfg.participation_distance_quantile,
            "distance_scale": distance_scale,
            "min": float(participation.min()),
            "mean": float(participation.mean()),
            "max": float(participation.max()),
        },
        "quality": {
            "proxy": "robustly normalized log historical in-region check-in count",
            "floor": cfg.quality_floor,
            "worker_min": float(worker_quality.min()),
            "worker_mean": float(worker_quality.mean()),
            "worker_max": float(worker_quality.max()),
            "historical_checkins_min": float(user_checkin_counts.min()),
            "historical_checkins_mean": float(user_checkin_counts.mean()),
            "historical_checkins_max": float(user_checkin_counts.max()),
        },
        "demand": {
            "proxy": "activity-weighted task intensity calibrated to candidate direct capacity",
            "load_factor": cfg.load_factor,
            "activity_power": cfg.demand_activity_power,
            "min": float(demand.min()),
            "mean": float(demand.mean()),
            "max": float(demand.max()),
            **demand_info,
        },
        "influence": {
            "protocol": "trivalency",
            "values": list(cfg.trivalency_values),
            "edge_probability_min": float(edge_probs.min()) if edge_probs.size else 0.0,
            "edge_probability_mean": float(edge_probs.mean()) if edge_probs.size else 0.0,
            "edge_probability_max": float(edge_probs.max()) if edge_probs.size else 0.0,
        },
        "notes": [
            "This is a journal scientific setting, not a claim of exact SIGIR-2024 numerical reproduction.",
            "Participation and quality use separate observable proxies; q is not copied from p.",
            "Demand scale is tied to candidate-pool capacity while spatial demand heterogeneity follows observed check-in activity.",
            "Friendship data provide no empirical cascade probabilities, so a low-probability trivalency IC protocol is explicit and will be sensitivity-tested.",
            "The legacy reproduction remains available separately for setting-ablation experiments.",
        ],
    }
    return instance, manifest
