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
class ScientificV2ReconstructionConfig:
    """Scientifically conservative journal benchmark construction.

    V2 keeps the original ECM optimization problem but avoids interpreting
    unobserved quantities as if they were measured labels. Participation is a
    frequency-aware spatial availability proxy, the main quality setting is
    uniform, target areas are sampled from empirically active cells, and social
    diffusion uses the classical TRI protocol unless explicitly overridden.

    ``structure_seed`` controls users, social edge probabilities, and the worker
    pool. ``task_seed`` controls target-area sampling only. Consequently one can
    vary task instances while holding the worker/social environment fixed.
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
    structure_seed: int = 2024
    task_seed: int = 2024
    bidirectional_edges: bool = True
    participation_distance_quantile: float = 0.50
    quality_mode: str = "uniform"
    quality_floor: float = 0.25
    quality_lower_quantile: float = 0.10
    quality_upper_quantile: float = 0.90
    load_factor: float = 1.50
    demand_activity_power: float = 0.50
    demand_floor_ratio: float = 0.05
    target_sampling: str = "active_uniform"
    trivalency_values: tuple[float, float, float] = (0.001, 0.01, 0.10)


def _validate_config(cfg: ScientificV2ReconstructionConfig) -> None:
    if not (0.0 < cfg.participation_distance_quantile <= 1.0):
        raise ValueError("participation_distance_quantile must lie in (0,1]")
    if cfg.quality_mode not in {"uniform", "activity_reliability"}:
        raise ValueError("quality_mode must be 'uniform' or 'activity_reliability'")
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
    if cfg.target_sampling not in {"active_uniform", "active_weighted", "all_uniform"}:
        raise ValueError("unknown target_sampling")
    if not cfg.trivalency_values or any(not (0.0 <= p <= 1.0) for p in cfg.trivalency_values):
        raise ValueError("trivalency probabilities must lie in [0,1]")


def _haversine_km(points_lon_lat: np.ndarray, centers_lon_lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance, [n_points, n_centers], in kilometers."""
    points = np.asarray(points_lon_lat, dtype=np.float64)
    centers = np.asarray(centers_lon_lat, dtype=np.float64)
    lon1 = np.deg2rad(points[:, 0])[:, None]
    lat1 = np.deg2rad(points[:, 1])[:, None]
    lon2 = np.deg2rad(centers[:, 0])[None, :]
    lat2 = np.deg2rad(centers[:, 1])[None, :]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 6371.0088 * (2.0 * np.arcsin(np.sqrt(a)))


def _frequency_aware_participation(
    user_locations: list[np.ndarray],
    centers: np.ndarray,
    distance_quantile: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Historical visitation-kernel participation proxy.

    The scale is estimated from nearest historical distances when possible, while
    the probability itself averages the distance kernel over all historical visits.
    If every user-target pair has an exact historical hit, the scale falls back to
    all non-zero visit-to-target distances rather than becoming undefined.
    """
    if not user_locations:
        raise ValueError("user_locations must not be empty")
    h = int(centers.shape[0])
    min_dist = np.empty((len(user_locations), h), dtype=np.float64)
    distance_mats: list[np.ndarray] = []
    for i, xy in enumerate(user_locations):
        d = _haversine_km(xy, centers)
        if d.shape[0] == 0:
            raise RuntimeError("sampled user has no in-region check-ins")
        distance_mats.append(d)
        min_dist[i] = d.min(axis=0)
    positive = min_dist[min_dist > 1e-12]
    if positive.size == 0:
        positive_parts = [d[d > 1e-12] for d in distance_mats if np.any(d > 1e-12)]
        if not positive_parts:
            raise RuntimeError("degenerate spatial distances: all visits coincide with all targets")
        positive = np.concatenate(positive_parts)
    tau_km = max(float(np.quantile(positive, distance_quantile)), 1e-12)
    participation = np.vstack(
        [np.exp(-d / tau_km).mean(axis=0, keepdims=True) for d in distance_mats]
    )
    return np.clip(participation, 0.0, 1.0), tau_km, min_dist


def _activity_reliability(
    user_checkin_counts: np.ndarray,
    cfg: ScientificV2ReconstructionConfig,
) -> np.ndarray:
    x = np.log1p(np.asarray(user_checkin_counts, dtype=np.float64))
    lo = float(np.quantile(x, cfg.quality_lower_quantile))
    hi = float(np.quantile(x, cfg.quality_upper_quantile))
    if hi <= lo + 1e-12:
        normalized = np.full_like(x, 0.5)
    else:
        normalized = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return cfg.quality_floor + (1.0 - cfg.quality_floor) * normalized


def _sample_target_cells(
    grid_counts: np.ndarray,
    cfg: ScientificV2ReconstructionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_counts = np.asarray(grid_counts, dtype=np.float64).reshape(-1)
    if cfg.target_sampling == "all_uniform":
        candidates = np.arange(flat_counts.size, dtype=np.int64)
        probs = None
    else:
        candidates = np.flatnonzero(flat_counts > 0).astype(np.int64, copy=False)
        probs = None
        if cfg.target_sampling == "active_weighted":
            w = np.sqrt(flat_counts[candidates])
            probs = w / w.sum()
    if candidates.size < cfg.n_target_subareas:
        raise ValueError(
            f"only {candidates.size} eligible target cells for {cfg.target_sampling}, "
            f"need {cfg.n_target_subareas}"
        )
    return rng.choice(candidates, size=cfg.n_target_subareas, replace=False, p=probs)


def _load_calibrated_demand(
    target_checkin_counts: np.ndarray,
    contribution: np.ndarray,
    worker_pool: set[int],
    cfg: ScientificV2ReconstructionConfig,
) -> tuple[np.ndarray, dict]:
    pool = np.asarray(sorted(worker_pool), dtype=np.int64)
    direct_capacity = contribution[pool].sum(axis=0)
    positive_capacity = direct_capacity[direct_capacity > 1e-12]
    if positive_capacity.size == 0:
        raise RuntimeError("candidate pool has zero direct capacity")
    capacity_scale = float(np.median(positive_capacity))
    counts = np.asarray(target_checkin_counts, dtype=np.float64)
    activity_weight = np.power(counts + 1.0, cfg.demand_activity_power)
    activity_weight /= max(float(activity_weight.mean()), 1e-12)
    demand = cfg.load_factor * capacity_scale * activity_weight
    floor = cfg.demand_floor_ratio * cfg.load_factor * capacity_scale
    demand = np.maximum(demand, floor)
    return demand, {
        "candidate_direct_capacity_min": float(direct_capacity.min()),
        "candidate_direct_capacity_mean": float(direct_capacity.mean()),
        "candidate_direct_capacity_median": float(np.median(direct_capacity)),
        "candidate_direct_capacity_max": float(direct_capacity.max()),
        "capacity_scale_median_positive": capacity_scale,
        "activity_weight_min": float(activity_weight.min()),
        "activity_weight_mean": float(activity_weight.mean()),
        "activity_weight_max": float(activity_weight.max()),
    }


def _graph_diagnostics(graph: nx.DiGraph) -> dict[str, float | int]:
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected)) if graph.number_of_nodes() else []
    largest = max((len(c) for c in components), default=0)
    n = max(graph.number_of_nodes(), 1)
    return {
        "nodes": int(graph.number_of_nodes()),
        "directed_edges": int(graph.number_of_edges()),
        "mean_out_degree": float(graph.number_of_edges() / n),
        "connected_components_undirected": int(len(components)),
        "largest_component_fraction": float(largest / n),
        "average_clustering_undirected": float(nx.average_clustering(undirected)) if n > 1 else 0.0,
    }


def reconstruct_scientific_v2_instance(
    edge_path: str | Path,
    checkin_path: str | Path,
    cfg: ScientificV2ReconstructionConfig,
) -> tuple[ECMInstance, dict]:
    _validate_config(cfg)
    structure_rng = np.random.default_rng(cfg.structure_seed)
    task_rng = np.random.default_rng(cfg.task_seed)
    source_edges, edge_nodes = _read_edge_nodes(edge_path)
    region_users, grid_counts, region_records = _scan_checkins(checkin_path, cfg)
    eligible = np.asarray(sorted(edge_nodes & region_users), dtype=np.int64)
    if eligible.size < cfg.n_users:
        raise ValueError(f"only {eligible.size} region-eligible users, need {cfg.n_users}")
    sampled_original = structure_rng.choice(eligible, size=cfg.n_users, replace=False)
    sampled_set = set(int(x) for x in sampled_original.tolist())
    locs = _collect_user_locations(checkin_path, sampled_set, cfg)
    if any(len(locs[u]) == 0 for u in sampled_set):
        raise RuntimeError("sampled region user unexpectedly has no in-region check-ins")
    flat_cells = _sample_target_cells(grid_counts, cfg, task_rng)
    cells = [(int(c // cfg.lat_cells), int(c % cfg.lat_cells)) for c in flat_cells]
    centers = np.asarray([_cell_center(x, y, cfg) for x, y in cells], dtype=np.float64)
    selected_counts = np.asarray([grid_counts[x, y] for x, y in cells], dtype=np.float64)
    original_to_new = {int(u): i for i, u in enumerate(sampled_original.tolist())}
    n, h = cfg.n_users, cfg.n_target_subareas
    user_locations: list[np.ndarray] = [np.empty((0, 2), dtype=np.float64) for _ in range(n)]
    user_checkin_counts = np.empty(n, dtype=np.float64)
    for original, new in original_to_new.items():
        xy = np.asarray(locs[original], dtype=np.float64)
        user_locations[new] = xy
        user_checkin_counts[new] = float(xy.shape[0])
    participation, tau_km, min_dist = _frequency_aware_participation(
        user_locations, centers, cfg.participation_distance_quantile
    )
    if cfg.quality_mode == "uniform":
        worker_quality = np.ones(n, dtype=np.float64)
    else:
        worker_quality = _activity_reliability(user_checkin_counts, cfg)
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
        graph.add_edge(u, v, weight=float(structure_rng.choice(tri)))
        if cfg.bidirectional_edges:
            graph.add_edge(v, u, weight=float(structure_rng.choice(tri)))
    pool_size = min(cfg.worker_pool_size, n)
    worker_pool = set(
        int(x) for x in structure_rng.choice(n, size=pool_size, replace=False).tolist()
    )
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
        "profile": "journal_scientific_v2",
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
            "proxy": "mean historical exp(-haversine_distance_km/tau_km)",
            "distance_scale_quantile": cfg.participation_distance_quantile,
            "distance_scale_km": tau_km,
            "nearest_distance_km_p50": float(np.quantile(min_dist, 0.50)),
            "nearest_distance_km_p90": float(np.quantile(min_dist, 0.90)),
            "min": float(participation.min()),
            "mean": float(participation.mean()),
            "max": float(participation.max()),
        },
        "quality": {
            "mode": cfg.quality_mode,
            "interpretation": (
                "uniform main setting; no empirical sensing-quality label is claimed"
                if cfg.quality_mode == "uniform"
                else "historical activity reliability proxy; sensitivity setting only"
            ),
            "worker_min": float(worker_quality.min()),
            "worker_mean": float(worker_quality.mean()),
            "worker_max": float(worker_quality.max()),
        },
        "demand": {
            "proxy": "activity-weighted normalized system load calibrated to candidate direct capacity",
            "interpretation": "normalized load protocol, not observed task-demand labels",
            "load_factor": cfg.load_factor,
            "activity_power": cfg.demand_activity_power,
            "min": float(demand.min()),
            "mean": float(demand.mean()),
            "max": float(demand.max()),
            **demand_info,
        },
        "targets": {
            "sampling": cfg.target_sampling,
            "all_have_historical_activity": bool(np.all(selected_counts > 0)),
            "checkin_count_min": float(selected_counts.min()),
            "checkin_count_mean": float(selected_counts.mean()),
            "checkin_count_max": float(selected_counts.max()),
        },
        "influence": {
            "protocol": "classical_trivalency" if tuple(cfg.trivalency_values) == (0.001, 0.01, 0.10) else "custom_trivalency",
            "values": list(cfg.trivalency_values),
            "interpretation": "simulation protocol; friendship data contain no empirical diffusion labels",
            "edge_probability_min": float(edge_probs.min()) if edge_probs.size else 0.0,
            "edge_probability_mean": float(edge_probs.mean()) if edge_probs.size else 0.0,
            "edge_probability_max": float(edge_probs.max()) if edge_probs.size else 0.0,
        },
        "graph_diagnostics": _graph_diagnostics(graph),
        "notes": [
            "V2 is a scientific benchmark protocol, not an exact SIGIR-2024 numerical reproduction.",
            "Structure and task randomness are separated so generalization can be tested without changing the worker/social environment.",
            "Participation is frequency-aware and uses great-circle distance rather than minimum Euclidean lon/lat distance.",
            "The main quality setting is uniform because the source data contain no sensing-quality labels.",
            "Targets are sampled only from historically active cells in the main setting.",
            "Demand is a normalized workload regime; load_factor must be sensitivity-tested.",
            "Classical TRI edge probabilities are a simulation convention rather than measured social influence.",
        ],
    }
    return instance, manifest
