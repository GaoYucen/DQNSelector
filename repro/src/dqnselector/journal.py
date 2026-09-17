from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path
from collections import defaultdict, deque

import networkx as nx
import numpy as np

EARTH_RADIUS_KM = 6371.0088


@dataclass
class JournalConfig:
    dataset_name: str
    n_users: int = 3000
    worker_pool_size: int = 300
    n_tasks: int = 100
    lon_min: float = -122.50
    lon_max: float = -118.00
    lat_min: float = 33.80
    lat_max: float = 37.90
    lon_cells: int = 80
    lat_cells: int = 90
    temporal_train_ratio: float = 0.90
    min_user_train_checkins: int = 10
    min_task_train_checkins: int = 20
    min_task_train_users: int = 5
    layout: str = "scattered"  # scattered | clustered | contiguous
    accessibility_tau_km: float = 10.0
    quality_radius_km: float = 5.0
    quality_beta: float = 3.0
    demand_min: float = 3.0
    demand_max: float = 10.0
    influence_min: float = 0.01
    influence_max: float = 0.20
    influence_mobility_tau_km: float = 50.0
    random_seed: int = 2026


@dataclass
class JournalInstance:
    graph: nx.DiGraph
    accessibility: np.ndarray  # [N,T]
    quality: np.ndarray  # [N,T]
    demand: np.ndarray  # [T]
    worker_pool: set[int]
    task_centers_lat_lon: np.ndarray  # [T,2]

    @property
    def suitability(self) -> np.ndarray:
        return self.accessibility * self.quality


def _open_text(path: str | Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return p.open("r", encoding="utf-8", errors="replace")


def _grid_cell(lon: float, lat: float, cfg: JournalConfig) -> tuple[int, int] | None:
    if not (cfg.lon_min <= lon <= cfg.lon_max and cfg.lat_min <= lat <= cfg.lat_max):
        return None
    x = min(cfg.lon_cells - 1, int((lon - cfg.lon_min) / (cfg.lon_max - cfg.lon_min) * cfg.lon_cells))
    y = min(cfg.lat_cells - 1, int((lat - cfg.lat_min) / (cfg.lat_max - cfg.lat_min) * cfg.lat_cells))
    return x, y


def _cell_center(x: int, y: int, cfg: JournalConfig) -> tuple[float, float]:
    lon_w = (cfg.lon_max - cfg.lon_min) / cfg.lon_cells
    lat_h = (cfg.lat_max - cfg.lat_min) / cfg.lat_cells
    return cfg.lat_min + (y + 0.5) * lat_h, cfg.lon_min + (x + 0.5) * lon_w


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _read_edges(path: str | Path) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    with _open_text(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 2:
                continue
            u, v = int(p[0]), int(p[1])
            if u == v:
                continue
            out.add((min(u, v), max(u, v)))
    return sorted(out)


def _read_region_checkins(path: str | Path, cfg: JournalConfig):
    # SNAP timestamps are ISO-like strings; lexical order equals chronological order.
    by_user: dict[int, list[tuple[str, float, float, tuple[int, int]]]] = defaultdict(list)
    with _open_text(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            try:
                uid = int(p[0]); ts = p[1]; lat = float(p[2]); lon = float(p[3])
            except ValueError:
                continue
            cell = _grid_cell(lon, lat, cfg)
            if cell is None:
                continue
            by_user[uid].append((ts, lat, lon, cell))
    return by_user


def _temporal_split(records, ratio: float):
    records = sorted(records, key=lambda r: r[0])
    if len(records) <= 1:
        return records, []
    cut = int(math.floor(len(records) * ratio))
    cut = min(max(cut, 1), len(records) - 1)
    return records[:cut], records[cut:]


def _connected_sample(edges: list[tuple[int, int]], eligible: set[int], activity: dict[int, int], n: int, seed: int):
    g = nx.Graph()
    g.add_edges_from((u, v) for u, v in edges if u in eligible and v in eligible)
    if g.number_of_nodes() < n:
        raise ValueError(f"only {g.number_of_nodes()} eligible graph users; need {n}")
    comp = max(nx.connected_components(g), key=len)
    if len(comp) < n:
        raise ValueError(f"largest eligible component has {len(comp)} users; need {n}")
    sg = g.subgraph(comp)
    rng = np.random.default_rng(seed)
    ranked = sorted(comp, key=lambda u: activity.get(u, 0), reverse=True)
    start = int(rng.choice(np.asarray(ranked[: min(50, len(ranked))], dtype=np.int64)))
    selected = {start}
    q = deque([start])
    while q and len(selected) < n:
        u = q.popleft()
        nbrs = list(sg.neighbors(u)); rng.shuffle(nbrs)
        for v in nbrs:
            if v not in selected:
                selected.add(int(v)); q.append(int(v))
                if len(selected) == n:
                    break
    if len(selected) < n:
        # The component is connected; this is only a defensive fallback.
        remain = np.asarray(sorted(set(comp) - selected), dtype=np.int64)
        need = n - len(selected)
        selected.update(int(x) for x in rng.choice(remain, size=need, replace=False))
    return sorted(selected)


def _rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones_like(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = (np.arange(len(values)) + 1) / len(values)
    return ranks


def _select_tasks(candidate_cells, cfg: JournalConfig):
    # candidate tuple: (x,y,train_checkins,train_users)
    if len(candidate_cells) < cfg.n_tasks:
        raise ValueError(f"only {len(candidate_cells)} task cells pass training filters; need {cfg.n_tasks}")
    cells = list(candidate_cells)
    centers = np.asarray([_cell_center(x, y, cfg) for x, y, *_ in cells], dtype=float)
    popularity = np.log1p(np.asarray([u for *_, u in cells], dtype=float))
    first = int(np.argmax(popularity))
    if cfg.layout in {"clustered", "contiguous"}:
        d = haversine_km(centers[:, 0], centers[:, 1], centers[first, 0], centers[first, 1])
        # contiguous is deliberately the tightest local concentration; clustered keeps a popularity term.
        if cfg.layout == "clustered":
            score = d / np.maximum(0.5 + 0.5 * popularity / max(popularity.max(), 1e-12), 1e-12)
        else:
            score = d
        idx = np.argsort(score)[: cfg.n_tasks]
        return [cells[int(i)] for i in idx]
    if cfg.layout != "scattered":
        raise ValueError(f"unknown layout={cfg.layout!r}")
    chosen = [first]
    min_d = haversine_km(centers[:, 0], centers[:, 1], centers[first, 0], centers[first, 1])
    while len(chosen) < cfg.n_tasks:
        score = min_d * (0.5 + 0.5 * popularity / max(popularity.max(), 1e-12))
        score[np.asarray(chosen, dtype=int)] = -1.0
        nxt = int(np.argmax(score)); chosen.append(nxt)
        d = haversine_km(centers[:, 0], centers[:, 1], centers[nxt, 0], centers[nxt, 1])
        min_d = np.minimum(min_d, d)
    return [cells[i] for i in chosen]


def _stratified_worker_pool(activity: np.ndarray, count: int, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    order = np.argsort(activity)
    bins = np.array_split(order, 3)
    alloc = [count // 3] * 3
    for i in range(count % 3):
        alloc[2 - i] += 1
    chosen = []
    for b, n in zip(bins, alloc):
        if len(b) < n:
            raise ValueError("not enough users in activity stratum")
        chosen.extend(int(x) for x in rng.choice(b, size=n, replace=False))
    return set(chosen)


def build_journal_instance(edge_path: str | Path, checkin_path: str | Path, cfg: JournalConfig):
    edges = _read_edges(edge_path)
    all_records = _read_region_checkins(checkin_path, cfg)
    split = {u: _temporal_split(rs, cfg.temporal_train_ratio) for u, rs in all_records.items()}
    eligible = {u for u, (tr, _) in split.items() if len(tr) >= cfg.min_user_train_checkins}
    activity_orig = {u: len(split[u][0]) for u in eligible}
    sampled = _connected_sample(edges, eligible, activity_orig, cfg.n_users, cfg.random_seed)
    sampled_set = set(sampled)
    old_to_new = {u: i for i, u in enumerate(sampled)}

    train = {u: split[u][0] for u in sampled}
    test = {u: split[u][1] for u in sampled}
    train_cell_count: dict[tuple[int, int], int] = defaultdict(int)
    train_cell_users: dict[tuple[int, int], set[int]] = defaultdict(set)
    test_cell_count: dict[tuple[int, int], int] = defaultdict(int)
    test_cell_users: dict[tuple[int, int], set[int]] = defaultdict(set)
    for u in sampled:
        for _, _, _, cell in train[u]:
            train_cell_count[cell] += 1; train_cell_users[cell].add(u)
        for _, _, _, cell in test[u]:
            test_cell_count[cell] += 1; test_cell_users[cell].add(u)
    candidates = []
    for cell, c in train_cell_count.items():
        nu = len(train_cell_users[cell])
        if c >= cfg.min_task_train_checkins and nu >= cfg.min_task_train_users:
            candidates.append((cell[0], cell[1], c, nu))
    selected = _select_tasks(candidates, cfg)
    task_cells = [(x, y) for x, y, *_ in selected]
    centers = np.asarray([_cell_center(x, y, cfg) for x, y in task_cells], dtype=float)

    future_density = np.asarray([len(test_cell_users.get(cell, set())) for cell in task_cells], dtype=float)
    z = np.log1p(future_density)
    if z.max() > z.min():
        zn = (z - z.min()) / (z.max() - z.min())
    else:
        zn = np.full_like(z, 0.5)
    demand = cfg.demand_min + (cfg.demand_max - cfg.demand_min) * zn

    n, t = cfg.n_users, cfg.n_tasks
    accessibility = np.zeros((n, t), dtype=np.float32)
    quality = np.zeros((n, t), dtype=np.float32)
    activity = np.zeros(n, dtype=float)
    home = np.zeros((n, 2), dtype=float)
    for old_u, new_u in old_to_new.items():
        rows = train[old_u]
        lat = np.asarray([r[1] for r in rows], dtype=float)
        lon = np.asarray([r[2] for r in rows], dtype=float)
        activity[new_u] = len(rows)
        home[new_u] = (float(np.median(lat)), float(np.median(lon)))
        d = haversine_km(lat[:, None], lon[:, None], centers[None, :, 0], centers[None, :, 1])
        min_d = d.min(axis=0)
        near_count = (d <= cfg.quality_radius_km).sum(axis=0)
        accessibility[new_u] = np.exp(-min_d / max(cfg.accessibility_tau_km, 1e-9))
        quality[new_u] = 1.0 - np.exp(-near_count / max(cfg.quality_beta, 1e-9))

    graph_u = nx.Graph()
    graph_u.add_nodes_from(range(n))
    retained = []
    for u0, v0 in edges:
        if u0 in sampled_set and v0 in sampled_set:
            u, v = old_to_new[u0], old_to_new[v0]
            graph_u.add_edge(u, v); retained.append((u, v))
    act_rank = _rank01(np.log1p(activity))
    nbrs = {u: set(graph_u.neighbors(u)) for u in graph_u.nodes()}
    graph = nx.DiGraph(); graph.add_nodes_from(range(n))
    for u, v in retained:
        union = len(nbrs[u] | nbrs[v]); jac = len(nbrs[u] & nbrs[v]) / union if union else 0.0
        hd = float(haversine_km(home[u, 0], home[u, 1], home[v, 0], home[v, 1]))
        mobility = math.exp(-hd / max(cfg.influence_mobility_tau_km, 1e-9))
        relation = 0.7 * mobility + 0.3 * jac
        for src, dst in ((u, v), (v, u)):
            directional = 0.50 * act_rank[src] + 0.20 * act_rank[dst] + 0.30 * relation
            w = cfg.influence_min + (cfg.influence_max - cfg.influence_min) * directional
            graph.add_edge(src, dst, weight=float(np.clip(w, cfg.influence_min, cfg.influence_max)))

    pool = _stratified_worker_pool(activity, min(cfg.worker_pool_size, n), cfg.random_seed + 991)
    inst = JournalInstance(graph, accessibility, quality, demand.astype(np.float32), pool, centers)
    manifest = {
        "config": asdict(cfg),
        "sampled_original_user_ids": sampled,
        "graph_edges": graph.number_of_edges(),
        "candidate_task_cells": len(candidates),
        "task_cells_xy": task_cells,
        "task_train_checkins": [int(train_cell_count[c]) for c in task_cells],
        "task_train_unique_users": [int(len(train_cell_users[c])) for c in task_cells],
        "task_test_checkins": [int(test_cell_count.get(c, 0)) for c in task_cells],
        "task_test_unique_users": future_density.astype(int).tolist(),
        "demand_definition": "future unique-user density mapped monotonically to [demand_min,demand_max] using log1p min-max scaling",
        "accessibility_definition": "exp(-historical minimum haversine distance / accessibility_tau_km)",
        "quality_definition": "1-exp(-historical visits within quality_radius_km / quality_beta)",
        "influence_definition": "global asymmetric campaign diffusion from activity ranks, home-location similarity and neighbor Jaccard",
        "worker_capacity_default": 1,
    }
    return inst, manifest


def save_journal_instance(inst: JournalInstance, manifest: dict, output_dir: str | Path):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    edges = np.asarray([(u, v, float(d.get("weight", 0.0))) for u, v, d in inst.graph.edges(data=True)], dtype=np.float64)
    np.savez_compressed(
        out / "instance.npz",
        accessibility=inst.accessibility,
        quality=inst.quality,
        demand=inst.demand,
        edges=edges,
        worker_pool=np.asarray(sorted(inst.worker_pool), dtype=np.int64),
        task_centers_lat_lon=inst.task_centers_lat_lon,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_journal_instance(output_dir: str | Path):
    out = Path(output_dir); z = np.load(out / "instance.npz")
    graph = nx.DiGraph(); graph.add_nodes_from(range(z["accessibility"].shape[0]))
    for row in z["edges"]:
        graph.add_edge(int(row[0]), int(row[1]), weight=float(row[2]))
    inst = JournalInstance(
        graph=graph,
        accessibility=z["accessibility"],
        quality=z["quality"],
        demand=z["demand"],
        worker_pool=set(int(x) for x in z["worker_pool"].tolist()),
        task_centers_lat_lon=z["task_centers_lat_lon"],
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return inst, manifest
