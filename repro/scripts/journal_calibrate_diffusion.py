#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'repro' / 'src'))

from dqnselector.journal import JournalInstance, load_journal_instance
from dqnselector.journal_oracle import JournalLiveEdgeOracle
from dqnselector.baselines import degree_greedy


def spectral_radius_power(graph, iterations: int = 120) -> float:
    """Perron power estimate of the weighted directed transmission matrix radius."""
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    x = np.ones(n, dtype=np.float64) / np.sqrt(n)
    lam = 0.0
    for _ in range(iterations):
        y = np.zeros(n, dtype=np.float64)
        for u, v, d in graph.edges(data=True):
            y[int(v)] += float(d.get('weight', 0.0)) * x[int(u)]
        norm = float(np.linalg.norm(y))
        if norm <= 1e-15:
            return 0.0
        x = y / norm
        lam = norm
    # Collatz-like ratio is unstable at zero coordinates; norm ratio is adequate for calibration.
    return float(lam)


def cascade_stats(oracle: JournalLiveEdgeOracle, workers: list[int]) -> dict[str, float]:
    vals = []
    for u in workers:
        for w in range(oracle.mc_times):
            vals.append(len(oracle._active_for_world((u,), w)))
    a = np.asarray(vals, dtype=np.float64)
    return {
        'singleton_active_mean': float(a.mean()),
        'singleton_active_median': float(np.median(a)),
        'singleton_active_p90': float(np.quantile(a, 0.90)),
        'singleton_active_p95': float(np.quantile(a, 0.95)),
        'singleton_active_p99': float(np.quantile(a, 0.99)),
        'singleton_active_max': float(a.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instance', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scales', type=float, nargs='+', default=[0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.0])
    p.add_argument('--mc', type=int, default=30)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--cascade-workers', type=int, default=80)
    a = p.parse_args()

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    base, _ = load_journal_instance(a.instance)
    pool = sorted(base.worker_pool)
    rng = np.random.default_rng(a.seed)
    probe_workers = sorted(rng.choice(np.asarray(pool, dtype=int), size=min(a.cascade_workers, len(pool)), replace=False).tolist())
    random_order = rng.permutation(pool).tolist()
    degree_order = degree_greedy(base.graph, 20, pool)
    direct_scores = (base.suitability / np.maximum(base.demand[None,:], 1e-12)).mean(axis=1)
    direct_order = sorted(pool, key=lambda u:(float(direct_scores[u]), -u), reverse=True)[:20]

    rows = []
    for scale in a.scales:
        g = base.graph.copy()
        for _, _, d in g.edges(data=True):
            d['weight'] = float(np.clip(float(d.get('weight',0.0)) * scale, 0.0, 1.0))
        inst = JournalInstance(g, base.accessibility, base.quality, base.demand, base.worker_pool, base.task_centers_lat_lon)
        weights = np.asarray([float(d['weight']) for _,_,d in g.edges(data=True)], dtype=float)
        rho = spectral_radius_power(g)
        oracle = JournalLiveEdgeOracle(inst, mc_times=a.mc, random_seed=a.seed + 7001, worker_capacity=1)
        cs = cascade_stats(oracle, probe_workers)
        base_row = {
            'scale': float(scale),
            'edge_weight_mean': float(weights.mean()),
            'edge_weight_p95': float(np.quantile(weights, .95)),
            'edge_weight_max': float(weights.max()),
            'spectral_radius_est': rho,
            **cs,
        }
        for method, order in [('Random', random_order), ('Degree', degree_order), ('DirectTask', direct_order)]:
            for k in [1,5,20]:
                r = oracle.evaluate(order[:k])
                rows.append({**base_row, 'method': method, 'k': k, **r})
                print(f"scale={scale:.3f} rho={rho:.3f} {method} k={k} mean={r['mean']:.4f} active={r['mean_active']:.1f} sat={r['saturated_ratio']:.3f}")

    with (out/'diffusion_calibration.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # A non-performance-based recommendation: choose the largest near/subcritical
    # scale (rho <= 0.90) whose 95th-percentile singleton cascade is <= 100 workers.
    candidates = []
    seen = set()
    for row in rows:
        key = row['scale']
        if key in seen: continue
        seen.add(key)
        if row['spectral_radius_est'] <= 0.90 and row['singleton_active_p95'] <= 100:
            candidates.append(row)
    rec = max(candidates, key=lambda r:r['scale']) if candidates else min(rows, key=lambda r:abs(r['spectral_radius_est']-0.90))
    summary = {
        'criterion': 'largest tested scale with weighted spectral radius <=0.90 and singleton cascade p95 <=100',
        'recommended_scale': float(rec['scale']),
        'spectral_radius_est': float(rec['spectral_radius_est']),
        'singleton_active_p95': float(rec['singleton_active_p95']),
        'singleton_active_mean': float(rec['singleton_active_mean']),
        'note': 'criterion depends only on diffusion stability/cascade size, not on DQN or baseline performance',
    }
    (out/'recommendation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
