#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from dqnselector.journal import JournalConfig, build_journal_instance, save_journal_instance


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['gowalla','brightkite'], required=True)
    p.add_argument('--edges', required=True)
    p.add_argument('--checkins', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--users', type=int, default=3000)
    p.add_argument('--workers', type=int, default=300)
    p.add_argument('--tasks', type=int, default=100)
    p.add_argument('--layout', choices=['scattered','clustered','contiguous'], default='scattered')
    p.add_argument('--demand-min', type=float, default=3.0)
    p.add_argument('--demand-max', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=2026)
    return p.parse_args()


def main():
    a=parse_args()
    cfg=JournalConfig(
        dataset_name=a.dataset,
        n_users=a.users,
        worker_pool_size=a.workers,
        n_tasks=a.tasks,
        layout=a.layout,
        demand_min=a.demand_min,
        demand_max=a.demand_max,
        random_seed=a.seed,
    )
    inst, manifest=build_journal_instance(a.edges,a.checkins,cfg)
    save_journal_instance(inst,manifest,a.output)
    print('saved',a.output)
    print('nodes',inst.graph.number_of_nodes(),'edges',inst.graph.number_of_edges(),'workers',len(inst.worker_pool),'tasks',len(inst.demand))
    print('demand',float(inst.demand.min()),float(inst.demand.mean()),float(inst.demand.max()))
    print('suitability_nonzero',float((inst.suitability>0).mean()),'mean',float(inst.suitability.mean()),'max',float(inst.suitability.max()))

if __name__=='__main__': main()
