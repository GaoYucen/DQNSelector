#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'repro'/'src'))

from dqnselector.journal import load_journal_instance
from dqnselector.journal_oracle import JournalLiveEdgeOracle
from dqnselector.baselines import degree_greedy


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--instance',required=True); p.add_argument('--output',required=True)
    p.add_argument('--mc',type=int,default=20); p.add_argument('--budgets',type=int,nargs='+',default=[1,3,5,10,20,50]); p.add_argument('--seed',type=int,default=2026)
    return p.parse_args()


def main():
    a=parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    inst,man=load_journal_instance(a.instance); pool=sorted(inst.worker_pool); maxk=min(max(a.budgets),len(pool))
    rng=np.random.default_rng(a.seed); random_order=rng.permutation(pool).tolist()
    degree_order=degree_greedy(inst.graph,maxk,pool)
    direct_scores=(inst.suitability/np.maximum(inst.demand[None,:],1e-12)).mean(axis=1)
    direct_order=sorted(pool,key=lambda u:(float(direct_scores[u]),-u),reverse=True)[:maxk]
    t0=time.perf_counter(); oracle=JournalLiveEdgeOracle(inst,mc_times=a.mc,random_seed=a.seed+100,worker_capacity=1); build=time.perf_counter()-t0
    rows=[]
    for name,order in [('Random',random_order),('Degree',degree_order),('DirectTask',direct_order)]:
        for k in sorted(set(a.budgets)):
            if k>len(order): continue
            t=time.perf_counter(); r=oracle.evaluate(order[:k]); sec=time.perf_counter()-t
            row={'method':name,'k':k,**r,'eval_seconds':sec}; rows.append(row)
            print(name,'k',k,'mean',f"{r['mean']:.6f}",'p10',f"{r['p10']:.6f}",'sat',f"{r['saturated_ratio']:.3f}",'parallel',f"{r['parallel_relaxation']:.6f}")
    with (out/'sanity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={
        'oracle_build_seconds':build,'graph_nodes':inst.graph.number_of_nodes(),'graph_edges':inst.graph.number_of_edges(),
        'demand':{'min':float(inst.demand.min()),'median':float(np.median(inst.demand)),'mean':float(inst.demand.mean()),'max':float(inst.demand.max())},
        'accessibility_nonzero':float((inst.accessibility>0).mean()),'quality_nonzero':float((inst.quality>0).mean()),
        'suitability_nonzero':float((inst.suitability>0).mean()),'suitability_mean':float(inst.suitability.mean()),'suitability_max':float(inst.suitability.max()),
        'task_train_checkins_min':min(man['task_train_checkins']),'task_test_unique_users_zero_fraction':float(np.mean(np.asarray(man['task_test_unique_users'])==0)),
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
