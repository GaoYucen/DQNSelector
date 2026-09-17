#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv
from pathlib import Path
import sys, copy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'repro'/'src'))

from dqnselector.journal import load_journal_instance, JournalInstance
from dqnselector.journal_oracle import JournalLiveEdgeOracle
from dqnselector.baselines import degree_greedy


def main():
    p=argparse.ArgumentParser(); p.add_argument('--instance',required=True); p.add_argument('--output',required=True)
    p.add_argument('--scales',type=float,nargs='+',default=[0.1,0.25,0.5,1.0]); p.add_argument('--mc',type=int,default=20); p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    base,_=load_journal_instance(a.instance); pool=sorted(base.worker_pool); rng=np.random.default_rng(a.seed)
    random_order=rng.permutation(pool).tolist(); degree_order=degree_greedy(base.graph,20,pool)
    direct_scores=(base.suitability/np.maximum(base.demand[None,:],1e-12)).mean(axis=1)
    direct_order=sorted(pool,key=lambda u:(float(direct_scores[u]),-u),reverse=True)[:20]
    rows=[]
    for scale in a.scales:
        g=base.graph.copy()
        for u,v,d in g.edges(data=True): d['weight']=float(np.clip(float(d.get('weight',0))*scale,0,1))
        inst=JournalInstance(g,base.accessibility,base.quality,base.demand,base.worker_pool,base.task_centers_lat_lon)
        weights=np.asarray([d['weight'] for _,_,d in g.edges(data=True)],float)
        oracle=JournalLiveEdgeOracle(inst,mc_times=a.mc,random_seed=a.seed+777,worker_capacity=1)
        for method,order in [('Random',random_order),('Degree',degree_order),('DirectTask',direct_order)]:
            for k in [1,5,20]:
                r=oracle.evaluate(order[:k]); row={'scale':scale,'edge_weight_mean':float(weights.mean()),'edge_weight_max':float(weights.max()),'method':method,'k':k,**r}; rows.append(row)
                print(scale,method,k,'mean',f"{r['mean']:.4f}",'active',f"{r['mean_active']:.1f}",'sat',f"{r['saturated_ratio']:.3f}")
    with (out/'influence_sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__=='__main__': main()
