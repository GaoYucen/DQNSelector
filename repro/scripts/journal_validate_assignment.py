#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'repro'/'src'))

from dqnselector.assignment import greedy_capacity_assignment, lp_relaxation_upper_bound
from dqnselector.baselines import degree_greedy
from dqnselector.journal import load_journal_instance
from dqnselector.journal_oracle import JournalLiveEdgeOracle


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--instance',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mc',type=int,default=12)
    p.add_argument('--worlds',type=int,default=8)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    inst,_=load_journal_instance(a.instance)
    pool=sorted(inst.worker_pool)
    rng=np.random.default_rng(a.seed)
    random_order=rng.permutation(pool).tolist()
    degree_order=degree_greedy(inst.graph,20,pool)
    direct_scores=(inst.suitability/np.maximum(inst.demand[None,:],1e-12)).mean(axis=1)
    direct_order=sorted(pool,key=lambda u:(float(direct_scores[u]),-u),reverse=True)[:20]
    oracle=JournalLiveEdgeOracle(inst,mc_times=max(a.mc,a.worlds),random_seed=a.seed+501,worker_capacity=1)

    rows=[]
    for method,order in [('Random',random_order),('Degree',degree_order),('DirectTask',direct_order)]:
        for k in [1,5,20]:
            seeds=order[:k]
            for w in range(min(a.worlds,oracle.mc_times)):
                active=oracle._active_for_world(seeds,w)
                greedy=greedy_capacity_assignment(active,inst.suitability,inst.demand,capacity=1)
                upper=lp_relaxation_upper_bound(active,inst.suitability,inst.demand,capacity=1)
                ratio=greedy.mean_satisfaction/upper if upper>1e-12 else 1.0
                rows.append({
                    'method':method,'k':k,'world':w,'active':len(active),
                    'greedy_mean':greedy.mean_satisfaction,'lp_upper':upper,
                    'greedy_over_lp':ratio,'absolute_gap':upper-greedy.mean_satisfaction,
                })
                print(method,'k',k,'world',w,'active',len(active),'greedy',f'{greedy.mean_satisfaction:.6f}','LP',f'{upper:.6f}','ratio',f'{ratio:.4f}')
    with (out/'assignment_validation.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    ratios=np.asarray([r['greedy_over_lp'] for r in rows],float)
    gaps=np.asarray([r['absolute_gap'] for r in rows],float)
    summary={
        'cases':len(rows),
        'greedy_over_lp_mean':float(ratios.mean()),
        'greedy_over_lp_min':float(ratios.min()),
        'greedy_over_lp_p10':float(np.quantile(ratios,.10)),
        'absolute_gap_mean':float(gaps.mean()),
        'absolute_gap_max':float(gaps.max()),
        'interpretation':'LP is an upper bound, so greedy_over_lp lower-bounds the greedy/optimal-integral ratio for each tested active set.',
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
