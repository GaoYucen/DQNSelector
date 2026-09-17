#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'repro'/'src'))

from dqnselector.journal import load_journal_instance
from dqnselector.journal_embeddings import build_journal_embeddings
from dqnselector.journal_oracle import JournalLiveEdgeOracle
from dqnselector.baselines import degree_greedy, celf, greedy_by_marginal_gain
from dqnselector.selector import RainbowSelector, greedy_select


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--instance',required=True); p.add_argument('--output',required=True); p.add_argument('--model')
    p.add_argument('--budgets',type=int,nargs='+',default=[1,3,5,10,20])
    p.add_argument('--selection-mc',type=int,default=10); p.add_argument('--evaluation-mc',type=int,default=50); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--hidden',type=int,default=64); p.add_argument('--atoms',type=int,default=31); p.add_argument('--device',default='cuda')
    p.add_argument('--include-greedy',action='store_true'); p.add_argument('--include-celf',action='store_true')
    return p.parse_args()


def main():
    a=parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    inst,man=load_journal_instance(a.instance); pool=sorted(inst.worker_pool); maxk=min(max(a.budgets),len(pool))
    sel_oracle=JournalLiveEdgeOracle(inst,mc_times=a.selection_mc,random_seed=a.seed+17,worker_capacity=1)
    eval_oracle=JournalLiveEdgeOracle(inst,mc_times=a.evaluation_mc,random_seed=a.seed+100003,worker_capacity=1)
    methods={}; selection_seconds={}

    rng=np.random.default_rng(a.seed)
    t=time.perf_counter(); methods['Random']=rng.permutation(pool).tolist()[:maxk]; selection_seconds['Random']=time.perf_counter()-t
    t=time.perf_counter(); methods['DegGreedy']=degree_greedy(inst.graph,maxk,pool); selection_seconds['DegGreedy']=time.perf_counter()-t
    direct_scores=(inst.suitability/np.maximum(inst.demand[None,:],1e-12)).mean(axis=1)
    t=time.perf_counter(); methods['DirectTask']=sorted(pool,key=lambda u:(float(direct_scores[u]),-u),reverse=True)[:maxk]; selection_seconds['DirectTask']=time.perf_counter()-t

    if a.include_greedy:
        t=time.perf_counter(); methods['GreedyMarginal']=greedy_by_marginal_gain(pool,maxk,sel_oracle.marginal_gain); selection_seconds['GreedyMarginal']=time.perf_counter()-t
    if a.include_celf:
        t=time.perf_counter(); methods['CELF']=celf(pool,maxk,sel_oracle.marginal_gain); selection_seconds['CELF']=time.perf_counter()-t

    if a.model:
        social,task=build_journal_embeddings(inst); device=a.device if (not a.device.startswith('cuda') or torch.cuda.is_available()) else 'cpu'
        model=RainbowSelector(social,task,worker_pool=inst.worker_pool,hidden_dim=a.hidden,atoms=a.atoms,v_min=0.0,v_max=1.0).to(device)
        model.load_state_dict(torch.load(a.model,map_location=device))
        t=time.perf_counter(); methods['DQNSelector']=greedy_select(model,maxk,device=device)
        if device.startswith('cuda'): torch.cuda.synchronize()
        selection_seconds['DQNSelector']=time.perf_counter()-t

    rows=[]
    for name,order in methods.items():
        for k in sorted(set(a.budgets)):
            if k>len(order): continue
            t=time.perf_counter(); r=eval_oracle.evaluate(order[:k]); sec=time.perf_counter()-t
            row={'method':name,'k':k,**r,'selection_seconds_max_k':selection_seconds[name],'evaluation_seconds':sec}; rows.append(row)
            print(name,'k',k,'mean',f"{r['mean']:.6f}",'p10',f"{r['p10']:.6f}",'unsat',f"{r['unsatisfied_ratio']:.3f}",'sat',f"{r['saturated_ratio']:.3f}",'active',f"{r['mean_active']:.1f}",'sel',f"{selection_seconds[name]:.3f}")
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={'selection_mc':a.selection_mc,'evaluation_mc':a.evaluation_mc,'selection_seconds':selection_seconds,'graph_nodes':inst.graph.number_of_nodes(),'graph_edges':inst.graph.number_of_edges(),'worker_pool':len(pool),'instance_config':man.get('config',{}),'variant':man.get('variant',{})}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__': main()
