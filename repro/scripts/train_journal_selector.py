#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'repro'/'src'))

from dqnselector.journal import load_journal_instance
from dqnselector.journal_embeddings import build_journal_embeddings
from dqnselector.journal_oracle import JournalLiveEdgeOracle
from dqnselector.selector import RainbowSelector, greedy_select, train_rainbow_selector


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--instance',required=True); p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=30); p.add_argument('--train-budget',type=int,default=20)
    p.add_argument('--eval-budgets',type=int,nargs='+',default=[5,10,20,30,50]); p.add_argument('--oracle-mc',type=int,default=10)
    p.add_argument('--eval-mc',type=int,default=30); p.add_argument('--hidden',type=int,default=64); p.add_argument('--atoms',type=int,default=31)
    p.add_argument('--learning-rate',type=float,default=3e-4); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--task-embedding',choices=['direct','singleton_mc'],default='direct')
    p.add_argument('--embedding-mc',type=int,default=20)
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--device',default='cuda')
    return p.parse_args()


def main():
    a=parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    inst,man=load_journal_instance(a.instance)
    t=time.perf_counter()
    social,task=build_journal_embeddings(
        inst,
        task_mode=a.task_embedding,
        mc_times=a.embedding_mc,
        random_seed=a.seed+3001,
        worker_capacity=1,
    )
    embedding_sec=time.perf_counter()-t
    np.savez_compressed(out/'embeddings.npz',social=social,task=task)
    device=a.device if (not a.device.startswith('cuda') or torch.cuda.is_available()) else 'cpu'
    t=time.perf_counter(); train_oracle=JournalLiveEdgeOracle(inst,mc_times=a.oracle_mc,random_seed=a.seed,worker_capacity=1); oracle_build=time.perf_counter()-t
    model=RainbowSelector(social,task,worker_pool=inst.worker_pool,hidden_dim=a.hidden,atoms=a.atoms,v_min=0.0,v_max=1.0)
    t=time.perf_counter(); model,stats=train_rainbow_selector(model,train_oracle.marginal_gain,seed_budget=a.train_budget,episodes=a.episodes,learning_rate=a.learning_rate,batch_size=a.batch_size,warmup=a.warmup,target_update_interval=50,random_seed=a.seed,device=device); train_sec=time.perf_counter()-t
    eval_oracle=JournalLiveEdgeOracle(inst,mc_times=a.eval_mc,random_seed=a.seed+100003,worker_capacity=1)
    order=greedy_select(model,min(max(a.eval_budgets),len(inst.worker_pool)),device=device)
    evaluation={}
    for k in sorted(set(a.eval_budgets)):
        if k>len(order): continue
        evaluation[str(k)]=eval_oracle.evaluate(order[:k]); print('k',k,evaluation[str(k)])
    torch.save(model.state_dict(),out/'model.pt'); np.save(out/'selection_order.npy',np.asarray(order,dtype=np.int64))
    metrics={'device':device,'episodes':a.episodes,'train_budget':a.train_budget,'oracle_mc':a.oracle_mc,'eval_mc':a.eval_mc,'task_embedding':a.task_embedding,'embedding_mc':a.embedding_mc,'embedding_seconds':embedding_sec,'oracle_build_seconds':oracle_build,'training_seconds':train_sec,'episode_returns':stats.episode_returns,'losses':stats.losses,'evaluation':evaluation,'selection_order':order,'instance_config':man.get('config',{}),'variant':man.get('variant',{})}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print('saved',out)

if __name__=='__main__': main()
