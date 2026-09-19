#!/usr/bin/env python3
"""Run and persist the original six-baseline comparison on fixed instances."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'repro/src'))
from dqnselector.baselines import celf, degree_greedy, one_step_coverage_greedy
from dqnselector.paper_baselines import mobility_profiles, fast_selector, kt_voting, load_mobility_profiles, save_mobility_profiles
from dqnselector.piano import PianoQNet, piano_select, train_piano
from dqnselector.reconstruction import load_reconstruction
from dqnselector.oracle import LiveEdgeECOracle
from dqnselector.selector import RainbowSelector, greedy_select
from dqnselector.multibudget import train_rainbow_selector_multibudget

BUDGETS = [50,60,70,80,90,100]


def save_json(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def paths(ds, n):
    inst=ROOT/'repro/data/scientific-v2-pilot'/f'{ds}-{n}-struct2024-task2024-rho150'
    emb=ROOT/'repro/results/scientific-v2-pilot'/f'{ds}-task2024-embedding/embeddings.npz'
    if n!=3000:
        emb=ROOT/'repro/results/sigir-a40-inputs'/f'{ds}-{n}-embedding/embeddings.npz'
    return inst,emb


def prepare(ds,n):
    instance,emb=paths(ds,n)
    raw=ROOT/'repro/data/raw'
    if not (instance/'instance.npz').exists():
        subprocess.run([sys.executable,str(ROOT/'repro/scripts/build_scientific_v2.py'),
            '--dataset',ds,'--users',str(n),'--structure-seed','2024','--task-seed','2024',
            '--load-factor','1.50','--edges',str(raw/f'loc-{ds}_edges.txt.gz'),
            '--checkins',str(raw/f'loc-{ds}_totalCheckins.txt.gz'),'--output',str(instance)],check=True)
    if not emb.exists():
        subprocess.run([sys.executable,str(ROOT/'repro/scripts/precompute_embeddings.py'),
            '--instance',str(instance),'--output',str(emb.parent),'--social-dim','128',
            '--mc','100','--graph-iterations','30','--seed','2024'],check=True)
    inst,manifest=load_reconstruction(instance)
    pool=sorted(inst.worker_pool)
    profile_file=emb.parent/'mobility_profiles.npz'
    try:
        if profile_file.exists():
            load_mobility_profiles(profile_file, pool)
        else:
            raise ValueError("missing profile cache")
    except (ValueError, KeyError):
        profiles=mobility_profiles(raw/f'loc-{ds}_totalCheckins.txt.gz',manifest,pool)
        save_mobility_profiles(profile_file, profiles, pool)
    print(f'PREPARED {ds} n={n} instance={instance} embeddings={emb}',flush=True)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def reuse_dqn(source, destination, config, variant, seed):
    """Reuse completed training only after verifying inputs and unchanged DQN code."""
    source=Path(source)
    old=json.loads((source/'config.json').read_text())
    for key in ['instance_sha256','embedding_sha256','train_mc','validation_mc','validation_seed','budgets']:
        if old[key]!=config[key]:
            raise ValueError(f'incompatible reusable DQN inputs: {key}')
    if seed not in old['seeds'] or variant not in old['dqn_variants']:
        return
    run=source/f'dqn-{variant}-seed{seed}'
    if not (run/'metrics.json').exists():
        return
    for name in ['selector.py','multibudget.py','rainbow.py','fusion.py','oracle.py']:
        relative='repro/src/dqnselector/'+name
        previous=subprocess.check_output(['git','show',old['git_head']+':'+relative],cwd=ROOT)
        if previous!=(ROOT/relative).read_bytes():
            raise ValueError(f'DQN implementation changed: {relative}')
    previous=subprocess.check_output(['git','show',old['git_head']+':repro/scripts/sigir_experiment.py'],cwd=ROOT,text=True)
    def training_block(script):
        return script.split('\n                def callback(episode,model,total):',1)[1].split('            metrics=json.loads',1)[0]
    if training_block(previous)!=training_block(Path(__file__).read_text()):
        raise ValueError('DQN training or checkpoint-selection configuration changed')
    for name in ['model.pt','metrics.json','progress.json']:
        shutil.copy2(run/name,destination/name)
    save_json(destination/'reuse-provenance.json',dict(source=str(run),source_commit=old['git_head'],
        checkpoint_sha256=sha256(run/'model.pt'),verified_inputs_and_dqn_code=True))
    print('REUSED_DQN',variant,seed,run,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['gowalla','brightkite'],required=True)
    p.add_argument('--users',type=int,default=3000)
    p.add_argument('--phase',choices=['prepare','run'],default='run')
    p.add_argument('--seeds',type=int,nargs='+',default=[2024,2025,2026])
    p.add_argument('--dqn-variants',nargs='+',default=['paper-horizon','multi-budget'])
    p.add_argument('--piano-episodes',type=int,default=200)
    p.add_argument('--reuse-dqn-from')
    p.add_argument('--evaluation-mc',type=int,default=1000)
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    if a.phase=='prepare':
        prepare(a.dataset,a.users)
        return
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; do not silently change timing device')
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    instance,emb_path=paths(a.dataset,a.users)
    inst,manifest=load_reconstruction(instance)
    emb=np.load(emb_path)
    pool=sorted(inst.worker_pool)
    config=vars(a)|dict(budgets=BUDGETS,instance_sha256=sha256(instance/'instance.npz'),
        embedding_sha256=sha256(emb_path),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=platform.python_version(),torch=torch.__version__,device_name=torch.cuda.get_device_name() if a.device.startswith('cuda') else platform.processor(),
        scope='instance-specific training; independent IC validation and evaluation; not unseen-task generalization',
        train_mc=100,validation_mc=300,validation_seed=314159,evaluation_seed=271828)
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for key in ['instance_sha256','embedding_sha256','git_head','evaluation_mc','piano_episodes','dqn_variants']:
            if old[key]!=config[key]:
                raise ValueError(f'resume configuration changed: {key}')
    else:
        save_json(out/'config.json',config)
    t=time.perf_counter()
    evaluation=LiveEdgeECOracle(inst,mc_times=a.evaluation_mc,random_seed=271828)
    validation=LiveEdgeECOracle(inst,mc_times=300,random_seed=314159)
    save_json(out/'evaluator.json',dict(construction_seconds=time.perf_counter()-t,
                                    evaluation_stats=evaluation.stats.__dict__))
    rows=[]
    def record(method,orders,times,seed=None,extra=None):
        for k in BUDGETS:
            selected=orders[k]
            if len(selected)!=k or len(set(selected))!=k or not set(selected)<=set(pool):
                raise ValueError(f'invalid {method} selection at {k}')
            rows.append(dict(dataset=a.dataset,users=a.users,method=method,seed=seed,k=k,
                ec=evaluation.score(selected),realized_ec=evaluation.realized_score(selected),
                selection_seconds=times[k],selected=selected,**(extra or {})))
        save_json(out/'results.json',rows)
        with (out/'comparison.csv').open('w',newline='') as f:
            fields=['dataset','users','method','seed','k','ec','realized_ec','selection_seconds']
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
        print(method,seed,[(r['k'],round(r['ec'],6)) for r in rows[-len(BUDGETS):]],flush=True)
    baseline_file=out/'baseline_selections.json'
    if baseline_file.exists():
        baseline=json.loads(baseline_file.read_text())
    else:
        baseline={}
        profiles=load_mobility_profiles(emb_path.parent/'mobility_profiles.npz', pool)
        selection=LiveEdgeECOracle(inst,mc_times=100,random_seed=2024+17)
        for name in ['DegGreedy','CovGreedy','FastSelector-SIGIR-adapted','KTVoting2-feasible','CELF']:
            orders,times={},{}
            for k in BUDGETS:
                if name=='CELF':
                    selection.clear_score_cache()
                start=time.perf_counter()
                if name=='DegGreedy': chosen=degree_greedy(inst.graph,k,pool)
                elif name=='CovGreedy': chosen=one_step_coverage_greedy(inst.graph,inst.participation*inst.quality,k,inst.nodes,pool)
                elif name=='FastSelector-SIGIR-adapted': chosen=fast_selector(inst.graph,profiles,pool,k,.56 if a.dataset=='gowalla' else .64)
                elif name=='KTVoting2-feasible': chosen=kt_voting(inst.graph,inst.participation*inst.quality,pool,k)
                else: chosen=celf(pool,k,selection.marginal_gain)
                times[k]=time.perf_counter()-start
                orders[k]=chosen
            baseline[name]=dict(orders=orders,times=times)
            print('BASELINE_SELECTED',name,flush=True)
        save_json(baseline_file,baseline)
    for name,info in baseline.items():
        record(name,{int(k):v for k,v in info['orders'].items()}, {int(k):v for k,v in info['times'].items()})
    for seed in a.seeds:
        training=LiveEdgeECOracle(inst,mc_times=100,random_seed=seed)
        candidates=[]
        for variant in a.dqn_variants:
            if variant not in ['paper-horizon','multi-budget']:
                raise ValueError('unknown DQN variant')
            run=out/f'dqn-{variant}-seed{seed}'
            run.mkdir(exist_ok=True)
            if a.reuse_dqn_from and not (run/'metrics.json').exists():
                reuse_dqn(a.reuse_dqn_from,run,config,variant,seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model=RainbowSelector(emb['social_s'],emb['coverage_r'],worker_pool=pool)
            trace=[]
            best=[-np.inf,0]
            if not (run/'metrics.json').exists():
                training.clear_score_cache()
                def callback(episode,model,total):
                    if episode%10 and episode!=1:
                        return
                    order=greedy_select(model,100,device=a.device)
                    score=float(np.mean([validation.score(order[:k]) for k in BUDGETS]))
                    trace.append(dict(episode=episode,train_return=total,validation_macro_ec=score))
                    if score>best[0]:
                        best[:]=[score,episode]
                        torch.save(model.state_dict(),run/'model.pt')
                    save_json(run/'progress.json',trace)
                    print('DQN_PROGRESS',a.dataset,a.users,variant,seed,trace[-1],flush=True)
                start=time.perf_counter()
                model,stats=train_rainbow_selector_multibudget(model,training.marginal_gain,
                    budgets=[50] if variant=='paper-horizon' else BUDGETS,
                    episodes=100 if variant=='paper-horizon' else 200,
                    random_seed=seed,device=a.device,episode_callback=callback)
                sync(a.device)
                save_json(run/'metrics.json',dict(training_seconds=time.perf_counter()-start,
                    best_validation_macro_ec=best[0],best_episode=best[1],
                    transitions=sum(stats.episode_budgets),trace=trace))
            metrics=json.loads((run/'metrics.json').read_text())
            model.load_state_dict(torch.load(run/'model.pt',map_location=a.device,weights_only=True))
            orders,times={},{}
            greedy_select(model,5,a.device)
            for k in BUDGETS:
                sync(a.device);start=time.perf_counter()
                orders[k]=greedy_select(model,k,a.device)
                sync(a.device);times[k]=time.perf_counter()-start
            candidates.append((metrics['best_validation_macro_ec'],variant,orders,times,run))
            record(f'DQNSelector-{variant}',orders,times,seed,dict(checkpoint_sha256=sha256(run/'model.pt')))
        if candidates:
            chosen=max(candidates,key=lambda x:x[0])
            record('DQNSelector',chosen[2],chosen[3],seed,dict(validation_chosen_variant=chosen[1],checkpoint_sha256=sha256(chosen[4]/'model.pt')))
        run=out/f'piano-seed{seed}'
        run.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        piano=PianoQNet(inst.graph,pool)
        if not (run/'metrics.json').exists():
            training.clear_score_cache()
            piano_best=[-np.inf,0]
            piano_trace=[]
            def reward(selected,v):
                return float(training.activation_probability(set(selected)|{v}).sum()-training.activation_probability(selected).sum())
            def progress(ep,total,seconds):
                order=piano_select(piano,100,a.device)
                score=float(np.mean([validation.score(order[:k]) for k in BUDGETS]))
                piano_trace.append(dict(episode=ep,validation_macro_ec=score))
                if score>piano_best[0]:
                    piano_best[:]=[score,ep]
                    torch.save(piano.state_dict(),run/'model.pt')
                print('PIANO_PROGRESS',a.dataset,a.users,seed,ep,total,round(seconds,2),flush=True)
            piano,metrics=train_piano(piano,reward,episodes=a.piano_episodes,seed=seed,
                device=a.device,progress=progress)
            metrics.update(best_validation_macro_ec=piano_best[0],best_episode=piano_best[1],trace=piano_trace)
            save_json(run/'metrics.json',metrics)
        piano.load_state_dict(torch.load(run/'model.pt',map_location=a.device,weights_only=True))
        orders,times={},{}
        piano_select(piano,5,a.device)
        for k in BUDGETS:
            sync(a.device);start=time.perf_counter()
            orders[k]=piano_select(piano,k,a.device)
            sync(a.device);times[k]=time.perf_counter()-start
        record('PIANO',orders,times,seed,dict(implementation='paper-equation reproduction',checkpoint_sha256=sha256(run/'model.pt')))
    save_json(out/'complete.json',dict(rows=len(rows),completed=True))
    print('COMPARISON_COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
