#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'repro/src'))
from dqnselector.reconstruction import load_reconstruction
from dqnselector.selector import RainbowSelector, greedy_select
from dqnselector.multibudget import train_rainbow_selector_multibudget
from dqnselector.oracle import LiveEdgeECOracle

torch.set_num_threads(4)
reference = json.loads((ROOT/'repro/results/migration-reference-4090.json').read_text())
rows = []
for ref in reference['rows']:
    ds = ref['dataset']
    inst, _ = load_reconstruction(ROOT/'repro/data/scientific-v2-pilot'/f'{ds}-3000-struct2024-task2024-rho150')
    emb = np.load(ROOT/'repro/results/scientific-v2-pilot'/f'{ds}-task2024-embedding/embeddings.npz')
    model = RainbowSelector(emb['social_s'],emb['coverage_r'],worker_pool=inst.worker_pool)
    state = torch.load(ROOT/'repro/results/scientific-v2-pilot/v3-full-baseline-equalcompute'/f'{ds}-trainseed2024/model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state)
    oracle = LiveEdgeECOracle(inst,mc_times=30,random_seed=777)
    cpu_order = greedy_select(model,100,'cpu')
    assert cpu_order == ref['order'], f'{ds}: CPU selections changed'
    ec = {str(k):oracle.score(cpu_order[:k]) for k in [5,50,100]}
    assert all(abs(ec[k]-ref['ec'][k]) < 1e-10 for k in ec)
    gpu_order = greedy_select(model,100,'cuda')
    gpu_ec = {str(k):oracle.score(gpu_order[:k]) for k in [5,50,100]}
    assert all(abs(gpu_ec[k]-ec[k]) < 1e-3 for k in ec), f'{ds}: GPU EC drift'
    torch.manual_seed(42)
    smoke = RainbowSelector(emb['social_s'],emb['coverage_r'],worker_pool=inst.worker_pool)
    smoke, stats = train_rainbow_selector_multibudget(smoke,oracle.marginal_gain,[5],episodes=3,
        batch_size=4,warmup=4,target_update_interval=4,random_seed=42,device='cuda')
    assert stats.losses and np.isfinite(stats.losses).all()
    rows.append(dict(dataset=ds,cpu_exact=True,cpu_ec=ec,gpu_ec=gpu_ec,
                     gpu_order_equal=gpu_order==cpu_order,smoke_losses=stats.losses))
result=dict(reference_environment={k:v for k,v in reference.items() if k!='rows'},
            target_torch=torch.__version__,gpu=torch.cuda.get_device_name(),rows=rows)
(ROOT/'repro/results/migration-validation-a40.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
