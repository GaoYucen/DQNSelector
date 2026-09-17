#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'repro'/'src'))

from dqnselector.journal import JournalInstance, load_journal_instance, save_journal_instance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--instance',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--edge-scale',type=float,default=1.0)
    p.add_argument('--demand-scale',type=float,default=1.0)
    a=p.parse_args()
    inst,man=load_journal_instance(a.instance)
    g=inst.graph.copy()
    for _,_,d in g.edges(data=True):
        d['weight']=float(np.clip(float(d.get('weight',0.0))*a.edge_scale,0.0,1.0))
    demand=(inst.demand.astype(np.float64)*a.demand_scale).astype(np.float32)
    out_inst=JournalInstance(g,inst.accessibility,inst.quality,demand,inst.worker_pool,inst.task_centers_lat_lon)
    out_man=copy.deepcopy(man)
    out_man['variant']={
        'source_instance':str(Path(a.instance).resolve()),
        'edge_scale':float(a.edge_scale),
        'demand_scale':float(a.demand_scale),
    }
    out_man['demand_definition_variant']='base demand multiplied by demand_scale'
    out_man['influence_definition_variant']='base edge probabilities multiplied by edge_scale and clipped to [0,1]'
    save_journal_instance(out_inst,out_man,a.output)
    w=np.asarray([float(d.get('weight',0.0)) for _,_,d in g.edges(data=True)],float)
    print(json.dumps({
        'output':str(Path(a.output).resolve()),
        'edge_scale':a.edge_scale,
        'edge_mean':float(w.mean()) if len(w) else 0.0,
        'edge_max':float(w.max()) if len(w) else 0.0,
        'demand_min':float(demand.min()),
        'demand_median':float(np.median(demand)),
        'demand_mean':float(demand.mean()),
        'demand_max':float(demand.max()),
    },indent=2))

if __name__=='__main__': main()
