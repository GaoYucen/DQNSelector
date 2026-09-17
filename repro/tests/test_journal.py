import networkx as nx
import numpy as np
import pytest

from dqnselector.assignment import (
    greedy_capacity_assignment,
    lp_relaxation_upper_bound,
    unconstrained_parallel_satisfaction,
)
from dqnselector.journal import JournalInstance
from dqnselector.journal_embeddings import singleton_service_embedding
from dqnselector.journal_oracle import JournalLiveEdgeOracle


def test_capacity_one_prevents_parallel_double_service():
    suitability=np.array([[1.0,1.0]])
    demand=np.array([1.0,1.0])
    r=greedy_capacity_assignment([0],suitability,demand,capacity=1)
    assert np.isclose(r.mean_satisfaction,0.5)
    assert np.isclose(unconstrained_parallel_satisfaction([0],suitability,demand),1.0)


def test_two_workers_can_fill_two_tasks():
    suitability=np.array([[1.0,0.0],[0.0,1.0]])
    demand=np.array([1.0,1.0])
    r=greedy_capacity_assignment([0,1],suitability,demand,capacity=1)
    assert np.isclose(r.mean_satisfaction,1.0)
    assert r.saturated_ratio == 1.0


def test_live_edge_oracle_respects_capacity():
    g=nx.DiGraph(); g.add_nodes_from([0,1]); g.add_edge(0,1,weight=1.0)
    a=np.ones((2,2),dtype=float); q=np.ones((2,2),dtype=float); d=np.ones(2,dtype=float)
    inst=JournalInstance(g,a,q,d,{0},np.zeros((2,2)))
    oracle=JournalLiveEdgeOracle(inst,mc_times=2,random_seed=1,worker_capacity=1)
    res=oracle.evaluate([0])
    assert np.isclose(res['mean'],1.0)  # two activated workers, one task each
    assert np.isclose(res['parallel_relaxation'],1.0)


def test_singleton_service_embedding_includes_downstream_assignment():
    g=nx.DiGraph(); g.add_nodes_from([0,1]); g.add_edge(0,1,weight=1.0)
    accessibility=np.ones((2,2),dtype=float)
    quality=np.array([[1.0,0.0],[0.0,1.0]],dtype=float)
    demand=np.ones(2,dtype=float)
    inst=JournalInstance(g,accessibility,quality,demand,{0},np.zeros((2,2)))
    emb=singleton_service_embedding(inst,mc_times=3,random_seed=7,worker_capacity=1)
    assert np.allclose(emb[0],[1.0,1.0])
    assert np.allclose(emb[1],[0.0,0.0])


def test_lp_relaxation_is_upper_bound_on_greedy():
    pytest.importorskip('scipy')
    suitability=np.array([
        [1.0,0.8,0.0],
        [0.8,1.0,0.2],
        [0.0,0.2,1.0],
    ])
    demand=np.array([1.0,1.0,1.0])
    greedy=greedy_capacity_assignment([0,1,2],suitability,demand,capacity=1)
    upper=lp_relaxation_upper_bound([0,1,2],suitability,demand,capacity=1)
    assert upper + 1e-9 >= greedy.mean_satisfaction
    assert upper <= 1.0 + 1e-9
