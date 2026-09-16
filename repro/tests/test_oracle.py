import networkx as nx
import numpy as np

from dqnselector.ecm import ECMInstance
from dqnselector.oracle import LiveEdgeECOracle


def test_live_edge_oracle_single_edge_expectation():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=0.5)
    inst = ECMInstance(
        graph=g,
        nodes=[0, 1],
        participation=np.ones((2, 1)),
        quality=np.ones((2, 1)),
        demand=np.array([10.0]),
        worker_pool={0, 1},
    )
    oracle = LiveEdgeECOracle(inst, mc_times=20000, random_seed=11)
    # Seed 0 always contributes one node and reaches node 1 with p=0.5.
    assert abs(oracle.score({0}) - 0.15) < 0.005
    assert abs(oracle.marginal_gain(set(), 0) - oracle.score({0})) < 1e-12


def test_live_edge_oracle_union_avoids_double_counting():
    g = nx.DiGraph()
    g.add_edge(0, 2, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    inst = ECMInstance(
        graph=g,
        nodes=[0, 1, 2],
        participation=np.ones((3, 1)),
        quality=np.ones((3, 1)),
        demand=np.array([3.0]),
        worker_pool={0, 1},
    )
    oracle = LiveEdgeECOracle(inst, mc_times=10, random_seed=0)
    assert np.isclose(oracle.score({0}), 2 / 3)
    assert np.isclose(oracle.score({0, 1}), 1.0)
    assert np.isclose(oracle.marginal_gain({0}, 1), 1 / 3)
