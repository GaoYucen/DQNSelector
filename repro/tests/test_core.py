import networkx as nx
import numpy as np

from dqnselector.ecm import ECMInstance, effective_coverage_from_activation
from dqnselector.influence import mc_activation_probabilities
from dqnselector.piic import enumerate_path_embedding_dag, piic


def test_ic_chain_probability():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=0.5)
    p = mc_activation_probabilities(g, [0], mc_times=20000, random_seed=7)
    assert p[0] == 1.0
    assert abs(p[1] - 0.5) < 0.02


def test_effective_coverage_cap_and_demand():
    g = nx.DiGraph()
    g.add_nodes_from([0, 1])
    inst = ECMInstance(
        graph=g,
        nodes=[0, 1],
        participation=np.array([[1.0], [0.5]]),
        quality=np.array([[2.0], [2.0]]),
        demand=np.array([2.5]),
    )
    ec, per_area, raw = effective_coverage_from_activation(inst, {0: 1.0, 1: 1.0})
    assert np.allclose(raw, [3.0])
    assert np.allclose(per_area, [1.0])
    assert ec == 1.0


def test_piic_matches_explicit_paths_on_dag():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=0.5)
    g.add_edge(1, 2, weight=0.2)
    g.add_edge(0, 2, weight=0.1)
    quality = np.array([[1.0], [2.0], [3.0]])
    fast = piic(g, quality, influence_range=2, nodes=[0, 1, 2])
    slow = enumerate_path_embedding_dag(g, quality, influence_range=2, nodes=[0, 1, 2])
    assert np.allclose(fast, slow)
    assert np.allclose(fast[0], [1.6])
