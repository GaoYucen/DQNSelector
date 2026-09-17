import networkx as nx
import numpy as np

from dqnselector.ecm import ECMInstance
from dqnselector.objective_features import (
    normalized_direct_contribution,
    objective_aware_coverage_embedding,
)


def test_normalized_direct_contribution_matches_ec_units():
    g = nx.DiGraph()
    g.add_nodes_from([0, 1])
    inst = ECMInstance(
        graph=g,
        nodes=[0, 1],
        participation=np.array([[0.5, 1.0], [1.0, 0.25]]),
        quality=np.array([[0.8, 0.5], [0.4, 0.8]]),
        demand=np.array([2.0, 4.0]),
        worker_pool={0, 1},
    )
    u = normalized_direct_contribution(inst)
    expected = np.array([[0.2, 0.125], [0.2, 0.05]])
    assert np.allclose(u, expected)


def test_objective_embedding_keeps_direct_value_for_isolated_worker():
    g = nx.DiGraph()
    g.add_nodes_from([0, 1])
    g.add_edge(0, 1, weight=0.5)
    inst = ECMInstance(
        graph=g,
        nodes=[0, 1],
        participation=np.array([[1.0], [0.5]]),
        quality=np.array([[0.8], [0.6]]),
        demand=np.array([2.0]),
        worker_pool={0, 1},
    )
    base = normalized_direct_contribution(inst)
    emb = objective_aware_coverage_embedding(inst, influence_range=1, include_direct=True)
    # Node 1 has no outgoing social edge but retains its own direct coverage utility.
    assert np.allclose(emb[1], base[1])
    # Node 0 additionally receives the one-hop propagated utility of node 1.
    assert np.allclose(emb[0], base[0] + 0.5 * base[1])
