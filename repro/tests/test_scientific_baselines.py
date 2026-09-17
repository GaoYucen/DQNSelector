import networkx as nx
import numpy as np

from dqnselector.baselines import (
    objective_aware_one_step_greedy,
    one_step_coverage_greedy,
)


def test_objective_aware_greedy_includes_direct_worker_utility():
    g = nx.DiGraph()
    g.add_nodes_from([0, 1, 2])
    g.add_edge(1, 2, weight=1.0)
    values = np.array([[0.9], [0.1], [0.1]], dtype=float)
    demand = np.array([1.0], dtype=float)

    # Conference-compatible CovGreedy ignores the candidate's own direct term,
    # therefore node 1 wins because it has an outgoing neighbor.
    legacy = one_step_coverage_greedy(g, values, 1, [0, 1, 2], [0, 1])
    assert legacy == [1]

    # The scientific heuristic must recognize node 0's much larger direct utility.
    scientific = objective_aware_one_step_greedy(
        g, values, demand, 1, [0, 1, 2], [0, 1]
    )
    assert scientific == [0]


def test_objective_aware_greedy_is_demand_aware():
    g = nx.DiGraph()
    g.add_nodes_from([0, 1])
    values = np.array([[0.8, 0.0], [0.0, 0.8]], dtype=float)
    # Area 0 has much larger demand, so the same absolute contribution there is
    # less useful than contribution to area 1.
    demand = np.array([10.0, 1.0], dtype=float)
    order = objective_aware_one_step_greedy(g, values, demand, 2, [0, 1], [0, 1])
    assert order == [1, 0]
