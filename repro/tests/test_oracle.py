import networkx as nx
import numpy as np

from dqnselector.ecm import ECMInstance, effective_coverage_from_activation
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
    assert abs(oracle.score({0}) - 0.15) < 0.005
    assert abs(oracle.realized_score({0}) - 0.15) < 0.005
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
    assert np.isclose(oracle.realized_score({0, 1}), 1.0)
    assert np.isclose(oracle.marginal_gain({0}, 1), 1 / 3)


def test_oracle_caps_expected_coverage_not_each_world():
    """Canonical EC and realized EC must remain distinct under nonlinear clipping."""
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=0.75)
    inst = ECMInstance(
        graph=g,
        nodes=[0, 1],
        participation=np.array([[0.0], [1.0]]),
        quality=np.array([[0.0], [2.0]]),
        demand=np.array([1.0]),
        worker_pool={0},
    )
    oracle = LiveEdgeECOracle(inst, mc_times=2000, random_seed=17)
    activation = oracle.activation_probability({0})
    ec, _, _ = effective_coverage_from_activation(
        inst, {v: float(activation[v]) for v in inst.nodes}
    )

    assert np.isclose(oracle.score({0}), ec)
    assert oracle.score({0}) > 0.95
    # A realized world contributes either 0 or 1 after clipping, hence ~=0.75.
    assert abs(oracle.realized_score({0}) - 0.75) < 0.04
    assert oracle.realized_score({0}) < oracle.score({0})


def test_activation_probability_cache_is_copy_safe():
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=0.5)
    inst = ECMInstance(
        graph=g,
        nodes=[0, 1],
        participation=np.ones((2, 1)),
        quality=np.ones((2, 1)),
        demand=np.ones(1),
        worker_pool={0},
    )
    oracle = LiveEdgeECOracle(inst, mc_times=50, random_seed=3)
    p = oracle.activation_probability({0})
    p[:] = 0.0
    p2 = oracle.activation_probability({0})
    assert p2[0] == 1.0
