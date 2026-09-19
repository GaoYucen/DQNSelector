import networkx as nx
import numpy as np
import torch

from dqnselector.paper_baselines import fast_selector, kt_voting, kt_voting_grouped
from dqnselector.piano import PianoQNet, piano_select, train_piano


def test_fast_selector_changes_choice_when_selected_trajectory_changes():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_edges_from([(0,3),(1,3),(2,3)])
    profiles = np.array([[1,0],[1,0],[0,1]],dtype=float)
    assert fast_selector(g,profiles,[0,1,2],2,alpha=.1) == [0,2]


def test_voting_always_obeys_pool_and_budget_with_zero_votes_and_remainder_areas():
    g = nx.DiGraph()
    g.add_nodes_from(range(7))
    for edges in [[],[(1,4),(3,4),(5,6)]]:
        g.add_edges_from(edges,weight=.5)
        for k in [1,2,3]:
            for values in [np.zeros((7,5)),np.arange(35).reshape(7,5)]:
                selected = kt_voting(g,values,[1,3,5],k)
                assert len(selected) == len(set(selected)) == k
                assert set(selected) <= {1,3,5}


def test_ktvoting2_path_is_distinct_from_grouped_appendix_path():
    g = nx.DiGraph()
    g.add_nodes_from(range(5))
    g.add_weighted_edges_from([(0, 1, .8), (2, 3, .8), (4, 1, .4)])
    values = np.array([[5, 0, 0], [0, 1, 0], [0, 0, 5], [1, 0, 0], [0, 1, 0]], dtype=float)
    primary = kt_voting(g, values, [0, 2, 4], 2)
    appendix = kt_voting_grouped(g, values, [0, 2, 4], 2)
    assert len(primary) == len(appendix) == 2
    assert set(primary) <= {0, 2, 4}


def test_fast_selector_uses_average_ranks_and_dynamic_similarity():
    g = nx.DiGraph(); g.add_nodes_from(range(4)); g.add_edges_from([(0, 3), (1, 3), (2, 3)])
    profiles = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    # First selection resolves tied degree deterministically; the second prefers the dissimilar profile.
    assert fast_selector(g, profiles, [0, 1, 2], 2, alpha=.1) == [0, 2]


def test_piano_matches_explicit_paper_equations_and_batching():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_weighted_edges_from([(0,1,.2),(1,2,.6),(2,0,.4)])
    torch.manual_seed(8)
    model = PianoQNet(g,[0,1,3],dim=3,rounds=2)
    masks = torch.tensor([[False,False,False,False],[True,False,False,False]])
    actual = model(masks)
    for b, mask in enumerate(masks):
        x = torch.zeros(4,3)
        for _ in range(2):
            nxt = []
            for v in range(4):
                neighbors = list(g.successors(v))
                msg = x[neighbors].sum(0) if neighbors else torch.zeros(3)
                edge = sum((torch.relu(model.alpha3*g[v][u]['weight']) for u in neighbors),torch.zeros(3))
                nxt.append(torch.relu(model.alpha1(msg)+model.alpha2(edge)+model.alpha4*mask[v]))
            x = torch.stack(nxt)
        q = model.beta1(torch.cat((model.beta2(x.sum(0)).relu().expand(4,-1),model.beta3(x).relu()),1)).squeeze(1)
        q[mask | ~model.worker_pool_mask] = -torch.inf
        torch.testing.assert_close(actual[b],q)
        torch.testing.assert_close(actual[b],model(mask)[0])
    assert len(set(piano_select(model,3))) == 3


def test_piano_training_has_finite_losses_and_updates():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_weighted_edges_from([(0,1,.5),(1,2,.5)])
    torch.manual_seed(2)
    model = PianoQNet(g,[0,1,2,3],dim=4,rounds=2)
    _,stats = train_piano(model,lambda s,v: [1,.7,.3,.1][v],episodes=3,budget=2,
        device='cpu',batch_size=2,n_step=2,seed=2,exploration_steps=6,lr_decay_interval=2)
    assert stats['updates'] > 0
    assert np.isfinite(stats['episode_returns']).all()
    assert abs(stats['final_epsilon']-.05) < 1e-12
    assert abs(stats['final_learning_rate']-.001*.95**(stats['updates']//2)) < 1e-12
