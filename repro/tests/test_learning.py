import numpy as np
import torch

from dqnselector.embedding import InfluencePairDataset, fit_social_influence_embedding
from dqnselector.fusion import GatedDualEmbedding
from dqnselector.rainbow import NoisyLinear, PairwiseDuelingC51, PrioritizedReplay, Transition
from dqnselector.selector import RainbowSelector, train_rainbow_selector, greedy_select


def test_social_embedding_reduces_mse():
    # A tiny directed probability matrix represented as balanced pairs.
    pairs = InfluencePairDataset(
        seeds=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        targets=np.array([1, 2, 0, 2, 0, 1], dtype=np.int64),
        probabilities=np.array([0.8, 0.0, 0.2, 0.7, 0.0, 0.1], dtype=np.float32),
        nodes=[0, 1, 2],
    )
    _, _, history = fit_social_influence_embedding(
        pairs, dim=4, epochs=80, learning_rate=0.1, batch_size=6, random_seed=2
    )
    assert history[-1] < history[0]


def test_gating_shapes_and_ranges():
    layer = GatedDualEmbedding(4, 3)
    s = torch.randn(5, 4)
    r = torch.randn(5, 3)
    gs, gr = layer.gates(s, r)
    out = layer(s, r)
    assert out.shape == (5, 7)
    assert torch.all((gs >= 0) & (gs <= 1))
    assert torch.all((gr >= 0) & (gr <= 1))


def test_noisy_layer_changes_after_reset():
    torch.manual_seed(0)
    layer = NoisyLinear(4, 3)
    layer.train()
    x = torch.ones(2, 4)
    y1 = layer(x).detach().clone()
    layer.reset_noise()
    y2 = layer(x).detach().clone()
    assert not torch.allclose(y1, y2)


def test_c51_distribution_normalizes():
    net = PairwiseDuelingC51(node_dim=6, hidden_dim=8, atoms=11, v_min=0.0, v_max=1.0)
    state = torch.randn(6)
    actions = torch.randn(4, 6)
    dist = net.distribution(state, actions)
    assert dist.shape == (1, 4, 11)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_prioritized_replay_roundtrip():
    replay = PrioritizedReplay(capacity=8, random_seed=3)
    for action in range(4):
        mask = np.zeros(4, dtype=bool)
        next_mask = mask.copy()
        next_mask[action] = True
        replay.add(Transition(mask, action, float(action), next_mask, False))
    batch, indices, weights = replay.sample(3)
    assert len(batch) == 3
    assert indices.shape == (3,)
    assert weights.shape == (3,)


def test_small_selector_learns_and_runs():
    # Deterministic modular objective: node 2 is best, then 1, then 0.
    social = np.eye(3, dtype=np.float32)
    coverage = np.array([[0.1], [0.5], [1.0]], dtype=np.float32)
    values = {0: 0.1, 1: 0.5, 2: 1.0}

    def reward_fn(selected, candidate):
        return values[candidate]

    model = RainbowSelector(
        social, coverage, hidden_dim=16, atoms=21, v_min=0.0, v_max=2.0
    )
    model, stats = train_rainbow_selector(
        model,
        reward_fn,
        seed_budget=2,
        episodes=6,
        batch_size=2,
        warmup=2,
        replay_capacity=64,
        target_update_interval=2,
        random_seed=1,
    )
    chosen = greedy_select(model, 2)
    assert len(chosen) == 2
    assert len(set(chosen)) == 2
    assert len(stats.episode_returns) == 6
