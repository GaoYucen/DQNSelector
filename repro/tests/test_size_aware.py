import numpy as np
import torch

from dqnselector.selector import greedy_select
from dqnselector.size_aware import SizeAwareRainbowSelector, train_size_aware_selector


def _model():
    social = np.eye(4, dtype=np.float32)
    coverage = np.array([[0.1], [0.3], [0.6], [1.0]], dtype=np.float32)
    return SizeAwareRainbowSelector(
        social, coverage, hidden_dim=16, atoms=21, v_min=0.0, v_max=2.0
    )


def test_size_aware_state_uses_mean_and_cardinality():
    model = _model()
    encoded = model.encode_nodes().detach()
    one = torch.tensor([True, False, False, False])
    two = torch.tensor([True, True, False, False])
    s1 = model.state_vector(encoded, one)
    s2 = model.state_vector(encoded, two)
    assert s1.shape[0] == encoded.shape[1] + 1
    assert torch.allclose(s1[:-1], encoded[0])
    assert torch.allclose(s2[:-1], encoded[:2].mean(dim=0))
    assert torch.isclose(s1[-1], torch.tensor(0.25))
    assert torch.isclose(s2[-1], torch.tensor(0.50))


def test_size_aware_multibudget_training_runs():
    model = _model()
    values = {0: 0.1, 1: 0.3, 2: 0.6, 3: 1.0}

    def reward_fn(selected, candidate):
        return values[candidate]

    model, stats = train_size_aware_selector(
        model,
        reward_fn,
        budgets=[1, 2, 3],
        episodes=6,
        batch_size=2,
        warmup=2,
        replay_capacity=64,
        target_update_interval=2,
        random_seed=7,
    )
    assert sorted(set(stats.episode_budgets)) == [1, 2, 3]
    assert [len(s) for s in stats.selected_sets] == stats.episode_budgets
    chosen = greedy_select(model, 3)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3
