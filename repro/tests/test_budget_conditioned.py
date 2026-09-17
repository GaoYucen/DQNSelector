import numpy as np
import torch

from dqnselector.budget_conditioned import (
    BudgetConditionedRainbowSelector,
    greedy_select_for_budget,
    train_budget_conditioned_selector,
)


def _tiny_model():
    social = np.eye(4, dtype=np.float32)
    coverage = np.array([[0.1], [0.4], [0.7], [1.0]], dtype=np.float32)
    return BudgetConditionedRainbowSelector(
        social, coverage, hidden_dim=16, atoms=21, v_min=0.0, v_max=2.0
    )


def test_budget_context_changes_with_target_and_remaining_horizon():
    model = _tiny_model()
    device = torch.device("cpu")
    c1 = model.budget_context(1, 2, device=device, dtype=torch.float32)
    c2 = model.budget_context(1, 4, device=device, dtype=torch.float32)
    c3 = model.budget_context(2, 4, device=device, dtype=torch.float32)
    assert not torch.allclose(c1, c2)
    assert not torch.allclose(c2, c3)
    assert torch.all((c1 >= 0) & (c1 <= 1))


def test_budget_conditioned_training_and_greedy_run():
    model = _tiny_model()
    values = {0: 0.1, 1: 0.4, 2: 0.7, 3: 1.0}

    def reward_fn(selected, candidate):
        return values[candidate]

    model, stats = train_budget_conditioned_selector(
        model,
        reward_fn,
        budgets=[1, 2, 3],
        episodes=6,
        batch_size=2,
        warmup=2,
        replay_capacity=64,
        target_update_interval=2,
        random_seed=5,
    )
    assert sorted(set(stats.episode_budgets)) == [1, 2, 3]
    assert [len(s) for s in stats.selected_sets] == stats.episode_budgets
    for k in [1, 2, 3]:
        chosen = greedy_select_for_budget(model, k)
        assert len(chosen) == k
        assert len(set(chosen)) == k
