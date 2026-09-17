import numpy as np

from dqnselector.multibudget import balanced_budget_schedule, train_rainbow_selector_multibudget
from dqnselector.selector import RainbowSelector


def test_balanced_budget_schedule_is_reproducible_and_balanced():
    s1 = balanced_budget_schedule([1, 2, 3], episodes=7, random_seed=9)
    s2 = balanced_budget_schedule([1, 2, 3], episodes=7, random_seed=9)
    assert s1 == s2
    assert len(s1) == 7
    counts = {k: s1.count(k) for k in [1, 2, 3]}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_multibudget_selector_uses_episode_budget():
    social = np.eye(3, dtype=np.float32)
    coverage = np.array([[0.1], [0.5], [1.0]], dtype=np.float32)
    values = {0: 0.1, 1: 0.5, 2: 1.0}

    def reward_fn(selected, candidate):
        return values[candidate]

    model = RainbowSelector(
        social, coverage, hidden_dim=16, atoms=21, v_min=0.0, v_max=2.0
    )
    _, stats = train_rainbow_selector_multibudget(
        model,
        reward_fn,
        budgets=[1, 2],
        episodes=4,
        batch_size=2,
        warmup=2,
        replay_capacity=64,
        target_update_interval=2,
        random_seed=3,
    )
    assert sorted(set(stats.episode_budgets)) == [1, 2]
    assert len(stats.selected_sets) == 4
    assert [len(s) for s in stats.selected_sets] == stats.episode_budgets
