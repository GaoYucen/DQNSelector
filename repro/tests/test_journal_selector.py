import copy

import numpy as np
import torch

from dqnselector.journal_selector import JournalSelector, TeacherState, journal_greedy_select, train_journal_selector


def test_journal_selector_is_invariant_to_area_column_permutation():
    torch.manual_seed(4)
    social = np.eye(4, dtype=np.float32)
    footprint = np.array([[.2, .1, .3], [.1, .4, .2], [.3, .2, .1], [.2, .3, .2]], dtype=np.float32)
    model = JournalSelector(social, footprint, [0, 1, 2, 3], hidden_dim=12, area_dim=5, atoms=11)
    permuted = copy.deepcopy(model)
    permuted.footprints.copy_(permuted.footprints[:, torch.tensor([2, 0, 1])])
    assert journal_greedy_select(model, 3) == journal_greedy_select(permuted, 3)


def test_journal_selector_trains_with_teacher_states():
    torch.manual_seed(3)
    model = JournalSelector(np.eye(3, dtype=np.float32), np.array([[.1], [.5], [.9]], dtype=np.float32), [0, 1, 2], hidden_dim=12, area_dim=4, atoms=11)
    teacher = TeacherState(mask=np.zeros(3, dtype=bool), gains=np.array([.1, .5, .9], dtype=np.float32), max_budget=2)
    model, stats = train_journal_selector(model, lambda selected, node: [.1, .5, .9][node], [1, 2], [teacher],
        episodes=4, batch_size=2, warmup=2, ranking_pretrain_epochs=2, seed=8)
    assert stats["updates"] > 0
    assert len(journal_greedy_select(model, 2)) == 2
