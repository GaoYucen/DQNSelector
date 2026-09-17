from copy import deepcopy

import numpy as np
import torch

from dqnselector.rainbow import Transition
from dqnselector.selector import (
    RainbowSelector,
    _transition_batch_losses,
    _transition_loss,
)


def _transition(n_nodes, selected, action, *, done=False, n_steps=3, reward=0.1):
    state = np.zeros(n_nodes, dtype=bool)
    state[list(selected)] = True
    next_state = state.copy()
    next_state[action] = True
    return Transition(state, action, reward, next_state, done, n_steps=n_steps)


def test_batched_replay_loss_matches_single_transition_reference():
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    n = 7
    social = rng.normal(size=(n, 4)).astype(np.float32)
    coverage = rng.normal(size=(n, 3)).astype(np.float32)
    pool = {0, 1, 2, 3, 4, 5}
    online = RainbowSelector(
        social,
        coverage,
        worker_pool=pool,
        hidden_dim=12,
        atoms=21,
        v_min=0.0,
        v_max=1.0,
    )
    target = deepcopy(online)
    online.train()
    target.eval()
    online.qnet.reset_noise()

    batch = [
        _transition(n, [], 1, n_steps=3, reward=0.08),
        _transition(n, [0], 2, n_steps=2, reward=0.12),
        _transition(n, [1], 4, n_steps=3, reward=0.07),
        _transition(n, [0, 2], 3, n_steps=1, reward=0.15),
        _transition(n, [1, 4, 5], 0, done=True, n_steps=2, reward=0.04),
    ]

    encoded_online = online.encode_nodes()
    with torch.no_grad():
        encoded_target = target.encode_nodes()

    batch_loss, batch_td = _transition_batch_losses(
        online, target, batch, encoded_online, encoded_target, gamma=0.99
    )
    ref = [
        _transition_loss(online, target, tr, encoded_online, encoded_target, gamma=0.99)
        for tr in batch
    ]
    ref_loss = torch.stack([x[0] for x in ref])
    ref_td = torch.stack([x[1] for x in ref])

    assert torch.allclose(batch_loss, ref_loss, atol=2e-5, rtol=2e-5)
    assert torch.allclose(batch_td, ref_td, atol=2e-5, rtol=2e-5)
