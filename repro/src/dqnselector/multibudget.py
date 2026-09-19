from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from .rainbow import NStepAccumulator, PrioritizedReplay, Transition
from .selector import RainbowSelector, _to_mask, _transition_batch_losses


@dataclass
class MultiBudgetTrainStats:
    episode_returns: list[float]
    losses: list[float]
    selected_sets: list[list[int]]
    episode_budgets: list[int]


def balanced_budget_schedule(
    budgets: Iterable[int], episodes: int, random_seed: int
) -> list[int]:
    """Build a reproducible, nearly balanced shuffled schedule over target budgets."""
    values = sorted({int(k) for k in budgets})
    if not values or any(k <= 0 for k in values):
        raise ValueError("budgets must contain positive integers")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(int(random_seed) + 104729)
    schedule: list[int] = []
    while len(schedule) < episodes:
        cycle = np.asarray(values, dtype=np.int64)
        rng.shuffle(cycle)
        schedule.extend(int(x) for x in cycle.tolist())
    return schedule[:episodes]


def train_rainbow_selector_multibudget(
    model: RainbowSelector,
    reward_fn: Callable[[set[int], int], float],
    budgets: Iterable[int],
    episodes: int = 200,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    n_step: int = 3,
    replay_capacity: int = 10000,
    batch_size: int = 32,
    warmup: int = 64,
    target_update_interval: int = 100,
    random_seed: int = 0,
    device: str | torch.device = "cpu",
    episode_callback: Callable | None = None,
) -> tuple[RainbowSelector, MultiBudgetTrainStats]:
    """Train one selector over a balanced mixture of deployment budgets.

    The deployment objective is evaluated at multiple k values. Earlier fixed-horizon
    training exposed the learner mostly to late-horizon states when k=50, which hurt
    the quality of small-k prefixes. This trainer keeps the architecture and reward
    unchanged while aligning the training-state distribution with the evaluation
    budgets.
    """
    budget_values = sorted({int(k) for k in budgets})
    if not budget_values or any(k <= 0 for k in budget_values):
        raise ValueError("budgets must contain positive integers")
    worker_pool_size = int(model.worker_pool_mask.sum().item())
    if max(budget_values) > worker_pool_size:
        raise ValueError(
            f"max training budget {max(budget_values)} exceeds worker pool {worker_pool_size}"
        )

    schedule = balanced_budget_schedule(budget_values, episodes, random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device(device)
    model = model.to(device)
    target = deepcopy(model).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay = PrioritizedReplay(replay_capacity, random_seed=random_seed)
    nstep = NStepAccumulator(n_step=n_step, gamma=gamma)
    episode_returns: list[float] = []
    losses: list[float] = []
    selected_sets: list[list[int]] = []
    learn_steps = 0

    for episode_budget in schedule:
        nstep.clear()
        mask = np.zeros(model.n_nodes, dtype=bool)
        selected: set[int] = set()
        total_reward = 0.0
        model.train()

        for t in range(episode_budget):
            model.qnet.reset_noise()
            encoded = model.encode_nodes()
            mask_t = _to_mask(mask, device)
            candidates, q = model.q_values_for_mask(mask_t, encoded)
            if candidates.numel() == 0:
                break
            action = int(candidates[int(torch.argmax(q).item())].item())
            reward = float(reward_fn(set(selected), action))
            next_mask = mask.copy()
            next_mask[action] = True
            done = (t + 1 >= episode_budget) or (len(selected) + 1 >= worker_pool_size)
            raw = Transition(mask.copy(), action, reward, next_mask.copy(), done, n_steps=1)
            for aggregated in nstep.push(raw):
                replay.add(aggregated)
            selected.add(action)
            mask = next_mask
            total_reward += reward

            if len(replay) >= max(warmup, batch_size):
                batch, indices, importance = replay.sample(batch_size)
                model.qnet.reset_noise()
                target.qnet.reset_noise()
                encoded_online = model.encode_nodes()
                with torch.no_grad():
                    encoded_target = target.encode_nodes()
                per_losses, td_errors = _transition_batch_losses(
                    model, target, batch, encoded_online, encoded_target, gamma
                )
                importance_t = torch.as_tensor(
                    importance, dtype=per_losses.dtype, device=device
                )
                loss = (per_losses * importance_t).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                replay.update_priorities(
                    indices, td_errors.detach().cpu().numpy().astype(np.float64, copy=False)
                )
                losses.append(float(loss.detach().cpu()))
                learn_steps += 1
                if learn_steps % target_update_interval == 0:
                    target.load_state_dict(model.state_dict())

            if done:
                break

        episode_returns.append(total_reward)
        selected_sets.append(sorted(selected))
        if episode_callback is not None:
            episode_callback(len(episode_returns), model, total_reward)

    return model, MultiBudgetTrainStats(
        episode_returns=episode_returns,
        losses=losses,
        selected_sets=selected_sets,
        episode_budgets=schedule,
    )
