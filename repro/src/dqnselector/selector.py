from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import nn

from .fusion import GatedDualEmbedding
from .rainbow import (
    NStepAccumulator,
    PairwiseDuelingC51,
    PrioritizedReplay,
    Transition,
    project_c51_distribution,
)


class RainbowSelector(nn.Module):
    """Gated dual embedding + pair-conditioned Rainbow Q network."""

    def __init__(
        self,
        social_embedding: np.ndarray | torch.Tensor,
        coverage_embedding: np.ndarray | torch.Tensor,
        worker_pool: Iterable[int] | None = None,
        hidden_dim: int = 128,
        atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ) -> None:
        super().__init__()
        social = torch.as_tensor(social_embedding, dtype=torch.float32)
        coverage = torch.as_tensor(coverage_embedding, dtype=torch.float32)
        if social.ndim != 2 or coverage.ndim != 2 or social.shape[0] != coverage.shape[0]:
            raise ValueError("social/coverage embeddings must be [n_nodes, dim]")
        self.register_buffer("social_embedding", social)
        self.register_buffer("coverage_embedding", coverage)
        pool = torch.zeros(social.shape[0], dtype=torch.bool)
        if worker_pool is None:
            pool[:] = True
        else:
            pool[list(worker_pool)] = True
        self.register_buffer("worker_pool_mask", pool)
        self.encoder = GatedDualEmbedding(social.shape[1], coverage.shape[1])
        self.qnet = PairwiseDuelingC51(
            self.encoder.output_dim,
            hidden_dim=hidden_dim,
            atoms=atoms,
            v_min=v_min,
            v_max=v_max,
        )

    @property
    def n_nodes(self) -> int:
        return int(self.social_embedding.shape[0])

    def encode_nodes(self) -> torch.Tensor:
        return self.encoder(self.social_embedding, self.coverage_embedding)

    def candidate_indices(self, selected_mask: torch.Tensor) -> torch.Tensor:
        selected_mask = selected_mask.to(self.worker_pool_mask.device, dtype=torch.bool)
        return torch.where(self.worker_pool_mask & ~selected_mask)[0]

    def state_vector(self, encoded_nodes: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        selected_mask = selected_mask.to(encoded_nodes.device, dtype=torch.bool)
        if selected_mask.any():
            return encoded_nodes[selected_mask].sum(dim=0)
        return torch.zeros(encoded_nodes.shape[1], device=encoded_nodes.device, dtype=encoded_nodes.dtype)

    def q_values_for_mask(
        self,
        selected_mask: torch.Tensor,
        encoded_nodes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded_nodes is None:
            encoded_nodes = self.encode_nodes()
        candidates = self.candidate_indices(selected_mask)
        if candidates.numel() == 0:
            return candidates, torch.empty(0, device=encoded_nodes.device)
        state = self.state_vector(encoded_nodes, selected_mask)
        q = self.qnet.q_values(state, encoded_nodes[candidates])[0]
        return candidates, q


@dataclass
class TrainStats:
    episode_returns: list[float]
    losses: list[float]
    selected_sets: list[list[int]]


def _to_mask(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _transition_loss(
    online: RainbowSelector,
    target: RainbowSelector,
    tr: Transition,
    encoded_online: torch.Tensor,
    encoded_target: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference single-transition Rainbow loss.

    This deliberately remains available as a correctness reference for the batched
    implementation below. Training uses `_transition_batch_losses` for efficiency.
    """
    device = encoded_online.device
    state_mask = _to_mask(tr.state_mask, device)
    candidates = online.candidate_indices(state_mask)
    if candidates.numel() == 0:
        raise RuntimeError("transition has no valid action")
    positions = torch.where(candidates == int(tr.action))[0]
    if positions.numel() != 1:
        raise RuntimeError("stored action is not valid for transition state")
    state = online.state_vector(encoded_online, state_mask)
    pred_all = online.qnet.distribution(state, encoded_online[candidates])[0]
    pred = pred_all[int(positions.item())]

    with torch.no_grad():
        if tr.done:
            next_dist = torch.full_like(pred, 1.0 / pred.numel())
        else:
            next_mask = _to_mask(tr.next_state_mask, device)
            next_candidates = online.candidate_indices(next_mask)
            if next_candidates.numel() == 0:
                next_dist = torch.full_like(pred, 1.0 / pred.numel())
            else:
                next_state_online = online.state_vector(encoded_online, next_mask)
                online_q = online.qnet.q_values(
                    next_state_online, encoded_online[next_candidates]
                )[0]
                best_pos = int(torch.argmax(online_q).item())
                next_state_target = target.state_vector(encoded_target, next_mask)
                target_dist_all = target.qnet.distribution(
                    next_state_target, encoded_target[next_candidates]
                )[0]
                next_dist = target_dist_all[best_pos]
        projected = project_c51_distribution(
            next_dist.unsqueeze(0),
            torch.tensor([tr.reward], dtype=torch.float32, device=device),
            torch.tensor([float(tr.done)], dtype=torch.float32, device=device),
            gamma ** int(tr.n_steps),
            online.qnet.support,
        )[0]
    loss = -(projected * pred.log()).sum()
    current_q = (pred.detach() * online.qnet.support).sum()
    target_q = (projected.detach() * online.qnet.support).sum()
    td_error = (target_q - current_q).abs()
    return loss, td_error


def _stack_masks(batch: list[Transition], attr: str, device: torch.device) -> torch.Tensor:
    arrays = np.stack([getattr(tr, attr) for tr in batch], axis=0)
    return torch.as_tensor(arrays, dtype=torch.bool, device=device)


def _state_vectors(encoded_nodes: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # The state definition is the sum of selected gated worker representations.
    return masks.to(dtype=encoded_nodes.dtype) @ encoded_nodes


def _transition_batch_losses(
    online: RainbowSelector,
    target: RainbowSelector,
    batch: list[Transition],
    encoded_online: torch.Tensor,
    encoded_target: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Rainbow loss for a prioritized replay minibatch.

    Every replay row has the same worker pool but a different subset of currently
    available candidates.  Earlier code grouped rows by candidate count, which was
    mathematically exact but launched many small GPU kernels.  Here all worker-pool
    actions are evaluated in one padded batch and a boolean action mask is used in
    the dueling advantage centering.  Invalid/padded actions therefore contribute
    neither to the dueling mean nor to Double-DQN action selection, making this
    equivalent to the variable-length reference calculation while substantially
    improving GPU utilization.
    """
    if not batch:
        raise ValueError("batch must not be empty")
    device = encoded_online.device
    bsz = len(batch)
    rows_all = torch.arange(bsz, device=device)
    state_masks = _stack_masks(batch, "state_mask", device)
    next_masks = _stack_masks(batch, "next_state_mask", device)
    actions = torch.tensor([tr.action for tr in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([tr.reward for tr in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([float(tr.done) for tr in batch], dtype=torch.float32, device=device)
    n_steps = torch.tensor([int(tr.n_steps) for tr in batch], dtype=torch.long, device=device)

    pool = torch.where(online.worker_pool_mask)[0]
    if pool.numel() == 0:
        raise RuntimeError("worker pool is empty")
    pool_positions = torch.full((online.n_nodes,), -1, dtype=torch.long, device=device)
    pool_positions[pool] = torch.arange(pool.numel(), device=device)
    action_positions = pool_positions[actions]
    if bool((action_positions < 0).any()):
        raise RuntimeError("stored action is outside the worker pool")

    candidate_mask = ~state_masks[:, pool]
    if not bool(candidate_mask[rows_all, action_positions].all()):
        raise RuntimeError("stored action is not valid for transition state")
    state_vecs = _state_vectors(encoded_online, state_masks)
    pool_actions_online = encoded_online[pool].unsqueeze(0)
    pred_all = online.qnet.distribution(
        state_vecs, pool_actions_online, action_mask=candidate_mask
    )
    pred = pred_all[rows_all, action_positions]

    with torch.no_grad():
        next_dist = torch.full_like(pred, 1.0 / online.qnet.atoms)
        nonterminal_rows = torch.where(dones == 0)[0]
        if nonterminal_rows.numel() > 0:
            next_candidate_mask = ~next_masks[nonterminal_rows][:, pool]
            has_candidate = next_candidate_mask.any(dim=1)
            usable_rows = nonterminal_rows[has_candidate]
            if usable_rows.numel() > 0:
                usable_mask = next_candidate_mask[has_candidate]
                next_state_online = _state_vectors(
                    encoded_online, next_masks[usable_rows]
                )
                online_q = online.qnet.q_values(
                    next_state_online,
                    pool_actions_online,
                    action_mask=usable_mask,
                )
                best_pos = torch.argmax(online_q, dim=1)
                next_state_target = _state_vectors(
                    encoded_target, next_masks[usable_rows]
                )
                target_dist_all = target.qnet.distribution(
                    next_state_target,
                    encoded_target[pool].unsqueeze(0),
                    action_mask=usable_mask,
                )
                next_dist[usable_rows] = target_dist_all[
                    torch.arange(usable_rows.numel(), device=device), best_pos
                ]

        projected = torch.empty_like(pred)
        for steps in torch.unique(n_steps).tolist():
            rows = torch.where(n_steps == steps)[0]
            projected[rows] = project_c51_distribution(
                next_dist[rows],
                rewards[rows],
                dones[rows],
                gamma ** int(steps),
                online.qnet.support,
            )

    per_loss = -(projected * pred.log()).sum(dim=1)
    current_q = (pred.detach() * online.qnet.support).sum(dim=1)
    target_q = (projected.detach() * online.qnet.support).sum(dim=1)
    td_error = (target_q - current_q).abs()
    return per_loss, td_error


def train_rainbow_selector(
    model: RainbowSelector,
    reward_fn: Callable[[set[int], int], float],
    seed_budget: int,
    episodes: int = 100,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    n_step: int = 3,
    replay_capacity: int = 10000,
    batch_size: int = 32,
    warmup: int = 64,
    target_update_interval: int = 100,
    random_seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[RainbowSelector, TrainStats]:
    """Correctness-first Rainbow training loop for one graph instance."""
    if seed_budget <= 0:
        raise ValueError("seed_budget must be positive")
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
    worker_pool_size = int(model.worker_pool_mask.sum().item())

    for _episode in range(episodes):
        nstep.clear()
        mask = np.zeros(model.n_nodes, dtype=bool)
        selected: set[int] = set()
        total_reward = 0.0
        model.train()

        for t in range(seed_budget):
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
            done = (t + 1 >= seed_budget) or (len(selected) + 1 >= worker_pool_size)
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

    return model, TrainStats(episode_returns, losses, selected_sets)


def greedy_select(model: RainbowSelector, k: int, device: str | torch.device = "cpu") -> list[int]:
    model = model.to(device)
    model.eval()
    mask = torch.zeros(model.n_nodes, dtype=torch.bool, device=device)
    chosen: list[int] = []
    with torch.no_grad():
        encoded = model.encode_nodes()
        while len(chosen) < k:
            candidates, q = model.q_values_for_mask(mask, encoded)
            if candidates.numel() == 0:
                break
            action = int(candidates[int(torch.argmax(q).item())].item())
            chosen.append(action)
            mask[action] = True
    return chosen
