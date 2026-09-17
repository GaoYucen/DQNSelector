from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from .multibudget import balanced_budget_schedule
from .rainbow import NStepAccumulator, PairwiseDuelingC51, PrioritizedReplay, Transition, project_c51_distribution
from .selector import RainbowSelector, _to_mask


class SizeAwareRainbowSelector(RainbowSelector):
    """Rainbow selector with scale-stable set pooling and explicit set size.

    The selected-worker state is represented by the mean gated worker embedding plus
    one normalized cardinality feature |S|/|W|.  This keeps the embedding scale
    stable across selection depths while retaining progress information explicitly.
    Candidate action embeddings are padded with a zero in the cardinality coordinate.
    """

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
        super().__init__(
            social_embedding,
            coverage_embedding,
            worker_pool=worker_pool,
            hidden_dim=hidden_dim,
            atoms=atoms,
            v_min=v_min,
            v_max=v_max,
        )
        self.base_node_dim = int(self.encoder.output_dim)
        self.qnet = PairwiseDuelingC51(
            self.base_node_dim + 1,
            hidden_dim=hidden_dim,
            atoms=atoms,
            v_min=v_min,
            v_max=v_max,
        )

    @property
    def worker_pool_size(self) -> int:
        return int(self.worker_pool_mask.sum().item())

    def augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(
            (*actions.shape[:-1], 1), dtype=actions.dtype, device=actions.device
        )
        return torch.cat((actions, zero), dim=-1)

    def state_vector(self, encoded_nodes: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        selected_mask = selected_mask.to(encoded_nodes.device, dtype=torch.bool)
        count = int(selected_mask.sum().item())
        if count:
            pooled = encoded_nodes[selected_mask].mean(dim=0)
        else:
            pooled = torch.zeros(
                encoded_nodes.shape[1], device=encoded_nodes.device, dtype=encoded_nodes.dtype
            )
        size = torch.tensor(
            [count / float(max(self.worker_pool_size, 1))],
            dtype=encoded_nodes.dtype,
            device=encoded_nodes.device,
        )
        return torch.cat((pooled, size), dim=0)

    def state_vectors(self, encoded_nodes: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        weights = masks.to(dtype=encoded_nodes.dtype)
        counts = weights.sum(dim=1, keepdim=True)
        summed = weights @ encoded_nodes
        pooled = summed / counts.clamp_min(1.0)
        size = counts / float(max(self.worker_pool_size, 1))
        return torch.cat((pooled, size), dim=1)

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
        q = self.qnet.q_values(state, self.augment_actions(encoded_nodes[candidates]))[0]
        return candidates, q


def _stack_masks(batch: list[Transition], attr: str, device: torch.device) -> torch.Tensor:
    arrays = np.stack([getattr(tr, attr) for tr in batch], axis=0)
    return torch.as_tensor(arrays, dtype=torch.bool, device=device)


def _transition_batch_losses(
    online: SizeAwareRainbowSelector,
    target: SizeAwareRainbowSelector,
    batch: list[Transition],
    encoded_online: torch.Tensor,
    encoded_target: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    state_vecs = online.state_vectors(encoded_online, state_masks)
    pool_actions_online = online.augment_actions(encoded_online[pool]).unsqueeze(0)
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
                next_state_online = online.state_vectors(
                    encoded_online, next_masks[usable_rows]
                )
                online_q = online.qnet.q_values(
                    next_state_online, pool_actions_online, action_mask=usable_mask
                )
                best_pos = torch.argmax(online_q, dim=1)
                next_state_target = target.state_vectors(
                    encoded_target, next_masks[usable_rows]
                )
                target_actions = target.augment_actions(encoded_target[pool]).unsqueeze(0)
                target_dist_all = target.qnet.distribution(
                    next_state_target, target_actions, action_mask=usable_mask
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


@dataclass
class SizeAwareTrainStats:
    episode_returns: list[float]
    losses: list[float]
    selected_sets: list[list[int]]
    episode_budgets: list[int]


def train_size_aware_selector(
    model: SizeAwareRainbowSelector,
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
) -> tuple[SizeAwareRainbowSelector, SizeAwareTrainStats]:
    budget_values = sorted({int(k) for k in budgets})
    if not budget_values or any(k <= 0 for k in budget_values):
        raise ValueError("budgets must contain positive integers")
    if max(budget_values) > model.worker_pool_size:
        raise ValueError("training budget exceeds worker pool")

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
    nstep_acc = NStepAccumulator(n_step=n_step, gamma=gamma)
    episode_returns: list[float] = []
    losses: list[float] = []
    selected_sets: list[list[int]] = []
    learn_steps = 0

    for episode_budget in schedule:
        nstep_acc.clear()
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
            done = (t + 1 >= episode_budget) or (
                len(selected) + 1 >= model.worker_pool_size
            )
            raw = Transition(mask.copy(), action, reward, next_mask.copy(), done, n_steps=1)
            for aggregated in nstep_acc.push(raw):
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

    return model, SizeAwareTrainStats(
        episode_returns=episode_returns,
        losses=losses,
        selected_sets=selected_sets,
        episode_budgets=schedule,
    )
