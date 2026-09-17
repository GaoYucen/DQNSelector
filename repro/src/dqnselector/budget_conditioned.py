from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from .rainbow import PairwiseDuelingC51, PrioritizedReplay, project_c51_distribution
from .selector import RainbowSelector, _to_mask


@dataclass
class BudgetTransition:
    state_mask: np.ndarray
    action: int
    reward: float
    next_state_mask: np.ndarray
    done: bool
    target_budget: int
    n_steps: int = 1


class BudgetNStepAccumulator:
    def __init__(self, n_step: int = 3, gamma: float = 0.99) -> None:
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.buffer: deque[BudgetTransition] = deque()

    def clear(self) -> None:
        self.buffer.clear()

    def _aggregate_prefix(self) -> BudgetTransition:
        first = self.buffer[0]
        reward = 0.0
        next_mask = first.next_state_mask
        done = False
        steps = 0
        for i, tr in enumerate(self.buffer):
            if int(tr.target_budget) != int(first.target_budget):
                raise RuntimeError("mixed target budgets inside one n-step trajectory")
            reward += (self.gamma**i) * float(tr.reward)
            next_mask = tr.next_state_mask
            done = bool(tr.done)
            steps = i + 1
            if done or steps >= self.n_step:
                break
        return BudgetTransition(
            state_mask=first.state_mask.copy(),
            action=int(first.action),
            reward=float(reward),
            next_state_mask=next_mask.copy(),
            done=done,
            target_budget=int(first.target_budget),
            n_steps=steps,
        )

    def push(self, transition: BudgetTransition) -> list[BudgetTransition]:
        self.buffer.append(transition)
        emitted: list[BudgetTransition] = []
        if len(self.buffer) >= self.n_step:
            emitted.append(self._aggregate_prefix())
            self.buffer.popleft()
        if transition.done:
            while self.buffer:
                emitted.append(self._aggregate_prefix())
                self.buffer.popleft()
        return emitted


class BudgetConditionedRainbowSelector(RainbowSelector):
    """Rainbow selector conditioned on deployment budget and remaining horizon.

    The base node representation remains unchanged. Two normalized context features are
    appended only to the state representation:
      1) target budget / worker-pool size;
      2) remaining selections / target budget.

    Candidate action vectors receive zeros in those two context coordinates. The
    existing PairwiseDuelingC51 therefore remains reusable while the state becomes
    Markov under mixed-budget training.
    """

    context_dim = 2

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
            self.base_node_dim + self.context_dim,
            hidden_dim=hidden_dim,
            atoms=atoms,
            v_min=v_min,
            v_max=v_max,
        )

    @property
    def worker_pool_size(self) -> int:
        return int(self.worker_pool_mask.sum().item())

    def budget_context(
        self,
        selected_count: torch.Tensor | int,
        target_budget: torch.Tensor | int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        selected = torch.as_tensor(selected_count, dtype=dtype, device=device)
        target = torch.as_tensor(target_budget, dtype=dtype, device=device)
        if bool((target <= 0).any()):
            raise ValueError("target budget must be positive")
        target_ratio = target / float(max(self.worker_pool_size, 1))
        remaining_ratio = ((target - selected).clamp_min(0.0)) / target
        return torch.stack((target_ratio, remaining_ratio), dim=-1)

    def augment_actions(self, action_vectors: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(
            (*action_vectors.shape[:-1], self.context_dim),
            dtype=action_vectors.dtype,
            device=action_vectors.device,
        )
        return torch.cat((action_vectors, zeros), dim=-1)

    def conditioned_state_vector(
        self,
        encoded_nodes: torch.Tensor,
        selected_mask: torch.Tensor,
        target_budget: int,
    ) -> torch.Tensor:
        selected_mask = selected_mask.to(encoded_nodes.device, dtype=torch.bool)
        if selected_mask.any():
            base = encoded_nodes[selected_mask].sum(dim=0)
        else:
            base = torch.zeros(
                encoded_nodes.shape[1], device=encoded_nodes.device, dtype=encoded_nodes.dtype
            )
        context = self.budget_context(
            int(selected_mask.sum().item()),
            int(target_budget),
            device=encoded_nodes.device,
            dtype=encoded_nodes.dtype,
        )
        return torch.cat((base, context), dim=-1)

    def q_values_for_budget(
        self,
        selected_mask: torch.Tensor,
        target_budget: int,
        encoded_nodes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded_nodes is None:
            encoded_nodes = self.encode_nodes()
        candidates = self.candidate_indices(selected_mask)
        if candidates.numel() == 0:
            return candidates, torch.empty(0, device=encoded_nodes.device)
        state = self.conditioned_state_vector(encoded_nodes, selected_mask, target_budget)
        actions = self.augment_actions(encoded_nodes[candidates])
        q = self.qnet.q_values(state, actions)[0]
        return candidates, q


def _balanced_budget_schedule(
    budgets: Iterable[int], episodes: int, random_seed: int
) -> list[int]:
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


def _stack_masks(
    batch: list[BudgetTransition], attr: str, device: torch.device
) -> torch.Tensor:
    arrays = np.stack([getattr(tr, attr) for tr in batch], axis=0)
    return torch.as_tensor(arrays, dtype=torch.bool, device=device)


def _conditioned_state_vectors(
    model: BudgetConditionedRainbowSelector,
    encoded_nodes: torch.Tensor,
    masks: torch.Tensor,
    target_budgets: torch.Tensor,
) -> torch.Tensor:
    base = masks.to(dtype=encoded_nodes.dtype) @ encoded_nodes
    selected_counts = masks.sum(dim=1)
    context = model.budget_context(
        selected_counts,
        target_budgets,
        device=encoded_nodes.device,
        dtype=encoded_nodes.dtype,
    )
    return torch.cat((base, context), dim=-1)


def _transition_batch_losses(
    online: BudgetConditionedRainbowSelector,
    target: BudgetConditionedRainbowSelector,
    batch: list[BudgetTransition],
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
    target_budgets = torch.tensor(
        [int(tr.target_budget) for tr in batch], dtype=torch.long, device=device
    )

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

    state_vecs = _conditioned_state_vectors(
        online, encoded_online, state_masks, target_budgets
    )
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
                usable_budgets = target_budgets[usable_rows]
                next_state_online = _conditioned_state_vectors(
                    online, encoded_online, next_masks[usable_rows], usable_budgets
                )
                online_q = online.qnet.q_values(
                    next_state_online,
                    pool_actions_online,
                    action_mask=usable_mask,
                )
                best_pos = torch.argmax(online_q, dim=1)
                next_state_target = _conditioned_state_vectors(
                    target, encoded_target, next_masks[usable_rows], usable_budgets
                )
                target_actions = target.augment_actions(encoded_target[pool]).unsqueeze(0)
                target_dist_all = target.qnet.distribution(
                    next_state_target,
                    target_actions,
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


@dataclass
class BudgetConditionedTrainStats:
    episode_returns: list[float]
    losses: list[float]
    selected_sets: list[list[int]]
    episode_budgets: list[int]


def train_budget_conditioned_selector(
    model: BudgetConditionedRainbowSelector,
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
) -> tuple[BudgetConditionedRainbowSelector, BudgetConditionedTrainStats]:
    budget_values = sorted({int(k) for k in budgets})
    if not budget_values or any(k <= 0 for k in budget_values):
        raise ValueError("budgets must contain positive integers")
    if max(budget_values) > model.worker_pool_size:
        raise ValueError("training budget exceeds worker pool")

    schedule = _balanced_budget_schedule(budget_values, episodes, random_seed)
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
    nstep = BudgetNStepAccumulator(n_step=n_step, gamma=gamma)
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
            candidates, q = model.q_values_for_budget(
                mask_t, episode_budget, encoded_nodes=encoded
            )
            if candidates.numel() == 0:
                break
            action = int(candidates[int(torch.argmax(q).item())].item())
            reward = float(reward_fn(set(selected), action))
            next_mask = mask.copy()
            next_mask[action] = True
            done = (t + 1 >= episode_budget) or (
                len(selected) + 1 >= model.worker_pool_size
            )
            raw = BudgetTransition(
                state_mask=mask.copy(),
                action=action,
                reward=reward,
                next_state_mask=next_mask.copy(),
                done=done,
                target_budget=int(episode_budget),
                n_steps=1,
            )
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

    return model, BudgetConditionedTrainStats(
        episode_returns=episode_returns,
        losses=losses,
        selected_sets=selected_sets,
        episode_budgets=schedule,
    )


def greedy_select_for_budget(
    model: BudgetConditionedRainbowSelector,
    target_budget: int,
    device: str | torch.device = "cpu",
) -> list[int]:
    if target_budget <= 0 or target_budget > model.worker_pool_size:
        raise ValueError("invalid target budget")
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    mask = torch.zeros(model.n_nodes, dtype=torch.bool, device=device)
    chosen: list[int] = []
    with torch.no_grad():
        encoded = model.encode_nodes()
        while len(chosen) < target_budget:
            candidates, q = model.q_values_for_budget(
                mask, target_budget, encoded_nodes=encoded
            )
            if candidates.numel() == 0:
                break
            action = int(candidates[int(torch.argmax(q).item())].item())
            chosen.append(action)
            mask[action] = True
    return chosen
