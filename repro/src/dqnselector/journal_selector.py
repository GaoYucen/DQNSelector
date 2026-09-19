"""Journal DQNSelector-J: permutation-invariant residual-demand Rainbow policy."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .rainbow import NStepAccumulator, PrioritizedReplay, Transition, project_c51_distribution


class JournalSelector(nn.Module):
    """A policy whose area representation is invariant to target-area column order.

    ``footprints[v, i]`` is a precomputed direct-plus-PIIC contribution in EC
    units.  The policy derives residual demand from the current seed mask rather
    than treating an arbitrary area-column position as semantic input.
    """

    checkpoint_version = "dqnselector-j-v1"

    def __init__(
        self, social_embedding: np.ndarray | torch.Tensor,
        footprints: np.ndarray | torch.Tensor, worker_pool: Iterable[int],
        hidden_dim: int = 128, area_dim: int = 32, atoms: int = 51,
        use_residual_state: bool = True,
    ) -> None:
        super().__init__()
        social = torch.as_tensor(social_embedding, dtype=torch.float32)
        footprint = torch.as_tensor(footprints, dtype=torch.float32)
        if social.ndim != 2 or footprint.ndim != 2 or social.shape[0] != footprint.shape[0]:
            raise ValueError("social and footprints must be [n_nodes, dim]")
        self.register_buffer("social_embedding", social)
        self.register_buffer("footprints", footprint.clamp_min(0.0))
        pool = torch.zeros(social.shape[0], dtype=torch.bool)
        pool[list(worker_pool)] = True
        self.register_buffer("worker_pool_mask", pool)
        self.use_residual_state = bool(use_residual_state)
        self.social_projection = nn.Sequential(nn.Linear(social.shape[1], area_dim), nn.ReLU())
        self.area_mlp = nn.Sequential(nn.Linear(4, area_dim), nn.ReLU(), nn.Linear(area_dim, area_dim), nn.ReLU())
        # candidate social + mean/max pooled area features
        self.action_projection = nn.Sequential(nn.Linear(area_dim * 3, hidden_dim), nn.ReLU())
        # selected social mean/max plus budget-progress scalars
        self.state_projection = nn.Sequential(nn.Linear(area_dim * 2 + 2, hidden_dim), nn.ReLU())
        from .rainbow import PairwiseDuelingC51
        self.qnet = PairwiseDuelingC51(hidden_dim, hidden_dim=hidden_dim, atoms=atoms, v_min=0.0, v_max=1.0)

    @property
    def n_nodes(self) -> int:
        return int(self.social_embedding.shape[0])

    @property
    def n_areas(self) -> int:
        return int(self.footprints.shape[1])

    def candidates(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(self.worker_pool_mask & ~mask.to(device=self.worker_pool_mask.device, dtype=torch.bool))[0]

    def _selected_social_state(self, mask: torch.Tensor, max_budget: int) -> torch.Tensor:
        social = self.social_projection(self.social_embedding)
        selected = social[mask]
        if selected.numel():
            mean, maximum = selected.mean(0), selected.max(0).values
        else:
            mean = maximum = torch.zeros(social.shape[1], device=social.device)
        total = max(int(self.worker_pool_mask.sum()), 1)
        progress = torch.tensor([float(mask.sum()) / total, float(mask.sum()) / max(max_budget, 1)], device=social.device)
        return self.state_projection(torch.cat([mean, maximum, progress]))

    def _area_action_state(self, mask: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        footprint = self.footprints
        candidate = footprint[candidates]
        if self.use_residual_state:
            accumulated = footprint[mask].sum(0) if bool(mask.any()) else torch.zeros(self.n_areas, device=footprint.device)
            residual = (1.0 - accumulated).clamp_min(0.0)
            token = torch.stack((candidate, residual.expand_as(candidate), torch.minimum(candidate, residual), torch.minimum(candidate, accumulated)), dim=-1)
        else:
            # Ranking-only ablation: retain the shared policy but remove all
            # residual-demand information from its regional state.
            token = torch.zeros((*candidate.shape, 4), dtype=candidate.dtype, device=candidate.device)
        encoded = self.area_mlp(token)
        pooled = torch.cat((encoded.mean(1), encoded.max(1).values), dim=1)
        social = self.social_projection(self.social_embedding[candidates])
        return self.action_projection(torch.cat((social, pooled), dim=1))

    def q_values_for_mask(self, mask: torch.Tensor, max_budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(self.social_embedding.device, torch.bool)
        candidates = self.candidates(mask)
        if not candidates.numel():
            return candidates, torch.empty(0, device=self.social_embedding.device)
        state = self._selected_social_state(mask, max_budget)
        actions = self._area_action_state(mask, candidates)
        return candidates, self.qnet.q_values(state, actions)[0]

    def distribution_for_mask(self, mask: torch.Tensor, max_budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(self.social_embedding.device, torch.bool)
        candidates = self.candidates(mask)
        if not candidates.numel():
            return candidates, torch.empty((0, self.qnet.atoms), device=self.social_embedding.device)
        state = self._selected_social_state(mask, max_budget)
        actions = self._area_action_state(mask, candidates)
        return candidates, self.qnet.distribution(state, actions)[0]


@torch.no_grad()
def journal_greedy_select(model: JournalSelector, k: int, device: str | torch.device = "cpu") -> list[int]:
    model = model.to(device).eval()
    mask = torch.zeros(model.n_nodes, dtype=torch.bool, device=device)
    order: list[int] = []
    while len(order) < min(k, int(model.worker_pool_mask.sum())):
        candidates, values = model.q_values_for_mask(mask, k)
        if not candidates.numel():
            break
        chosen = int(candidates[values.argmax()].item())
        order.append(chosen)
        mask[chosen] = True
    return order


@dataclass(frozen=True)
class TeacherState:
    mask: np.ndarray
    gains: np.ndarray
    max_budget: int


def build_teacher_states(oracle, pool: Iterable[int], budgets: Iterable[int], orders: Iterable[list[int]], states_per_order: int = 4) -> list[TeacherState]:
    """Materialize training-world marginal-gain labels without consulting validation/test worlds."""
    pool = np.asarray(sorted(pool), dtype=np.int64)
    states: list[TeacherState] = []
    for order in orders:
        for budget in sorted(set(int(v) for v in budgets)):
            prefixes = np.linspace(0, max(0, budget - 1), num=min(states_per_order, budget), dtype=int)
            for prefix in sorted(set(int(x) for x in prefixes.tolist())):
                selected = set(int(v) for v in order[:prefix])
                mask = np.zeros(max(oracle.nodes) + 1, dtype=bool)
                mask[list(selected)] = True
                gains = np.full(mask.shape[0], -np.inf, dtype=np.float32)
                for candidate in pool:
                    if candidate not in selected:
                        gains[candidate] = float(oracle.marginal_gain(selected, int(candidate)))
                states.append(TeacherState(mask=mask, gains=gains, max_budget=budget))
    return states


def _c51_loss(model: JournalSelector, target: JournalSelector, transition: Transition, max_budget: int, gamma: float) -> torch.Tensor:
    device = model.social_embedding.device
    mask = torch.as_tensor(transition.state_mask, dtype=torch.bool, device=device)
    candidates, pred_all = model.distribution_for_mask(mask, max_budget)
    position = torch.where(candidates == int(transition.action))[0]
    if position.numel() != 1:
        raise RuntimeError("invalid JournalSelector replay action")
    pred = pred_all[int(position.item())]
    with torch.no_grad():
        if transition.done:
            future = torch.full_like(pred, 1.0 / pred.numel())
        else:
            next_mask = torch.as_tensor(transition.next_state_mask, dtype=torch.bool, device=device)
            next_candidates, online_q = model.q_values_for_mask(next_mask, max_budget)
            if not next_candidates.numel():
                future = torch.full_like(pred, 1.0 / pred.numel())
            else:
                target_candidates, target_dist = target.distribution_for_mask(next_mask, max_budget)
                if not torch.equal(next_candidates, target_candidates):
                    raise RuntimeError("target candidate mismatch")
                future = target_dist[int(online_q.argmax().item())]
        projected = project_c51_distribution(
            future[None], torch.tensor([transition.reward], device=device),
            torch.tensor([float(transition.done)], device=device), gamma ** transition.n_steps,
            model.qnet.support,
        )[0]
    return -(projected * pred.log()).sum()


def _ranking_loss(model: JournalSelector, teacher: TeacherState, temperature: float) -> torch.Tensor:
    device = model.social_embedding.device
    mask = torch.as_tensor(teacher.mask, dtype=torch.bool, device=device)
    candidates, q = model.q_values_for_mask(mask, teacher.max_budget)
    gains = torch.as_tensor(teacher.gains[candidates.cpu().numpy()], dtype=q.dtype, device=device)
    target = torch.softmax((gains - gains.max()) / temperature, dim=0)
    return -(target * torch.log_softmax(q / temperature, dim=0)).sum()


def train_journal_selector(
    model: JournalSelector, reward_fn: Callable[[set[int], int], float], budgets: Iterable[int],
    teacher_states: list[TeacherState], episodes: int = 400, learning_rate: float = 1e-4,
    gamma: float = .99, n_step: int = 3, batch_size: int = 32, warmup: int = 64,
    target_update: int = 100, ranking_weight: float = .3, ranking_temperature: float = .1,
    ranking_pretrain_epochs: int = 20, seed: int = 0, device: str | torch.device = "cpu",
) -> tuple[JournalSelector, dict]:
    """Train DQNSelector-J with fixed-world EC rewards and optional teacher ranking."""
    budgets = sorted(set(int(k) for k in budgets))
    if not budgets or not teacher_states:
        raise ValueError("budgets and teacher_states are required")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    target = deepcopy(model).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(ranking_pretrain_epochs):
        teacher = teacher_states[int(rng.integers(len(teacher_states)))]
        optimizer.zero_grad(set_to_none=True)
        _ranking_loss(model, teacher, ranking_temperature).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    replay = PrioritizedReplay(10000, random_seed=seed)
    accumulator = NStepAccumulator(n_step=n_step, gamma=gamma)
    schedule = [budgets[i % len(budgets)] for i in rng.permutation(np.resize(np.arange(len(budgets)), episodes))]
    returns: list[float] = []; losses: list[float] = []; updates = 0
    for budget in schedule:
        accumulator.clear(); selected: set[int] = set(); mask = np.zeros(model.n_nodes, dtype=bool); total = 0.0
        for step in range(budget):
            model.qnet.reset_noise()
            candidates, values = model.q_values_for_mask(torch.as_tensor(mask, device=device), budget)
            action = int(candidates[values.argmax()].item())
            reward = float(reward_fn(set(selected), action))
            nxt = mask.copy(); nxt[action] = True
            done = step + 1 == budget
            for aggregated in accumulator.push(Transition(mask.copy(), action, reward, nxt.copy(), done)):
                replay.add(aggregated)
            mask = nxt; selected.add(action); total += reward
            if len(replay) >= max(batch_size, warmup):
                batch, indices, importance = replay.sample(batch_size)
                model.qnet.reset_noise(); target.qnet.reset_noise()
                rl_losses = torch.stack([_c51_loss(model, target, transition, budget, gamma) for transition in batch])
                teacher = teacher_states[int(rng.integers(len(teacher_states)))]
                loss = (rl_losses * torch.as_tensor(importance, device=device)).mean() + ranking_weight * _ranking_loss(model, teacher, ranking_temperature)
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
                replay.update_priorities(indices, rl_losses.detach().cpu().numpy())
                losses.append(float(loss.detach().cpu())); updates += 1
                if updates % target_update == 0:
                    target.load_state_dict(model.state_dict())
        returns.append(total)
    return model, {"episodes": episodes, "updates": updates, "returns": returns, "losses": losses,
                   "ranking_pretrain_epochs": ranking_pretrain_epochs, "ranking_weight": ranking_weight,
                   "ranking_temperature": ranking_temperature, "checkpoint_version": model.checkpoint_version}
