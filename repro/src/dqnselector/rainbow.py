from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer (Fortunato et al.)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma0 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma0 / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class PairwiseDuelingC51(nn.Module):
    """Distributional Q(s,a) for a variable candidate set."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ) -> None:
        super().__init__()
        if atoms < 2 or v_max <= v_min:
            raise ValueError("invalid C51 support")
        self.node_dim = node_dim
        self.atoms = atoms
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.state_fc = NoisyLinear(node_dim, hidden_dim)
        self.action_fc = NoisyLinear(node_dim, hidden_dim)
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value_out = NoisyLinear(hidden_dim, atoms)
        self.adv_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.adv_out = NoisyLinear(hidden_dim, atoms)
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def logits(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if state.shape[0] != actions.shape[0]:
            if state.shape[0] == 1:
                state = state.expand(actions.shape[0], -1)
            else:
                raise ValueError("state/actions batch size mismatch")
        s = F.relu(self.state_fc(state))
        a = F.relu(self.action_fc(actions))
        joint = F.relu(s.unsqueeze(1) + a)
        value = self.value_out(F.relu(self.value_hidden(s))).unsqueeze(1)
        advantage = self.adv_out(F.relu(self.adv_hidden(joint)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def distribution(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(state, actions), dim=-1).clamp_min(1e-8)

    def q_values(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(state, actions)
        return (dist * self.support).sum(dim=-1)


@dataclass
class Transition:
    state_mask: np.ndarray
    action: int
    reward: float
    next_state_mask: np.ndarray
    done: bool
    n_steps: int = 1


class NStepAccumulator:
    def __init__(self, n_step: int = 3, gamma: float = 0.99) -> None:
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque[Transition] = deque()

    def _aggregate_prefix(self) -> Transition:
        first = self.buffer[0]
        reward = 0.0
        next_mask = first.next_state_mask
        done = False
        steps = 0
        for i, tr in enumerate(self.buffer):
            reward += (self.gamma**i) * float(tr.reward)
            next_mask = tr.next_state_mask
            done = bool(tr.done)
            steps = i + 1
            if done or steps >= self.n_step:
                break
        return Transition(
            state_mask=first.state_mask.copy(),
            action=int(first.action),
            reward=float(reward),
            next_state_mask=next_mask.copy(),
            done=done,
            n_steps=steps,
        )

    def push(self, transition: Transition) -> list[Transition]:
        self.buffer.append(transition)
        emitted: list[Transition] = []
        if len(self.buffer) >= self.n_step:
            emitted.append(self._aggregate_prefix())
            self.buffer.popleft()
        if transition.done:
            while self.buffer:
                emitted.append(self._aggregate_prefix())
                self.buffer.popleft()
        return emitted

    def clear(self) -> None:
        self.buffer.clear()


class PrioritizedReplay:
    """Simple proportional prioritized replay suitable for correctness-first runs."""

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta0: float = 0.4,
        beta_steps: int = 100000,
        epsilon: float = 1e-6,
        random_seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.beta_steps = max(int(beta_steps), 1)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(random_seed)
        self.data: list[Transition] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.position = 0
        self.sample_steps = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition, priority: float | None = None) -> None:
        if priority is None:
            priority = float(self.priorities[: len(self.data)].max()) if self.data else 1.0
        priority = max(float(priority), self.epsilon)
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        if not self.data:
            raise ValueError("cannot sample an empty replay buffer")
        n = len(self.data)
        p = self.priorities[:n] ** self.alpha
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            p = np.full(n, 1.0 / n)
        else:
            p = p / p_sum
        replace = n < batch_size
        indices = self.rng.choice(n, size=batch_size, replace=replace, p=p)
        beta = min(1.0, self.beta0 + (1.0 - self.beta0) * self.sample_steps / self.beta_steps)
        self.sample_steps += 1
        weights = (n * p[indices]) ** (-beta)
        weights /= weights.max()
        return [self.data[int(i)] for i in indices], indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for i, p in zip(indices.tolist(), priorities.tolist()):
            self.priorities[int(i)] = max(float(abs(p)), self.epsilon)


def project_c51_distribution(
    next_distribution: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma_n: float,
    support: torch.Tensor,
) -> torch.Tensor:
    """Categorical Bellman projection used by distributional Rainbow."""
    if next_distribution.ndim != 2:
        raise ValueError("next_distribution must be [batch, atoms]")
    v_min = float(support[0])
    v_max = float(support[-1])
    atoms = int(support.numel())
    delta_z = (v_max - v_min) / (atoms - 1)
    tz = rewards[:, None] + (1.0 - dones[:, None]) * gamma_n * support[None, :]
    tz = tz.clamp(v_min, v_max)
    b = (tz - v_min) / delta_z
    lower = b.floor().long()
    upper = b.ceil().long()
    projected = torch.zeros_like(next_distribution)
    for j in range(atoms):
        l = lower[:, j]
        u = upper[:, j]
        p = next_distribution[:, j]
        same = l == u
        projected[same, l[same]] += p[same]
        diff = ~same
        projected[diff, l[diff]] += p[diff] * (u[diff].float() - b[diff, j])
        projected[diff, u[diff]] += p[diff] * (b[diff, j] - l[diff].float())
    return projected
