"""Paper-based PyTorch PIANO reproduction, not a DQNSelector ablation.

Li et al., TCSS 2023, Eqs. (1)--(3): structure2vec with selected-node flags,
weighted edge features, global sum readout, and n-step influence-reward DQN.
The original repository differs from the paper; this implementation follows
the published equations. The experiment records all adapter choices.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn

from .rainbow import NStepAccumulator, Transition


class PianoQNet(nn.Module):
    def __init__(self, graph, worker_pool, dim=64, rounds=4):
        super().__init__()
        self.n_nodes = graph.number_of_nodes()
        self.rounds = rounds
        edges = list(graph.edges(data=True))
        idx = torch.tensor([[u for u, _, _ in edges], [v for _, v, _ in edges]], dtype=torch.long)
        if not edges:
            idx = torch.empty((2, 0), dtype=torch.long)
        adjacency = torch.sparse_coo_tensor(idx, torch.ones(len(edges)), (self.n_nodes, self.n_nodes)).coalesce()
        weighted_degree = torch.zeros(self.n_nodes)
        if edges:
            weighted_degree.index_add_(0, idx[0], torch.tensor([d.get('weight', 1.) for _, _, d in edges]))
        self.register_buffer('adjacency', adjacency)
        self.register_buffer('weighted_degree', weighted_degree)
        pool = torch.zeros(self.n_nodes, dtype=torch.bool)
        pool[sorted(worker_pool)] = True
        self.register_buffer('worker_pool_mask', pool)
        self.alpha1 = nn.Linear(dim, dim, bias=False)
        self.alpha2 = nn.Linear(dim, dim, bias=False)
        self.alpha3 = nn.Parameter(torch.empty(dim))
        self.alpha4 = nn.Parameter(torch.empty(dim))
        self.beta2 = nn.Linear(dim, dim, bias=False)
        self.beta3 = nn.Linear(dim, dim, bias=False)
        self.beta1 = nn.Linear(2 * dim, 1, bias=False)
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=.01)

    def forward(self, selected):
        if selected.ndim == 1:
            selected = selected[None]
        selected = selected.to(device=self.alpha3.device, dtype=torch.bool)
        b, n = selected.shape
        d = self.alpha3.numel()
        # w>=0 implies sum_u ReLU(alpha3*w_vu)=ReLU(alpha3)*sum_u w_vu.
        edge_term = self.alpha2(self.weighted_degree[:, None] * self.alpha3.relu())
        static = edge_term[None] + selected[:, :, None] * self.alpha4
        x = torch.zeros((b, n, d), device=selected.device)
        for _ in range(self.rounds):
            neighbors = torch.sparse.mm(self.adjacency, x.permute(1, 0, 2).reshape(n, b*d))
            neighbors = neighbors.reshape(n, b, d).permute(1, 0, 2)
            x = torch.relu(self.alpha1(neighbors) + static)
        global_readout = self.beta2(x.sum(dim=1)).relu()
        local_readout = self.beta3(x).relu()
        q = self.beta1(torch.cat((global_readout[:, None].expand(-1, n, -1), local_readout), dim=-1)).squeeze(-1)
        return q.masked_fill(selected | ~self.worker_pool_mask[None], -torch.inf)


@torch.no_grad()
def piano_select(model, k, device='cpu'):
    model = model.to(device).eval()
    mask = torch.zeros(model.n_nodes, dtype=torch.bool, device=device)
    order = []
    for _ in range(min(k, int(model.worker_pool_mask.sum()))):
        a = int(model(mask)[0].argmax())
        order.append(a)
        mask[a] = True
    return order


def train_piano(model, reward_fn, episodes=200, budget=50, seed=2024, device='cuda',
                learning_rate=.001, gamma=.95, n_step=5, batch_size=64,
                target_update=100, replay_capacity=10000, progress=None,
                exploration_steps=10000, lr_decay_interval=1000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    target = deepcopy(model).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay = []
    acc = NStepAccumulator(n_step=n_step, gamma=gamma)
    pool = torch.where(model.worker_pool_mask)[0].cpu().numpy()
    updates = 0
    trace = []
    start = time.perf_counter()
    for episode in range(episodes):
        acc.clear()
        selected = set()
        mask = np.zeros(model.n_nodes, dtype=bool)
        total = 0.
        model.train()
        for t in range(budget):
            elapsed_steps = episode * budget + t
            epsilon = max(.05, 1. - .95 * elapsed_steps / max(1, exploration_steps-1))
            if rng.random() < epsilon:
                a = int(rng.choice(pool[~mask[pool]]))
            else:
                with torch.no_grad():
                    a = int(model(torch.as_tensor(mask, device=device))[0].argmax())
            reward = float(reward_fn(selected, a))
            nxt = mask.copy()
            nxt[a] = True
            done = t + 1 == budget
            replay.extend(acc.push(Transition(mask.copy(), a, reward, nxt.copy(), done, 1)))
            if len(replay) > replay_capacity:
                del replay[:len(replay)-replay_capacity]
            selected.add(a)
            mask = nxt
            total += reward
            if len(replay) >= batch_size:
                batch = [replay[i] for i in rng.choice(len(replay), batch_size, replace=False)]
                states = torch.as_tensor(np.stack([tr.state_mask for tr in batch]), device=device)
                actions = torch.tensor([tr.action for tr in batch], device=device)
                predicted = model(states).gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    next_states = torch.as_tensor(np.stack([tr.next_state_mask for tr in batch]), device=device)
                    future = target(next_states).max(dim=1).values
                    terminals = torch.tensor([tr.done for tr in batch], device=device)
                    future = torch.where(terminals, torch.zeros_like(future), future)
                    rewards = torch.tensor([tr.reward for tr in batch], device=device)
                    discount = torch.tensor([gamma**tr.n_steps for tr in batch], device=device)
                    desired = rewards + discount * future
                loss = nn.functional.mse_loss(predicted, desired)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite PIANO loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                updates += 1
                if lr_decay_interval and updates % lr_decay_interval == 0:
                    for group in optimizer.param_groups:
                        group['lr'] *= .95
                if updates % target_update == 0:
                    target.load_state_dict(model.state_dict())
        trace.append(total)
        if progress is not None and ((episode + 1) % 10 == 0 or episode == 0):
            progress(episode + 1, total, time.perf_counter() - start)
    return model, dict(episode_returns=trace,updates=updates,transitions=episodes*budget,
                       training_seconds=time.perf_counter()-start,seed=seed,
                       exploration_steps=exploration_steps,final_epsilon=epsilon,
                       lr_decay_interval=lr_decay_interval,final_learning_rate=optimizer.param_groups[0]['lr'])
