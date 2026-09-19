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


def sample_piano_subgraph(graph, max_nodes: int, seed: int, roots=None):
    """Public-repository-compatible connected BFS subgraph sampler.

    PIANO trains on connected graph samples and transfers only trainable
    parameters to the full deployment graph.  The returned graph is relabelled
    densely and carries ``original_node_ids`` for reward adapters.
    """
    import networkx as nx
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    rng = np.random.default_rng(seed)
    undirected = graph.to_undirected()
    components = [component for component in nx.connected_components(undirected) if component]
    if not components:
        return nx.DiGraph(), []
    component = components[int(rng.integers(len(components)))]
    valid_roots = sorted(set(component) & set(roots or component))
    root = int(rng.choice(np.asarray(valid_roots or sorted(component), dtype=np.int64)))
    queue, seen, nodes = [root], {root}, []
    while queue and len(nodes) < max_nodes:
        node = queue.pop(0); nodes.append(node)
        neighbors = list(undirected.neighbors(node)); rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in component and neighbor not in seen:
                seen.add(neighbor); queue.append(neighbor)
    sampled = graph.subgraph(nodes).copy()
    mapping = {node: index for index, node in enumerate(nodes)}
    return nx.relabel_nodes(sampled, mapping, copy=True), nodes


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
                exploration_steps=10000, lr_decay_interval=1000, progress_interval=1):
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
        if progress is not None and ((episode + 1) % max(1, progress_interval) == 0 or episode == 0):
            progress(episode + 1, total, time.perf_counter() - start)
    return model, dict(episode_returns=trace,updates=updates,transitions=episodes*budget,
                       training_seconds=time.perf_counter()-start,seed=seed,
                       exploration_steps=exploration_steps,final_epsilon=epsilon,
                       lr_decay_interval=lr_decay_interval,final_learning_rate=optimizer.param_groups[0]['lr'])


def train_piano_on_subgraphs(full_model, full_graph, worker_pool, reward_fn, episodes=100,
                             budget=50, subgraph_nodes=1024, games_per_subgraph=2,
                             seed=2024, device='cuda', progress=None):
    """Train paper-equation PIANO on BFS samples and transfer parameters to full graph.

    The public implementation rotates connected BFS samples during training.  Its
    C++ replay memory stores a graph with every transition; this compact PyTorch
    adapter instead keeps replay local to each sampled graph, while transferring
    only named trainable parameters back to the deployment model.  The adapter is
    deliberately recorded by callers and never presented as byte-identical code.
    """
    full_pool = set(int(v) for v in worker_pool)
    total_updates = total_transitions = 0
    returns: list[float] = []
    start = time.perf_counter()
    for episode in range(episodes):
        graph, original_nodes = sample_piano_subgraph(
            full_graph, subgraph_nodes, seed + episode, roots=full_pool
        )
        original_pool = [node for node in original_nodes if node in full_pool]
        if len(original_pool) < 2:
            continue
        remap = {node: position for position, node in enumerate(original_nodes)}
        local_pool = [remap[node] for node in original_pool]
        local_budget = min(int(budget), len(local_pool))
        local = PianoQNet(graph, local_pool, dim=full_model.alpha3.numel(), rounds=full_model.rounds)
        local_params = dict(local.named_parameters())
        with torch.no_grad():
            for name, parameter in full_model.named_parameters():
                local_params[name].copy_(parameter.detach().cpu())

        def local_reward(selected, candidate):
            selected_original = {original_nodes[v] for v in selected}
            return float(reward_fn(selected_original, original_nodes[int(candidate)]))

        local, stats = train_piano(
            local, local_reward, episodes=games_per_subgraph, budget=local_budget,
            seed=seed + episode, device=device, progress_interval=games_per_subgraph + 1,
        )
        with torch.no_grad():
            trained = dict(local.named_parameters())
            for name, parameter in full_model.named_parameters():
                parameter.copy_(trained[name].detach().to(parameter.device))
        total_updates += int(stats["updates"])
        total_transitions += int(stats["transitions"])
        returns.extend(stats["episode_returns"])
        if progress is not None:
            progress(episode + 1, float(np.mean(stats["episode_returns"])), time.perf_counter() - start)
    return full_model, {"episode_returns": returns, "updates": total_updates, "transitions": total_transitions,
                        "training_seconds": time.perf_counter() - start, "seed": seed,
                        "subgraph_nodes": subgraph_nodes, "games_per_subgraph": games_per_subgraph,
                        "adapter": "BFS-subgraph parameter-transfer with local replay"}
