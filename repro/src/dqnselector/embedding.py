from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import nn

from .influence import mc_activation_probabilities


@dataclass
class InfluencePairDataset:
    seeds: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    nodes: list[int]

    def __len__(self) -> int:
        return int(self.seeds.shape[0])


def build_balanced_influence_pairs(
    graph: nx.DiGraph,
    mc_times: int = 100,
    graph_iterations: int = 1,
    random_seed: int = 0,
) -> InfluencePairDataset:
    """Build a dense balanced pair set for diagnostics/alternative fitting.

    This utility follows the paper's positive/negative definition but stores all
    positive pairs plus an equal number of negative pairs. The more literal
    Algorithm-1 update path is `fit_social_influence_embedding_algorithm1` below.
    """
    if graph_iterations <= 0:
        raise ValueError("graph_iterations must be positive")
    nodes = list(graph.nodes())
    node_to_index = {v: i for i, v in enumerate(nodes)}
    rng = np.random.default_rng(random_seed)
    seed_idx: list[int] = []
    target_idx: list[int] = []
    probs_out: list[float] = []

    for _ in range(graph_iterations):
        order = list(nodes)
        rng.shuffle(order)
        for seed in order:
            probs = mc_activation_probabilities(
                graph,
                [seed],
                mc_times=mc_times,
                random_seed=int(rng.integers(0, 2**31 - 1)),
            )
            positives = [v for v in nodes if v != seed and probs.get(v, 0.0) > 0.0]
            if not positives:
                continue
            negatives = [v for v in nodes if v != seed and probs.get(v, 0.0) == 0.0]
            if not negatives:
                negatives = [v for v in nodes if v != seed]
            neg_sample = rng.choice(negatives, size=len(positives), replace=True)
            for target in positives:
                seed_idx.append(node_to_index[seed])
                target_idx.append(node_to_index[target])
                probs_out.append(float(probs[target]))
            for target in neg_sample.tolist():
                seed_idx.append(node_to_index[seed])
                target_idx.append(node_to_index[int(target)])
                probs_out.append(0.0)

    return InfluencePairDataset(
        seeds=np.asarray(seed_idx, dtype=np.int64),
        targets=np.asarray(target_idx, dtype=np.int64),
        probabilities=np.asarray(probs_out, dtype=np.float32),
        nodes=nodes,
    )


def fit_social_influence_embedding_algorithm1(
    graph: nx.DiGraph,
    dim: int = 128,
    mc_times: int = 100,
    graph_iterations: int = 30,
    learning_rate: float = 0.01,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Literal implementation of the published Algorithm 1 / Eqs. (8)--(9).

    For every graph iteration, nodes are shuffled. For each seed v, MC estimates
    p_u({v}); an equal-size negative pool is sampled with replacement; then one
    target u is uniformly drawn from the positive+negative pool and one SGD update
    is performed. The returned trace is the mean squared error of those performed
    updates in each graph iteration.

    The paper's pseudocode performs one randomly selected pair update per seed,
    which is materially different from iterating over every positive/negative pair.
    Both versions are kept so this distinction is testable rather than hidden.
    """
    if dim <= 0 or mc_times <= 0 or graph_iterations <= 0:
        raise ValueError("dim, mc_times and graph_iterations must be positive")
    nodes = list(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError("Algorithm-1 implementation currently requires dense node ids 0..n-1")
    rng = np.random.default_rng(random_seed)
    bound = 1.0 / float(dim)
    s = rng.uniform(-bound, bound, size=(len(nodes), dim)).astype(np.float64)
    t = rng.uniform(-bound, bound, size=(len(nodes), dim)).astype(np.float64)
    history: list[float] = []

    for _ in range(graph_iterations):
        order = np.asarray(nodes, dtype=np.int64)
        rng.shuffle(order)
        losses: list[float] = []
        for seed_np in order:
            seed = int(seed_np)
            probs = mc_activation_probabilities(
                graph,
                [seed],
                mc_times=mc_times,
                random_seed=int(rng.integers(0, 2**31 - 1)),
            )
            positives = [u for u in nodes if u != seed and probs.get(u, 0.0) > 0.0]
            if positives:
                negatives = [u for u in nodes if u != seed and probs.get(u, 0.0) == 0.0]
                if negatives:
                    neg_sample = rng.choice(negatives, size=len(positives), replace=True).tolist()
                else:
                    neg_sample = []
                sample_pool = positives + [int(x) for x in neg_sample]
                target = int(sample_pool[int(rng.integers(0, len(sample_pool)))])
                label = float(probs[target]) if target in positives else 0.0
            else:
                negatives = [u for u in nodes if u != seed]
                if not negatives:
                    continue
                target = int(negatives[int(rng.integers(0, len(negatives)))])
                label = 0.0

            old_s = s[seed].copy()
            old_t = t[target].copy()
            pred = float(old_s @ old_t)
            error = pred - label
            # Paper Eqs. (8)--(9) omit the constant factor 2 from differentiating
            # squared error, so we follow the equations literally here.
            s[seed] = old_s - learning_rate * error * old_t
            t[target] = old_t - learning_rate * error * old_s
            losses.append(error * error)
        history.append(float(np.mean(losses)) if losses else 0.0)
    return s.astype(np.float32), t.astype(np.float32), history


class SocialInfluenceEmbedding(nn.Module):
    """Two directed embeddings s_v and t_u whose dot product predicts p_u({v})."""

    def __init__(self, n_nodes: int, dim: int = 128) -> None:
        super().__init__()
        self.seed_embedding = nn.Embedding(n_nodes, dim)
        self.target_embedding = nn.Embedding(n_nodes, dim)
        bound = 1.0 / float(dim)
        nn.init.uniform_(self.seed_embedding.weight, -bound, bound)
        nn.init.uniform_(self.target_embedding.weight, -bound, bound)

    def forward(self, seeds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        s = self.seed_embedding(seeds)
        t = self.target_embedding(targets)
        return (s * t).sum(dim=-1)


def fit_social_influence_embedding(
    pairs: InfluencePairDataset,
    dim: int = 128,
    epochs: int = 30,
    learning_rate: float = 0.01,
    batch_size: int = 1024,
    random_seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit Eq. (7) over an explicit pair dataset with mini-batch SGD."""
    if len(pairs) == 0:
        raise ValueError("no influence pairs were generated")
    torch.manual_seed(random_seed)
    device = torch.device(device)
    model = SocialInfluenceEmbedding(len(pairs.nodes), dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    g = torch.Generator(device="cpu").manual_seed(random_seed)

    seeds = torch.from_numpy(pairs.seeds)
    targets = torch.from_numpy(pairs.targets)
    labels = torch.from_numpy(pairs.probabilities)
    history: list[float] = []

    for _ in range(epochs):
        order = torch.randperm(len(pairs), generator=g)
        epoch_loss = 0.0
        count = 0
        for start in range(0, len(pairs), batch_size):
            idx = order[start : start + batch_size]
            b_seed = seeds[idx].to(device)
            b_target = targets[idx].to(device)
            b_label = labels[idx].to(device)
            pred = model(b_seed, b_target)
            loss = mse(pred, b_label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(idx)
            count += len(idx)
        history.append(epoch_loss / max(count, 1))

    s = model.seed_embedding.weight.detach().cpu().numpy().copy()
    t = model.target_embedding.weight.detach().cpu().numpy().copy()
    return s, t, history
