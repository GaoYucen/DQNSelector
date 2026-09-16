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
    """Construct positive/negative pairs for Eqs. (7)--(9) / Algorithm 1.

    A positive target is any non-seed node observed with non-zero estimated
    activation probability. For each seed and graph iteration we sample the same
    number of zero-probability targets with replacement, matching Algorithm 1's
    balanced-pair description.
    """
    if graph_iterations <= 0:
        raise ValueError("graph_iterations must be positive")
    nodes = list(graph.nodes())
    node_to_index = {v: i for i, v in enumerate(nodes)}
    rng = np.random.default_rng(random_seed)
    seed_idx: list[int] = []
    target_idx: list[int] = []
    probs_out: list[float] = []

    for gi in range(graph_iterations):
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
    """Fit Eq. (7) with mini-batch SGD and return s, t, loss history."""
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
