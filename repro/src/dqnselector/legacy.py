from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .influence import layered_product_activation_probabilities


@dataclass
class LegacyProcessedInstance:
    graph: nx.DiGraph
    nodes: list[int]
    node_values: np.ndarray

    @property
    def n_subareas(self) -> int:
        return int(self.node_values.shape[1])


def load_legacy_processed_instance(
    node_file: str | Path,
    edge_file: str | Path,
    n_subareas: int = 100,
) -> LegacyProcessedInstance:
    """Read the released `input_node_*` and `input_edge_*` files.

    The public repository stores one vector per node but does not document enough
    information to invert that vector into separate p_v^i, q_v^i and d_i. We
    therefore call it `node_values` rather than pretending it is the full ECM
    instance defined in the paper.
    """
    graph = nx.DiGraph()
    values: dict[int, np.ndarray] = {}
    with Path(node_file).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < n_subareas + 1:
                raise ValueError(f"node file line {line_no} has too few columns")
            node = int(fields[0])
            vec = np.asarray([float(x) for x in fields[1 : n_subareas + 1]], dtype=np.float64)
            values[node] = vec
            graph.add_node(node)
    with Path(edge_file).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"edge file line {line_no} has too few columns")
            u, v, w = int(fields[0]), int(fields[1]), float(fields[2])
            graph.add_edge(u, v, weight=w)
    nodes = list(values.keys())
    matrix = np.stack([values[v] for v in nodes], axis=0)
    return LegacyProcessedInstance(graph=graph, nodes=nodes, node_values=matrix)


def legacy_effective_coverage(
    instance: LegacyProcessedInstance,
    seeds: set[int] | list[int],
    clip_per_subarea: bool = False,
) -> tuple[float, np.ndarray]:
    """Compatibility score matching the released script as closely as possible.

    The released `effective_cov` propagates approximate activation probabilities,
    multiplies them by each node's stored vector, sums by subarea and returns the
    mean *without* the paper's explicit min(C_i/d_i,1) cap. `clip_per_subarea=True`
    provides a diagnostic capped variant when the stored vector is already
    normalized by demand.
    """
    probs = layered_product_activation_probabilities(instance.graph, seeds)
    idx = {v: i for i, v in enumerate(instance.nodes)}
    p = np.asarray([probs.get(v, 0.0) for v in instance.nodes], dtype=np.float64)
    coverage = (p[:, None] * instance.node_values).sum(axis=0)
    if clip_per_subarea:
        coverage = np.minimum(coverage, 1.0)
    return float(coverage.mean()), coverage
