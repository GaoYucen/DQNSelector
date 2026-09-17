from __future__ import annotations

import numpy as np

from .ecm import ECMInstance
from .piic import piic


def normalized_direct_contribution(instance: ECMInstance) -> np.ndarray:
    """Return each worker's direct contribution in EC units.

    Before the per-subarea clipping in ECM, worker v contributes
    p(v,i) * q(v,i) / d_i to subarea i. Using this quantity as the base
    coverage feature keeps the learner representation aligned with the actual
    objective and naturally accounts for heterogeneous demand.
    """
    demand = np.asarray(instance.demand, dtype=np.float64)
    if np.any(demand <= 0):
        raise ValueError("all demands must be positive")
    return (
        np.asarray(instance.participation, dtype=np.float64)
        * np.asarray(instance.quality, dtype=np.float64)
        / demand[None, :]
    )


def objective_aware_coverage_embedding(
    instance: ECMInstance,
    influence_range: int,
    include_direct: bool = True,
) -> np.ndarray:
    """PIIC-style feature built from objective-aligned direct utility.

    PIIC is a linear path/walk approximation, so we propagate the normalized
    direct EC contribution rather than quality alone. The direct term is kept by
    default: an isolated candidate can still be valuable even if it influences
    nobody else.
    """
    base = normalized_direct_contribution(instance)
    propagated = piic(
        instance.graph,
        base,
        influence_range=influence_range,
        nodes=instance.nodes,
    )
    return base + propagated if include_direct else propagated
