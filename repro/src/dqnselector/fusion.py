from __future__ import annotations

import torch
from torch import nn


class GatedDualEmbedding(nn.Module):
    """Paper Eqs. (12)--(14).

    G_s = sigmoid(W1 s + W2 r + b1)
    G_r = sigmoid(W3 s + W4 r + b2)
    x_v = [G_s * s || G_r * r]
    """

    def __init__(self, social_dim: int, coverage_dim: int) -> None:
        super().__init__()
        self.social_from_social = nn.Linear(social_dim, social_dim, bias=False)
        self.social_from_coverage = nn.Linear(coverage_dim, social_dim, bias=True)
        self.coverage_from_social = nn.Linear(social_dim, coverage_dim, bias=False)
        self.coverage_from_coverage = nn.Linear(coverage_dim, coverage_dim, bias=True)

    @property
    def output_dim(self) -> int:
        return self.social_from_social.in_features + self.coverage_from_coverage.in_features

    def gates(
        self, social: torch.Tensor, coverage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gs = torch.sigmoid(
            self.social_from_social(social) + self.social_from_coverage(coverage)
        )
        gr = torch.sigmoid(
            self.coverage_from_social(social) + self.coverage_from_coverage(coverage)
        )
        return gs, gr

    def forward(self, social: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        gs, gr = self.gates(social, coverage)
        return torch.cat([gs * social, gr * coverage], dim=-1)
