"""Reusable neural layers for Graph-SDF modeling."""

from typing import Optional

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Encodes per-node point-wise features into one node embedding."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Builds shared MLP layers for point features and node projection."""
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        point_features: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds [B, N, P, F] point features into [B, N, H] node features."""

        features = self.point_mlp(point_features)
        if point_mask is not None:
            masked = features.masked_fill(point_mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=2).values
            pooled[torch.isinf(pooled)] = 0.0
        else:
            pooled = features.max(dim=2).values
        return self.output_proj(pooled)


class GraphTransformerEncoder(nn.Module):
    """Applies transformer message passing across graph nodes."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Builds transformer layers for node-level contextualization."""
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns contextualized node features [B, N, H]."""

        output = self.encoder(node_features, src_key_padding_mask=node_mask)
        return self.norm(output)
