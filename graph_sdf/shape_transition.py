"""Lightweight node-level shape transition head for Graph-SDF planning."""

from typing import Optional

import torch
import torch.nn as nn

from .config import GraphSdfModelConfig


class ShapeTransitionHead(nn.Module):
    """Predicts a coarse next-state node SDF as an auxiliary planning signal."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        """Initializes a residual MLP that updates node-wise residual thickness."""
        super().__init__()
        config.validate()
        self.sdf_channel_index = config.sdf_channel_index
        hidden = config.hidden_dim
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        state_points: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts next node-level SDF from the current graph state."""

        current_node_sdf = state_points[..., self.sdf_channel_index].mean(dim=2, keepdim=True)
        pred_delta_sdf = self.delta_head(node_embeddings)
        pred_next_node_sdf = current_node_sdf + pred_delta_sdf

        if node_mask is not None:
            pred_delta_sdf = pred_delta_sdf.masked_fill(node_mask[:, :, None], 0.0)
            pred_next_node_sdf = pred_next_node_sdf.masked_fill(node_mask[:, :, None], 0.0)
            current_node_sdf = current_node_sdf.masked_fill(node_mask[:, :, None], 0.0)

        return {
            "current_node_sdf": current_node_sdf,
            "pred_delta_sdf": pred_delta_sdf,
            "pred_next_node_sdf": pred_next_node_sdf,
        }
