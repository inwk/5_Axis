"""State encoder for per-face local SDF graph states."""

from typing import Optional

import torch
import torch.nn as nn

from .config import GraphSdfModelConfig
from .layers import GraphTransformerEncoder, PointNetEncoder


class StateEncoder(nn.Module):
    """Encodes [B, N, P, F] local state tensors into [B, N, H] graph embeddings."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        """Builds point encoder, node index embeddings, and graph transformer."""
        super().__init__()
        config.validate()
        self.point_encoder = PointNetEncoder(
            input_dim=config.point_feature_dim,
            hidden_dim=config.hidden_dim,
        )
        self.node_process_projection = (
            nn.Linear(config.node_process_feature_dim, config.hidden_dim)
            if config.node_process_feature_dim > 0
            else None
        )
        self.node_index_embedding = nn.Embedding(config.num_nodes, config.hidden_dim)
        self.graph_encoder = GraphTransformerEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout=config.transformer_dropout,
        )

    def forward(
        self,
        node_point_features: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes node embeddings from geometry and optional process-state features."""

        batch_size, node_count = node_point_features.shape[:2]
        node_features = self.point_encoder(node_point_features, point_mask=point_mask)
        if self.node_process_projection is not None:
            if node_process_state is None:
                node_process_state = node_features.new_zeros(
                    batch_size,
                    node_count,
                    self.node_process_projection.in_features,
                )
            node_features = node_features + self.node_process_projection(node_process_state)

        node_index = torch.arange(node_count, device=node_point_features.device).unsqueeze(0)
        node_index = node_index.expand(batch_size, node_count)
        node_features = node_features + self.node_index_embedding(node_index)

        return self.graph_encoder(node_features, node_mask=node_mask)
