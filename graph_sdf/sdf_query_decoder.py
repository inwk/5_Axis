"""Query-point TSDF decoder for action-conditioned shape transitions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import GraphSdfModelConfig
from .octree_decoder import ScaleAwarePosEncoding, _OctreeDecoderBlock


class SdfQueryDecoder(nn.Module):
    """Predicts after-operation TSDF at arbitrary 3D query points."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        super().__init__()
        config.validate()
        hidden = config.hidden_dim
        self.pos_enc = ScaleAwarePosEncoding(num_bands=config.octree_fourier_bands)
        self.input_proj = nn.Sequential(
            nn.Linear(self.pos_enc.output_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.query_state_proj = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.query_action_proj = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.decoder_blocks = nn.ModuleList([
            _OctreeDecoderBlock(
                hidden_dim=hidden,
                num_heads=config.octree_cross_attn_heads,
                dropout=config.octree_dropout,
            )
            for _ in range(config.octree_decoder_layers)
        ])
        self.tsdf_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        action_context: torch.Tensor,
        query_points: torch.Tensor,
        query_state: Optional[torch.Tensor] = None,
        query_action_features: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dummy_scale = torch.zeros((*query_points.shape[:-1], 1), device=query_points.device, dtype=query_points.dtype)
        xyzd = torch.cat([query_points, dummy_scale], dim=-1)
        features = self.input_proj(self.pos_enc(xyzd))

        if query_state is not None:
            q = query_state.float()
            if q.shape[-1] != 4:
                raise ValueError(f"query_state must have last dim 4, got {tuple(q.shape)}")
            features = features + self.query_state_proj(q.to(features.device))
        if query_action_features is not None:
            q_action = query_action_features.float()
            if q_action.shape[-1] != 5:
                raise ValueError(f"query_action_features must have last dim 5, got {tuple(q_action.shape)}")
            features = features + self.query_action_proj(q_action.to(features.device))

        for block in self.decoder_blocks:
            features = block(
                query_features=features,
                node_embeddings=node_embeddings,
                action_context=action_context,
                node_key_padding_mask=node_mask,
            )
        return self.tsdf_head(features).squeeze(-1)
