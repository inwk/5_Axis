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
    """Applies graph-aware transformer message passing across graph nodes."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_spatial_pos: int = 255,
    ) -> None:
        """Builds graph-aware transformer layers for node-level contextualization."""
        super().__init__()
        self.num_heads = num_heads
        self.max_spatial_pos = max_spatial_pos
        self.spatial_pos_embedding = nn.Embedding(max_spatial_pos + 2, num_heads)
        self.layers = nn.ModuleList(
            [
                _GraphTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _build_attn_bias(
        self,
        node_count: int,
        spatial_pos: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Builds additive attention bias [B*H, N, N] from spatial positions."""

        if spatial_pos is None:
            return None

        clipped = spatial_pos.long().clamp(min=0, max=self.max_spatial_pos + 1)
        bias = self.spatial_pos_embedding(clipped).permute(0, 3, 1, 2).contiguous()
        return bias.view(-1, node_count, node_count).to(device)

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns contextualized node features [B, N, H]."""

        node_count = node_features.shape[1]
        attn_bias = self._build_attn_bias(
            node_count=node_count,
            spatial_pos=spatial_pos,
            device=node_features.device,
        )

        output = node_features
        for layer in self.layers:
            output = layer(output, node_mask=node_mask, attn_bias=attn_bias)
        return self.norm(output)


class _GraphTransformerLayer(nn.Module):
    """One graph-aware transformer layer with optional additive attention bias."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        """Initializes multi-head attention and feed-forward blocks."""
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one attention + feed-forward update."""

        normalized = self.attn_norm(node_features)
        attn_mask = attn_bias
        key_padding_mask = node_mask
        if attn_mask is not None and key_padding_mask is not None:
            # Convert key padding into additive attention mask to keep mask dtypes consistent.
            expanded_padding = key_padding_mask[:, None, None, :].float() * (-1e4)
            expanded_padding = expanded_padding.expand(-1, self.self_attn.num_heads, normalized.shape[1], -1)
            attn_mask = attn_mask + expanded_padding.reshape_as(attn_mask)
            key_padding_mask = None
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        node_features = node_features + self.dropout(attended)
        node_features = node_features + self.dropout(self.ffn(self.ffn_norm(node_features)))
        return node_features
