"""Action-conditioned decoder for one-step Graph-SDF state transitions."""

from typing import Optional

import torch
import torch.nn as nn

from .config import GraphSdfModelConfig


class ActionEmbedding(nn.Module):
    """Builds an action context from operation class, tool kind, geometry, and anchor-face state."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        """Initializes projections for macro class, tool kind, geometry, and action-face context."""
        super().__init__()
        self.sdf_channel_index = config.sdf_channel_index
        self.node_process_feature_dim = config.node_process_feature_dim
        action_dim = config.action_embedding_dim

        self.macro_embedding = nn.Embedding(config.macro_class_count, action_dim)
        self.tool_kind_embedding = nn.Embedding(config.tool_kind_count, action_dim)
        self.tool_geometry_projection = nn.Sequential(
            nn.Linear(8, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
        )
        self.target_projection = nn.Sequential(
            nn.Linear(config.hidden_dim + 3 + 4 + config.node_process_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, action_dim),
        )
        self.region_projection = nn.Sequential(
            nn.Linear(config.hidden_dim + 4 + config.node_process_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, action_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(action_dim * 5, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.action_dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    @staticmethod
    def _gather_node_features(node_features: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        """Selects one node feature vector per batch item."""
        gather_index = node_ids[:, None, None].expand(-1, 1, node_features.shape[-1])
        return torch.gather(node_features, 1, gather_index).squeeze(1)

    @staticmethod
    def _masked_mean(node_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Computes a mask-aware mean over graph nodes."""
        weights = valid_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (node_features * weights).sum(dim=1) / denom

    def forward(
        self,
        node_embeddings: torch.Tensor,
        state_points: torch.Tensor,
        macro_class_id: torch.Tensor,
        tool_choice_id: torch.Tensor,
        action_face_id: torch.Tensor,
        tool_kind_id: Optional[torch.Tensor] = None,
        axis_visible: Optional[torch.Tensor] = None,
        axis_dir: Optional[torch.Tensor] = None,
        tool_diameter_norm: Optional[torch.Tensor] = None,
        tool_radius_norm: Optional[torch.Tensor] = None,
        tool_length_norm: Optional[torch.Tensor] = None,
        holder_diameter_norm: Optional[torch.Tensor] = None,
        holder_radius_norm: Optional[torch.Tensor] = None,
        holder_length_norm: Optional[torch.Tensor] = None,
        node_process_state: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Fuses action choices with anchor-face geometry into one decoder context."""

        if node_process_state is None:
            node_process_state = node_embeddings.new_zeros(
                node_embeddings.shape[0],
                node_embeddings.shape[1],
                self.node_process_feature_dim,
            )

        target_node_embedding = self._gather_node_features(node_embeddings, action_face_id)
        target_normals = self._gather_node_features(state_points[..., 3:6].mean(dim=2), action_face_id)
        target_process_state = self._gather_node_features(node_process_state, action_face_id)

        per_node_sdf = state_points[..., self.sdf_channel_index]
        current_node_sdf = per_node_sdf.mean(dim=2)
        sdf_stats = torch.stack(
            [
                current_node_sdf,
                per_node_sdf.min(dim=2).values,
                per_node_sdf.max(dim=2).values,
                per_node_sdf.std(dim=2, unbiased=False),
            ],
            dim=-1,
        )
        target_sdf_stats = self._gather_node_features(sdf_stats, action_face_id)

        macro_feature = self.macro_embedding(macro_class_id)
        del tool_choice_id
        if tool_kind_id is None:
            tool_kind_id = macro_class_id.new_zeros(macro_class_id.shape)
        tool_kind_feature = self.tool_kind_embedding(tool_kind_id.clamp_min(0))
        if axis_dir is None:
            axis_feature = target_normals
        else:
            axis_feature = axis_dir.to(node_embeddings.device).float().reshape(node_embeddings.shape[0], 3)
        axis_feature = axis_feature / axis_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if tool_diameter_norm is None:
            if tool_radius_norm is None:
                tool_diameter_feature = node_embeddings.new_zeros(node_embeddings.shape[0], 1)
            else:
                tool_diameter_feature = 2.0 * tool_radius_norm.to(node_embeddings.device).float().reshape(node_embeddings.shape[0], -1)[:, :1]
        else:
            tool_diameter_feature = tool_diameter_norm.to(node_embeddings.device).float().reshape(node_embeddings.shape[0], -1)[:, :1]
        if tool_length_norm is None:
            tool_length_feature = node_embeddings.new_zeros(node_embeddings.shape[0], 1)
        else:
            tool_length_feature = tool_length_norm.to(node_embeddings.device).float().reshape(node_embeddings.shape[0], -1)[:, :1]
        if holder_diameter_norm is None:
            holder_diameter_feature = node_embeddings.new_zeros(node_embeddings.shape[0], 1)
        else:
            holder_diameter_feature = holder_diameter_norm.to(node_embeddings.device).float().reshape(node_embeddings.shape[0], -1)[:, :1]
        if holder_radius_norm is None:
            holder_radius_feature = node_embeddings.new_zeros(node_embeddings.shape[0], 1)
        else:
            holder_radius_feature = holder_radius_norm.to(node_embeddings.device).float().reshape(node_embeddings.shape[0], -1)[:, :1]
        if holder_length_norm is None:
            holder_length_feature = node_embeddings.new_zeros(node_embeddings.shape[0], 1)
        else:
            holder_length_feature = holder_length_norm.to(node_embeddings.device).float().reshape(node_embeddings.shape[0], -1)[:, :1]
        tool_geometry_feature = self.tool_geometry_projection(
            torch.cat(
                [
                    tool_diameter_feature,
                    tool_length_feature,
                    holder_diameter_feature,
                    holder_radius_feature,
                    holder_length_feature,
                    axis_feature,
                ],
                dim=-1,
            )
        )

        valid_nodes = torch.ones_like(current_node_sdf, dtype=torch.bool)
        if node_mask is not None:
            valid_nodes = valid_nodes & (~node_mask)
        visible_nodes = valid_nodes
        if axis_visible is not None:
            visible_nodes = visible_nodes & (axis_visible > 0)
        unfinished_nodes = current_node_sdf > 1e-6
        region_nodes = visible_nodes & unfinished_nodes
        has_region = region_nodes.any(dim=1, keepdim=True)
        region_nodes = torch.where(has_region, region_nodes, visible_nodes)
        has_visible = region_nodes.any(dim=1, keepdim=True)
        region_nodes = torch.where(has_visible, region_nodes, valid_nodes)

        region_embedding = self._masked_mean(node_embeddings, region_nodes)
        region_process_state = self._masked_mean(node_process_state, region_nodes)
        region_count = region_nodes.float().sum(dim=1).clamp_min(1.0)
        visible_count = visible_nodes.float().sum(dim=1).clamp_min(1.0)
        region_mean_sdf = (current_node_sdf * region_nodes.float()).sum(dim=1) / region_count
        region_max_sdf = current_node_sdf.masked_fill(~region_nodes, 0.0).max(dim=1).values
        region_fraction = region_count / valid_nodes.float().sum(dim=1).clamp_min(1.0)
        visible_fraction = visible_count / valid_nodes.float().sum(dim=1).clamp_min(1.0)
        region_stats = torch.stack(
            [region_mean_sdf, region_max_sdf, region_fraction, visible_fraction],
            dim=-1,
        )
        region_feature = self.region_projection(
            torch.cat([region_embedding, region_stats, region_process_state], dim=-1)
        )
        target_feature = self.target_projection(
            torch.cat(
                [target_node_embedding, target_normals, target_sdf_stats, target_process_state],
                dim=-1,
            )
        )
        action_context = self.fuse(
            torch.cat([macro_feature, tool_kind_feature, tool_geometry_feature, target_feature, region_feature], dim=-1)
        )
        return {
            "action_context": action_context,
            "target_node_embedding": target_node_embedding,
            "region_embedding": region_embedding,
        }


class _FiLMDecoderBlock(nn.Module):
    """Residual decoder block conditioned by the action context."""

    def __init__(self, hidden_dim: int) -> None:
        """Creates one FiLM-style residual refinement block."""
        super().__init__()
        self.condition = nn.Linear(hidden_dim, hidden_dim * 2)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, node_features: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        """Applies action-conditioned feature modulation and residual refinement."""

        gamma, beta = self.condition(action_context).chunk(2, dim=-1)
        conditioned = node_features * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        return node_features + self.block(conditioned)


class ShapeTransitionHead(nn.Module):
    """Predicts a one-step coarse next-state SDF conditioned on the chosen action."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        """Initializes action embedding and a FiLM decoder over graph tokens."""
        super().__init__()
        config.validate()
        self.sdf_channel_index = config.sdf_channel_index
        self.action_embedding = ActionEmbedding(config)
        hidden = config.hidden_dim
        self.decoder_input_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.decoder_blocks = nn.ModuleList(
            [_FiLMDecoderBlock(hidden) for _ in range(config.transition_decoder_layers)]
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.changed_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.point_delta_head = nn.Sequential(
            nn.Linear(hidden + config.point_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        state_points: torch.Tensor,
        macro_class_id: torch.Tensor,
        tool_choice_id: torch.Tensor,
        action_face_id: torch.Tensor,
        tool_kind_id: Optional[torch.Tensor] = None,
        axis_visible: Optional[torch.Tensor] = None,
        tool_diameter_norm: Optional[torch.Tensor] = None,
        tool_radius_norm: Optional[torch.Tensor] = None,
        tool_length_norm: Optional[torch.Tensor] = None,
        holder_diameter_norm: Optional[torch.Tensor] = None,
        holder_radius_norm: Optional[torch.Tensor] = None,
        holder_length_norm: Optional[torch.Tensor] = None,
        node_process_state: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts next node-level SDF from the current graph state and action."""

        action_outputs = self.action_embedding(
            node_embeddings=node_embeddings,
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            tool_kind_id=tool_kind_id,
            axis_visible=axis_visible,
            tool_diameter_norm=tool_diameter_norm,
            tool_radius_norm=tool_radius_norm,
            tool_length_norm=tool_length_norm,
            holder_diameter_norm=holder_diameter_norm,
            holder_radius_norm=holder_radius_norm,
            holder_length_norm=holder_length_norm,
            node_process_state=node_process_state,
            node_mask=node_mask,
        )
        target_embedding = action_outputs["target_node_embedding"][:, None, :].expand_as(node_embeddings)
        decoded_features = self.decoder_input_projection(
            torch.cat([node_embeddings, target_embedding], dim=-1)
        )
        for decoder_block in self.decoder_blocks:
            decoded_features = decoder_block(decoded_features, action_outputs["action_context"])

        current_node_sdf = state_points[..., self.sdf_channel_index].mean(dim=2)
        pred_delta_sdf = self.delta_head(decoded_features).squeeze(-1)
        pred_next_node_sdf = current_node_sdf + pred_delta_sdf
        pred_changed_logits = self.changed_head(decoded_features).squeeze(-1)
        points_per_node = state_points.shape[2]
        expanded_decoded = decoded_features[:, :, None, :].expand(-1, -1, points_per_node, -1)
        point_decoder_input = torch.cat([expanded_decoded, state_points], dim=-1)
        pred_delta_point_sdf = self.point_delta_head(point_decoder_input).squeeze(-1)
        current_point_sdf = state_points[..., self.sdf_channel_index]
        pred_next_point_sdf = current_point_sdf + pred_delta_point_sdf

        if node_mask is not None:
            pred_delta_sdf = pred_delta_sdf.masked_fill(node_mask, 0.0)
            pred_next_node_sdf = pred_next_node_sdf.masked_fill(node_mask, 0.0)
            pred_changed_logits = pred_changed_logits.masked_fill(node_mask, 0.0)
            current_node_sdf = current_node_sdf.masked_fill(node_mask, 0.0)
            expanded_node_mask = node_mask[:, :, None].expand_as(pred_next_point_sdf)
            pred_delta_point_sdf = pred_delta_point_sdf.masked_fill(expanded_node_mask, 0.0)
            pred_next_point_sdf = pred_next_point_sdf.masked_fill(expanded_node_mask, 0.0)
            current_point_sdf = current_point_sdf.masked_fill(expanded_node_mask, 0.0)

        return {
            "current_node_sdf": current_node_sdf,
            "pred_delta_sdf": pred_delta_sdf,
            "pred_next_node_sdf": pred_next_node_sdf,
            "pred_changed_logits": pred_changed_logits,
            "current_point_sdf": current_point_sdf,
            "pred_delta_point_sdf": pred_delta_point_sdf,
            "pred_next_point_sdf": pred_next_point_sdf,
            "action_context": action_outputs["action_context"],
        }
