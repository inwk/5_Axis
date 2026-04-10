"""NX-oriented process skeleton planner head for Graph-SDF planning."""

from typing import Optional

import torch
import torch.nn as nn

from .config import GraphSdfModelConfig


def _masked_mean_pool(node_features: torch.Tensor, node_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Computes a mask-aware mean pool across graph nodes."""

    if node_mask is None:
        return node_features.mean(dim=1)

    valid = (~node_mask).float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (node_features * valid).sum(dim=1) / denom


class ProcessPlannerHead(nn.Module):
    """Predicts an executable process skeleton from the current graph state."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        """Initializes global planning heads and a node-selection head."""
        super().__init__()
        hidden = config.hidden_dim

        self.global_trunk = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.global_process_projection = (
            nn.Linear(config.global_process_feature_dim, hidden)
            if config.global_process_feature_dim > 0
            else None
        )
        self.macro_class_head = nn.Linear(hidden, config.macro_class_count)
        self.tool_choice_head = nn.Linear(hidden, config.tool_choice_count)

        self.node_selector = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        global_process_state: Optional[torch.Tensor] = None,
        action_face_mask: Optional[torch.Tensor] = None,
        macro_class_mask: Optional[torch.Tensor] = None,
        tool_choice_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Produces macro class, action-face, and tool-choice logits."""

        pooled_state = _masked_mean_pool(state_embedding, node_mask)
        global_hidden = self.global_trunk(pooled_state)
        if self.global_process_projection is not None:
            if global_process_state is None:
                global_process_state = global_hidden.new_zeros(
                    global_hidden.shape[0],
                    self.global_process_projection.in_features,
                )
            global_hidden = global_hidden + self.global_process_projection(global_process_state)

        node_count = state_embedding.shape[1]
        repeated_global = global_hidden[:, None, :].expand(-1, node_count, -1)
        selector_input = torch.cat([state_embedding, repeated_global], dim=-1)
        action_face_logits = self.node_selector(selector_input).squeeze(-1)

        if node_mask is not None:
            action_face_logits = action_face_logits.masked_fill(node_mask, -1e9)
        if action_face_mask is not None:
            action_face_logits = action_face_logits.masked_fill(action_face_mask, -1e9)

        macro_class_logits = self.macro_class_head(global_hidden)
        tool_choice_logits = self.tool_choice_head(global_hidden)
        if macro_class_mask is not None:
            macro_class_logits = macro_class_logits.masked_fill(macro_class_mask, -1e9)
        if tool_choice_mask is not None:
            tool_choice_logits = tool_choice_logits.masked_fill(tool_choice_mask, -1e9)

        return {
            "macro_class_logits": macro_class_logits,
            "tool_choice_logits": tool_choice_logits,
            "action_face_logits": action_face_logits,
            "global_feature": global_hidden,
        }
