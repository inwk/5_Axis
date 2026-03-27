"""Loss helpers for process-planning training."""

from typing import Optional

import torch
import torch.nn.functional as F


def masked_classification_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Computes cross-entropy with optional per-sample validity masking."""
    per_item = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    if valid_mask is None:
        valid_mask = (target != ignore_index)
    valid = valid_mask.float()
    return (per_item * valid).sum() / valid.sum().clamp_min(1.0)


def process_planning_loss(
    pred_macro_class_logits: torch.Tensor,
    target_macro_class: torch.Tensor,
    pred_target_node_logits: torch.Tensor,
    target_target_node: torch.Tensor,
    pred_tool_choice_logits: torch.Tensor,
    target_tool_choice: torch.Tensor,
    target_node_valid: Optional[torch.Tensor] = None,
    tool_choice_valid: Optional[torch.Tensor] = None,
    macro_class_weight: float = 1.0,
    target_node_weight: float = 1.0,
    tool_choice_weight: float = 1.0,
) -> torch.Tensor:
    """Combines classification losses for macro class, target node, and tool choice."""
    macro_class_loss = masked_classification_ce(pred_macro_class_logits, target_macro_class)
    target_node_loss = masked_classification_ce(
        pred_target_node_logits,
        target_target_node,
        valid_mask=target_node_valid,
        ignore_index=-1,
    )
    tool_choice_loss = masked_classification_ce(
        pred_tool_choice_logits,
        target_tool_choice,
        valid_mask=tool_choice_valid,
        ignore_index=-1,
    )
    return (
        macro_class_weight * macro_class_loss
        + target_node_weight * target_node_loss
        + tool_choice_weight * tool_choice_loss
    )

