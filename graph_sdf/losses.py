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


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes SmoothL1 loss with an optional validity mask."""

    per_item = F.smooth_l1_loss(prediction, target, reduction="none")
    if valid_mask is None:
        valid_mask = torch.ones_like(prediction, dtype=torch.bool)
    valid = valid_mask.float()
    return (per_item * valid).sum() / valid.sum().clamp_min(1.0)


def process_planning_loss(
    pred_macro_class_logits: torch.Tensor,
    target_macro_class: torch.Tensor,
    pred_action_face_logits: torch.Tensor,
    target_action_face: torch.Tensor,
    pred_tool_choice_logits: torch.Tensor,
    target_tool_choice: torch.Tensor,
    action_face_valid: Optional[torch.Tensor] = None,
    tool_choice_valid: Optional[torch.Tensor] = None,
    macro_class_weight: float = 1.0,
    action_face_weight: float = 1.0,
    tool_choice_weight: float = 1.0,
) -> torch.Tensor:
    """Combines classification losses for planner heads."""
    macro_class_loss = masked_classification_ce(pred_macro_class_logits, target_macro_class)
    action_face_loss = masked_classification_ce(
        pred_action_face_logits,
        target_action_face,
        valid_mask=action_face_valid,
        ignore_index=-1,
    )
    tool_choice_loss = masked_classification_ce(
        pred_tool_choice_logits,
        target_tool_choice,
        valid_mask=tool_choice_valid,
        ignore_index=-1,
    )
    total_loss = (
        macro_class_weight * macro_class_loss
        + action_face_weight * action_face_loss
        + tool_choice_weight * tool_choice_loss
    )
    return total_loss


def transition_reconstruction_loss(
    pred_next_node_sdf: torch.Tensor,
    target_next_node_sdf: torch.Tensor,
    current_node_sdf: torch.Tensor,
    pred_next_point_sdf: Optional[torch.Tensor] = None,
    target_next_point_sdf: Optional[torch.Tensor] = None,
    pred_changed_logits: Optional[torch.Tensor] = None,
    node_mask: Optional[torch.Tensor] = None,
    changed_face_threshold: float = 0.01,
    sdf_weight: float = 1.0,
    point_sdf_weight: float = 1.0,
    changed_mask_weight: float = 0.25,
) -> torch.Tensor:
    """Supervises one-step next-state reconstruction and optional changed-face logits."""

    valid_mask = torch.ones_like(target_next_node_sdf, dtype=torch.bool)
    if node_mask is not None:
        valid_mask = ~node_mask

    sdf_loss = masked_huber_loss(
        pred_next_node_sdf,
        target_next_node_sdf,
        valid_mask=valid_mask,
    )
    total_loss = sdf_weight * sdf_loss

    if pred_next_point_sdf is not None and target_next_point_sdf is not None and point_sdf_weight > 0.0:
        point_valid_mask = torch.ones_like(target_next_point_sdf, dtype=torch.bool)
        if node_mask is not None:
            point_valid_mask = ~node_mask[:, :, None].expand_as(target_next_point_sdf)
        point_sdf_loss = masked_huber_loss(
            pred_next_point_sdf,
            target_next_point_sdf,
            valid_mask=point_valid_mask,
        )
        total_loss = total_loss + point_sdf_weight * point_sdf_loss

    if pred_changed_logits is not None and changed_mask_weight > 0.0:
        target_changed = (target_next_node_sdf - current_node_sdf).abs() > changed_face_threshold
        per_item = F.binary_cross_entropy_with_logits(
            pred_changed_logits,
            target_changed.float(),
            reduction="none",
        )
        valid = valid_mask.float()
        changed_loss = (per_item * valid).sum() / valid.sum().clamp_min(1.0)
        total_loss = total_loss + changed_mask_weight * changed_loss

    return total_loss
