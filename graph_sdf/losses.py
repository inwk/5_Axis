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


def occupancy_bce_loss(
    occ_logits: torch.Tensor,
    occ_labels: torch.Tensor,
    occ_valid_mask: Optional[torch.Tensor] = None,
    pos_weight_factor: float = 1.0,
) -> torch.Tensor:
    """Binary cross-entropy loss for volumetric occupancy prediction.

    .. deprecated::
        Use :func:`octree_bce_loss` for depth-weighted occupancy loss.

    Args:
        occ_logits:       [B, Q] raw logits from OccupancyDecoder.
        occ_labels:       [B, Q] float32 labels — 1.0 = inside material after
                          the operation, 0.0 = empty / removed.
        occ_valid_mask:   Optional [B, Q] bool; True = include this query point.
                          When None, all points are used.
        pos_weight_factor: Scalar multiplier for the positive-class BCE weight.
                          Increase (e.g. 2–5) when empty-space samples outnumber
                          inside-material samples.

    Returns:
        Scalar loss tensor.
    """
    if occ_valid_mask is None:
        occ_valid_mask = torch.ones_like(occ_labels, dtype=torch.bool)

    pos_weight = occ_labels.new_tensor([pos_weight_factor])
    per_point = F.binary_cross_entropy_with_logits(
        occ_logits,
        occ_labels,
        pos_weight=pos_weight,
        reduction="none",
    )  # [B, Q]
    valid = occ_valid_mask.float()
    return (per_point * valid).sum() / valid.sum().clamp_min(1.0)


def octree_bce_loss(
    occ_logits: torch.Tensor,
    occ_labels: torch.Tensor,
    octree_depths: torch.Tensor,
    pos_weight_factor: float = 1.0,
    depth_weight_base: float = 2.0,
) -> torch.Tensor:
    """Depth-weighted binary cross-entropy for adaptive octree occupancy prediction.

    Fine-depth cells (at the material surface boundary) carry more information
    than coarse-depth cells (bulk interior/exterior).  Each cell is weighted by
    ``depth_weight_base ** depth`` before the loss is normalised, so deeper cells
    receive proportionally higher gradient.

    Args:
        occ_logits:        [B, K] raw BCE logits from OctreeDecoder.
        occ_labels:        [B, K] float32 — 1.0 = inside material after op,
                           0.0 = empty / removed.
        octree_depths:     [B, K] int64 — octree level of each cell.
        pos_weight_factor: Scalar multiplier for the positive-class BCE weight.
                           Increase (e.g. 2–5) to compensate for class imbalance.
        depth_weight_base: Base for exponential depth weighting.  ``2.0`` means
                           each additional octree level doubles the cell's weight.
                           Use ``1.0`` to disable depth weighting (uniform).

    Returns:
        Scalar weighted loss tensor.
    """
    pos_weight = occ_labels.new_tensor([pos_weight_factor])
    per_cell = F.binary_cross_entropy_with_logits(
        occ_logits,
        occ_labels,
        pos_weight=pos_weight,
        reduction="none",
    )  # [B, K]

    # Depth weights: 2^depth (or base^depth) normalised per sample so the total
    # weight per sample sums to K (same effective batch normalisation as mean).
    depth_w = depth_weight_base ** octree_depths.float()   # [B, K]
    depth_w = depth_w / depth_w.mean(dim=1, keepdim=True).clamp_min(1e-6)  # re-scale to mean=1

    return (per_cell * depth_w).mean()


def monotonicity_occupancy_loss(
    occ_logits: torch.Tensor,
    occ_labels_before: torch.Tensor,
) -> torch.Tensor:
    """Penalises predicting occupied material in cells that were already empty.

    Machining is a strictly subtractive process: once a cell is empty (air),
    no subsequent operation can fill it back with material.

    Violation = max(0, sigmoid(logit) - before_label)

    When before_label = 0.0 (already air):
        penalty = sigmoid(logit)         ← any occupancy prediction is penalised
    When before_label = 1.0 (currently solid):
        penalty = max(0, pred - 1.0) = 0 ← no constraint; material may stay or be cut

    Args:
        occ_logits:        [B, K] raw logits for the *after* state.
        occ_labels_before: [B, K] float32 occupancy labels for the *before* state
                           (1.0 = inside material, 0.0 = outside/air).

    Returns:
        Scalar mean violation.
    """
    occ_pred = torch.sigmoid(occ_logits)                       # [B, K]
    violation = F.relu(occ_pred - occ_labels_before)           # [B, K]
    return violation.mean()
