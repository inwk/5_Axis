"""Training-step utilities for planner and octree transition learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import (
    monotonicity_occupancy_loss,
    occupancy_bce_loss,
    octree_bce_loss,
    process_planning_loss,
)
from .model import GraphSdfPlanningModel


def _loss_value(loss: torch.Tensor) -> float:
    """Returns a detached Python scalar for optional loss breakdown logging."""
    return float(loss.detach().item())


# ─────────────────────────────────────────────────────────────────────────────
# EMA-based loss scale balancer
# ─────────────────────────────────────────────────────────────────────────────

class EMALossBalancer:
    """Tracks the exponential moving average of each named loss to produce
    per-loss normalisation scales.

    Each loss is divided by its EMA before being summed, so all losses
    contribute with roughly equal gradient magnitude regardless of their
    absolute scale.  The user-supplied weights then set the desired *ratio*
    between terms rather than fighting against raw magnitude differences.

    Example::

        balancer = EMALossBalancer(momentum=0.99)
        # inside train loop:
        sp = balancer.scale("planner", float(planner_loss))
        so = balancer.scale("octree",  float(octree_loss))
        loss = sp * planner_loss + octree_weight * so * octree_loss

    Args:
        momentum:  EMA decay (0.99 → slow-adapting, 0.9 → fast-adapting).
        eps:       Numerical floor to avoid division by zero.
    """

    def __init__(self, momentum: float = 0.99, eps: float = 1e-6) -> None:
        self._ema: dict[str, float] = {}
        self._momentum = float(momentum)
        self._eps = float(eps)

    def scale(self, name: str, value: float) -> float:
        """Updates the EMA for *name* and returns ``1 / EMA(name)``."""
        value = max(abs(float(value)), self._eps)
        if name not in self._ema:
            self._ema[name] = value
        else:
            self._ema[name] = (
                self._momentum * self._ema[name]
                + (1.0 - self._momentum) * value
            )
        return 1.0 / (self._ema[name] + self._eps)

    def ema_values(self) -> dict[str, float]:
        """Returns a snapshot of all current EMA values (for logging)."""
        return dict(self._ema)


def _to_device(optional_tensor, device: torch.device):
    """Moves an optional tensor to the requested device."""
    if optional_tensor is None:
        return None
    return optional_tensor.to(device)


def _collect_common_inputs(batch: dict, device: torch.device) -> dict:
    """Moves shared model inputs and targets to device."""
    state_points = batch["state_points"].to(device)
    target_macro_class = batch["macro_class_id"].to(device)
    target_tool_choice = batch["tool_choice_id"].to(device)
    target_action_face = _to_device(batch.get("action_face_id"), device)
    if target_action_face is None:
        target_action_face = torch.full_like(target_macro_class, -1)

    return {
        "state_points": state_points,
        "node_process_state": _to_device(batch.get("node_process_state"), device),
        "node_centrality": _to_device(batch.get("node_centrality"), device),
        "spatial_pos": _to_device(batch.get("spatial_pos"), device),
        "face_area": _to_device(batch.get("face_area"), device),
        "node_face_type": _to_device(batch.get("node_face_type"), device),
        "axis_visible": _to_device(batch.get("axis_visible"), device),
        "axis_dir": _to_device(batch.get("axis_dir"), device),
        "tool_radius_norm": _to_device(batch.get("tool_radius_norm"), device),
        "tool_length_norm": _to_device(batch.get("tool_length_norm"), device),
        "holder_diameter_norm": _to_device(batch.get("holder_diameter_norm"), device),
        "holder_radius_norm": _to_device(batch.get("holder_radius_norm"), device),
        "holder_length_norm": _to_device(batch.get("holder_length_norm"), device),
        "global_process_state": _to_device(batch.get("global_process_state"), device),
        "target_macro_class": target_macro_class,
        "target_tool_choice": target_tool_choice,
        "target_action_face": target_action_face,
        "node_mask": _to_device(batch.get("node_mask"), device),
        "point_mask": _to_device(batch.get("point_mask"), device),
        "action_face_mask": _to_device(batch.get("action_face_mask"), device),
        "macro_class_mask": _to_device(batch.get("macro_class_mask"), device),
        "tool_choice_mask": _to_device(batch.get("tool_choice_mask"), device),
        "action_face_valid": _to_device(batch.get("action_face_valid"), device),
        "tool_choice_valid": _to_device(batch.get("tool_choice_valid"), device),
        "is_chosen": _to_device(batch.get("is_chosen"), device),
    }


def _encode_state_only(model: GraphSdfPlanningModel, inputs: dict) -> torch.Tensor:
    """Runs only the state encoder, bypassing planner heads entirely."""
    return model.encode_state(
        state_points=inputs["state_points"],
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_face_type=inputs["node_face_type"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
    )


def _run_planner(model: GraphSdfPlanningModel, inputs: dict) -> dict:
    """Runs the planner/encoder path once and returns logits plus state embedding."""
    return model.forward_process_planner(
        state_points=inputs["state_points"],
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_face_type=inputs["node_face_type"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
        axis_visible=inputs["axis_visible"],
        global_process_state=inputs["global_process_state"],
        action_face_mask=inputs["action_face_mask"],
        macro_class_mask=inputs["macro_class_mask"],
        tool_choice_mask=inputs["tool_choice_mask"],
    )


def _compute_planner_loss(
    outputs: dict,
    inputs: dict,
    macro_class_loss_weight: float,
    action_face_loss_weight: float,
    tool_choice_loss_weight: float,
) -> torch.Tensor:
    """Computes planner CE loss, using only chosen rows when is_chosen exists."""
    is_chosen = inputs["is_chosen"]
    state_points = inputs["state_points"]
    target_macro_class = inputs["target_macro_class"]
    target_tool_choice = inputs["target_tool_choice"]
    target_action_face = inputs["target_action_face"]
    action_face_valid = inputs["action_face_valid"]
    tool_choice_valid = inputs["tool_choice_valid"]

    if is_chosen is not None:
        chosen = is_chosen.bool()
        if not bool(chosen.any()):
            return torch.zeros((), device=state_points.device, dtype=state_points.dtype)
        idx = chosen.nonzero(as_tuple=True)[0]
        return process_planning_loss(
            pred_macro_class_logits=outputs["macro_class_logits"][idx],
            target_macro_class=target_macro_class[idx],
            pred_action_face_logits=outputs["action_face_logits"][idx],
            target_action_face=target_action_face[idx],
            pred_tool_choice_logits=outputs["tool_choice_logits"][idx],
            target_tool_choice=target_tool_choice[idx],
            action_face_valid=action_face_valid[idx] if action_face_valid is not None else None,
            tool_choice_valid=tool_choice_valid[idx] if tool_choice_valid is not None else None,
            macro_class_weight=macro_class_loss_weight,
            action_face_weight=action_face_loss_weight,
            tool_choice_weight=tool_choice_loss_weight,
        )

    return process_planning_loss(
        pred_macro_class_logits=outputs["macro_class_logits"],
        target_macro_class=target_macro_class,
        pred_action_face_logits=outputs["action_face_logits"],
        target_action_face=target_action_face,
        pred_tool_choice_logits=outputs["tool_choice_logits"],
        target_tool_choice=target_tool_choice,
        action_face_valid=action_face_valid,
        tool_choice_valid=tool_choice_valid,
        macro_class_weight=macro_class_loss_weight,
        action_face_weight=action_face_loss_weight,
        tool_choice_weight=tool_choice_loss_weight,
    )


def _compute_octree_loss(
    model: GraphSdfPlanningModel,
    batch: dict,
    inputs: dict,
    outputs: dict,
    device: torch.device,
    octree_loss_weight: float,
    octree_pos_weight_factor: float,
    octree_depth_weight_base: float = 2.0,
    monotonicity_weight: float = 0.1,
    fill_fraction_weight: float = 0.5,
    removed_fraction_weight: float = 2.0,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
    occupancy_loss_weight: float = 0.1,
    affected_face_loss_weight: float = 1.0,
    affected_delta_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Computes octree occupancy loss (depth-weighted BCE + optional monotonicity).

    When ``octree_occ_labels_before`` is available, it is also passed to the
    decoder as current-state occupancy at each query cell.

    Adds two loss terms when the data is available:
    1. ``octree_bce_loss``: depth-weighted BCE — finer octree cells get higher
       gradient weight because they carry more surface detail.
    2. ``monotonicity_occupancy_loss`` (scaled by *monotonicity_weight*): soft
       penalty that discourages predicting occupied material in cells that were
       already empty *before* the operation (material cannot be added back).
       Requires ``octree_occ_labels_before`` in the batch.
    """
    if octree_loss_weight <= 0.0:
        return torch.zeros((), device=device, dtype=inputs["state_points"].dtype)
    if "sdf_query_points" in batch and "sdf_tsdf_after" in batch:
        return _compute_transition_only_sdf_query_loss(
            model=model,
            batch=batch,
            inputs=inputs,
            device=device,
            tsdf_loss_weight=tsdf_loss_weight,
            delta_tsdf_loss_weight=delta_tsdf_loss_weight,
            tsdf_monotonicity_weight=tsdf_monotonicity_weight,
            tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
            affected_face_loss_weight=affected_face_loss_weight,
            affected_delta_loss_weight=affected_delta_loss_weight,
            state_embedding=outputs.get("state_embedding"),
        )
    if model.octree_decoder is None:
        return torch.zeros((), device=device, dtype=inputs["state_points"].dtype)
    if "octree_centers" not in batch or "octree_depths" not in batch or "octree_occ_labels" not in batch:
        return torch.zeros((), device=device, dtype=inputs["state_points"].dtype)

    octree_centers = batch["octree_centers"].to(device)
    octree_depths = batch["octree_depths"].to(device)
    octree_labels = batch["octree_occ_labels"].to(device)
    octree_occ_before = _octree_before_decoder_input(batch, device)

    octree_outputs = model.forward_octree(
        state_points=inputs["state_points"],
        macro_class_id=inputs["target_macro_class"],
        tool_choice_id=inputs["target_tool_choice"],
        action_face_id=inputs["target_action_face"].clamp_min(0),
        octree_centers=octree_centers,
        octree_depths=octree_depths,
        octree_occ_before=octree_occ_before,
        axis_visible=inputs["axis_visible"],
        axis_dir=inputs["axis_dir"],
        tool_radius_norm=inputs["tool_radius_norm"],
        tool_length_norm=inputs["tool_length_norm"],
        holder_diameter_norm=inputs["holder_diameter_norm"],
        holder_radius_norm=inputs["holder_radius_norm"],
        holder_length_norm=inputs["holder_length_norm"],
        sdf_axis_clearance_before=batch["sdf_axis_clearance_before"].to(device) if "sdf_axis_clearance_before" in batch else None,
        sdf_axis_blocked_before=batch["sdf_axis_blocked_before"].to(device) if "sdf_axis_blocked_before" in batch else None,
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_face_type=inputs["node_face_type"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
        state_embedding=outputs.get("state_embedding"),
    )
    occ_logits = octree_outputs["occ_logits"]

    # ── 1. Depth-weighted BCE ─────────────────────────────────────────────
    bce = octree_bce_loss(
        occ_logits=occ_logits,
        occ_labels=octree_labels,
        octree_depths=octree_depths,
        pos_weight_factor=octree_pos_weight_factor,
        depth_weight_base=octree_depth_weight_base,
    )

    # ── 2. Monotonicity penalty (soft: already-empty cells must stay empty) ─
    mono = torch.zeros((), device=device, dtype=occ_logits.dtype)
    if monotonicity_weight > 0.0 and "octree_occ_labels_before" in batch:
        labels_before = batch["octree_occ_labels_before"].to(device)
        before_valid = (
            batch["octree_occ_labels_before_valid"].to(device)
            if "octree_occ_labels_before_valid" in batch
            else None
        )
        mono = monotonicity_occupancy_loss(
            occ_logits=occ_logits,
            occ_labels_before=labels_before,
            valid_mask=before_valid,
        )

    fill = _compute_octree_fill_fraction_loss(
        occ_logits=occ_logits,
        batch=batch,
        device=device,
        fill_fraction_weight=fill_fraction_weight,
        removed_fraction_weight=removed_fraction_weight,
    )
    tsdf = _compute_octree_tsdf_loss(
        octree_outputs=octree_outputs,
        batch=batch,
        device=device,
        tsdf_loss_weight=tsdf_loss_weight,
        delta_tsdf_loss_weight=delta_tsdf_loss_weight,
        tsdf_monotonicity_weight=tsdf_monotonicity_weight,
        tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
    )

    return float(occupancy_loss_weight) * bce + monotonicity_weight * mono + fill + tsdf


def _octree_before_decoder_input(batch: dict, device: torch.device) -> torch.Tensor | None:
    """Returns per-query condition: before TSDF, target TSDF, before fill, known."""
    before_fill = None
    if "octree_fill_before" in batch:
        before_fill = batch["octree_fill_before"].to(device).float().clamp(0.0, 1.0)
    elif "octree_occ_labels_before" in batch:
        labels_before = batch["octree_occ_labels_before"].to(device).float().clamp(0.0, 1.0)
        before_valid = (
            batch["octree_occ_labels_before_valid"].to(device)
            if "octree_occ_labels_before_valid" in batch
            else None
        )
        if before_valid is None or bool(before_valid.sum().item() >= before_valid.numel()):
            before_fill = labels_before

    before_tsdf = (
        batch["octree_tsdf_before"].to(device).float().clamp(-1.0, 1.0)
        if "octree_tsdf_before" in batch
        else None
    )
    target_tsdf = (
        batch["octree_target_tsdf"].to(device).float().clamp(-1.0, 1.0)
        if "octree_target_tsdf" in batch
        else None
    )
    if before_fill is None and before_tsdf is None and target_tsdf is None:
        return None

    ref = before_tsdf if before_tsdf is not None else target_tsdf if target_tsdf is not None else before_fill
    zeros = torch.zeros_like(ref)
    ones = torch.ones_like(ref)
    return torch.stack([
        before_tsdf if before_tsdf is not None else zeros,
        target_tsdf if target_tsdf is not None else zeros,
        before_fill if before_fill is not None else zeros,
        ones,
    ], dim=-1)


def _compute_octree_tsdf_loss(
    octree_outputs: dict,
    batch: dict,
    device: torch.device,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
) -> torch.Tensor:
    """Primary TSDF supervision for 3D transition learning."""
    loss = torch.zeros((), device=device)
    if "tsdf" not in octree_outputs or "octree_tsdf_after" not in batch:
        return loss

    pred_after = octree_outputs["tsdf"]
    target_after = batch["octree_tsdf_after"].to(device).float().clamp(-1.0, 1.0)
    if tsdf_loss_weight > 0.0:
        loss = loss + float(tsdf_loss_weight) * F.smooth_l1_loss(
            pred_after,
            target_after,
            reduction="mean",
        )

    if "octree_tsdf_before" in batch:
        before = batch["octree_tsdf_before"].to(device).float().clamp(-1.0, 1.0)
        target_delta = (
            batch["octree_delta_tsdf"].to(device).float().clamp(-2.0, 2.0)
            if "octree_delta_tsdf" in batch
            else target_after - before
        )
        if delta_tsdf_loss_weight > 0.0:
            pred_delta = pred_after - before
            loss = loss + float(delta_tsdf_loss_weight) * F.smooth_l1_loss(
                pred_delta,
                target_delta,
                reduction="mean",
            )
        if tsdf_monotonicity_weight > 0.0:
            empty_mask = before > float(tsdf_monotonicity_empty_margin)
            if bool(empty_mask.any().item()):
                loss = loss + float(tsdf_monotonicity_weight) * F.relu(before - pred_after)[empty_mask].mean()
    return loss


def _compute_octree_fill_fraction_loss(
    occ_logits: torch.Tensor,
    batch: dict,
    device: torch.device,
    fill_fraction_weight: float = 0.5,
    removed_fraction_weight: float = 2.0,
) -> torch.Tensor:
    """Adds dense fill-fraction supervision when those parquet targets exist."""
    loss = torch.zeros((), device=device, dtype=occ_logits.dtype)
    if fill_fraction_weight <= 0.0 and removed_fraction_weight <= 0.0:
        return loss

    pred_after_fill = torch.sigmoid(occ_logits)
    if fill_fraction_weight > 0.0 and "octree_fill_after" in batch:
        target_after_fill = batch["octree_fill_after"].to(device).float().clamp(0.0, 1.0)
        loss = loss + float(fill_fraction_weight) * F.smooth_l1_loss(
            pred_after_fill,
            target_after_fill,
            reduction="mean",
        )

    if removed_fraction_weight > 0.0 and "octree_removed_fraction" in batch:
        before_fill = None
        if "octree_fill_before" in batch:
            before_fill = batch["octree_fill_before"].to(device).float().clamp(0.0, 1.0)
        elif "octree_occ_labels_before" in batch:
            before_fill = batch["octree_occ_labels_before"].to(device).float().clamp(0.0, 1.0)
        if before_fill is not None:
            target_removed = batch["octree_removed_fraction"].to(device).float().clamp(0.0, 1.0)
            pred_removed = F.relu(before_fill - pred_after_fill)
            loss = loss + float(removed_fraction_weight) * F.smooth_l1_loss(
                pred_removed,
                target_removed,
                reduction="mean",
            )
    return loss


def _sdf_query_state_input(
    batch: dict,
    device: torch.device,
    use_target_tsdf_input: bool = True,
) -> torch.Tensor | None:
    """Builds [before_tsdf, target_tsdf, removable_prior, known]."""
    before = (
        batch["sdf_tsdf_before"].to(device).float().clamp(-1.0, 1.0)
        if "sdf_tsdf_before" in batch
        else None
    )
    target = (
        batch["sdf_target_tsdf"].to(device).float().clamp(-1.0, 1.0)
        if bool(use_target_tsdf_input) and "sdf_target_tsdf" in batch
        else None
    )
    if before is None and target is None:
        return None
    ref = before if before is not None else target
    zeros = torch.zeros_like(ref)
    ones = torch.ones_like(ref)
    removable = (
        (target - before).clamp(-2.0, 2.0)
        if before is not None and target is not None
        else zeros
    )
    return torch.stack([
        before if before is not None else zeros,
        target if target is not None else zeros,
        removable,
        ones,
    ], dim=-1)


def _compute_affected_face_loss(
    model: GraphSdfPlanningModel,
    batch: dict,
    inputs: dict,
    device: torch.device,
    state_embedding: torch.Tensor,
    affected_face_loss_weight: float = 1.0,
    affected_delta_loss_weight: float = 0.0,
    return_components: bool = False,
) -> torch.Tensor:
    """Auxiliary loss: predict which faces change under the current action."""
    components: dict[str, float] = {}
    if affected_face_loss_weight <= 0.0 and affected_delta_loss_weight <= 0.0:
        loss = torch.zeros((), device=device, dtype=state_embedding.dtype)
        return (loss, components) if return_components else loss
    if "affected_face_mask" not in batch:
        loss = torch.zeros((), device=device, dtype=state_embedding.dtype)
        return (loss, components) if return_components else loss

    out = model.forward_affected_faces(
        state_points=inputs["state_points"],
        macro_class_id=inputs["target_macro_class"],
        tool_choice_id=inputs["target_tool_choice"],
        action_face_id=inputs["target_action_face"].clamp_min(0),
        axis_visible=inputs["axis_visible"],
        axis_dir=inputs["axis_dir"],
        tool_radius_norm=inputs["tool_radius_norm"],
        tool_length_norm=inputs["tool_length_norm"],
        holder_diameter_norm=inputs["holder_diameter_norm"],
        holder_radius_norm=inputs["holder_radius_norm"],
        holder_length_norm=inputs["holder_length_norm"],
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_face_type=inputs["node_face_type"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
        state_embedding=state_embedding,
    )
    logits = out["affected_logits"]
    target = batch["affected_face_mask"].to(device).float().clamp(0.0, 1.0)
    valid = ~inputs["node_mask"] if inputs["node_mask"] is not None else torch.ones_like(target, dtype=torch.bool)

    loss = torch.zeros((), device=device, dtype=logits.dtype)
    if affected_face_loss_weight > 0.0:
        valid_target = target[valid]
        pos = valid_target.sum()
        neg = valid_target.numel() - pos
        pos_weight = torch.clamp(neg / pos.clamp_min(1.0), min=1.0, max=50.0)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
        face_loss = float(affected_face_loss_weight) * (
            bce[valid].sum() / valid.float().sum().clamp_min(1.0)
        )
        loss = loss + face_loss
        components["affected_face"] = _loss_value(face_loss)

    if affected_delta_loss_weight > 0.0 and "affected_face_delta" in batch:
        positive = valid & (target > 0.5)
        if bool(positive.any().item()):
            target_delta = batch["affected_face_delta"].to(device).float().clamp_min(0.0)
            delta_loss = float(affected_delta_loss_weight) * F.smooth_l1_loss(
                out["affected_delta"][positive],
                target_delta[positive],
                reduction="mean",
            )
            loss = loss + delta_loss
            components["affected_delta"] = _loss_value(delta_loss)
    return (loss, components) if return_components else loss


def _compute_transition_only_sdf_query_loss(
    model: GraphSdfPlanningModel,
    batch: dict,
    inputs: dict,
    device: torch.device,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    changed_tsdf_loss_weight: float = 0.0,
    changed_delta_tsdf_loss_weight: float = 0.0,
    tsdf_change_eps: float = 1e-3,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
    affected_face_loss_weight: float = 1.0,
    affected_delta_loss_weight: float = 0.0,
    use_target_tsdf_input: bool = True,
    state_embedding: torch.Tensor | None = None,
    return_components: bool = False,
) -> torch.Tensor:
    """Computes SDF-only transition loss with GT action labels."""
    if getattr(model, "sdf_query_decoder", None) is None:
        raise RuntimeError("SDF-only training requires model.sdf_query_decoder to be enabled.")
    if "sdf_query_points" not in batch or "sdf_tsdf_after" not in batch:
        raise RuntimeError("SDF-only training requires 'sdf_query_points' and 'sdf_tsdf_after'.")

    if state_embedding is None:
        state_embedding = _encode_state_only(model, inputs)
    outputs = model.forward_sdf_query(
        state_points=inputs["state_points"],
        macro_class_id=inputs["target_macro_class"],
        tool_choice_id=inputs["target_tool_choice"],
        action_face_id=inputs["target_action_face"].clamp_min(0),
        sdf_query_points=batch["sdf_query_points"].to(device),
        sdf_query_state=_sdf_query_state_input(
            batch,
            device,
            use_target_tsdf_input=use_target_tsdf_input,
        ),
        axis_visible=inputs["axis_visible"],
        axis_dir=inputs["axis_dir"],
        tool_radius_norm=inputs["tool_radius_norm"],
        tool_length_norm=inputs["tool_length_norm"],
        holder_diameter_norm=inputs["holder_diameter_norm"],
        holder_radius_norm=inputs["holder_radius_norm"],
        holder_length_norm=inputs["holder_length_norm"],
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_face_type=inputs["node_face_type"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
        state_embedding=state_embedding,
    )
    pred_after = outputs["sdf_tsdf"]
    target_after = batch["sdf_tsdf_after"].to(device).float().clamp(-1.0, 1.0)
    components: dict[str, float] = {}
    loss = torch.zeros((), device=device, dtype=pred_after.dtype)
    if tsdf_loss_weight > 0.0:
        tsdf_loss = float(tsdf_loss_weight) * F.smooth_l1_loss(
            pred_after,
            target_after,
            reduction="mean",
        )
        loss = loss + tsdf_loss
        components["tsdf"] = _loss_value(tsdf_loss)

    if "sdf_tsdf_before" in batch:
        before = batch["sdf_tsdf_before"].to(device).float().clamp(-1.0, 1.0)
        target_delta = (
            batch["sdf_delta_tsdf"].to(device).float().clamp(-2.0, 2.0)
            if "sdf_delta_tsdf" in batch
            else target_after - before
        )
        if delta_tsdf_loss_weight > 0.0:
            delta_loss = float(delta_tsdf_loss_weight) * F.smooth_l1_loss(
                pred_after - before,
                target_delta,
                reduction="mean",
            )
            loss = loss + delta_loss
            components["delta_tsdf"] = _loss_value(delta_loss)
        changed_mask = target_delta.abs() > float(tsdf_change_eps)
        if bool(changed_mask.any().item()):
            if changed_tsdf_loss_weight > 0.0:
                changed_tsdf_loss = float(changed_tsdf_loss_weight) * F.smooth_l1_loss(
                    pred_after[changed_mask],
                    target_after[changed_mask],
                    reduction="mean",
                )
                loss = loss + changed_tsdf_loss
                components["changed_tsdf"] = _loss_value(changed_tsdf_loss)
            if changed_delta_tsdf_loss_weight > 0.0:
                changed_delta_loss = float(changed_delta_tsdf_loss_weight) * F.smooth_l1_loss(
                    (pred_after - before)[changed_mask],
                    target_delta[changed_mask],
                    reduction="mean",
                )
                loss = loss + changed_delta_loss
                components["changed_delta_tsdf"] = _loss_value(changed_delta_loss)
        if tsdf_monotonicity_weight > 0.0:
            empty_mask = before > float(tsdf_monotonicity_empty_margin)
            if bool(empty_mask.any().item()):
                mono_loss = float(tsdf_monotonicity_weight) * F.relu(before - pred_after)[empty_mask].mean()
                loss = loss + mono_loss
                components["tsdf_mono"] = _loss_value(mono_loss)
    affected_result = _compute_affected_face_loss(
        model=model,
        batch=batch,
        inputs=inputs,
        device=device,
        state_embedding=state_embedding,
        affected_face_loss_weight=affected_face_loss_weight,
        affected_delta_loss_weight=affected_delta_loss_weight,
        return_components=return_components,
    )
    if return_components:
        affected_loss, affected_components = affected_result
        components.update(affected_components)
    else:
        affected_loss = affected_result
    loss = loss + affected_loss
    return (loss, components) if return_components else loss


def _compute_transition_only_octree_loss(
    model: GraphSdfPlanningModel,
    batch: dict,
    inputs: dict,
    device: torch.device,
    octree_pos_weight_factor: float,
    octree_depth_weight_base: float = 2.0,
    monotonicity_weight: float = 0.1,
    fill_fraction_weight: float = 0.5,
    removed_fraction_weight: float = 2.0,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
    occupancy_loss_weight: float = 0.1,
) -> torch.Tensor:
    """Computes octree occupancy loss with action labels as inputs and no planner path.

    Uses depth-weighted BCE + optional monotonicity penalty.  When available,
    before-state occupancy is also used as a decoder query input.
    """
    if model.octree_decoder is None:
        raise RuntimeError("Transition-only training requires model.octree_decoder to be enabled.")
    if "octree_centers" not in batch or "octree_depths" not in batch or "octree_occ_labels" not in batch:
        raise RuntimeError(
            "Transition-only training requires batch keys: "
            "'octree_centers', 'octree_depths', and 'octree_occ_labels'."
        )

    octree_depths = batch["octree_depths"].to(device)
    octree_occ_before = _octree_before_decoder_input(batch, device)
    state_embedding = _encode_state_only(model, inputs)
    octree_outputs = model.forward_octree(
        state_points=inputs["state_points"],
        macro_class_id=inputs["target_macro_class"],
        tool_choice_id=inputs["target_tool_choice"],
        action_face_id=inputs["target_action_face"].clamp_min(0),
        octree_centers=batch["octree_centers"].to(device),
        octree_depths=octree_depths,
        octree_occ_before=octree_occ_before,
        axis_visible=inputs["axis_visible"],
        axis_dir=inputs["axis_dir"],
        tool_radius_norm=inputs["tool_radius_norm"],
        tool_length_norm=inputs["tool_length_norm"],
        holder_diameter_norm=inputs["holder_diameter_norm"],
        holder_radius_norm=inputs["holder_radius_norm"],
        holder_length_norm=inputs["holder_length_norm"],
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_face_type=inputs["node_face_type"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
        state_embedding=state_embedding,
    )
    occ_logits = octree_outputs["occ_logits"]

    # ── 1. Depth-weighted BCE ─────────────────────────────────────────────
    bce = octree_bce_loss(
        occ_logits=occ_logits,
        occ_labels=batch["octree_occ_labels"].to(device),
        octree_depths=octree_depths,
        pos_weight_factor=octree_pos_weight_factor,
        depth_weight_base=octree_depth_weight_base,
    )

    # ── 2. Monotonicity penalty ───────────────────────────────────────────
    mono = torch.zeros((), device=device, dtype=occ_logits.dtype)
    if monotonicity_weight > 0.0 and "octree_occ_labels_before" in batch:
        before_valid = (
            batch["octree_occ_labels_before_valid"].to(device)
            if "octree_occ_labels_before_valid" in batch
            else None
        )
        mono = monotonicity_occupancy_loss(
            occ_logits=occ_logits,
            occ_labels_before=batch["octree_occ_labels_before"].to(device),
            valid_mask=before_valid,
        )

    fill = _compute_octree_fill_fraction_loss(
        occ_logits=occ_logits,
        batch=batch,
        device=device,
        fill_fraction_weight=fill_fraction_weight,
        removed_fraction_weight=removed_fraction_weight,
    )
    tsdf = _compute_octree_tsdf_loss(
        octree_outputs=octree_outputs,
        batch=batch,
        device=device,
        tsdf_loss_weight=tsdf_loss_weight,
        delta_tsdf_loss_weight=delta_tsdf_loss_weight,
        tsdf_monotonicity_weight=tsdf_monotonicity_weight,
        tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
    )

    return float(occupancy_loss_weight) * bce + monotonicity_weight * mono + fill + tsdf


def planner_train_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    macro_class_loss_weight: float = 1.0,
    action_face_loss_weight: float = 1.0,
    tool_choice_loss_weight: float = 1.0,
    transition_loss_weight: float = 0.0,
    point_sdf_loss_weight: float = 0.0,
    changed_mask_loss_weight: float = 0.0,
    octree_loss_weight: float = 1.0,
    octree_pos_weight_factor: float = 2.0,
    octree_depth_weight_base: float = 2.0,
    monotonicity_weight: float = 0.1,
    fill_fraction_weight: float = 0.5,
    removed_fraction_weight: float = 2.0,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
    occupancy_loss_weight: float | None = None,
    occ_pos_weight_factor: float | None = None,
    affected_face_loss_weight: float = 1.0,
    affected_delta_loss_weight: float = 0.0,
    balancer: "EMALossBalancer | None" = None,
) -> float:
    """Runs one optimizer step for planner plus octree-only transition learning.

    Args:
        octree_depth_weight_base: Exponential base for per-cell depth weighting in
            the octree BCE loss.  ``2.0`` doubles the weight per octree level.
        monotonicity_weight: Scale for the monotonicity penalty loss term.
            Requires ``octree_occ_labels_before`` to be present in the batch.
            Set to ``0.0`` to disable.
        balancer: Optional :class:`EMALossBalancer`.  When provided, each loss is
            divided by its running EMA scale before summation so that planner and
            octree terms contribute with comparable gradient magnitudes.
    """
    del transition_loss_weight, point_sdf_loss_weight, changed_mask_loss_weight
    if occ_pos_weight_factor is not None:
        octree_pos_weight_factor = occ_pos_weight_factor

    model.train()
    optimizer.zero_grad(set_to_none=True)

    inputs = _collect_common_inputs(batch, device)
    outputs = _run_planner(model, inputs)

    planner_loss = _compute_planner_loss(
        outputs,
        inputs,
        macro_class_loss_weight,
        action_face_loss_weight,
        tool_choice_loss_weight,
    )
    octree_loss = _compute_octree_loss(
        model,
        batch,
        inputs,
        outputs,
        device,
        octree_loss_weight,
        octree_pos_weight_factor,
        octree_depth_weight_base=octree_depth_weight_base,
        monotonicity_weight=monotonicity_weight,
        fill_fraction_weight=fill_fraction_weight,
        removed_fraction_weight=removed_fraction_weight,
        tsdf_loss_weight=tsdf_loss_weight,
        delta_tsdf_loss_weight=delta_tsdf_loss_weight,
        tsdf_monotonicity_weight=tsdf_monotonicity_weight,
        tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
        occupancy_loss_weight=float(occupancy_loss_weight if occupancy_loss_weight is not None else 0.1),
        affected_face_loss_weight=affected_face_loss_weight,
        affected_delta_loss_weight=affected_delta_loss_weight,
    )

    # ── Loss combination with optional EMA balancing ──────────────────────
    if balancer is not None:
        sp = balancer.scale("planner", float(planner_loss.item()))
        so = balancer.scale("octree",  float(octree_loss.item()))
        loss = sp * planner_loss + octree_loss_weight * so * octree_loss
    else:
        loss = planner_loss + octree_loss_weight * octree_loss

    if not loss.requires_grad:
        return float(loss.item())
    loss.backward()
    optimizer.step()
    return float(loss.item())


def transition_train_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    octree_pos_weight_factor: float = 2.0,
    octree_depth_weight_base: float = 2.0,
    monotonicity_weight: float = 0.1,
    fill_fraction_weight: float = 0.5,
    removed_fraction_weight: float = 2.0,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    changed_tsdf_loss_weight: float = 0.0,
    changed_delta_tsdf_loss_weight: float = 0.0,
    tsdf_change_eps: float = 1e-3,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
    occupancy_loss_weight: float = 0.1,
    affected_face_loss_weight: float = 1.0,
    affected_delta_loss_weight: float = 0.0,
    use_target_tsdf_input: bool = True,
    return_components: bool = False,
) -> float:
    """Runs one optimizer step for transition-only octree learning.

    This path bypasses planner heads entirely:
        current_state + GT action labels + octree queries → next occupancy

    Args:
        octree_depth_weight_base: Exponential base for per-cell depth weighting.
        monotonicity_weight: Scale for the monotonicity penalty.  Requires
            ``octree_occ_labels_before`` in the batch; set to ``0.0`` to disable.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    inputs = _collect_common_inputs(batch, device)
    if "sdf_query_points" in batch and "sdf_tsdf_after" in batch:
        loss_result = _compute_transition_only_sdf_query_loss(
            model=model,
            batch=batch,
            inputs=inputs,
            device=device,
            tsdf_loss_weight=tsdf_loss_weight,
            delta_tsdf_loss_weight=delta_tsdf_loss_weight,
            changed_tsdf_loss_weight=changed_tsdf_loss_weight,
            changed_delta_tsdf_loss_weight=changed_delta_tsdf_loss_weight,
            tsdf_change_eps=tsdf_change_eps,
            tsdf_monotonicity_weight=tsdf_monotonicity_weight,
            tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
            affected_face_loss_weight=affected_face_loss_weight,
            affected_delta_loss_weight=affected_delta_loss_weight,
            use_target_tsdf_input=use_target_tsdf_input,
            return_components=return_components,
        )
    else:
        loss_result = _compute_transition_only_octree_loss(
            model=model,
            batch=batch,
            inputs=inputs,
            device=device,
            octree_pos_weight_factor=octree_pos_weight_factor,
            octree_depth_weight_base=octree_depth_weight_base,
            monotonicity_weight=monotonicity_weight,
            fill_fraction_weight=fill_fraction_weight,
            removed_fraction_weight=removed_fraction_weight,
            tsdf_loss_weight=tsdf_loss_weight,
            delta_tsdf_loss_weight=delta_tsdf_loss_weight,
            tsdf_monotonicity_weight=tsdf_monotonicity_weight,
            tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
            occupancy_loss_weight=occupancy_loss_weight,
        )
    if return_components and isinstance(loss_result, tuple):
        loss, components = loss_result
    else:
        loss, components = loss_result, {}
    loss.backward()
    optimizer.step()
    loss_value = float(loss.item())
    return (loss_value, components) if return_components else loss_value


@torch.no_grad()
def planner_validation_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    device: torch.device,
    macro_class_loss_weight: float = 1.0,
    action_face_loss_weight: float = 1.0,
    tool_choice_loss_weight: float = 1.0,
    transition_loss_weight: float = 0.0,
    point_sdf_loss_weight: float = 0.0,
    changed_mask_loss_weight: float = 0.0,
    octree_loss_weight: float = 1.0,
    octree_pos_weight_factor: float = 2.0,
    octree_depth_weight_base: float = 2.0,
    monotonicity_weight: float = 0.1,
    fill_fraction_weight: float = 0.5,
    removed_fraction_weight: float = 2.0,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
    occupancy_loss_weight: float | None = None,
    occ_pos_weight_factor: float | None = None,
    affected_face_loss_weight: float = 1.0,
    affected_delta_loss_weight: float = 0.0,
) -> float:
    """Computes validation loss for planner plus octree-only transition learning."""
    del transition_loss_weight, point_sdf_loss_weight, changed_mask_loss_weight
    if occ_pos_weight_factor is not None:
        octree_pos_weight_factor = occ_pos_weight_factor

    model.eval()
    inputs = _collect_common_inputs(batch, device)
    outputs = _run_planner(model, inputs)

    planner_loss = _compute_planner_loss(
        outputs,
        inputs,
        macro_class_loss_weight,
        action_face_loss_weight,
        tool_choice_loss_weight,
    )
    octree_loss = _compute_octree_loss(
        model,
        batch,
        inputs,
        outputs,
        device,
        octree_loss_weight,
        octree_pos_weight_factor,
        octree_depth_weight_base=octree_depth_weight_base,
        monotonicity_weight=monotonicity_weight,
        fill_fraction_weight=fill_fraction_weight,
        removed_fraction_weight=removed_fraction_weight,
        tsdf_loss_weight=tsdf_loss_weight,
        delta_tsdf_loss_weight=delta_tsdf_loss_weight,
        tsdf_monotonicity_weight=tsdf_monotonicity_weight,
        tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
        occupancy_loss_weight=float(occupancy_loss_weight if occupancy_loss_weight is not None else 0.1),
        affected_face_loss_weight=affected_face_loss_weight,
        affected_delta_loss_weight=affected_delta_loss_weight,
    )
    loss = planner_loss + octree_loss_weight * octree_loss
    return float(loss.item())


@torch.no_grad()
def transition_validation_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    device: torch.device,
    octree_pos_weight_factor: float = 2.0,
    octree_depth_weight_base: float = 2.0,
    monotonicity_weight: float = 0.1,
    fill_fraction_weight: float = 0.5,
    removed_fraction_weight: float = 2.0,
    tsdf_loss_weight: float = 1.0,
    delta_tsdf_loss_weight: float = 0.5,
    changed_tsdf_loss_weight: float = 0.0,
    changed_delta_tsdf_loss_weight: float = 0.0,
    tsdf_change_eps: float = 1e-3,
    tsdf_monotonicity_weight: float = 0.1,
    tsdf_monotonicity_empty_margin: float = 0.2,
    occupancy_loss_weight: float = 0.1,
    affected_face_loss_weight: float = 1.0,
    affected_delta_loss_weight: float = 0.0,
    use_target_tsdf_input: bool = True,
    return_components: bool = False,
) -> float:
    """Computes validation loss for transition-only octree learning."""
    model.eval()
    inputs = _collect_common_inputs(batch, device)
    if "sdf_query_points" in batch and "sdf_tsdf_after" in batch:
        loss_result = _compute_transition_only_sdf_query_loss(
            model=model,
            batch=batch,
            inputs=inputs,
            device=device,
            tsdf_loss_weight=tsdf_loss_weight,
            delta_tsdf_loss_weight=delta_tsdf_loss_weight,
            changed_tsdf_loss_weight=changed_tsdf_loss_weight,
            changed_delta_tsdf_loss_weight=changed_delta_tsdf_loss_weight,
            tsdf_change_eps=tsdf_change_eps,
            tsdf_monotonicity_weight=tsdf_monotonicity_weight,
            tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
            affected_face_loss_weight=affected_face_loss_weight,
            affected_delta_loss_weight=affected_delta_loss_weight,
            use_target_tsdf_input=use_target_tsdf_input,
            return_components=return_components,
        )
    else:
        loss_result = _compute_transition_only_octree_loss(
            model=model,
            batch=batch,
            inputs=inputs,
            device=device,
            octree_pos_weight_factor=octree_pos_weight_factor,
            octree_depth_weight_base=octree_depth_weight_base,
            monotonicity_weight=monotonicity_weight,
            fill_fraction_weight=fill_fraction_weight,
            removed_fraction_weight=removed_fraction_weight,
            tsdf_loss_weight=tsdf_loss_weight,
            delta_tsdf_loss_weight=delta_tsdf_loss_weight,
            tsdf_monotonicity_weight=tsdf_monotonicity_weight,
            tsdf_monotonicity_empty_margin=tsdf_monotonicity_empty_margin,
            occupancy_loss_weight=occupancy_loss_weight,
        )
    if return_components and isinstance(loss_result, tuple):
        loss, components = loss_result
    else:
        loss, components = loss_result, {}
    loss_value = float(loss.item())
    return (loss_value, components) if return_components else loss_value
