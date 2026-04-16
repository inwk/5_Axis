"""Training-step utilities for planner and octree transition learning."""

from __future__ import annotations

import torch

from .losses import occupancy_bce_loss, process_planning_loss
from .model import GraphSdfPlanningModel


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
        "axis_visible": _to_device(batch.get("axis_visible"), device),
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
) -> torch.Tensor:
    """Computes octree occupancy BCE for all transition rows in the batch."""
    if octree_loss_weight <= 0.0:
        return torch.zeros((), device=device, dtype=inputs["state_points"].dtype)
    if model.octree_decoder is None:
        return torch.zeros((), device=device, dtype=inputs["state_points"].dtype)
    if "octree_centers" not in batch or "octree_depths" not in batch or "octree_occ_labels" not in batch:
        return torch.zeros((), device=device, dtype=inputs["state_points"].dtype)

    octree_centers = batch["octree_centers"].to(device)
    octree_depths = batch["octree_depths"].to(device)
    octree_labels = batch["octree_occ_labels"].to(device)

    octree_outputs = model.forward_octree(
        state_points=inputs["state_points"],
        macro_class_id=inputs["target_macro_class"],
        tool_choice_id=inputs["target_tool_choice"],
        action_face_id=inputs["target_action_face"].clamp_min(0),
        octree_centers=octree_centers,
        octree_depths=octree_depths,
        axis_visible=inputs["axis_visible"],
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
        state_embedding=outputs.get("state_embedding"),
    )
    return occupancy_bce_loss(
        occ_logits=octree_outputs["occ_logits"],
        occ_labels=octree_labels,
        pos_weight_factor=octree_pos_weight_factor,
    )


def _compute_transition_only_octree_loss(
    model: GraphSdfPlanningModel,
    batch: dict,
    inputs: dict,
    device: torch.device,
    octree_pos_weight_factor: float,
) -> torch.Tensor:
    """Computes octree occupancy BCE with action labels as inputs and no planner path."""
    if model.octree_decoder is None:
        raise RuntimeError("Transition-only training requires model.octree_decoder to be enabled.")
    if "octree_centers" not in batch or "octree_depths" not in batch or "octree_occ_labels" not in batch:
        raise RuntimeError(
            "Transition-only training requires batch keys: "
            "'octree_centers', 'octree_depths', and 'octree_occ_labels'."
        )

    state_embedding = _encode_state_only(model, inputs)
    octree_outputs = model.forward_octree(
        state_points=inputs["state_points"],
        macro_class_id=inputs["target_macro_class"],
        tool_choice_id=inputs["target_tool_choice"],
        action_face_id=inputs["target_action_face"].clamp_min(0),
        octree_centers=batch["octree_centers"].to(device),
        octree_depths=batch["octree_depths"].to(device),
        axis_visible=inputs["axis_visible"],
        node_process_state=inputs["node_process_state"],
        node_centrality=inputs["node_centrality"],
        spatial_pos=inputs["spatial_pos"],
        face_area=inputs["face_area"],
        node_mask=inputs["node_mask"],
        point_mask=inputs["point_mask"],
        state_embedding=state_embedding,
    )
    return occupancy_bce_loss(
        occ_logits=octree_outputs["occ_logits"],
        occ_labels=batch["octree_occ_labels"].to(device),
        pos_weight_factor=octree_pos_weight_factor,
    )


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
    occupancy_loss_weight: float | None = None,
    occ_pos_weight_factor: float | None = None,
) -> float:
    """Runs one optimizer step for planner plus octree-only transition learning."""
    del transition_loss_weight, point_sdf_loss_weight, changed_mask_loss_weight
    if occupancy_loss_weight is not None:
        octree_loss_weight = occupancy_loss_weight
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
    )
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
) -> float:
    """Runs one optimizer step for transition-only octree learning.

    This path bypasses planner heads entirely:
        current_state + action labels + octree queries -> next occupancy
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    inputs = _collect_common_inputs(batch, device)
    loss = _compute_transition_only_octree_loss(
        model=model,
        batch=batch,
        inputs=inputs,
        device=device,
        octree_pos_weight_factor=octree_pos_weight_factor,
    )
    loss.backward()
    optimizer.step()
    return float(loss.item())


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
    occupancy_loss_weight: float | None = None,
    occ_pos_weight_factor: float | None = None,
) -> float:
    """Computes validation loss for planner plus octree-only transition learning."""
    del transition_loss_weight, point_sdf_loss_weight, changed_mask_loss_weight
    if occupancy_loss_weight is not None:
        octree_loss_weight = occupancy_loss_weight
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
    )
    loss = planner_loss + octree_loss_weight * octree_loss
    return float(loss.item())


@torch.no_grad()
def transition_validation_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    device: torch.device,
    octree_pos_weight_factor: float = 2.0,
) -> float:
    """Computes validation loss for transition-only octree learning."""
    model.eval()
    inputs = _collect_common_inputs(batch, device)
    loss = _compute_transition_only_octree_loss(
        model=model,
        batch=batch,
        inputs=inputs,
        device=device,
        octree_pos_weight_factor=octree_pos_weight_factor,
    )
    return float(loss.item())
