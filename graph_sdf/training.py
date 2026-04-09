"""Training-step utilities for planner and one-step transition learning."""

import torch

from .losses import process_planning_loss, transition_reconstruction_loss
from .model import GraphSdfPlanningModel


def _to_device(optional_tensor, device: torch.device):
    """Moves an optional tensor to the requested device."""

    if optional_tensor is None:
        return None
    return optional_tensor.to(device)


def planner_train_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    macro_class_loss_weight: float = 1.0,
    target_node_loss_weight: float = 1.0,
    tool_choice_loss_weight: float = 1.0,
    strategy_loss_weight: float = 0.25,
    transition_loss_weight: float = 1.0,
    point_sdf_loss_weight: float = 1.0,
    changed_mask_loss_weight: float = 0.25,
) -> float:
    """Runs one optimizer step for process planning and optional transition learning."""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    state_points = batch["state_points"].to(device)
    node_process_state = _to_device(batch.get("node_process_state"), device)
    node_centrality = _to_device(batch.get("node_centrality"), device)
    spatial_pos = _to_device(batch.get("spatial_pos"), device)
    face_area = _to_device(batch.get("face_area"), device)
    axis_visible = _to_device(batch.get("axis_visible"), device)
    global_process_state = _to_device(batch.get("global_process_state"), device)

    target_macro_class = batch["macro_class_id"].to(device)
    target_tool_choice = batch["tool_choice_id"].to(device)
    target_strategy = _to_device(batch.get("strategy_id"), device)
    target_target_node = _to_device(batch.get("target_node_id"), device)

    node_mask = _to_device(batch.get("node_mask"), device)
    point_mask = _to_device(batch.get("point_mask"), device)
    target_node_mask = _to_device(batch.get("target_node_mask"), device)
    macro_class_mask = _to_device(batch.get("macro_class_mask"), device)
    tool_choice_mask = _to_device(batch.get("tool_choice_mask"), device)
    strategy_mask = _to_device(batch.get("strategy_mask"), device)
    target_node_valid = _to_device(batch.get("target_node_valid"), device)
    tool_choice_valid = _to_device(batch.get("tool_choice_valid"), device)
    strategy_valid = _to_device(batch.get("strategy_valid"), device)

    if target_target_node is None:
        target_target_node = torch.full_like(target_macro_class, -1)
    if target_strategy is None:
        target_strategy = torch.full_like(target_macro_class, -1)

    # is_chosen: True for the committed best action, False for candidates.
    # Planner loss is only meaningful on chosen rows (correct labels).
    # Transition loss uses ALL rows (each candidate has a valid next_node_sdf).
    is_chosen_raw = batch.get("is_chosen")
    if is_chosen_raw is not None:
        is_chosen = is_chosen_raw.to(device).bool()
    else:
        is_chosen = None

    outputs = model.forward_process_planner(
        state_points=state_points,
        node_process_state=node_process_state,
        node_centrality=node_centrality,
        spatial_pos=spatial_pos,
        face_area=face_area,
        node_mask=node_mask,
        point_mask=point_mask,
        axis_visible=axis_visible,
        global_process_state=global_process_state,
        target_node_mask=target_node_mask,
        macro_class_mask=macro_class_mask,
        tool_choice_mask=tool_choice_mask,
        strategy_mask=strategy_mask,
    )

    # Apply planner loss only on chosen rows when is_chosen is available.
    if is_chosen is not None and is_chosen.any():
        idx = is_chosen.nonzero(as_tuple=True)[0]
        planner_loss = process_planning_loss(
            pred_macro_class_logits=outputs["macro_class_logits"][idx],
            target_macro_class=target_macro_class[idx],
            pred_target_node_logits=outputs["target_node_logits"][idx],
            target_target_node=target_target_node[idx],
            pred_tool_choice_logits=outputs["tool_choice_logits"][idx],
            target_tool_choice=target_tool_choice[idx],
            pred_strategy_logits=outputs["strategy_logits"][idx],
            target_strategy=target_strategy[idx],
            target_node_valid=target_node_valid[idx] if target_node_valid is not None else None,
            tool_choice_valid=tool_choice_valid[idx] if tool_choice_valid is not None else None,
            strategy_valid=strategy_valid[idx] if strategy_valid is not None else None,
            macro_class_weight=macro_class_loss_weight,
            target_node_weight=target_node_loss_weight,
            tool_choice_weight=tool_choice_loss_weight,
            strategy_weight=strategy_loss_weight,
        )
    else:
        # Fallback: use all rows (e.g., old data without is_chosen column).
        planner_loss = process_planning_loss(
            pred_macro_class_logits=outputs["macro_class_logits"],
            target_macro_class=target_macro_class,
            pred_target_node_logits=outputs["target_node_logits"],
            target_target_node=target_target_node,
            pred_tool_choice_logits=outputs["tool_choice_logits"],
            target_tool_choice=target_tool_choice,
            pred_strategy_logits=outputs["strategy_logits"],
            target_strategy=target_strategy,
            target_node_valid=target_node_valid,
            tool_choice_valid=tool_choice_valid,
            strategy_valid=strategy_valid,
            macro_class_weight=macro_class_loss_weight,
            target_node_weight=target_node_loss_weight,
            tool_choice_weight=tool_choice_loss_weight,
            strategy_weight=strategy_loss_weight,
        )
    loss = planner_loss

    if "next_node_sdf" in batch and transition_loss_weight > 0.0:
        target_next_node_sdf = batch["next_node_sdf"].to(device)
        target_next_point_sdf = _to_device(batch.get("next_point_sdf"), device)

        full_outputs = model(
            state_points=state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_mask=node_mask,
            point_mask=point_mask,
            axis_visible=axis_visible,
            global_process_state=global_process_state,
            target_node_mask=target_node_mask,
            macro_class_mask=macro_class_mask,
            tool_choice_mask=tool_choice_mask,
            strategy_mask=strategy_mask,
            target_macro_class=target_macro_class,
            target_tool_choice=target_tool_choice,
            target_strategy=target_strategy.clamp_min(0),
            target_node_id=target_target_node.clamp_min(0),
            use_teacher_forcing_action=True,
            run_transition=True,
        )
        transition_loss = transition_reconstruction_loss(
            pred_next_node_sdf=full_outputs["pred_next_node_sdf"],
            target_next_node_sdf=target_next_node_sdf,
            current_node_sdf=full_outputs["current_node_sdf"],
            pred_next_point_sdf=full_outputs["pred_next_point_sdf"],
            target_next_point_sdf=target_next_point_sdf,
            pred_changed_logits=full_outputs["pred_changed_logits"],
            node_mask=node_mask,
            changed_face_threshold=model.config.changed_face_threshold,
            point_sdf_weight=point_sdf_loss_weight,
            changed_mask_weight=changed_mask_loss_weight,
        )
        loss = loss + transition_loss_weight * transition_loss

    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def planner_validation_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    device: torch.device,
    macro_class_loss_weight: float = 1.0,
    target_node_loss_weight: float = 1.0,
    tool_choice_loss_weight: float = 1.0,
    strategy_loss_weight: float = 0.25,
    transition_loss_weight: float = 1.0,
    point_sdf_loss_weight: float = 1.0,
    changed_mask_loss_weight: float = 0.25,
) -> float:
    """Computes validation loss for process planning and optional transition learning."""

    model.eval()

    state_points = batch["state_points"].to(device)
    node_process_state = _to_device(batch.get("node_process_state"), device)
    node_centrality = _to_device(batch.get("node_centrality"), device)
    spatial_pos = _to_device(batch.get("spatial_pos"), device)
    face_area = _to_device(batch.get("face_area"), device)
    axis_visible = _to_device(batch.get("axis_visible"), device)
    global_process_state = _to_device(batch.get("global_process_state"), device)

    target_macro_class = batch["macro_class_id"].to(device)
    target_tool_choice = batch["tool_choice_id"].to(device)
    target_strategy = _to_device(batch.get("strategy_id"), device)
    target_target_node = _to_device(batch.get("target_node_id"), device)

    node_mask = _to_device(batch.get("node_mask"), device)
    point_mask = _to_device(batch.get("point_mask"), device)
    target_node_mask = _to_device(batch.get("target_node_mask"), device)
    macro_class_mask = _to_device(batch.get("macro_class_mask"), device)
    tool_choice_mask = _to_device(batch.get("tool_choice_mask"), device)
    strategy_mask = _to_device(batch.get("strategy_mask"), device)
    target_node_valid = _to_device(batch.get("target_node_valid"), device)
    tool_choice_valid = _to_device(batch.get("tool_choice_valid"), device)
    strategy_valid = _to_device(batch.get("strategy_valid"), device)

    if target_target_node is None:
        target_target_node = torch.full_like(target_macro_class, -1)
    if target_strategy is None:
        target_strategy = torch.full_like(target_macro_class, -1)

    is_chosen_raw = batch.get("is_chosen")
    if is_chosen_raw is not None:
        is_chosen = is_chosen_raw.to(device).bool()
    else:
        is_chosen = None

    outputs = model(
        state_points=state_points,
        node_process_state=node_process_state,
        node_centrality=node_centrality,
        spatial_pos=spatial_pos,
        face_area=face_area,
        node_mask=node_mask,
        point_mask=point_mask,
        axis_visible=axis_visible,
        global_process_state=global_process_state,
        target_node_mask=target_node_mask,
        macro_class_mask=macro_class_mask,
        tool_choice_mask=tool_choice_mask,
        strategy_mask=strategy_mask,
        target_macro_class=target_macro_class,
        target_tool_choice=target_tool_choice,
        target_strategy=target_strategy.clamp_min(0),
        target_node_id=target_target_node.clamp_min(0),
        use_teacher_forcing_action=True,
        run_transition=True,
    )

    if is_chosen is not None and is_chosen.any():
        idx = is_chosen.nonzero(as_tuple=True)[0]
        loss = process_planning_loss(
            pred_macro_class_logits=outputs["macro_class_logits"][idx],
            target_macro_class=target_macro_class[idx],
            pred_target_node_logits=outputs["target_node_logits"][idx],
            target_target_node=target_target_node[idx],
            pred_tool_choice_logits=outputs["tool_choice_logits"][idx],
            target_tool_choice=target_tool_choice[idx],
            pred_strategy_logits=outputs["strategy_logits"][idx],
            target_strategy=target_strategy[idx],
            target_node_valid=target_node_valid[idx] if target_node_valid is not None else None,
            tool_choice_valid=tool_choice_valid[idx] if tool_choice_valid is not None else None,
            strategy_valid=strategy_valid[idx] if strategy_valid is not None else None,
            macro_class_weight=macro_class_loss_weight,
            target_node_weight=target_node_loss_weight,
            tool_choice_weight=tool_choice_loss_weight,
            strategy_weight=strategy_loss_weight,
        )
    else:
        loss = process_planning_loss(
            pred_macro_class_logits=outputs["macro_class_logits"],
            target_macro_class=target_macro_class,
            pred_target_node_logits=outputs["target_node_logits"],
            target_target_node=target_target_node,
            pred_tool_choice_logits=outputs["tool_choice_logits"],
            target_tool_choice=target_tool_choice,
            pred_strategy_logits=outputs["strategy_logits"],
            target_strategy=target_strategy,
            target_node_valid=target_node_valid,
            tool_choice_valid=tool_choice_valid,
            strategy_valid=strategy_valid,
            macro_class_weight=macro_class_loss_weight,
            target_node_weight=target_node_loss_weight,
            tool_choice_weight=tool_choice_loss_weight,
            strategy_weight=strategy_loss_weight,
        )

    if "next_node_sdf" in batch and transition_loss_weight > 0.0:
        target_next_node_sdf = batch["next_node_sdf"].to(device)
        target_next_point_sdf = _to_device(batch.get("next_point_sdf"), device)
        transition_loss = transition_reconstruction_loss(
            pred_next_node_sdf=outputs["pred_next_node_sdf"],
            target_next_node_sdf=target_next_node_sdf,
            current_node_sdf=outputs["current_node_sdf"],
            pred_next_point_sdf=outputs["pred_next_point_sdf"],
            target_next_point_sdf=target_next_point_sdf,
            pred_changed_logits=outputs["pred_changed_logits"],
            node_mask=node_mask,
            changed_face_threshold=model.config.changed_face_threshold,
            point_sdf_weight=point_sdf_loss_weight,
            changed_mask_weight=changed_mask_loss_weight,
        )
        loss = loss + transition_loss_weight * transition_loss

    return float(loss.item())
