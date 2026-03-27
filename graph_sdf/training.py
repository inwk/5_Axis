"""Training-step utilities for process-planning only."""

import torch

from .losses import process_planning_loss
from .model import GraphSdfPlanningModel


def planner_train_step(
    model: GraphSdfPlanningModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    macro_class_loss_weight: float = 1.0,
    target_node_loss_weight: float = 1.0,
    tool_choice_loss_weight: float = 1.0,
) -> float:
    """Runs one optimizer step for process skeleton planning."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    state_points = batch["state_points"].to(device)
    node_process_state = batch.get("node_process_state")
    global_process_state = batch.get("global_process_state")
    target_macro_class = batch["macro_class_id"].to(device)
    target_tool_choice = batch["tool_choice_id"].to(device)
    target_target_node = batch.get("target_node_id")

    node_mask = batch.get("node_mask")
    point_mask = batch.get("point_mask")
    target_node_mask = batch.get("target_node_mask")
    macro_class_mask = batch.get("macro_class_mask")
    tool_choice_mask = batch.get("tool_choice_mask")
    target_node_valid = batch.get("target_node_valid")
    tool_choice_valid = batch.get("tool_choice_valid")

    if node_process_state is not None:
        node_process_state = node_process_state.to(device)
    if global_process_state is not None:
        global_process_state = global_process_state.to(device)
    if node_mask is not None:
        node_mask = node_mask.to(device)
    if point_mask is not None:
        point_mask = point_mask.to(device)
    if target_node_mask is not None:
        target_node_mask = target_node_mask.to(device)
    if macro_class_mask is not None:
        macro_class_mask = macro_class_mask.to(device)
    if tool_choice_mask is not None:
        tool_choice_mask = tool_choice_mask.to(device)
    if target_node_valid is not None:
        target_node_valid = target_node_valid.to(device)
    if tool_choice_valid is not None:
        tool_choice_valid = tool_choice_valid.to(device)
    if target_target_node is not None:
        target_target_node = target_target_node.to(device)
    else:
        target_target_node = torch.full_like(target_macro_class, -1)

    outputs = model.forward_process_planner(
        state_points=state_points,
        node_process_state=node_process_state,
        node_mask=node_mask,
        point_mask=point_mask,
        global_process_state=global_process_state,
        target_node_mask=target_node_mask,
        macro_class_mask=macro_class_mask,
        tool_choice_mask=tool_choice_mask,
    )

    loss = process_planning_loss(
        pred_macro_class_logits=outputs["macro_class_logits"],
        target_macro_class=target_macro_class,
        pred_target_node_logits=outputs["target_node_logits"],
        target_target_node=target_target_node,
        pred_tool_choice_logits=outputs["tool_choice_logits"],
        target_tool_choice=target_tool_choice,
        target_node_valid=target_node_valid,
        tool_choice_valid=tool_choice_valid,
        macro_class_weight=macro_class_loss_weight,
        target_node_weight=target_node_loss_weight,
        tool_choice_weight=tool_choice_loss_weight,
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
    target_node_loss_weight: float = 1.0,
    tool_choice_loss_weight: float = 1.0,
) -> float:
    """Computes validation loss for process skeleton planning."""
    model.eval()

    state_points = batch["state_points"].to(device)
    node_process_state = batch.get("node_process_state")
    global_process_state = batch.get("global_process_state")
    target_macro_class = batch["macro_class_id"].to(device)
    target_tool_choice = batch["tool_choice_id"].to(device)
    target_target_node = batch.get("target_node_id")

    node_mask = batch.get("node_mask")
    point_mask = batch.get("point_mask")
    target_node_mask = batch.get("target_node_mask")
    macro_class_mask = batch.get("macro_class_mask")
    tool_choice_mask = batch.get("tool_choice_mask")
    target_node_valid = batch.get("target_node_valid")
    tool_choice_valid = batch.get("tool_choice_valid")

    if node_process_state is not None:
        node_process_state = node_process_state.to(device)
    if global_process_state is not None:
        global_process_state = global_process_state.to(device)
    if node_mask is not None:
        node_mask = node_mask.to(device)
    if point_mask is not None:
        point_mask = point_mask.to(device)
    if target_node_mask is not None:
        target_node_mask = target_node_mask.to(device)
    if macro_class_mask is not None:
        macro_class_mask = macro_class_mask.to(device)
    if tool_choice_mask is not None:
        tool_choice_mask = tool_choice_mask.to(device)
    if target_node_valid is not None:
        target_node_valid = target_node_valid.to(device)
    if tool_choice_valid is not None:
        tool_choice_valid = tool_choice_valid.to(device)
    if target_target_node is not None:
        target_target_node = target_target_node.to(device)
    else:
        target_target_node = torch.full_like(target_macro_class, -1)

    outputs = model.forward_process_planner(
        state_points=state_points,
        node_process_state=node_process_state,
        node_mask=node_mask,
        point_mask=point_mask,
        global_process_state=global_process_state,
        target_node_mask=target_node_mask,
        macro_class_mask=macro_class_mask,
        tool_choice_mask=tool_choice_mask,
    )

    loss = process_planning_loss(
        pred_macro_class_logits=outputs["macro_class_logits"],
        target_macro_class=target_macro_class,
        pred_target_node_logits=outputs["target_node_logits"],
        target_target_node=target_target_node,
        pred_tool_choice_logits=outputs["tool_choice_logits"],
        target_tool_choice=target_tool_choice,
        target_node_valid=target_node_valid,
        tool_choice_valid=tool_choice_valid,
        macro_class_weight=macro_class_loss_weight,
        target_node_weight=target_node_loss_weight,
        tool_choice_weight=tool_choice_loss_weight,
    )
    return float(loss.item())
