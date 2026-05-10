"""Training runner for Graph-SDF process skeleton planning."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from graph_sdf import GraphSdfModelConfig, GraphSdfPlanningModel
from graph_sdf.training import (
    EMALossBalancer,
    planner_train_step,
    planner_validation_step,
    transition_train_step,
    transition_validation_step,
)


@dataclass
class TrainConfig:
    """Holds basic training hyperparameters."""

    lr: float = 1e-4
    epochs: int = 20
    macro_class_loss_weight: float = 1.0
    target_node_loss_weight: float = 1.0
    tool_choice_loss_weight: float = 1.0
    strategy_loss_weight: float = 0.25
    transition_loss_weight: float = 0.0
    point_sdf_loss_weight: float = 0.0
    changed_mask_loss_weight: float = 0.0
    # Octree-only transition supervision.
    octree_loss_weight: float = 1.0
    octree_pos_weight_factor: float = 2.0
    # Depth-weighted BCE: each octree level multiplies the cell weight by this base.
    # 2.0 = fine cells (surface boundary) get 4× the weight of coarse cells at depth 3 vs 5.
    # 1.0 = uniform (legacy behaviour).
    octree_depth_weight_base: float = 2.0
    # Monotonicity penalty: penalises predicting occupied for already-empty cells.
    # Requires octree_occ_labels_before in the dataset.  Set 0.0 to disable.
    monotonicity_weight: float = 0.1
    occupancy_loss_weight: float = 0.1
    fill_fraction_weight: float = 0.5
    removed_fraction_weight: float = 2.0
    tsdf_loss_weight: float = 1.0
    delta_tsdf_loss_weight: float = 0.5
    tsdf_monotonicity_weight: float = 0.1
    tsdf_monotonicity_empty_margin: float = 0.2
    affected_face_loss_weight: float = 1.0
    affected_delta_loss_weight: float = 0.0
    # EMA-based gradient balancing between planner and octree losses.
    use_loss_balancer: bool = True
    balancer_momentum: float = 0.99


def run_training(
    model: GraphSdfPlanningModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> None:
    """Trains the process skeleton planner."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    balancer = EMALossBalancer(momentum=config.balancer_momentum) if config.use_loss_balancer else None

    for epoch in range(config.epochs):
        train_losses = [
            planner_train_step(
                model,
                batch,
                optimizer,
                device,
                macro_class_loss_weight=config.macro_class_loss_weight,
                action_face_loss_weight=config.target_node_loss_weight,
                tool_choice_loss_weight=config.tool_choice_loss_weight,
                transition_loss_weight=config.transition_loss_weight,
                point_sdf_loss_weight=config.point_sdf_loss_weight,
                changed_mask_loss_weight=config.changed_mask_loss_weight,
                octree_loss_weight=config.octree_loss_weight,
                octree_pos_weight_factor=config.octree_pos_weight_factor,
                octree_depth_weight_base=config.octree_depth_weight_base,
                monotonicity_weight=config.monotonicity_weight,
                fill_fraction_weight=config.fill_fraction_weight,
                removed_fraction_weight=config.removed_fraction_weight,
                tsdf_loss_weight=config.tsdf_loss_weight,
                delta_tsdf_loss_weight=config.delta_tsdf_loss_weight,
                tsdf_monotonicity_weight=config.tsdf_monotonicity_weight,
                tsdf_monotonicity_empty_margin=config.tsdf_monotonicity_empty_margin,
                occupancy_loss_weight=config.occupancy_loss_weight,
                affected_face_loss_weight=config.affected_face_loss_weight,
                affected_delta_loss_weight=config.affected_delta_loss_weight,
                balancer=balancer,
            )
            for batch in train_loader
        ]
        val_losses = [
            planner_validation_step(
                model,
                batch,
                device,
                macro_class_loss_weight=config.macro_class_loss_weight,
                action_face_loss_weight=config.target_node_loss_weight,
                tool_choice_loss_weight=config.tool_choice_loss_weight,
                transition_loss_weight=config.transition_loss_weight,
                point_sdf_loss_weight=config.point_sdf_loss_weight,
                changed_mask_loss_weight=config.changed_mask_loss_weight,
                octree_loss_weight=config.octree_loss_weight,
                octree_pos_weight_factor=config.octree_pos_weight_factor,
                octree_depth_weight_base=config.octree_depth_weight_base,
                monotonicity_weight=config.monotonicity_weight,
                fill_fraction_weight=config.fill_fraction_weight,
                removed_fraction_weight=config.removed_fraction_weight,
                tsdf_loss_weight=config.tsdf_loss_weight,
                delta_tsdf_loss_weight=config.delta_tsdf_loss_weight,
                tsdf_monotonicity_weight=config.tsdf_monotonicity_weight,
                tsdf_monotonicity_empty_margin=config.tsdf_monotonicity_empty_margin,
                occupancy_loss_weight=config.occupancy_loss_weight,
                affected_face_loss_weight=config.affected_face_loss_weight,
                affected_delta_loss_weight=config.affected_delta_loss_weight,
            )
            for batch in val_loader
        ]

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        ema_str = ""
        if balancer is not None:
            ema_vals = balancer.ema_values()
            ema_str = "  ema=" + " ".join(f"{k}:{v:.4f}" for k, v in sorted(ema_vals.items()))
        print(f"[Planner+Octree][Epoch {epoch + 1}] train={train_loss:.6f} val={val_loss:.6f}{ema_str}")


def run_transition_training(
    model: GraphSdfPlanningModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> None:
    """Trains only the action-conditioned octree transition model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        train_losses = [
            transition_train_step(
                model,
                batch,
                optimizer,
                device,
                octree_pos_weight_factor=config.octree_pos_weight_factor,
                octree_depth_weight_base=config.octree_depth_weight_base,
                monotonicity_weight=config.monotonicity_weight,
                fill_fraction_weight=config.fill_fraction_weight,
                removed_fraction_weight=config.removed_fraction_weight,
                tsdf_loss_weight=config.tsdf_loss_weight,
                delta_tsdf_loss_weight=config.delta_tsdf_loss_weight,
                tsdf_monotonicity_weight=config.tsdf_monotonicity_weight,
                tsdf_monotonicity_empty_margin=config.tsdf_monotonicity_empty_margin,
                occupancy_loss_weight=config.occupancy_loss_weight,
                affected_face_loss_weight=config.affected_face_loss_weight,
                affected_delta_loss_weight=config.affected_delta_loss_weight,
            )
            for batch in train_loader
        ]
        val_losses = [
            transition_validation_step(
                model,
                batch,
                device,
                octree_pos_weight_factor=config.octree_pos_weight_factor,
                octree_depth_weight_base=config.octree_depth_weight_base,
                monotonicity_weight=config.monotonicity_weight,
                fill_fraction_weight=config.fill_fraction_weight,
                removed_fraction_weight=config.removed_fraction_weight,
                tsdf_loss_weight=config.tsdf_loss_weight,
                delta_tsdf_loss_weight=config.delta_tsdf_loss_weight,
                tsdf_monotonicity_weight=config.tsdf_monotonicity_weight,
                tsdf_monotonicity_empty_margin=config.tsdf_monotonicity_empty_margin,
                occupancy_loss_weight=config.occupancy_loss_weight,
                affected_face_loss_weight=config.affected_face_loss_weight,
                affected_delta_loss_weight=config.affected_delta_loss_weight,
            )
            for batch in val_loader
        ]

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"[TransitionOnly-Octree][Epoch {epoch + 1}] train={train_loss:.6f} val={val_loss:.6f}")


def build_model(device: torch.device) -> GraphSdfPlanningModel:
    """Builds and returns the planning model on the requested device."""

    model_cfg = GraphSdfModelConfig()
    model = GraphSdfPlanningModel(model_cfg)
    return model.to(device)


if __name__ == "__main__":
    print(
        "This script provides training runners only. "
        "Connect your DataLoader instances and call run_training or run_transition_training "
        "from your training entrypoint."
    )
