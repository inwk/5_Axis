"""Training runner for Graph-SDF process skeleton planning."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from graph_sdf import GraphSdfModelConfig, GraphSdfPlanningModel
from graph_sdf.training import (
    planner_train_step,
    planner_validation_step,
)


@dataclass
class TrainConfig:
    """Holds basic training hyperparameters."""

    lr: float = 1e-4
    epochs: int = 20
    macro_class_loss_weight: float = 1.0
    target_node_loss_weight: float = 1.0
    tool_choice_loss_weight: float = 1.0


def run_training(
    model: GraphSdfPlanningModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> None:
    """Trains the process skeleton planner."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        train_losses = [
            planner_train_step(
                model,
                batch,
                optimizer,
                device,
                macro_class_loss_weight=config.macro_class_loss_weight,
                target_node_loss_weight=config.target_node_loss_weight,
                tool_choice_loss_weight=config.tool_choice_loss_weight,
            )
            for batch in train_loader
        ]
        val_losses = [
            planner_validation_step(
                model,
                batch,
                device,
                macro_class_loss_weight=config.macro_class_loss_weight,
                target_node_loss_weight=config.target_node_loss_weight,
                tool_choice_loss_weight=config.tool_choice_loss_weight,
            )
            for batch in val_loader
        ]

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"[Planner][Epoch {epoch + 1}] train={train_loss:.6f} val={val_loss:.6f}")


def build_model(device: torch.device) -> GraphSdfPlanningModel:
    """Builds and returns the planning model on the requested device."""

    model_cfg = GraphSdfModelConfig()
    model = GraphSdfPlanningModel(model_cfg)
    return model.to(device)


if __name__ == "__main__":
    print(
        "This script provides the planner training runner only. "
        "Connect your DataLoader instances and call run_training from your training entrypoint."
    )
