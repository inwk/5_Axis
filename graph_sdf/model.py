"""Unified model that combines state encoding and process planning."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphSdfModelConfig
from .process_planner import ProcessPlannerHead
from .state_encoder import StateEncoder


class GraphSdfPlanningModel(nn.Module):
    """Implements a process skeleton planner."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        """Builds the encoder and process planner head."""
        super().__init__()
        config.validate()
        self.config = config

        self.state_encoder = StateEncoder(config)
        self.process_planner = ProcessPlannerHead(config)

    def encode_state(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes graph state tensors into node embeddings [B, N, H]."""

        return self.state_encoder(
            state_points,
            node_process_state=node_process_state,
            node_mask=node_mask,
            point_mask=point_mask,
        )

    def forward_process_planner(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        global_process_state: Optional[torch.Tensor] = None,
        target_node_mask: Optional[torch.Tensor] = None,
        macro_class_mask: Optional[torch.Tensor] = None,
        tool_choice_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts an NX-oriented process skeleton from the current state."""

        state_embedding = self.encode_state(
            state_points,
            node_process_state=node_process_state,
            node_mask=node_mask,
            point_mask=point_mask,
        )
        outputs = self.process_planner(
            state_embedding=state_embedding,
            node_mask=node_mask,
            global_process_state=global_process_state,
            target_node_mask=target_node_mask,
            macro_class_mask=macro_class_mask,
            tool_choice_mask=tool_choice_mask,
        )
        pred_target_node = outputs["target_node_logits"].argmax(dim=1)
        node_normals = state_points[:, :, :, 3:6].mean(dim=2)
        gather_index = pred_target_node[:, None, None].expand(-1, 1, 3)
        pred_axis = torch.gather(node_normals, 1, gather_index).squeeze(1)
        outputs["pred_axis_from_target"] = F.normalize(pred_axis, dim=-1, eps=1e-8)
        outputs["pred_target_node"] = pred_target_node
        outputs["state_embedding"] = state_embedding
        return outputs

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Runs planner inference/training forward."""
        return self.forward_process_planner(**kwargs)
