"""Unified model that combines state encoding, planning, and transition decoding."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphSdfModelConfig
from .process_planner import ProcessPlannerHead
from .schema import MACRO_CLASS_TO_ID
from .shape_transition import ShapeTransitionHead
from .state_encoder import StateEncoder


class GraphSdfPlanningModel(nn.Module):
    """Implements process skeleton planning plus one-step state transition decoding."""

    AXIS_MODE_INDEXED = 0
    AXIS_MODE_SIMULTANEOUS_5_AXIS = 1

    def __init__(self, config: GraphSdfModelConfig) -> None:
        """Builds the encoder, planner head, and action-conditioned decoder."""
        super().__init__()
        config.validate()
        self.config = config

        self.state_encoder = StateEncoder(config)
        self.process_planner = ProcessPlannerHead(config)
        self.shape_transition = ShapeTransitionHead(config)

        self.indexed_ids = torch.tensor(
            [
                MACRO_CLASS_TO_ID["indexed_rough"],
                MACRO_CLASS_TO_ID["indexed_finish"],
            ],
            dtype=torch.long,
        )

    def _derive_axis_from_action(
        self,
        macro_class_id: torch.Tensor,
        action_face_id: torch.Tensor,
        state_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps the selected action to a deterministic axis and axis-mode id."""

        node_normals = state_points[:, :, :, 3:6].mean(dim=2)
        gather_index = action_face_id[:, None, None].expand(-1, 1, 3)
        pred_axis = torch.gather(node_normals, 1, gather_index).squeeze(1)
        pred_axis = F.normalize(pred_axis, dim=-1, eps=1e-8)

        axis_mode = torch.full_like(
            macro_class_id,
            fill_value=self.AXIS_MODE_SIMULTANEOUS_5_AXIS,
            dtype=torch.long,
        )
        is_indexed = (macro_class_id[:, None] == self.indexed_ids.to(macro_class_id.device)).any(dim=1)
        axis_mode[is_indexed] = self.AXIS_MODE_INDEXED
        return pred_axis, axis_mode

    def encode_state(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        face_area: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes graph state tensors into node embeddings [B, N, H]."""

        return self.state_encoder(
            state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_mask=node_mask,
            point_mask=point_mask,
        )

    def forward_process_planner(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        face_area: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        axis_visible: Optional[torch.Tensor] = None,
        global_process_state: Optional[torch.Tensor] = None,
        action_face_mask: Optional[torch.Tensor] = None,
        macro_class_mask: Optional[torch.Tensor] = None,
        tool_choice_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts an NX-oriented process skeleton from the current state."""

        _ = axis_visible

        state_embedding = self.encode_state(
            state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_mask=node_mask,
            point_mask=point_mask,
        )
        outputs = self.process_planner(
            state_embedding=state_embedding,
            node_mask=node_mask,
            global_process_state=global_process_state,
            action_face_mask=action_face_mask,
            macro_class_mask=macro_class_mask,
            tool_choice_mask=tool_choice_mask,
        )
        pred_macro_class = outputs["macro_class_logits"].argmax(dim=1)
        pred_action_face = outputs["action_face_logits"].argmax(dim=1)
        pred_tool_choice = outputs["tool_choice_logits"].argmax(dim=1)
        pred_axis, pred_axis_mode = self._derive_axis_from_action(
            macro_class_id=pred_macro_class,
            action_face_id=pred_action_face,
            state_points=state_points,
        )

        outputs["pred_macro_class"] = pred_macro_class
        outputs["pred_tool_choice"] = pred_tool_choice
        outputs["pred_action_face"] = pred_action_face
        outputs["pred_axis_from_face"] = pred_axis
        outputs["pred_axis_mode"] = pred_axis_mode
        outputs["state_embedding"] = state_embedding
        return outputs

    def forward_transition(
        self,
        state_points: torch.Tensor,
        macro_class_id: torch.Tensor,
        tool_choice_id: torch.Tensor,
        action_face_id: torch.Tensor,
        axis_visible: Optional[torch.Tensor] = None,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        face_area: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        state_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Runs the action-conditioned one-step state transition decoder."""

        if state_embedding is None:
            state_embedding = self.encode_state(
                state_points,
                node_process_state=node_process_state,
                node_centrality=node_centrality,
                spatial_pos=spatial_pos,
                face_area=face_area,
                node_mask=node_mask,
                point_mask=point_mask,
            )
        outputs = self.shape_transition(
            node_embeddings=state_embedding,
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            axis_visible=axis_visible,
            node_process_state=node_process_state,
            node_mask=node_mask,
        )
        outputs["state_embedding"] = state_embedding
        return outputs

    def forward(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        face_area: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        axis_visible: Optional[torch.Tensor] = None,
        global_process_state: Optional[torch.Tensor] = None,
        action_face_mask: Optional[torch.Tensor] = None,
        macro_class_mask: Optional[torch.Tensor] = None,
        tool_choice_mask: Optional[torch.Tensor] = None,
        target_macro_class: Optional[torch.Tensor] = None,
        target_tool_choice: Optional[torch.Tensor] = None,
        target_action_face: Optional[torch.Tensor] = None,
        use_teacher_forcing_action: bool = False,
        run_transition: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Runs planner forward and, optionally, one-step shape decoding."""

        outputs = self.forward_process_planner(
            state_points=state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_mask=node_mask,
            point_mask=point_mask,
            global_process_state=global_process_state,
            action_face_mask=action_face_mask,
            macro_class_mask=macro_class_mask,
            tool_choice_mask=tool_choice_mask,
        )
        if not run_transition:
            return outputs

        if use_teacher_forcing_action:
            if target_macro_class is None or target_tool_choice is None or target_action_face is None:
                raise ValueError(
                    "Teacher forcing requires target_macro_class, target_tool_choice, and target_action_face"
                )
            macro_class_id = target_macro_class
            tool_choice_id = target_tool_choice
            action_face_id = target_action_face
        else:
            macro_class_id = outputs["pred_macro_class"]
            tool_choice_id = outputs["pred_tool_choice"]
            action_face_id = outputs["pred_action_face"]

        transition_outputs = self.forward_transition(
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            axis_visible=axis_visible,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_mask=node_mask,
            point_mask=point_mask,
            state_embedding=outputs["state_embedding"],
        )
        outputs.update(transition_outputs)
        return outputs
