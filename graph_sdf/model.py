"""Unified model: face-graph encoder + process planner + octree decoder."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphSdfModelConfig
from .octree_decoder import OctreeDecoder
from .process_planner import ProcessPlannerHead
from .schema import MACRO_CLASS_TO_ID
from .sdf_query_decoder import SdfQueryDecoder
from .shape_transition import ShapeTransitionHead
from .state_encoder import StateEncoder


class GraphSdfPlanningModel(nn.Module):
    """Process skeleton planner + action-conditioned octree decoder.

    Architecture (hybrid):
    ┌──────────────────────────────────────────────────────────────┐
    │  StateEncoder  (face-graph transformer)         [unchanged]  │
    │    input:  state_points [B, N, P, 7]                         │
    │    output: node_embeddings [B, N, H]                         │
    ├──────────────────────────────────────────────────────────────┤
    │  ProcessPlannerHead                             [unchanged]  │
    │    output: macro_class / tool / action_face logits           │
    ├──────────────────────────────────────────────────────────────┤
    │  ShapeTransitionHead.ActionEmbedding            [unchanged]  │
    │    output: action_context [B, H]                             │
    │    (SDF prediction skipped when use_sdf_decoder=False)       │
    ├──────────────────────────────────────────────────────────────┤
    │  OctreeDecoder                                  [NEW]        │
    │    input:  node_embeddings, action_context,                  │
    │            octree_centers [B,K,3], octree_depths [B,K]       │
    │    output: occ_logits [B, K]                                 │
    │    → sigmoid → Marching Cubes → 3-D mesh                     │
    └──────────────────────────────────────────────────────────────┘
    """

    AXIS_MODE_INDEXED = 0
    AXIS_MODE_SIMULTANEOUS_5_AXIS = 1

    def __init__(self, config: GraphSdfModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.state_encoder    = StateEncoder(config)
        self.process_planner  = ProcessPlannerHead(config)
        # ShapeTransitionHead is always instantiated so that ActionEmbedding
        # (which lives inside it) is available for producing action_context.
        # The SDF head parts (FiLM blocks, delta heads) are only *called*
        # when config.use_sdf_decoder=True.
        self.shape_transition = ShapeTransitionHead(config)

        self.octree_decoder: Optional[OctreeDecoder] = (
            OctreeDecoder(config) if config.use_octree_decoder else None
        )
        self.sdf_query_decoder: Optional[SdfQueryDecoder] = (
            SdfQueryDecoder(config) if config.use_sdf_query_decoder else None
        )
        self.affected_face_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.affected_delta_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

        self.indexed_ids = torch.tensor(
            [MACRO_CLASS_TO_ID["indexed_rough"], MACRO_CLASS_TO_ID["indexed_finish"]],
            dtype=torch.long,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _derive_axis_from_action(
        self,
        macro_class_id: torch.Tensor,
        action_face_id: torch.Tensor,
        state_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_normals  = state_points[:, :, :, 3:6].mean(dim=2)
        gather_index  = action_face_id[:, None, None].expand(-1, 1, 3)
        pred_axis     = torch.gather(node_normals, 1, gather_index).squeeze(1)
        pred_axis     = F.normalize(pred_axis, dim=-1, eps=1e-8)
        axis_mode     = torch.full_like(
            macro_class_id, self.AXIS_MODE_SIMULTANEOUS_5_AXIS, dtype=torch.long,
        )
        is_indexed = (macro_class_id[:, None] == self.indexed_ids.to(macro_class_id.device)).any(dim=1)
        axis_mode[is_indexed] = self.AXIS_MODE_INDEXED
        return pred_axis, axis_mode

    def _build_action_context(
        self,
        state_embedding: torch.Tensor,
        state_points: torch.Tensor,
        macro_class_id: torch.Tensor,
        tool_choice_id: torch.Tensor,
        action_face_id: torch.Tensor,
        tool_kind_id: Optional[torch.Tensor],
        axis_visible: Optional[torch.Tensor],
        axis_dir: Optional[torch.Tensor],
        tool_diameter_norm: Optional[torch.Tensor],
        tool_radius_norm: Optional[torch.Tensor],
        tool_length_norm: Optional[torch.Tensor],
        holder_diameter_norm: Optional[torch.Tensor],
        holder_radius_norm: Optional[torch.Tensor],
        holder_length_norm: Optional[torch.Tensor],
        node_process_state: Optional[torch.Tensor],
        node_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Calls ActionEmbedding to produce action_context (always runs)."""
        return self.shape_transition.action_embedding(
            node_embeddings=state_embedding,
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            tool_kind_id=tool_kind_id,
            axis_visible=axis_visible,
            axis_dir=axis_dir,
            tool_diameter_norm=tool_diameter_norm,
            tool_radius_norm=tool_radius_norm,
            tool_length_norm=tool_length_norm,
            holder_diameter_norm=holder_diameter_norm,
            holder_radius_norm=holder_radius_norm,
            holder_length_norm=holder_length_norm,
            node_process_state=node_process_state,
            node_mask=node_mask,
        )

    @staticmethod
    def _gather_action_face_vectors(values: torch.Tensor, action_face_id: torch.Tensor) -> torch.Tensor:
        """Gathers one per-face vector per batch item."""
        gather_index = action_face_id[:, None, None].expand(-1, 1, values.shape[-1])
        return torch.gather(values, 1, gather_index).squeeze(1)

    def _build_sdf_query_action_features(
        self,
        state_points: torch.Tensor,
        action_face_id: torch.Tensor,
        query_points: torch.Tensor,
        axis_dir: Optional[torch.Tensor],
        tool_radius_norm: Optional[torch.Tensor],
        tool_length_norm: Optional[torch.Tensor],
        holder_diameter_norm: Optional[torch.Tensor],
        holder_radius_norm: Optional[torch.Tensor],
        holder_length_norm: Optional[torch.Tensor],
        sdf_axis_clearance_before: Optional[torch.Tensor] = None,
        sdf_axis_blocked_before: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Builds query-local cutter geometry features in normalized part units."""
        action_centers = self._gather_action_face_vectors(state_points[..., 0:3].mean(dim=2), action_face_id)
        action_normals = self._gather_action_face_vectors(state_points[..., 3:6].mean(dim=2), action_face_id)
        if axis_dir is None:
            axis = action_normals
        else:
            axis = axis_dir.to(state_points.device).float().reshape(state_points.shape[0], 3)
        axis = F.normalize(axis, dim=-1, eps=1e-6)
        action_normals = F.normalize(action_normals, dim=-1, eps=1e-6)

        rel = query_points - action_centers[:, None, :]
        along_axis = (rel * axis[:, None, :]).sum(dim=-1, keepdim=True)
        radial_vec = rel - along_axis * axis[:, None, :]
        radial_to_axis = radial_vec.norm(dim=-1, keepdim=True)
        if tool_radius_norm is None:
            radius = query_points.new_zeros(query_points.shape[0], 1)
        else:
            radius = tool_radius_norm.to(query_points.device).float().reshape(query_points.shape[0], -1)[:, :1]
        if tool_length_norm is None:
            tool_length = query_points.new_zeros(query_points.shape[0], 1)
        else:
            tool_length = tool_length_norm.to(query_points.device).float().reshape(query_points.shape[0], -1)[:, :1]
        if holder_diameter_norm is None:
            holder_diameter = query_points.new_zeros(query_points.shape[0], 1)
        else:
            holder_diameter = holder_diameter_norm.to(query_points.device).float().reshape(query_points.shape[0], -1)[:, :1]
        if holder_radius_norm is None:
            holder_radius = query_points.new_zeros(query_points.shape[0], 1)
        else:
            holder_radius = holder_radius_norm.to(query_points.device).float().reshape(query_points.shape[0], -1)[:, :1]
        if holder_length_norm is None:
            holder_length = query_points.new_zeros(query_points.shape[0], 1)
        else:
            holder_length = holder_length_norm.to(query_points.device).float().reshape(query_points.shape[0], -1)[:, :1]
        radius = radius[:, None, :].expand(-1, query_points.shape[1], -1)
        tool_length = tool_length[:, None, :].expand(-1, query_points.shape[1], -1)
        holder_diameter = holder_diameter[:, None, :].expand(-1, query_points.shape[1], -1)
        holder_radius = holder_radius[:, None, :].expand(-1, query_points.shape[1], -1)
        holder_length = holder_length[:, None, :].expand(-1, query_points.shape[1], -1)
        normal_axis_dot = (action_normals * axis).sum(dim=-1, keepdim=True)
        normal_axis_dot = normal_axis_dot[:, None, :].expand(-1, query_points.shape[1], -1)
        if sdf_axis_clearance_before is None:
            axis_clearance = query_points.new_zeros(query_points.shape[0], query_points.shape[1], 1)
        else:
            axis_clearance = sdf_axis_clearance_before.to(query_points.device).float().reshape(query_points.shape[0], query_points.shape[1], 1)
        if sdf_axis_blocked_before is None:
            axis_blocked = query_points.new_zeros(query_points.shape[0], query_points.shape[1], 1)
        else:
            axis_blocked = sdf_axis_blocked_before.to(query_points.device).float().reshape(query_points.shape[0], query_points.shape[1], 1)
        return torch.cat(
            [
                along_axis,
                radial_to_axis,
                radial_to_axis - radius,
                radius,
                tool_length,
                holder_diameter,
                holder_radius,
                holder_length,
                normal_axis_dot,
                axis_clearance,
                axis_blocked,
            ],
            dim=-1,
        )

    @staticmethod
    def apply_rollout_tsdf_constraints(
        pred_after: torch.Tensor,
        before_tsdf: torch.Tensor,
        target_tsdf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hard-clamps rollout TSDF so material cannot be added back."""
        out = torch.maximum(pred_after, before_tsdf.to(pred_after.device).float())
        if target_tsdf is not None:
            out = torch.minimum(out, target_tsdf.to(pred_after.device).float())
        return out

    # ── Public API ─────────────────────────────────────────────────────────

    def encode_state(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality:    Optional[torch.Tensor] = None,
        spatial_pos:        Optional[torch.Tensor] = None,
        face_area:          Optional[torch.Tensor] = None,
        node_face_type:     Optional[torch.Tensor] = None,
        node_mask:          Optional[torch.Tensor] = None,
        point_mask:         Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns node embeddings [B, N, H]."""
        return self.state_encoder(
            state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_face_type=node_face_type,
            node_mask=node_mask,
            point_mask=point_mask,
        )

    def forward_process_planner(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality:    Optional[torch.Tensor] = None,
        spatial_pos:        Optional[torch.Tensor] = None,
        face_area:          Optional[torch.Tensor] = None,
        node_face_type:     Optional[torch.Tensor] = None,
        node_mask:          Optional[torch.Tensor] = None,
        point_mask:         Optional[torch.Tensor] = None,
        axis_visible:       Optional[torch.Tensor] = None,
        global_process_state: Optional[torch.Tensor] = None,
        action_face_mask:   Optional[torch.Tensor] = None,
        macro_class_mask:   Optional[torch.Tensor] = None,
        tool_choice_mask:   Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts process skeleton from the current state."""
        _ = axis_visible   # reserved

        state_embedding = self.encode_state(
            state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_face_type=node_face_type,
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

        pred_macro  = outputs["macro_class_logits"].argmax(dim=1)
        pred_face   = outputs["action_face_logits"].argmax(dim=1)
        pred_tool   = outputs["tool_choice_logits"].argmax(dim=1)
        pred_axis, pred_axis_mode = self._derive_axis_from_action(
            pred_macro, pred_face, state_points,
        )
        outputs["pred_macro_class"]   = pred_macro
        outputs["pred_tool_choice"]   = pred_tool
        outputs["pred_action_face"]   = pred_face
        outputs["pred_axis_from_face"] = pred_axis
        outputs["pred_axis_mode"]     = pred_axis_mode
        outputs["state_embedding"]    = state_embedding
        return outputs

    def forward_octree(
        self,
        state_points:    torch.Tensor,
        macro_class_id:  torch.Tensor,
        tool_choice_id:  torch.Tensor,
        action_face_id:  torch.Tensor,
        octree_centers:  torch.Tensor,   # [B, K, 3]
        octree_depths:   torch.Tensor,   # [B, K]
        octree_occ_before: Optional[torch.Tensor] = None,  # [B, K]
        tool_kind_id:    Optional[torch.Tensor] = None,
        axis_visible:         Optional[torch.Tensor] = None,
        axis_dir:             Optional[torch.Tensor] = None,
        tool_diameter_norm:   Optional[torch.Tensor] = None,
        tool_radius_norm:     Optional[torch.Tensor] = None,
        tool_length_norm:     Optional[torch.Tensor] = None,
        holder_diameter_norm: Optional[torch.Tensor] = None,
        holder_radius_norm:   Optional[torch.Tensor] = None,
        holder_length_norm:   Optional[torch.Tensor] = None,
        node_process_state:   Optional[torch.Tensor] = None,
        node_centrality:      Optional[torch.Tensor] = None,
        spatial_pos:          Optional[torch.Tensor] = None,
        face_area:            Optional[torch.Tensor] = None,
        node_face_type:       Optional[torch.Tensor] = None,
        node_mask:            Optional[torch.Tensor] = None,
        point_mask:           Optional[torch.Tensor] = None,
        state_embedding:      Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts occupancy at every octree node.

        Returns dict with keys:
            occ_logits     [B, K]  – raw BCE logits
            action_context [B, H]
            state_embedding [B, N, H]
        """
        if self.octree_decoder is None:
            raise RuntimeError("OctreeDecoder is disabled (use_octree_decoder=False).")

        if state_embedding is None:
            state_embedding = self.encode_state(
                state_points,
                node_process_state=node_process_state,
                node_centrality=node_centrality,
                spatial_pos=spatial_pos,
                face_area=face_area,
                node_face_type=node_face_type,
                node_mask=node_mask,
                point_mask=point_mask,
            )

        action_out     = self._build_action_context(
            state_embedding=state_embedding,
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            tool_kind_id=tool_kind_id,
            axis_visible=axis_visible,
            axis_dir=axis_dir,
            tool_diameter_norm=tool_diameter_norm,
            tool_radius_norm=tool_radius_norm,
            tool_length_norm=tool_length_norm,
            holder_diameter_norm=holder_diameter_norm,
            holder_radius_norm=holder_radius_norm,
            holder_length_norm=holder_length_norm,
            node_process_state=node_process_state,
            node_mask=node_mask,
        )
        action_context = action_out["action_context"]

        octree_pred = self.octree_decoder.forward_outputs(
            node_embeddings=state_embedding,
            action_context=action_context,
            octree_centers=octree_centers,
            octree_depths=octree_depths,
            octree_occ_before=octree_occ_before,
            node_mask=node_mask,
        )
        return {
            "occ_logits":      octree_pred["occ_logits"],
            "tsdf":            octree_pred["tsdf"],
            "action_context":  action_context,
            "state_embedding": state_embedding,
        }

    def forward_sdf_query(
        self,
        state_points: torch.Tensor,
        macro_class_id: torch.Tensor,
        tool_choice_id: torch.Tensor,
        action_face_id: torch.Tensor,
        sdf_query_points: torch.Tensor,
        sdf_query_state: Optional[torch.Tensor] = None,
        tool_kind_id: Optional[torch.Tensor] = None,
        axis_visible: Optional[torch.Tensor] = None,
        axis_dir: Optional[torch.Tensor] = None,
        tool_diameter_norm: Optional[torch.Tensor] = None,
        tool_radius_norm: Optional[torch.Tensor] = None,
        tool_length_norm: Optional[torch.Tensor] = None,
        holder_diameter_norm: Optional[torch.Tensor] = None,
        holder_radius_norm: Optional[torch.Tensor] = None,
        holder_length_norm: Optional[torch.Tensor] = None,
        sdf_axis_clearance_before: Optional[torch.Tensor] = None,
        sdf_axis_blocked_before: Optional[torch.Tensor] = None,
        rollout_before_tsdf: Optional[torch.Tensor] = None,
        rollout_target_tsdf: Optional[torch.Tensor] = None,
        apply_rollout_clamp: bool = False,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        face_area: Optional[torch.Tensor] = None,
        node_face_type: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        state_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts after-operation TSDF at arbitrary query points."""
        if self.sdf_query_decoder is None:
            raise RuntimeError("SdfQueryDecoder is disabled (use_sdf_query_decoder=False).")
        if state_embedding is None:
            state_embedding = self.encode_state(
                state_points,
                node_process_state=node_process_state,
                node_centrality=node_centrality,
                spatial_pos=spatial_pos,
                face_area=face_area,
                node_face_type=node_face_type,
                node_mask=node_mask,
                point_mask=point_mask,
            )
        action_out = self._build_action_context(
            state_embedding=state_embedding,
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            tool_kind_id=tool_kind_id,
            axis_visible=axis_visible,
            axis_dir=axis_dir,
            tool_diameter_norm=tool_diameter_norm,
            tool_radius_norm=tool_radius_norm,
            tool_length_norm=tool_length_norm,
            holder_diameter_norm=holder_diameter_norm,
            holder_radius_norm=holder_radius_norm,
            holder_length_norm=holder_length_norm,
            node_process_state=node_process_state,
            node_mask=node_mask,
        )
        query_action_features = self._build_sdf_query_action_features(
            state_points=state_points,
            action_face_id=action_face_id,
            query_points=sdf_query_points,
            axis_dir=axis_dir,
            tool_radius_norm=tool_radius_norm,
            tool_length_norm=tool_length_norm,
            holder_diameter_norm=holder_diameter_norm,
            holder_radius_norm=holder_radius_norm,
            holder_length_norm=holder_length_norm,
            sdf_axis_clearance_before=sdf_axis_clearance_before,
            sdf_axis_blocked_before=sdf_axis_blocked_before,
        )
        tsdf = self.sdf_query_decoder(
            node_embeddings=state_embedding,
            action_context=action_out["action_context"],
            query_points=sdf_query_points,
            query_state=sdf_query_state,
            query_action_features=query_action_features,
            node_mask=node_mask,
        )
        raw_tsdf = tsdf
        if apply_rollout_clamp and rollout_before_tsdf is not None:
            tsdf = self.apply_rollout_tsdf_constraints(
                pred_after=tsdf,
                before_tsdf=rollout_before_tsdf,
                target_tsdf=rollout_target_tsdf,
            )
        return {
            "sdf_tsdf": tsdf,
            "sdf_tsdf_raw": raw_tsdf,
            "action_context": action_out["action_context"],
            "state_embedding": state_embedding,
        }

    def forward_affected_faces(
        self,
        state_points: torch.Tensor,
        macro_class_id: torch.Tensor,
        tool_choice_id: torch.Tensor,
        action_face_id: torch.Tensor,
        tool_kind_id: Optional[torch.Tensor] = None,
        axis_visible: Optional[torch.Tensor] = None,
        axis_dir: Optional[torch.Tensor] = None,
        tool_diameter_norm: Optional[torch.Tensor] = None,
        tool_radius_norm: Optional[torch.Tensor] = None,
        tool_length_norm: Optional[torch.Tensor] = None,
        holder_diameter_norm: Optional[torch.Tensor] = None,
        holder_radius_norm: Optional[torch.Tensor] = None,
        holder_length_norm: Optional[torch.Tensor] = None,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality: Optional[torch.Tensor] = None,
        spatial_pos: Optional[torch.Tensor] = None,
        face_area: Optional[torch.Tensor] = None,
        node_face_type: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        state_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predicts which graph nodes/faces are affected by the GT action."""
        if state_embedding is None:
            state_embedding = self.encode_state(
                state_points,
                node_process_state=node_process_state,
                node_centrality=node_centrality,
                spatial_pos=spatial_pos,
                face_area=face_area,
                node_face_type=node_face_type,
                node_mask=node_mask,
                point_mask=point_mask,
            )
        action_out = self._build_action_context(
            state_embedding=state_embedding,
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            tool_kind_id=tool_kind_id,
            axis_visible=axis_visible,
            axis_dir=axis_dir,
            tool_diameter_norm=tool_diameter_norm,
            tool_radius_norm=tool_radius_norm,
            tool_length_norm=tool_length_norm,
            holder_diameter_norm=holder_diameter_norm,
            holder_radius_norm=holder_radius_norm,
            holder_length_norm=holder_length_norm,
            node_process_state=node_process_state,
            node_mask=node_mask,
        )
        action_context = action_out["action_context"]
        action_per_node = action_context[:, None, :].expand(-1, state_embedding.shape[1], -1)
        features = torch.cat([state_embedding, action_per_node], dim=-1)
        return {
            "affected_logits": self.affected_face_head(features).squeeze(-1),
            "affected_delta": F.relu(self.affected_delta_head(features).squeeze(-1)),
            "action_context": action_context,
            "state_embedding": state_embedding,
        }

    def extract_mesh_from_state(
        self,
        state_points:   torch.Tensor,
        macro_class_id: torch.Tensor,
        tool_choice_id: torch.Tensor,
        action_face_id: torch.Tensor,
        tool_kind_id: Optional[torch.Tensor] = None,
        axis_visible:   Optional[torch.Tensor] = None,
        axis_dir:       Optional[torch.Tensor] = None,
        tool_diameter_norm: Optional[torch.Tensor] = None,
        tool_radius_norm: Optional[torch.Tensor] = None,
        tool_length_norm: Optional[torch.Tensor] = None,
        holder_diameter_norm: Optional[torch.Tensor] = None,
        holder_radius_norm: Optional[torch.Tensor] = None,
        holder_length_norm: Optional[torch.Tensor] = None,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality:    Optional[torch.Tensor] = None,
        spatial_pos:        Optional[torch.Tensor] = None,
        face_area:          Optional[torch.Tensor] = None,
        node_face_type:     Optional[torch.Tensor] = None,
        node_mask:          Optional[torch.Tensor] = None,
        point_mask:         Optional[torch.Tensor] = None,
        adaptive: bool = True,
        grid_resolution: int = 64,
        occupancy_threshold: float = 0.5,
        bbox_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bbox_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """Full pipeline: encode → build action context → Marching Cubes mesh.

        Requires batch_size=1.  Returns (vertices, faces) or None.
        """
        assert state_points.shape[0] == 1, "extract_mesh_from_state requires batch_size=1"
        if self.octree_decoder is None:
            raise RuntimeError("OctreeDecoder is disabled.")

        state_embedding = self.encode_state(
            state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_face_type=node_face_type,
            node_mask=node_mask,
            point_mask=point_mask,
        )
        action_out     = self._build_action_context(
            state_embedding=state_embedding,
            state_points=state_points,
            macro_class_id=macro_class_id,
            tool_choice_id=tool_choice_id,
            action_face_id=action_face_id,
            tool_kind_id=tool_kind_id,
            axis_visible=axis_visible,
            axis_dir=axis_dir,
            tool_diameter_norm=tool_diameter_norm,
            tool_radius_norm=tool_radius_norm,
            tool_length_norm=tool_length_norm,
            holder_diameter_norm=holder_diameter_norm,
            holder_radius_norm=holder_radius_norm,
            holder_length_norm=holder_length_norm,
            node_process_state=node_process_state,
            node_mask=node_mask,
        )
        action_context = action_out["action_context"]

        if adaptive:
            return self.octree_decoder.extract_mesh_adaptive(
                node_embeddings=state_embedding,
                action_context=action_context,
                node_mask=node_mask,
                coarse_depth=self.config.octree_coarse_depth,
                occupancy_threshold=occupancy_threshold,
                bbox_min=bbox_min, bbox_max=bbox_max,
            )
        else:
            return self.octree_decoder.extract_mesh(
                node_embeddings=state_embedding,
                action_context=action_context,
                node_mask=node_mask,
                grid_resolution=grid_resolution,
                occupancy_threshold=occupancy_threshold,
                bbox_min=bbox_min, bbox_max=bbox_max,
            )

    def forward(
        self,
        state_points: torch.Tensor,
        node_process_state: Optional[torch.Tensor] = None,
        node_centrality:    Optional[torch.Tensor] = None,
        spatial_pos:        Optional[torch.Tensor] = None,
        face_area:          Optional[torch.Tensor] = None,
        node_face_type:     Optional[torch.Tensor] = None,
        node_mask:          Optional[torch.Tensor] = None,
        point_mask:         Optional[torch.Tensor] = None,
        axis_visible:       Optional[torch.Tensor] = None,
        global_process_state: Optional[torch.Tensor] = None,
        action_face_mask:   Optional[torch.Tensor] = None,
        macro_class_mask:   Optional[torch.Tensor] = None,
        tool_choice_mask:   Optional[torch.Tensor] = None,
        target_macro_class: Optional[torch.Tensor] = None,
        target_tool_choice: Optional[torch.Tensor] = None,
        target_tool_kind: Optional[torch.Tensor] = None,
        target_action_face: Optional[torch.Tensor] = None,
        use_teacher_forcing_action: bool = False,
        run_transition: bool = True,
        # Octree decoder inputs (None → skipped)
        octree_centers: Optional[torch.Tensor] = None,   # [B, K, 3]
        octree_depths:  Optional[torch.Tensor] = None,   # [B, K]
        octree_occ_before: Optional[torch.Tensor] = None, # [B, K]
        axis_dir: Optional[torch.Tensor] = None,
        tool_diameter_norm: Optional[torch.Tensor] = None,
        tool_radius_norm: Optional[torch.Tensor] = None,
        tool_length_norm: Optional[torch.Tensor] = None,
        holder_diameter_norm: Optional[torch.Tensor] = None,
        holder_radius_norm: Optional[torch.Tensor] = None,
        holder_length_norm: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run planner; optionally run octree decoder.

        When ``octree_centers`` and ``octree_depths`` are provided and
        ``config.use_octree_decoder=True``, adds ``occ_logits [B,K]``.
        """
        outputs = self.forward_process_planner(
            state_points=state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_face_type=node_face_type,
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
                    "Teacher forcing requires target_macro_class, target_tool_choice, target_action_face"
                )
            macro_class_id = target_macro_class
            tool_choice_id = target_tool_choice
            tool_kind_id = target_tool_kind
            action_face_id = target_action_face
        else:
            macro_class_id = outputs["pred_macro_class"]
            tool_choice_id = outputs["pred_tool_choice"]
            tool_kind_id = None
            action_face_id = outputs["pred_action_face"]

        # ── Optional: per-face SDF decoder (legacy) ───────────────────────
        if self.config.use_sdf_decoder:
            sdf_out = self.shape_transition(
                node_embeddings=outputs["state_embedding"],
                state_points=state_points,
                macro_class_id=macro_class_id,
                tool_choice_id=tool_choice_id,
                tool_kind_id=tool_kind_id,
                action_face_id=action_face_id,
                axis_visible=axis_visible,
                tool_diameter_norm=tool_diameter_norm,
                tool_radius_norm=tool_radius_norm,
                tool_length_norm=tool_length_norm,
                holder_diameter_norm=holder_diameter_norm,
                holder_radius_norm=holder_radius_norm,
                holder_length_norm=holder_length_norm,
                node_process_state=node_process_state,
                node_mask=node_mask,
            )
            outputs.update(sdf_out)

        # ── Optional: octree decoder ───────────────────────────────────────
        if (
            octree_centers is not None
            and octree_depths is not None
            and self.octree_decoder is not None
        ):
            oct_out = self.forward_octree(
                state_points=state_points,
                macro_class_id=macro_class_id,
                tool_choice_id=tool_choice_id,
                tool_kind_id=tool_kind_id,
                action_face_id=action_face_id,
                octree_centers=octree_centers,
                octree_depths=octree_depths,
                octree_occ_before=octree_occ_before,
                axis_visible=axis_visible,
                axis_dir=axis_dir,
                tool_diameter_norm=tool_diameter_norm,
                tool_radius_norm=tool_radius_norm,
                tool_length_norm=tool_length_norm,
                holder_diameter_norm=holder_diameter_norm,
                holder_radius_norm=holder_radius_norm,
                holder_length_norm=holder_length_norm,
                node_process_state=node_process_state,
                node_face_type=node_face_type,
                node_mask=node_mask,
                point_mask=point_mask,
                state_embedding=outputs["state_embedding"],
            )
            outputs["occ_logits"]     = oct_out["occ_logits"]
            outputs["tsdf"]           = oct_out.get("tsdf")
            outputs["action_context"] = oct_out["action_context"]

        return outputs
