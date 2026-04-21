# Graph-SDF Process Skeleton Planner

This package implements a planner plus an action-conditioned octree transition model.

1. `StateEncoder`: encodes the current face graph state `[B, N, P, F] -> [B, N, H]`.
2. `ProcessPlannerHead`: predicts process type, action face, and tool choice.
3. `OctreeDecoder`: predicts next-state material occupancy at adaptive octree leaf centers.

Current action order is `macro process + action face + tool choice`.  The axis is mapped deterministically from the selected face normal where needed.

## Main Files

- `config.py`: hyperparameter dataclass.
- `dataset.py`: parquet dataset loader for planner and octree transition schema.
- `state_encoder.py`: PointNet + graph transformer state encoder.
- `process_planner.py`: skeleton planner head.
- `octree_decoder.py`: action-conditioned adaptive octree occupancy decoder.
- `model.py`: unified model wrapper.
- `losses.py`: planner CE and occupancy BCE losses.
- `training.py`: planner + octree train/validation utilities.

## Expected Batch Keys

- `state_points`: `[B, N, P, F]`; channel 6 remains the current-state SDF/residual input feature.
- optional `node_process_state`: `[B, N, 2]` where channels are `rough_done`, `finish_ready`.
- optional `global_process_state`: `[B, 11]` where features are `prev_macro_onehot(5) + cumulative_removed_ratio + remaining_volume_ratio + bbox_extent_xyz + log_ref_scale`.
- optional `node_centrality`: `[B, N]`.
- optional `spatial_pos`: `[B, N, N]`.
- optional `face_area`: `[B, N, 1]`.
- optional `node_face_type`: `[B, N]` discrete NX surface/face type ids.
- `macro_class_id`: `[B]`.
- `action_face_id`: `[B]`.
- `tool_choice_id`: `[B]`.
- `octree_centers`: `[B, K, 3]` normalized adaptive octree leaf centers.
- `octree_depths`: `[B, K]` integer octree level per leaf.
- optional `octree_occ_labels_before`: `[B, K]` current-state occupancy at the same octree cells; used as decoder input and monotonicity supervision.
- `octree_occ_labels`: `[B, K]` next-state occupancy labels, `1=inside material`, `0=empty`.
- optional `octree_bbox_min`, `octree_bbox_max`: normalized octree sampling bounds for mesh extraction.
- optional `action_face_valid`, `tool_choice_valid`, `node_mask`, `point_mask`.
- optional `macro_class_mask`, `tool_choice_mask`, `action_face_mask`; these are invalid masks, `True` means masked out.

## Outputs

- `macro_class_logits`: `[B, C_macro]`.
- `action_face_logits`: `[B, N]`.
- `tool_choice_logits`: `[B, C_tool]`.
- `pred_action_face`: `[B]`.
- `pred_axis_from_face`: `[B, 3]` computed from the selected face normal.
- `pred_axis_mode`: `[B]` (`0=indexed`, `1=simultaneous-5-axis`).
- `occ_logits`: `[B, K]` next-state occupancy logits when octree inputs are provided.

## Dataset Contract

The dataset stores one executable NX operation per row.

- Roughing rows provide `action_face_id` as an indexed-setup anchor face.
- Finishing rows provide `action_face_id` as the local target face.
- `tool_choice_id` is a single discrete id from the configured tool library.
- No continuous process-parameter regression is used.
- Next-state geometry supervision is octree occupancy only; SDF is used as current-state input, not as a transition output target.
