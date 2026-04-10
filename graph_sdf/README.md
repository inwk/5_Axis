# Graph-SDF Process Skeleton Planner

This package implements a joint planning + action-conditioned transition draft:

1. `StateEncoder`: Encodes local graph state `[B, N, P, F] -> [B, N, H]`.
2. `ProcessPlannerHead`: Predicts process skeleton logits from the current graph state.
3. `ShapeTransitionHead`: Uses teacher-forced or predicted actions to decode one-step next state.

Current decoding order: `macro process + action face + tool choice`.
Axis is mapped deterministically from the selected face normal.

## Main Files

- `config.py`: Hyperparameter dataclass.
- `dataset.py`: Parquet dataset loader for the planner schema.
- `state_encoder.py`: PointNet + graph transformer state encoder.
- `process_planner.py`: Skeleton planner head.
- `shape_transition.py`: Action-conditioned one-step transition decoder.
- `model.py`: Unified model wrapper.
- `losses.py`: Planner and transition loss functions.
- `training.py`: Joint train/validation utilities.

## Expected Batch Keys

- `state_points`: `[B, N, P, F]`
- optional `node_process_state`: `[B, N, 2]` where channels are `rough_done`, `finish_ready`
- optional `global_process_state`: `[B, 11]` where features are
  `prev_macro_onehot(5) + rough_ratio + finish_ratio + bbox_extent_xyz + log_ref_scale`
- optional `node_centrality`: `[B, N]`
- optional `spatial_pos`: `[B, N, N]`
- optional `face_area`: `[B, N, 1]`
- `macro_class_id`: `[B]`
- `action_face_id`: `[B]`
- `tool_choice_id`: `[B]`
- `next_node_sdf`: `[B, N]` coarse face-level next-state supervision
- optional `next_point_sdf`: `[B, N, P]` full sampled next-state supervision
- optional `action_face_valid`, `tool_choice_valid`, `node_mask`, `point_mask`
- optional `macro_class_mask`, `tool_choice_mask`, `action_face_mask`
  (all invalid-masks, `True` means masked out)

## Planner Outputs

- `macro_class_logits`: `[B, C_macro]`
- `action_face_logits`: `[B, N]`
- `tool_choice_logits`: `[B, C_tool]`
- `pred_action_face`: `[B]`
- `pred_axis_from_face`: `[B, 3]` computed from the selected face normal
- `pred_axis_mode`: `[B]` (`0=indexed`, `1=simultaneous-5-axis`)
- `pred_next_node_sdf`: `[B, N]` one-step coarse next-state prediction
- `pred_next_point_sdf`: `[B, N, P]` one-step sampled next-state prediction
- `pred_changed_logits`: `[B, N]` optional changed-face logits

## Dataset Contract

The model is designed for one executable NX operation per row.

- Roughing rows provide `action_face_id` as an indexed-setup anchor face.
- Finishing rows provide `action_face_id` as the local target face.
- `tool_choice_id` is a single discrete id from the configured tool library
  (tool type and diameter are not predicted separately).
- No continuous process-parameter regression is used.

This draft supports both coarse `[B, N]` and sampled `[B, N, P]` transition supervision.
