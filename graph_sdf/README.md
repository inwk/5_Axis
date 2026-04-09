# Graph-SDF Process Skeleton Planner

This package implements a joint planning + action-conditioned transition draft:

1. `StateEncoder`: Encodes local graph state `[B, N, P, F] -> [B, N, H]`.
2. `ProcessPlannerHead`: Predicts process skeleton logits from the current graph state.
3. `ShapeTransitionHead`: Uses teacher-forced or predicted actions to decode one-step next state.

Current decoding order: `macro process + target node + tool choice + strategy`.
Axis is mapped deterministically from action:
`3-axis -> +Z`, `3+2-axis -> target-face normal`, `simultaneous-5-axis -> strategy law`.

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
- optional `global_process_state`: `[B, 9]` where features are
  `prev_macro_onehot(7) + rough_ratio + finish_ratio`
- optional `node_centrality`: `[B, N]`
- optional `spatial_pos`: `[B, N, N]`
- optional `face_area`: `[B, N, 1]`
- `macro_class_id`: `[B]`
- `target_node_id`: `[B]`
- `tool_choice_id`: `[B]`
- optional `strategy_id`: `[B]`
- `next_node_sdf`: `[B, N]` coarse face-level next-state supervision
- optional `next_point_sdf`: `[B, N, P]` full sampled next-state supervision
- optional `target_node_valid`, `tool_choice_valid`, `node_mask`, `point_mask`
- optional `macro_class_mask`, `tool_choice_mask`, `strategy_mask`, `target_node_mask`
  (all invalid-masks, `True` means masked out)

## Planner Outputs

- `macro_class_logits`: `[B, C_macro]`
- `target_node_logits`: `[B, N]`
- `tool_choice_logits`: `[B, C_tool]`
- `strategy_logits`: `[B, C_strategy]`
- `pred_target_node`: `[B]`
- `pred_axis_from_target`: `[B, 3]` computed from selected face normal
- `pred_axis_mode`: `[B]` (`0=3-axis`, `1=3+2-axis`, `2=simultaneous-5-axis`)
- `pred_next_node_sdf`: `[B, N]` one-step coarse next-state prediction
- `pred_next_point_sdf`: `[B, N, P]` one-step sampled next-state prediction
- `pred_changed_logits`: `[B, N]` optional changed-face logits

## Dataset Contract

The model is designed for one executable NX operation per row.

- Roughing rows are global operations but still provide `target_node_id` as an axis-anchor face.
- Finishing rows are local operations and provide the primary target face as `target_node_id`.
- `tool_choice_id` is a single discrete id from the configured tool library
  (tool type and diameter are not predicted separately).
- No continuous process-parameter regression is used.

This draft supports both coarse `[B, N]` and sampled `[B, N, P]` transition supervision.
