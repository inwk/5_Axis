# Graph-SDF Process Skeleton Planner

This package implements a planner-only framework:

1. `StateEncoder`: Encodes local graph state `[B, N, P, F] -> [B, N, H]`.
2. `ProcessPlannerHead`: Predicts process skeleton logits from the current state.

## Main Files

- `config.py`: Hyperparameter dataclass.
- `dataset.py`: Parquet dataset loader for the planner schema.
- `state_encoder.py`: PointNet + graph transformer state encoder.
- `process_planner.py`: Skeleton planner head.
- `model.py`: Unified model wrapper.
- `losses.py`: Planner loss functions.
- `training.py`: Planner train/validation utilities.

## Expected Batch Keys

- `state_points`: `[B, N, P, F]`
- optional `node_process_state`: `[B, N, 2]` where channels are `rough_done`, `finish_ready`
- optional `global_process_state`: `[B, 9]` where features are
  `prev_macro_onehot(7) + rough_ratio + finish_ratio`
- `macro_class_id`: `[B]`
- `target_node_id`: `[B]`
- `tool_choice_id`: `[B]`
- optional `target_node_valid`, `tool_choice_valid`, `node_mask`, `point_mask`
- optional `macro_class_mask`, `tool_choice_mask`, `target_node_mask` (all invalid-masks, `True` means masked out)

## Planner Outputs

- `macro_class_logits`: `[B, C_macro]`
- `target_node_logits`: `[B, N]`
- `tool_choice_logits`: `[B, C_tool]`
- `pred_target_node`: `[B]`
- `pred_axis_from_target`: `[B, 3]` computed from selected face normal

## Dataset Contract

The model is designed for one executable NX operation per row.

- Roughing rows are global operations but still provide `target_node_id` as an axis-anchor face.
- Finishing rows are local operations and provide the primary target face as `target_node_id`.
- `tool_choice_id` is a single discrete id from the configured tool library
  (tool type and diameter are not predicted separately).
- No continuous process-parameter regression is used.
