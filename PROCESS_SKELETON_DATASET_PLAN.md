# Process Skeleton Dataset Plan

This note captures the dataset contract for the current planner-only model.

## Row Definition

Each training row represents exactly one executed NX operation.

## Required Supervision

Each row stores:

- `state_points`
  `[512, 100, 7] = xyz(3) + normal(3) + normalized_residual_sdf(1)`
- `node_process_state`
  `[512, 2] = rough_done(1) + finish_ready(1)`
- `global_process_state`
  `[9] = prev_macro_onehot(7) + rough_ratio + finish_ratio`
- `macro_class_id`
  `0..6` for:
  `3-axis rough`, `3-axis finish`, `3+2 rough`, `3+2 finish`, `5-axis point`, `5-axis flank`, `STOP`
- `target_node_id`
  face index used as local target or axis-anchor
- `tool_choice_id`
  discrete tool-library id (tool type + diameter together)
- `target_node_valid`
  usually `1`
- `tool_choice_valid`
  usually `1`
- `macro_class_mask`
  `[7]` invalid-mask for macro choices
- `tool_choice_mask`
  `[len(tool_library)]` invalid-mask for tool choices
- `target_node_mask`
  `[512]` invalid-mask for node selection

## Optional Diagnostics

- `next_node_sdf`
- `node_mask`, `point_mask`
- `axis_visible_512`
- `state_node_sdf_raw_512`, `next_node_sdf_raw_512`
- `rough_done_mask_512`, `finish_ready_mask_512`

## Labeling Rules

- Per decision state, generate a bounded set of executable action candidates.
- Score candidates with short-horizon lookahead (default depth 2), then execute/store the best first action.
- Roughing operations:
  `target_node_id` is selected as an axis-anchor face whose normal aligns with the chosen tool axis.
- Finishing operations:
  `target_node_id` is selected from the largest positive residual drop among valid visible nodes.
