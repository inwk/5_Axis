# 5_Axis Codebase (Clean Core)

This workspace is now organized around a clean **Graph-SDF process skeleton planning core**.

## Active Core

- `graph_sdf/`: main model package
  - `state_encoder.py`: PointNet + graph transformer state encoder
  - `process_planner.py`: operation-level skeleton prediction head
  - `dataset.py`: parquet loader for planner-schema rows
  - `schema.py`: shared class and tool-library constants
  - `model.py`: unified planner model wrapper
  - `losses.py`, `training.py`: planner losses and train-step helpers
- `train_graph_sdf.py`: planner training runner skeleton
- `smoke_test_graph_sdf.py`: quick planner smoke test

## Data Generation (optional)

- `collect_axis_dataset.py`: NX rollout collector that now emits one row per executed operation
- `run_parallel_axis_collection.py`
- `graph_face_compression.py`
- `PROCESS_SKELETON_DATASET_PLAN.md`: dataset contract for the planner schema
- `CAM/`
  - `session.py`: NX part open + CAM session bootstrap
  - `geometry.py`: workpiece/IPW geometry creation
  - `operations.py`: machining operation builders
  - `measurements.py`: deviation/volume/point sampling
  - `utils.py`: graph/visibility/tool/IPW helper utilities

## Removed During Cleanup

- Broken runner: `run_axis_dataset.py`
- Parquet utility scripts not used in model training path:
  - `clean.py`
  - `merge_parquet.py`
  - `plot_parquet.py`
  - `read_parquet.py`
  - `update_parquet.py`
- Removed unused CAM legacy scripts:
  - `CAM/Graph2SDF.py`
  - `CAM/Graph2SDF_loop.py`
  - `CAM/main.py`
  - `CAM/main_loop.py`
  - `CAM/main_loop_v2.py`
  - `CAM/main_test.py`
  - `CAM/main_v2.py`
  - `CAM/mesh2sdf.py`
  - `CAM/postprocess.py`
  - `CAM/pyrender_wrapper.py`
  - `CAM/scan.py`
  - `CAM/stl_viewer.py`
  - `CAM/surface_point_cloud.py`
  - `CAM/shaders/`
- Merged duplicate CAM utility modules:
  - `CAM/utils3.py` + `CAM/utils.py` -> `CAM/utils.py`

## Notes

- Legacy experimental code is still under `old/` and can be migrated or deleted in a second cleanup pass.
- `smoke_test_graph_sdf.py` requires `torch` to run.
- The current active schema is planner-oriented: one parquet row corresponds to one executed NX operation.
