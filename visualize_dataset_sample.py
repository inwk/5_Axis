r"""Visualize one collected CAM transition row as CAD-proxy points and SDF fields.

This script does not require NX. It visualizes the stored face sample points from
the parquet dataset:

    state_points[..., 0:3] = normalized xyz
    state_points[..., 3:6] = face normal
    state_points[..., 6]   = normalized current SDF/residual

Edit the module-level configuration below, then run:
    python visualize_dataset_sample.py
"""

from __future__ import annotations

from datetime import datetime
import glob
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv


# Edit these values directly when running from VSCode/debug mode.
RUN_DIR = r""
PARQUET_PATH = r"C:\Users\inwoo\Desktop\3dDataset1760_seed0_process_skeleton_dataset.parquet"
ROW_INDEX = 0
CHOSEN_ONLY = False
DENORMALIZE = True
SCREENSHOT_PATH = r""
SHOW_WINDOW = True
TOP_CHANGED_FACES = 10
STOCK_OFFSET_SCALE = 0.25
CHANGED_DELTA_REL_TOL = 0.10
CHANGED_DELTA_ABS_TOL = 0.0
MAX_REMOVAL_VECTORS = 500
REMOVAL_VECTOR_TOP_FACES = 5
STOCK_PROXY_CHANGED_ONLY = True
SDF_PANEL_USE_STOCK_PROXY = False
SDF_PANEL_CHANGED_POINT_SIZE = 11
SDF_PANEL_BASE_POINT_SIZE = 4
SHOW_TARGET_MESH = True
TARGET_MESH_PATH = r""
TARGET_MESH_OPACITY = 0.18
TARGET_MESH_CONTEXT_OPACITY = 0.20
TARGET_MESH_SHOW_EDGES = True
BATCH_VISUALIZE_ALL_ROWS = True
BATCH_OUTPUT_DIR = r""
BATCH_MAX_ROWS = 0  # 0 means all rows after CHOSEN_ONLY filtering.


def _find_parquet(run_dir: str) -> str:
    matches = sorted(glob.glob(os.path.join(run_dir, "*_process_skeleton_dataset.parquet")))
    if not matches:
        matches = sorted(glob.glob(os.path.join(run_dir, "*.parquet")))
    if not matches:
        raise FileNotFoundError(f"No parquet file found in: {run_dir}")
    return matches[0]


def _array(value: Any, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    def _flatten_nested(x: Any) -> np.ndarray:
        chunks: list[np.ndarray] = []
        stack = [x]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            if isinstance(item, np.ndarray):
                if item.dtype == object:
                    stack.extend(item.ravel()[::-1].tolist())
                else:
                    chunks.append(item.astype(dtype, copy=False).reshape(-1))
            elif isinstance(item, (list, tuple)):
                stack.extend(reversed(item))
            else:
                chunks.append(np.asarray([item], dtype=dtype))
        if not chunks:
            return np.asarray([], dtype=dtype)
        return np.concatenate(chunks).astype(dtype, copy=False)

    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        arr = _flatten_nested(value)
    if shape is not None:
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"Expected {shape} ({expected} values), got {arr.shape} ({arr.size} values)")
        arr = arr.reshape(shape)
    return arr


def _row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row.index:
        return default
    value = row[key]
    if isinstance(value, float) and np.isnan(value):
        return default
    return value


def _row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = _row_value(row, key, default)
    return float(value)


def _row_int(row: pd.Series, key: str, default: int = -1) -> int:
    value = _row_value(row, key, default)
    return int(value)


def _row_vector(row: pd.Series, key: str, length: int, default: float = 0.0) -> np.ndarray:
    value = _row_value(row, key, [default] * length)
    return _array(value, np.float32, (length,))


def _load_meta(run_dir: str | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_row(df: pd.DataFrame, row_index: int, chosen_only: bool) -> pd.Series:
    if chosen_only:
        if "is_chosen" not in df.columns:
            raise KeyError("--chosen-only was requested, but parquet has no is_chosen column")
        df = df[df["is_chosen"].astype(int) == 1].reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows with is_chosen=1")
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row index {row_index} out of range for {len(df)} rows")
    return df.iloc[int(row_index)]


def _filter_rows(df: pd.DataFrame, chosen_only: bool) -> pd.DataFrame:
    if not chosen_only:
        return df.reset_index(drop=False).rename(columns={"index": "source_row_index"})
    if "is_chosen" not in df.columns:
        raise KeyError("CHOSEN_ONLY=True was requested, but parquet has no is_chosen column")
    return (
        df[df["is_chosen"].astype(int) == 1]
        .reset_index(drop=False)
        .rename(columns={"index": "source_row_index"})
    )


def _get_scale_and_center(row: pd.Series, meta: dict[str, Any]) -> tuple[float, np.ndarray]:
    scale = row.get("normalization_scale", None)
    center = row.get("normalization_center_xyz", None)

    if scale is None and meta:
        scale = meta.get("note", {}).get("normalization", {}).get("reference_scale", 1.0)
    if center is None and meta:
        center = meta.get("note", {}).get("normalization", {}).get("center_xyz", [0.0, 0.0, 0.0])

    scale_f = float(scale) if scale is not None else 1.0
    center_arr = _array(center if center is not None else [0.0, 0.0, 0.0], np.float32, (3,))
    return scale_f, center_arr


def _resolve_target_mesh_path(row: pd.Series, meta: dict[str, Any]) -> str | None:
    base_dirs = [
        meta.get("_run_dir", ""),
        meta.get("_parquet_dir", ""),
        RUN_DIR,
    ]
    part_name = str(_row_value(row, "part_name", "") or "")
    candidates = [
        TARGET_MESH_PATH,
        _row_value(row, "target_body_mesh_path", ""),
        meta.get("target_body_mesh_path", ""),
        meta.get("note", {}).get("target_body_mesh_path", "") if meta else "",
    ]

    for base_dir in base_dirs:
        if not base_dir:
            continue
        base_dir = os.path.abspath(os.path.expanduser(str(base_dir)))
        candidates.append(os.path.join(base_dir, "target_body.obj"))
        if part_name:
            candidates.extend(sorted(glob.glob(os.path.join(base_dir, f"{part_name}*target*.obj"))))
            candidates.extend(sorted(glob.glob(os.path.join(base_dir, f"{part_name}*.obj"))))
        candidates.extend(sorted(glob.glob(os.path.join(base_dir, "target*.obj"))))
        obj_matches = sorted(glob.glob(os.path.join(base_dir, "*.obj")))
        if len(obj_matches) == 1:
            candidates.extend(obj_matches)

    for value in candidates:
        if not value:
            continue
        raw_path = os.path.expanduser(str(value))
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(str(meta.get("_run_dir", "") or meta.get("_parquet_dir", "") or "."), raw_path)
        path = os.path.abspath(raw_path)
        if os.path.exists(path):
            return path
    return None


def _load_target_mesh(row: pd.Series, meta: dict[str, Any]) -> pv.DataSet | None:
    if not bool(SHOW_TARGET_MESH):
        return None
    mesh_path = _resolve_target_mesh_path(row, meta)
    if mesh_path is None:
        return None
    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    return mesh


def _normalize_normals_grid(normals: np.ndarray) -> np.ndarray:
    normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(
        normals,
        np.maximum(normal_norm, 1e-9),
        out=np.zeros_like(normals),
        where=normal_norm > 1e-9,
    )


def _orient_normals_outward_from_center(
    xyz: np.ndarray,
    normals: np.ndarray,
    valid_mask: np.ndarray,
    center_xyz: np.ndarray,
) -> np.ndarray:
    """Flips each face normal so stock offsets point away from the part center."""
    out = _normalize_normals_grid(np.asarray(normals, dtype=np.float32).copy())
    center = np.asarray(center_xyz, dtype=np.float32).reshape(3)
    valid = np.asarray(valid_mask, dtype=bool)

    for face_id in range(min(out.shape[0], valid.shape[0])):
        mask = valid[face_id]
        if not bool(np.any(mask)):
            continue
        face_center = np.asarray(xyz[face_id][mask], dtype=np.float32).reshape(-1, 3).mean(axis=0)
        face_normal = out[face_id][mask].reshape(-1, 3).mean(axis=0)
        n_norm = float(np.linalg.norm(face_normal))
        outward_hint = face_center - center
        if n_norm <= 1e-9 or float(np.linalg.norm(outward_hint)) <= 1e-9:
            continue
        if float(np.dot(face_normal / n_norm, outward_hint)) < 0.0:
            out[face_id] *= -1.0

    return out


def _load_transition(row: pd.Series, denormalize: bool, meta: dict[str, Any]) -> dict[str, Any]:
    state_points = _array(row["state_points"], np.float32, (512, 100, 7))
    node_mask = _array(row.get("node_mask", np.zeros((512,), dtype=np.int16)), np.int16, (512,))
    point_mask = _array(row.get("point_mask", np.zeros((512, 100), dtype=np.int16)), np.int16, (512, 100))

    xyz = state_points[..., 0:3].copy()
    normals = state_points[..., 3:6].copy()
    sdf_before = state_points[..., 6].copy()
    sdf_after = _array(row["next_point_sdf"], np.float32, (512, 100))
    face_area = _array(
        row.get("face_area_512x1", np.zeros((512, 1), dtype=np.float32)),
        np.float32,
        (512, 1),
    ).reshape(512)

    valid = (node_mask[:, None] == 0) & (point_mask == 0)
    face_ids_grid = np.broadcast_to(np.arange(512, dtype=np.int16)[:, None], valid.shape)
    scale, center = _get_scale_and_center(row, meta)
    if denormalize:
        xyz = xyz * scale + center.reshape(1, 1, 3)
        sdf_before = sdf_before * scale
        sdf_after = sdf_after * scale
        face_area = face_area * (scale ** 2)
        orientation_center = center
    else:
        orientation_center = np.zeros((3,), dtype=np.float32)

    normals = _orient_normals_outward_from_center(xyz, normals, valid, orientation_center)

    delta = np.maximum(sdf_before - sdf_after, 0.0)
    stock_before = xyz + normals * sdf_before[..., None] * float(STOCK_OFFSET_SCALE)
    stock_after = xyz + normals * sdf_after[..., None] * float(STOCK_OFFSET_SCALE)
    valid_counts = valid.sum(axis=1).astype(np.float32)
    delta_sum_by_face = np.where(valid, delta, 0.0).sum(axis=1)
    delta_mean_by_face = np.divide(
        delta_sum_by_face,
        np.maximum(valid_counts, 1.0),
        out=np.zeros((512,), dtype=np.float32),
        where=valid_counts > 0,
    )
    delta_max_by_face = np.where(valid, delta, 0.0).max(axis=1)
    approx_removed_volume_by_face = face_area * delta_mean_by_face
    return {
        "xyz": xyz[valid],
        "normals": normals[valid],
        "sdf_before": sdf_before[valid],
        "sdf_after": sdf_after[valid],
        "delta": delta[valid],
        "stock_before": stock_before[valid],
        "stock_after": stock_after[valid],
        "face_ids": face_ids_grid[valid],
        "xyz_grid": xyz,
        "normals_grid": normals,
        "sdf_before_grid": sdf_before,
        "sdf_after_grid": sdf_after,
        "delta_grid": delta,
        "stock_before_grid": stock_before,
        "stock_after_grid": stock_after,
        "valid_mask": valid,
        "face_area": face_area,
        "delta_mean_by_face": delta_mean_by_face,
        "delta_max_by_face": delta_max_by_face,
        "approx_removed_volume_by_face": approx_removed_volume_by_face,
        "scale": scale,
        "center": center,
        "valid_count": int(valid.sum()),
        "denormalized": bool(denormalize),
        "target_mesh_path": _resolve_target_mesh_path(row, meta),
        "target_mesh": _load_target_mesh(row, meta),
    }


def _make_cloud(points: np.ndarray, **arrays: np.ndarray) -> pv.PolyData:
    cloud = pv.PolyData(points)
    for name, values in arrays.items():
        cloud[name] = np.asarray(values)
    return cloud


def _make_line_segments(starts: np.ndarray, ends: np.ndarray, **cell_arrays: np.ndarray) -> pv.PolyData | None:
    starts = np.asarray(starts, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)
    if starts.size == 0 or ends.size == 0:
        return None
    n = min(len(starts), len(ends))
    points = np.empty((2 * n, 3), dtype=np.float32)
    points[0::2] = starts[:n]
    points[1::2] = ends[:n]
    lines = np.column_stack(
        [
            np.full(n, 2, dtype=np.int64),
            np.arange(0, 2 * n, 2, dtype=np.int64),
            np.arange(1, 2 * n, 2, dtype=np.int64),
        ]
    ).reshape(-1)
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = lines
    for name, values in cell_arrays.items():
        mesh.cell_data[name] = np.asarray(values)[:n]
    return mesh


def _changed_indices(data: dict[str, Any], scope_faces: list[int] | None = None) -> np.ndarray:
    delta = np.asarray(data["delta"], dtype=np.float32)
    if delta.size == 0:
        return np.asarray([], dtype=np.int64)
    max_delta = float(np.max(delta))
    if max_delta <= 0.0:
        return np.asarray([], dtype=np.int64)
    threshold = max(float(CHANGED_DELTA_ABS_TOL), float(CHANGED_DELTA_REL_TOL) * max_delta)
    candidate = delta >= threshold
    if scope_faces:
        face_ids = np.asarray(data["face_ids"], dtype=np.int32)
        scope = np.asarray([int(f) for f in scope_faces if 0 <= int(f) < 512], dtype=np.int32)
        if scope.size > 0:
            candidate = candidate & np.isin(face_ids, scope)
    indices = np.flatnonzero(candidate)
    if indices.size > int(MAX_REMOVAL_VECTORS):
        top = np.argsort(-delta[indices])[: int(MAX_REMOVAL_VECTORS)]
        indices = indices[top]
    return indices.astype(np.int64, copy=False)


def _face_cloud(data: dict[str, Any], face_id: int) -> pv.PolyData | None:
    if not (0 <= int(face_id) < 512):
        return None
    mask = data["valid_mask"][int(face_id)]
    if not bool(np.any(mask)):
        return None
    points = data["xyz_grid"][int(face_id)][mask]
    return _make_cloud(
        points,
        sdf_before=data["sdf_before_grid"][int(face_id)][mask],
        sdf_after=data["sdf_after_grid"][int(face_id)][mask],
        delta=data["delta_grid"][int(face_id)][mask],
    )


def _top_changed_faces(data: dict[str, Any], n: int) -> list[int]:
    approx = np.asarray(data["approx_removed_volume_by_face"], dtype=np.float32)
    valid_face = data["valid_mask"].any(axis=1)
    score = np.where(valid_face, approx, -np.inf)
    order = np.argsort(-score)
    return [int(i) for i in order[: max(0, int(n))] if np.isfinite(score[int(i)]) and score[int(i)] > 0]


def print_row_report(row: pd.Series, data: dict[str, Any]) -> None:
    action_face = _row_int(row, "action_face_id", -1)
    axis_dir = _row_vector(row, "axis_dir", 3)
    tool_name = _row_value(row, "tool_choice_name", None)
    if tool_name is None:
        tool_name = f"{_row_value(row, 'tool_type_name', _row_value(row, 'tool_kind', '?'))}_{_row_value(row, 'tool_diameter', '?')}"

    print("\n[ACTION]")
    print(f"  part              : {_row_value(row, 'part_name', '?')}")
    print(f"  scenario          : {_row_value(row, 'scenario_id', '?')}  parent={_row_value(row, 'parent_scenario_id', '?')}")
    print(f"  step/candidate    : {_row_value(row, 'decision_step', '?')} / {_row_value(row, 'candidate_index', '?')}  chosen={_row_value(row, 'is_chosen', '?')}")
    print(f"  macro / operation : {_row_value(row, 'macro_class_name', '?')} / {_row_value(row, 'operation_name', '?')}")
    print(f"  tool              : {tool_name}  diameter={_row_value(row, 'tool_diameter', '?')}")
    print(f"  face              : action={action_face}  anchor={_row_value(row, 'anchor_face_id', '?')}  target={_row_value(row, 'target_face_id', '?')}")
    print(f"  axis_dir          : [{axis_dir[0]:.6g}, {axis_dir[1]:.6g}, {axis_dir[2]:.6g}]")

    print("\n[GLOBAL RESULT]")
    print(f"  volume            : {_row_float(row, 'state_volume', 0.0):.6g} -> {_row_float(row, 'next_state_volume', 0.0):.6g}")
    print(f"  removed_volume    : {_row_float(row, 'out_removed_volume', 0.0):.6g}")
    print(f"  removed_ratio     : {_row_float(row, 'out_removed_ratio', 0.0):.6g}")
    print(f"  cycle_time        : {_row_float(row, 'out_cycle_time', 0.0):.6g}")
    print(f"  done_ratio        : {_row_float(row, 'state_done_ratio', 0.0):.6g} -> {_row_float(row, 'next_done_ratio', 0.0):.6g}")

    if 0 <= action_face < 512:
        face_valid = bool(data["valid_mask"][action_face].any())
        print("\n[ACTION FACE]")
        print(f"  valid_points      : {int(data['valid_mask'][action_face].sum()) if face_valid else 0}")
        print(f"  area              : {float(data['face_area'][action_face]):.6g}")
        print(f"  mean_delta_sdf    : {float(data['delta_mean_by_face'][action_face]):.6g}")
        print(f"  max_delta_sdf     : {float(data['delta_max_by_face'][action_face]):.6g}")
        print(
            f"  area_weighted_delta: {float(data['approx_removed_volume_by_face'][action_face]):.6g}  "
            "(proxy = face_area * mean_delta_sdf; not true NX per-face removed volume)"
        )

    top_faces = _top_changed_faces(data, TOP_CHANGED_FACES)
    if top_faces:
        print("\n[TOP CHANGED FACES] proxy ranking, sorted by face_area * mean_delta_sdf")
        print("  This is for localization only; true per-face removed volume is not stored in the parquet row.")
        print("  face_id | area | mean_delta | max_delta | area_weighted_delta")
        for face_id in top_faces:
            print(
                f"  {face_id:7d} | "
                f"{float(data['face_area'][face_id]):.6g} | "
                f"{float(data['delta_mean_by_face'][face_id]):.6g} | "
                f"{float(data['delta_max_by_face'][face_id]):.6g} | "
                f"{float(data['approx_removed_volume_by_face'][face_id]):.6g}"
            )
    else:
        print("\n[TOP CHANGED FACES] no positive SDF decrease found")


def build_row_summary(row: pd.Series, data: dict[str, Any], filtered_index: int) -> dict[str, Any]:
    action_face = _row_int(row, "action_face_id", -1)
    top_faces = _top_changed_faces(data, 1)
    top_face = top_faces[0] if top_faces else -1
    return {
        "filtered_index": int(filtered_index),
        "source_row_index": _row_int(row, "source_row_index", int(filtered_index)),
        "part_name": _row_value(row, "part_name", ""),
        "scenario_id": _row_value(row, "scenario_id", ""),
        "parent_scenario_id": _row_value(row, "parent_scenario_id", ""),
        "decision_step": _row_int(row, "decision_step", -1),
        "candidate_index": _row_int(row, "candidate_index", -1),
        "is_chosen": _row_int(row, "is_chosen", 0),
        "macro_class_name": _row_value(row, "macro_class_name", ""),
        "operation_name": _row_value(row, "operation_name", ""),
        "tool_choice_name": _row_value(row, "tool_choice_name", ""),
        "tool_diameter": _row_float(row, "tool_diameter", 0.0),
        "action_face_id": int(action_face),
        "anchor_face_id": _row_int(row, "anchor_face_id", -1),
        "target_face_id": _row_int(row, "target_face_id", -1),
        "state_volume": _row_float(row, "state_volume", 0.0),
        "next_state_volume": _row_float(row, "next_state_volume", 0.0),
        "out_removed_volume": _row_float(row, "out_removed_volume", 0.0),
        "out_removed_ratio": _row_float(row, "out_removed_ratio", 0.0),
        "out_cycle_time": _row_float(row, "out_cycle_time", 0.0),
        "state_done_ratio": _row_float(row, "state_done_ratio", 0.0),
        "next_done_ratio": _row_float(row, "next_done_ratio", 0.0),
        "action_face_area": float(data["face_area"][action_face]) if 0 <= action_face < 512 else 0.0,
        "action_face_mean_delta": float(data["delta_mean_by_face"][action_face]) if 0 <= action_face < 512 else 0.0,
        "action_face_max_delta": float(data["delta_max_by_face"][action_face]) if 0 <= action_face < 512 else 0.0,
        "action_face_area_weighted_delta": (
            float(data["approx_removed_volume_by_face"][action_face]) if 0 <= action_face < 512 else 0.0
        ),
        "top_changed_face_id": int(top_face),
        "top_changed_face_area_weighted_delta": (
            float(data["approx_removed_volume_by_face"][top_face]) if 0 <= top_face < 512 else 0.0
        ),
        "target_mesh_path": data.get("target_mesh_path") or "",
    }


def visualize(row: pd.Series, data: dict[str, Any], screenshot: str | None, show: bool) -> None:
    points = data["xyz"]
    if points.size == 0:
        raise ValueError("No valid points to visualize")
    target_mesh = data.get("target_mesh")
    if target_mesh is None:
        print("[WARN] target/background OBJ was not found. Set TARGET_MESH_PATH or use a parquet row with target_body_mesh_path.")
    sdf_clim = [
        float(min(np.min(data["sdf_before"]), np.min(data["sdf_after"]))),
        float(max(np.max(data["sdf_before"]), np.max(data["sdf_after"]))),
    ]
    delta_max = float(np.max(data["delta"])) if len(data["delta"]) else 0.0
    delta_clim = [0.0, max(delta_max, 1e-9)]

    def _add_target_mesh(opacity: float | None = None, show_edges: bool | None = None) -> None:
        if target_mesh is None:
            return
        plotter.add_mesh(
            target_mesh,
            color="silver",
            opacity=float(TARGET_MESH_CONTEXT_OPACITY if opacity is None else opacity),
            show_edges=bool(TARGET_MESH_SHOW_EDGES if show_edges is None else show_edges),
            edge_color="white",
        )

    cloud = _make_cloud(
        points,
        sdf_before=data["sdf_before"],
        sdf_after=data["sdf_after"],
        delta=data["delta"],
    )
    action_face = _row_int(row, "action_face_id", -1)
    action_cloud = _face_cloud(data, action_face)
    top_faces = _top_changed_faces(data, 1)
    top_changed_cloud = _face_cloud(data, top_faces[0]) if top_faces else None
    scoped_faces = []
    if 0 <= action_face < 512:
        scoped_faces.append(action_face)
    scoped_faces.extend(_top_changed_faces(data, int(REMOVAL_VECTOR_TOP_FACES)))
    scoped_faces = list(dict.fromkeys(scoped_faces))
    changed_idx = _changed_indices(data, scoped_faces)
    if changed_idx.size == 0:
        changed_idx = _changed_indices(data)
    stock_idx = changed_idx if bool(STOCK_PROXY_CHANGED_ONLY) and changed_idx.size > 0 else np.arange(len(data["stock_before"]))
    stock_before_cloud = _make_cloud(
        data["stock_before"][stock_idx],
        sdf_before=data["sdf_before"][stock_idx],
        sdf_after=data["sdf_after"][stock_idx],
        delta=data["delta"][stock_idx],
    )
    stock_after_cloud = _make_cloud(
        data["stock_after"][stock_idx],
        sdf_before=data["sdf_before"][stock_idx],
        sdf_after=data["sdf_after"][stock_idx],
        delta=data["delta"][stock_idx],
    )
    sdf_before_points = data["stock_before"] if bool(SDF_PANEL_USE_STOCK_PROXY) else data["xyz"]
    sdf_after_points = data["stock_after"] if bool(SDF_PANEL_USE_STOCK_PROXY) else data["xyz"]
    sdf_before_panel_cloud = _make_cloud(
        sdf_before_points,
        sdf_before=data["sdf_before"],
        sdf_after=data["sdf_after"],
        delta=data["delta"],
    )
    sdf_after_panel_cloud = _make_cloud(
        sdf_after_points,
        sdf_before=data["sdf_before"],
        sdf_after=data["sdf_after"],
        delta=data["delta"],
    )
    changed_before_cloud = (
        _make_cloud(sdf_before_points[changed_idx], delta=data["delta"][changed_idx])
        if changed_idx.size > 0
        else None
    )
    changed_after_cloud = (
        _make_cloud(sdf_after_points[changed_idx], delta=data["delta"][changed_idx])
        if changed_idx.size > 0
        else None
    )
    removal_lines = (
        _make_line_segments(
            data["stock_before"][changed_idx],
            data["stock_after"][changed_idx],
            delta=data["delta"][changed_idx],
        )
        if changed_idx.size > 0
        else None
    )
    axis_dir = _row_vector(row, "axis_dir", 3)
    if 0 <= action_face < 512:
        mask = data["valid_mask"][action_face]
        if bool(np.any(mask)):
            outward_axis = data["normals_grid"][action_face][mask].reshape(-1, 3).mean(axis=0)
            outward_axis_norm = float(np.linalg.norm(outward_axis))
            if outward_axis_norm > 1e-9:
                axis_dir = (outward_axis / outward_axis_norm).astype(np.float32)

    tool_name = _row_value(row, "tool_choice_name", None)
    if tool_name is None:
        tool_name = f"{_row_value(row, 'tool_type_name', _row_value(row, 'tool_kind', '?'))}_{_row_value(row, 'tool_diameter', '?')}"
    title = (
        f"step={row.get('decision_step', '?')} "
        f"candidate={row.get('candidate_index', '?')} "
        f"chosen={row.get('is_chosen', '?')} "
        f"{row.get('macro_class_name', '?')} "
        f"{tool_name} "
        f"face={action_face} "
        f"removed={_row_float(row, 'out_removed_volume', 0.0):.4g}"
    )

    off_screen = screenshot is not None and not show
    plotter = pv.Plotter(shape=(2, 3), off_screen=off_screen, window_size=(2200, 1400))
    plotter.set_background("white")
    plotter.add_text(title, position="upper_edge", font_size=10)

    plotter.subplot(0, 0)
    plotter.add_text(
        "Target CAD mesh + sampled points: red=action face, cyan=top changed face",
        font_size=9,
    )
    _add_target_mesh(opacity=float(TARGET_MESH_OPACITY), show_edges=True)
    plotter.add_mesh(cloud, color="lightgray", point_size=3, render_points_as_spheres=True)
    if action_cloud is not None:
        plotter.add_mesh(action_cloud, color="red", point_size=10, render_points_as_spheres=True)
    if top_changed_cloud is not None:
        plotter.add_mesh(top_changed_cloud, color="cyan", point_size=6, render_points_as_spheres=True)
    if action_cloud is not None and np.linalg.norm(axis_dir) > 1e-9:
        center = np.asarray(action_cloud.points).mean(axis=0, keepdims=True)
        axis = (axis_dir / np.linalg.norm(axis_dir)).reshape(1, 3)
        mag = 0.1 * float(data["scale"]) if data["denormalized"] else 0.25
        plotter.add_arrows(center, axis, mag=mag, color="red")
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text("Stock proxy overlay: orange=before, blue=after, red=removed", font_size=9)
    _add_target_mesh()
    plotter.add_mesh(cloud, color="lightgray", point_size=2, opacity=0.12, render_points_as_spheres=True)
    plotter.add_mesh(stock_before_cloud, color="orange", point_size=4, opacity=0.35, render_points_as_spheres=True)
    plotter.add_mesh(stock_after_cloud, color="dodgerblue", point_size=4, opacity=0.75, render_points_as_spheres=True)
    if removal_lines is not None:
        plotter.add_mesh(removal_lines, color="red", line_width=2)
    else:
        plotter.add_text("No positive removal above threshold", position="lower_left", font_size=9)
    plotter.add_axes()

    plotter.subplot(0, 2)
    plotter.add_text("Removed displacement vectors", font_size=10)
    _add_target_mesh()
    plotter.add_mesh(cloud, color="lightgray", point_size=2, opacity=0.10, render_points_as_spheres=True)
    if removal_lines is not None:
        plotter.add_mesh(removal_lines, scalars="delta", cmap="inferno", line_width=3)
    if changed_after_cloud is not None:
        plotter.add_mesh(
            changed_after_cloud,
            scalars="delta",
            cmap="inferno",
            point_size=5,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
    if removal_lines is None:
        plotter.add_text("No positive removal above threshold", position="lower_left", font_size=9)
    plotter.add_axes()

    plotter.subplot(1, 0)
    plotter.add_text("Current SDF on CAD surface: changed points highlighted", font_size=10)
    _add_target_mesh()
    plotter.add_mesh(
        sdf_before_panel_cloud,
        scalars="sdf_before",
        cmap="viridis",
        clim=sdf_clim,
        point_size=int(SDF_PANEL_BASE_POINT_SIZE),
        render_points_as_spheres=True,
    )
    if removal_lines is not None:
        plotter.add_mesh(removal_lines, color="red", line_width=2)
    if changed_before_cloud is not None:
        plotter.add_mesh(
            changed_before_cloud,
            scalars="delta",
            cmap="inferno",
            clim=delta_clim,
            point_size=int(SDF_PANEL_CHANGED_POINT_SIZE),
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
    if action_cloud is not None:
        plotter.add_mesh(action_cloud, color="red", point_size=8, render_points_as_spheres=True)
    plotter.add_axes()

    plotter.subplot(1, 1)
    plotter.add_text("Next SDF on CAD surface: changed points highlighted", font_size=10)
    _add_target_mesh()
    plotter.add_mesh(
        sdf_after_panel_cloud,
        scalars="sdf_after",
        cmap="viridis",
        clim=sdf_clim,
        point_size=int(SDF_PANEL_BASE_POINT_SIZE),
        render_points_as_spheres=True,
    )
    if removal_lines is not None:
        plotter.add_mesh(removal_lines, color="red", line_width=2)
    if changed_after_cloud is not None:
        plotter.add_mesh(
            changed_after_cloud,
            scalars="delta",
            cmap="inferno",
            clim=delta_clim,
            point_size=int(SDF_PANEL_CHANGED_POINT_SIZE),
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
    if action_cloud is not None:
        plotter.add_mesh(action_cloud, color="red", point_size=8, render_points_as_spheres=True)
    plotter.add_axes()

    plotter.subplot(1, 2)
    plotter.add_text("Removed amount: max(before - after, 0)", font_size=10)
    _add_target_mesh()
    plotter.add_mesh(
        cloud,
        scalars="delta",
        cmap="inferno",
        clim=delta_clim,
        point_size=4,
        render_points_as_spheres=True,
    )
    if action_cloud is not None:
        plotter.add_mesh(action_cloud, color="red", point_size=8, render_points_as_spheres=True)
    plotter.add_axes()

    plotter.link_views()
    plotter.view_isometric()

    if screenshot:
        os.makedirs(os.path.dirname(os.path.abspath(screenshot)) or ".", exist_ok=True)
        plotter.screenshot(screenshot, transparent_background=False)
        print(f"[OK] screenshot saved: {os.path.abspath(screenshot)}")

    if show:
        plotter.show()
    else:
        plotter.close()


def _write_summary_csv(summaries: list[dict[str, Any]], summary_path: str) -> str:
    try:
        pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_path
    except PermissionError:
        root, ext = os.path.splitext(summary_path)
        fallback_path = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"
        pd.DataFrame(summaries).to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] summary CSV is locked, wrote fallback: {fallback_path}")
        return fallback_path


def batch_visualize(df: pd.DataFrame, meta: dict[str, Any], parquet_path: str, out_dir: str) -> None:
    rows = _filter_rows(df, CHOSEN_ONLY)
    if rows.empty:
        raise ValueError("No rows selected for batch visualization")
    if BATCH_MAX_ROWS > 0:
        rows = rows.iloc[: int(BATCH_MAX_ROWS)].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    summary_path = os.path.join(out_dir, "visualization_summary.csv")
    print(f"[INFO] batch parquet={parquet_path}")
    print(f"[INFO] batch rows={len(rows)} output_dir={out_dir}")

    for idx, row in rows.iterrows():
        data = _load_transition(row, DENORMALIZE, meta)
        summary = build_row_summary(row, data, int(idx))
        summaries.append(summary)

        file_name = (
            f"row_{int(summary['filtered_index']):05d}"
            f"_src_{int(summary['source_row_index']):05d}"
            f"_t{int(summary['decision_step']):02d}"
            f"_c{int(summary['candidate_index']):03d}"
            f"_{summary['macro_class_name']}"
            f"_face{int(summary['action_face_id'])}.png"
        )
        file_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in file_name)
        screenshot_path = os.path.join(out_dir, file_name)
        summary["screenshot_path"] = os.path.abspath(screenshot_path)
        summary["visualization_status"] = "pending"
        summary_path = _write_summary_csv(summaries, summary_path)

        visualize(row=row, data=data, screenshot=screenshot_path, show=False)
        summary["visualization_status"] = "ok"
        summary_path = _write_summary_csv(summaries, summary_path)
        print(
            f"[{int(idx) + 1}/{len(rows)}] "
            f"src={summary['source_row_index']} "
            f"step={summary['decision_step']} cand={summary['candidate_index']} "
            f"{summary['macro_class_name']} {summary['tool_choice_name']} "
            f"face={summary['action_face_id']} removed={summary['out_removed_volume']:.6g}"
        )

    summary_path = _write_summary_csv(summaries, summary_path)
    print(f"[OK] summary saved: {summary_path}")


def main() -> None:
    if PARQUET_PATH:
        parquet_path = os.path.abspath(PARQUET_PATH)
        run_dir = os.path.dirname(parquet_path)
    elif RUN_DIR:
        run_dir = os.path.abspath(RUN_DIR)
        parquet_path = _find_parquet(run_dir)
    else:
        raise ValueError("Set either PARQUET_PATH or RUN_DIR at the top of visualize_dataset_sample.py")

    meta = _load_meta(run_dir)
    meta["_run_dir"] = os.path.abspath(run_dir)
    meta["_parquet_dir"] = os.path.dirname(os.path.abspath(parquet_path))

    df = pd.read_parquet(parquet_path)
    if BATCH_VISUALIZE_ALL_ROWS:
        out_dir = BATCH_OUTPUT_DIR
        if not out_dir:
            base = os.path.splitext(os.path.basename(parquet_path))[0]
            out_dir = os.path.join(os.path.dirname(parquet_path), f"{base}_visualizations")
        batch_visualize(df, meta, parquet_path, os.path.abspath(out_dir))
        return

    row = _select_row(df, ROW_INDEX, CHOSEN_ONLY)
    data = _load_transition(row, DENORMALIZE, meta)

    print(f"[INFO] parquet={parquet_path}")
    print(f"[INFO] row={ROW_INDEX} chosen_only={CHOSEN_ONLY} valid_points={data['valid_count']}")
    print(f"[INFO] normalization_scale={data['scale']:.6g} center={data['center'].tolist()}")
    print(
        "[INFO] sdf ranges "
        f"before=[{float(np.min(data['sdf_before'])):.6g}, {float(np.max(data['sdf_before'])):.6g}] "
        f"after=[{float(np.min(data['sdf_after'])):.6g}, {float(np.max(data['sdf_after'])):.6g}] "
        f"delta=[{float(np.min(data['delta'])):.6g}, {float(np.max(data['delta'])):.6g}]"
    )
    print_row_report(row, data)

    screenshot = SCREENSHOT_PATH if SCREENSHOT_PATH else None
    visualize(row=row, data=data, screenshot=screenshot, show=SHOW_WINDOW)


if __name__ == "__main__":
    main()
