r"""Visualize octree-only CAM transition parquet rows.

This script does not require NX.  It visualizes the stored face sample state and
next-state octree occupancy target:

    state_points[..., 0:3] = normalized face sample xyz
    state_points[..., 3:6] = face normal
    state_points[..., 6]   = normalized current SDF/residual

    octree_centers         = normalized adaptive octree leaf centers, flat [K*3]
    octree_depths          = integer octree depth per leaf, [K]
    octree_occ_labels      = next-state material occupancy, [K]

Edit the module-level configuration below, then run:
    python visualize_octree_dataset_sample.py
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
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# Edit these values directly when running from VSCode/debug mode.
RUN_DIR = r""
PARQUET_PATH = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\3dDataset0055_seed0_20260415_204610\3dDataset0055_seed0_process_skeleton_dataset.parquet"
ROW_INDEX = 0
CHOSEN_ONLY = False
DENORMALIZE = True
SCREENSHOT_PATH = r""
SHOW_WINDOW = True
BATCH_VISUALIZE_ALL_ROWS = False
BATCH_OUTPUT_DIR = r""
BATCH_MAX_ROWS = 0  # 0 means all rows after CHOSEN_ONLY filtering.

VISUAL_STYLE = "paper"  # "paper" for DeepMill-like surface panels, "debug" for octree diagnostics.
SHOW_TARGET_MESH = True
TARGET_MESH_PATH = r""  # Optional override. Otherwise uses row['target_body_mesh_path'] if present.
TARGET_MESH_OPACITY = 0.10
TARGET_MESH_SHOW_EDGES = True

SHOW_OCTREE_CELL_GLYPHS = False
SHOW_BOUNDARY_CELL_WIREFRAMES = True
SHOW_RECONSTRUCTED_OCCUPANCY_SURFACE = True
MAX_GLYPH_CELLS = 600
MAX_BOUNDARY_POINTS = 12000
MAX_BOUNDARY_WIREFRAME_CELLS = 900
MAX_CONTEXT_FACE_POINTS = 12000
MAX_RESIDUAL_POINTS = 12000
OCCUPIED_POINT_SIZE = 5
EMPTY_POINT_SIZE = 3
FACE_POINT_SIZE = 2
ACTION_FACE_POINT_SIZE = 12
BOUNDARY_POINT_SIZE = 5
RECON_SURFACE_OPACITY = 0.58
BOUNDARY_POINT_OPACITY = 0.72
BOUNDARY_WIREFRAME_OPACITY = 0.34

# If True, panel 2 draws only occupied leaves. If False, it draws occupied and empty leaves.
CELL_GLYPH_OCCUPIED_ONLY = True


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _find_parquet(run_dir: str) -> str:
    matches = sorted(glob.glob(os.path.join(run_dir, "*_process_skeleton_dataset.parquet")))
    if not matches:
        matches = sorted(glob.glob(os.path.join(run_dir, "*.parquet")))
    if not matches:
        raise FileNotFoundError(f"No parquet file found in: {run_dir}")
    return matches[0]


def _array(value: Any, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Robustly converts parquet nested/object values into a numpy array."""

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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row.index:
        return default
    value = row[key]
    if _is_missing(value):
        return default
    return value


def _row_int(row: pd.Series, key: str, default: int = -1) -> int:
    return int(_row_value(row, key, default))


def _row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    return float(_row_value(row, key, default))


def _load_meta(run_dir: str | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _select_row(df: pd.DataFrame, row_index: int, chosen_only: bool) -> pd.Series:
    rows = _filter_rows(df, chosen_only)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row index {row_index} out of range for {len(rows)} selected rows")
    return rows.iloc[int(row_index)]


def _get_scale_and_center(row: pd.Series, meta: dict[str, Any]) -> tuple[float, np.ndarray]:
    scale = _row_value(row, "normalization_scale", None)
    center = _row_value(row, "normalization_center_xyz", None)
    if scale is None and meta:
        scale = meta.get("note", {}).get("normalization", {}).get("reference_scale", 1.0)
    if center is None and meta:
        center = meta.get("note", {}).get("normalization", {}).get("center_xyz", [0.0, 0.0, 0.0])
    return float(scale if scale is not None else 1.0), _array(center if center is not None else [0.0, 0.0, 0.0], np.float32, (3,))


def _resolve_target_mesh_path(row: pd.Series, meta: dict[str, Any]) -> str | None:
    candidates = [
        TARGET_MESH_PATH,
        _row_value(row, "target_body_mesh_path", ""),
        meta.get("target_body_mesh_path", "") if meta else "",
        meta.get("note", {}).get("target_body_mesh_path", "") if meta else "",
    ]
    for value in candidates:
        if not value:
            continue
        path = os.path.abspath(os.path.expanduser(str(value)))
        if os.path.exists(path):
            return path
    return None


def _load_target_mesh(row: pd.Series, meta: dict[str, Any], denormalize: bool, scale: float, center: np.ndarray) -> pv.DataSet | None:
    if not bool(SHOW_TARGET_MESH):
        return None
    mesh_path = _resolve_target_mesh_path(row, meta)
    if mesh_path is None:
        return None
    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not denormalize:
        mesh = mesh.copy()
        mesh.points = (np.asarray(mesh.points, dtype=np.float32) - center.reshape(1, 3)) / max(scale, 1e-9)
    return mesh


# ---------------------------------------------------------------------------
# Dataset row conversion
# ---------------------------------------------------------------------------

def _load_state(row: pd.Series, denormalize: bool, meta: dict[str, Any]) -> dict[str, Any]:
    state_points = _array(row["state_points"], np.float32, (512, 100, 7))
    node_mask = _array(_row_value(row, "node_mask", np.ones((512,), dtype=np.int16)), np.int16, (512,))
    point_mask = _array(_row_value(row, "point_mask", np.ones((512, 100), dtype=np.int16)), np.int16, (512, 100))

    xyz = state_points[..., 0:3].copy()
    normals = state_points[..., 3:6].copy()
    current_sdf = state_points[..., 6].copy()
    next_sdf_raw = _row_value(row, "next_point_sdf", None)
    next_sdf = None
    if next_sdf_raw is not None:
        try:
            next_sdf = _array(next_sdf_raw, np.float32, (512, 100)).copy()
        except Exception:
            next_sdf = None
    scale, center = _get_scale_and_center(row, meta)
    if denormalize:
        xyz = xyz * scale + center.reshape(1, 1, 3)
        current_sdf = current_sdf * scale
        if next_sdf is not None:
            next_sdf = next_sdf * scale

    valid = (node_mask[:, None] == 0) & (point_mask == 0)
    face_ids = np.broadcast_to(np.arange(512, dtype=np.int32)[:, None], valid.shape)
    next_sdf_valid = next_sdf[valid] if next_sdf is not None else np.zeros((int(valid.sum()),), dtype=np.float32)
    removed_delta = np.maximum(current_sdf[valid] - next_sdf_valid, 0.0).astype(np.float32)
    return {
        "xyz": xyz[valid],
        "normals": normals[valid],
        "current_sdf": current_sdf[valid],
        "next_sdf": next_sdf_valid,
        "removed_delta": removed_delta,
        "face_ids": face_ids[valid],
        "xyz_grid": xyz,
        "current_sdf_grid": current_sdf,
        "next_sdf_grid": next_sdf,
        "valid_mask": valid,
        "scale": scale,
        "center": center,
    }


def _load_octree(row: pd.Series, denormalize: bool, scale: float, center: np.ndarray) -> dict[str, Any]:
    centers_raw = _row_value(row, "octree_centers", None)
    depths_raw = _row_value(row, "octree_depths", None)
    labels_raw = _row_value(row, "octree_occ_labels", None)

    # Backward-compatible fallback for earlier random occupancy datasets.
    if centers_raw is None:
        centers_raw = _row_value(row, "occ_query_xyz", None)
    if labels_raw is None:
        labels_raw = _row_value(row, "next_occ_labels", None)

    if centers_raw is None or labels_raw is None:
        raise KeyError("This parquet row has no octree_centers/octree_occ_labels columns")

    centers = _array(centers_raw, np.float32).reshape(-1, 3)
    labels = _array(labels_raw, np.float32).reshape(-1)
    if depths_raw is None:
        depths = np.full((centers.shape[0],), 5, dtype=np.int32)
    else:
        depths = _array(depths_raw, np.int32).reshape(-1)

    count = min(len(centers), len(labels), len(depths))
    centers = centers[:count]
    labels = labels[:count]
    depths = depths[:count]

    bbox_min_raw = _row_value(row, "octree_bbox_min", None)
    bbox_max_raw = _row_value(row, "octree_bbox_max", None)
    if bbox_min_raw is None:
        bbox_min_raw = _row_value(row, "occ_bbox_min", None)
    if bbox_max_raw is None:
        bbox_max_raw = _row_value(row, "occ_bbox_max", None)
    if bbox_min_raw is not None and bbox_max_raw is not None:
        bbox_min = _array(bbox_min_raw, np.float32, (3,))
        bbox_max = _array(bbox_max_raw, np.float32, (3,))
    else:
        bbox_min = centers.min(axis=0) if len(centers) else np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        bbox_max = centers.max(axis=0) if len(centers) else np.array([1.0, 1.0, 1.0], dtype=np.float32)

    if denormalize:
        centers = centers * scale + center.reshape(1, 3)
        bbox_min = bbox_min * scale + center
        bbox_max = bbox_max * scale + center

    extent = np.maximum(bbox_max - bbox_min, 1e-9)
    scalar_cell_size = np.max(extent) / np.maximum(2.0 ** depths.astype(np.float32), 1.0)

    return {
        "centers": centers.astype(np.float32),
        "depths": depths.astype(np.int32),
        "labels": labels.astype(np.float32),
        "bbox_min": bbox_min.astype(np.float32),
        "bbox_max": bbox_max.astype(np.float32),
        "cell_size": scalar_cell_size.astype(np.float32),
        "occupied_mask": labels >= 0.5,
    }


def _make_cloud(points: np.ndarray, **arrays: np.ndarray) -> pv.PolyData:
    cloud = pv.PolyData(np.asarray(points, dtype=np.float32))
    for name, values in arrays.items():
        cloud[name] = np.asarray(values)
    return cloud


def _sample_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_count <= 0 or indices.size <= max_count:
        return indices
    rng = np.random.default_rng(0)
    return rng.choice(indices, size=max_count, replace=False)


def _sample_cloud(cloud: pv.PolyData, max_points: int) -> pv.PolyData:
    if cloud is None or cloud.n_points == 0 or max_points <= 0 or cloud.n_points <= max_points:
        return cloud
    indices = _sample_indices(np.arange(cloud.n_points, dtype=np.int64), int(max_points))
    return cloud.extract_points(indices, adjacent_cells=False, include_cells=False)


def _make_octree_cell_glyph(octree: dict[str, Any], occupied_only: bool) -> pv.PolyData | None:
    if not bool(SHOW_OCTREE_CELL_GLYPHS):
        return None
    labels = octree["labels"]
    if labels.size == 0:
        return None
    if occupied_only:
        indices = np.flatnonzero(octree["occupied_mask"])
    else:
        indices = np.arange(labels.size, dtype=np.int64)
    indices = _sample_indices(indices, MAX_GLYPH_CELLS)
    if indices.size == 0:
        return None
    cloud = _make_cloud(
        octree["centers"][indices],
        occ=octree["labels"][indices],
        depth=octree["depths"][indices].astype(np.float32),
        cell_size=octree["cell_size"][indices],
    )
    try:
        return cloud.glyph(scale="cell_size", geom=pv.Cube(), orient=False)
    except Exception:
        return cloud


def _octree_to_dense_grid(octree: dict[str, Any], max_depth_cap: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Rasterizes mixed-depth octree leaves to a dense grid at max depth."""
    centers = np.asarray(octree["centers"], dtype=np.float32)
    depths = np.asarray(octree["depths"], dtype=np.int32)
    labels = np.asarray(octree["labels"], dtype=np.float32)
    if centers.size == 0 or depths.size == 0 or labels.size == 0:
        return None

    max_depth = int(np.max(depths))
    max_depth = min(max_depth, int(max_depth_cap))
    n = int(2 ** max_depth)
    if n < 2:
        return None

    bbox_min = np.asarray(octree["bbox_min"], dtype=np.float32).reshape(3)
    bbox_max = np.asarray(octree["bbox_max"], dtype=np.float32).reshape(3)
    extent = np.maximum(bbox_max - bbox_min, 1e-9)
    grid = np.zeros((n, n, n), dtype=np.float32)

    for center, depth, label in zip(centers, depths, labels):
        depth_i = int(min(max(int(depth), 0), max_depth))
        cells_at_depth = int(2 ** depth_i)
        span = int(2 ** (max_depth - depth_i))
        rel = (center - bbox_min) / extent
        parent_idx = np.floor(rel * cells_at_depth).astype(np.int32)
        parent_idx = np.clip(parent_idx, 0, cells_at_depth - 1)
        start = parent_idx * span
        end = np.minimum(start + span, n)
        grid[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = float(label)

    spacing = extent / float(max(n - 1, 1))
    return grid, bbox_min, spacing


def _boundary_voxel_indices(octree: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns boundary occupied dense-grid indices plus grid origin/spacing."""
    rasterized = _octree_to_dense_grid(octree)
    if rasterized is None:
        return None
    grid, origin, spacing = rasterized
    occ = grid >= 0.5
    if not bool(np.any(occ)):
        return None

    boundary = np.zeros_like(occ, dtype=bool)
    for axis in range(3):
        diff = np.diff(occ.astype(np.int8), axis=axis) != 0
        left = [slice(None), slice(None), slice(None)]
        right = [slice(None), slice(None), slice(None)]
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        boundary[tuple(left)] |= diff
        boundary[tuple(right)] |= diff

    # Domain boundary can also be a visible stock boundary if it is occupied.
    boundary[0, :, :] |= occ[0, :, :]
    boundary[-1, :, :] |= occ[-1, :, :]
    boundary[:, 0, :] |= occ[:, 0, :]
    boundary[:, -1, :] |= occ[:, -1, :]
    boundary[:, :, 0] |= occ[:, :, 0]
    boundary[:, :, -1] |= occ[:, :, -1]
    boundary &= occ

    indices = np.argwhere(boundary)
    if indices.size == 0:
        return None
    return indices.astype(np.int32), origin.astype(np.float32), spacing.astype(np.float32)


def _make_boundary_cloud(octree: dict[str, Any]) -> pv.PolyData | None:
    """Returns occupied boundary voxel centers only, hiding interior occupied cells."""
    boundary_data = _boundary_voxel_indices(octree)
    if boundary_data is None:
        return None
    indices, origin, spacing = boundary_data
    if len(indices) > MAX_BOUNDARY_POINTS:
        keep = _sample_indices(np.arange(len(indices), dtype=np.int64), int(MAX_BOUNDARY_POINTS))
        indices = indices[keep]
    points = origin.reshape(1, 3) + indices.astype(np.float32) * spacing.reshape(1, 3)
    return _make_cloud(points, boundary=np.ones((points.shape[0],), dtype=np.float32))


def _make_boundary_cell_wireframes(octree: dict[str, Any]) -> pv.PolyData | None:
    """Builds wireframe cube outlines for occupied boundary voxels."""
    if not bool(SHOW_BOUNDARY_CELL_WIREFRAMES):
        return None
    boundary_data = _boundary_voxel_indices(octree)
    if boundary_data is None:
        return None
    indices, origin, spacing = boundary_data
    if len(indices) > MAX_BOUNDARY_WIREFRAME_CELLS:
        keep = _sample_indices(np.arange(len(indices), dtype=np.int64), int(MAX_BOUNDARY_WIREFRAME_CELLS))
        indices = indices[keep]
    if len(indices) == 0:
        return None

    half = spacing.reshape(1, 3) * 0.5
    centers = origin.reshape(1, 3) + indices.astype(np.float32) * spacing.reshape(1, 3)
    corners_local = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    edge_pairs = np.asarray(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int64,
    )

    all_points = []
    all_lines = []
    for cell_id, center in enumerate(centers):
        base = cell_id * 8
        all_points.append(center.reshape(1, 3) + corners_local * half)
        for a, b in edge_pairs:
            all_lines.extend([2, int(base + a), int(base + b)])

    mesh = pv.PolyData()
    mesh.points = np.vstack(all_points).astype(np.float32)
    mesh.lines = np.asarray(all_lines, dtype=np.int64)
    return mesh


def _make_bbox_outline(octree: dict[str, Any]) -> pv.PolyData:
    bbox_min = np.asarray(octree["bbox_min"], dtype=np.float32).reshape(3)
    bbox_max = np.asarray(octree["bbox_max"], dtype=np.float32).reshape(3)
    return pv.Box(bounds=(bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1], bbox_min[2], bbox_max[2])).outline()


def _make_occupancy_surface(octree: dict[str, Any]) -> pv.PolyData | None:
    """Extracts a visual surface from octree occupancy using a dense grid contour."""
    if not bool(SHOW_RECONSTRUCTED_OCCUPANCY_SURFACE):
        return None
    rasterized = _octree_to_dense_grid(octree)
    if rasterized is None:
        return None
    grid, origin, spacing = rasterized
    if float(grid.min()) >= 0.5 or float(grid.max()) < 0.5:
        return None
    try:
        image = pv.ImageData(dimensions=grid.shape, spacing=tuple(spacing), origin=tuple(origin))
    except Exception:
        image = pv.UniformGrid(dimensions=grid.shape, spacing=tuple(spacing), origin=tuple(origin))
    image.point_data["occ"] = grid.ravel(order="F")
    try:
        surface = image.contour([0.5], scalars="occ")
    except Exception:
        return None
    if surface.n_points == 0:
        return None
    return surface


def _action_face_cloud(state: dict[str, Any], action_face: int) -> pv.PolyData | None:
    if action_face < 0 or action_face >= 512:
        return None
    mask = state["valid_mask"][action_face]
    if not bool(np.any(mask)):
        return None
    return _make_cloud(
        state["xyz_grid"][action_face][mask],
        current_sdf=state["current_sdf_grid"][action_face][mask],
    )


def _mesh_with_nearest_sample_scalar(
    mesh: pv.DataSet | None,
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    scalar_name: str,
) -> pv.DataSet | None:
    """Copies a mesh and assigns per-vertex values from nearest sampled points."""
    if mesh is None or cKDTree is None:
        return None
    pts = np.asarray(sample_points, dtype=np.float32).reshape(-1, 3)
    vals = np.asarray(sample_values, dtype=np.float32).reshape(-1)
    if pts.size == 0 or vals.size == 0:
        return None
    count = min(len(pts), len(vals))
    pts = pts[:count]
    vals = vals[:count]
    out = mesh.copy()
    vertices = np.asarray(out.points, dtype=np.float32).reshape(-1, 3)
    if vertices.size == 0:
        return None
    tree = cKDTree(pts)
    _, nearest = tree.query(vertices, k=1)
    out.point_data[scalar_name] = vals[np.asarray(nearest, dtype=np.int64)]
    return out


def _extract_nearest_action_region(
    mesh: pv.DataSet | None,
    action_cloud: pv.PolyData | None,
    scale: float,
) -> pv.DataSet | None:
    """Extracts an approximate action-face patch on the target mesh."""
    if mesh is None or action_cloud is None or action_cloud.n_points == 0 or cKDTree is None:
        return None
    out = mesh.copy()
    vertices = np.asarray(out.points, dtype=np.float32).reshape(-1, 3)
    action_pts = np.asarray(action_cloud.points, dtype=np.float32).reshape(-1, 3)
    if vertices.size == 0 or action_pts.size == 0:
        return None
    tree = cKDTree(action_pts)
    distances, _ = tree.query(vertices, k=1)
    bbox_extent = np.asarray(out.bounds, dtype=np.float32)
    diag = float(np.linalg.norm([bbox_extent[1] - bbox_extent[0], bbox_extent[3] - bbox_extent[2], bbox_extent[5] - bbox_extent[4]]))
    radius = max(0.012 * diag, 0.018 * float(scale), 1e-6)
    out.point_data["action_region"] = (distances <= radius).astype(np.float32)
    try:
        region = out.threshold(0.5, scalars="action_region")
    except Exception:
        return None
    if region.n_points == 0:
        return None
    return region


def _add_scalar_mesh(
    plotter: pv.Plotter,
    mesh: pv.DataSet | None,
    scalar_name: str,
    title: str,
    cmap: str,
    opacity: float = 1.0,
    clim: tuple[float, float] | None = None,
) -> None:
    if mesh is None or scalar_name not in mesh.point_data:
        return
    scalars = np.asarray(mesh.point_data[scalar_name], dtype=np.float32)
    if clim is None:
        vmax = float(np.nanpercentile(scalars, 98)) if scalars.size else 1.0
        if vmax <= 1e-9:
            vmax = 1.0
        clim = (0.0, vmax)
    plotter.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={"title": title},
    )


# ---------------------------------------------------------------------------
# Reporting / visualization
# ---------------------------------------------------------------------------

def _row_title(row: pd.Series) -> str:
    tool = _row_value(row, "tool_choice_name", None)
    if tool is None:
        tool = f"{_row_value(row, 'tool_type_name', '?')}_{_row_value(row, 'tool_diameter', '?')}"
    return (
        f"src={_row_value(row, 'source_row_index', '?')} "
        f"t={_row_value(row, 'decision_step', '?')} "
        f"cand={_row_value(row, 'candidate_index', '?')} "
        f"chosen={_row_value(row, 'is_chosen', '?')} "
        f"{_row_value(row, 'macro_class_name', '?')} {tool} "
        f"face={_row_value(row, 'action_face_id', '?')}"
    )


def print_report(row: pd.Series, state: dict[str, Any], octree: dict[str, Any]) -> None:
    labels = octree["labels"]
    depths = octree["depths"]
    occ_ratio = float((labels >= 0.5).mean()) if labels.size else 0.0
    unique_depths, depth_counts = np.unique(depths, return_counts=True) if depths.size else ([], [])
    depth_summary = ", ".join(f"d{int(d)}={int(c)}" for d, c in zip(unique_depths, depth_counts))

    print("\n[ACTION]")
    print(f"  part              : {_row_value(row, 'part_name', '?')}")
    print(f"  scenario          : {_row_value(row, 'scenario_id', '?')}  parent={_row_value(row, 'parent_scenario_id', '?')}")
    print(f"  step/candidate    : {_row_value(row, 'decision_step', '?')} / {_row_value(row, 'candidate_index', '?')}  chosen={_row_value(row, 'is_chosen', '?')}")
    print(f"  macro             : {_row_value(row, 'macro_class_name', '?')}")
    print(f"  tool              : {_row_value(row, 'tool_choice_name', _row_value(row, 'tool_type_name', '?'))}  diameter={_row_value(row, 'tool_diameter', '?')}")
    print(f"  action_face_id    : {_row_value(row, 'action_face_id', '?')}")
    print(f"  volume            : {_row_float(row, 'state_volume', 0.0):.6g} -> {_row_float(row, 'next_state_volume', 0.0):.6g}")
    print(f"  removed_volume    : {_row_float(row, 'out_removed_volume', 0.0):.6g}")
    print(f"  cycle_time        : {_row_float(row, 'out_cycle_time', 0.0):.6g}")

    print("\n[OCTREE TARGET]")
    print(f"  K leaves          : {len(labels)}")
    print(f"  occupied ratio    : {occ_ratio:.4f}")
    print(f"  depth counts      : {depth_summary}")
    print(f"  bbox min          : {octree['bbox_min'].tolist()}")
    print(f"  bbox max          : {octree['bbox_max'].tolist()}")
    if occ_ratio <= 0.001 or occ_ratio >= 0.999:
        print("  warning           : occupancy is nearly constant; no surface can be extracted from this row")
    print(f"  state valid pts   : {len(state['xyz'])}")


def build_summary(row: pd.Series, state: dict[str, Any], octree: dict[str, Any], filtered_index: int) -> dict[str, Any]:
    labels = octree["labels"]
    occ_ratio = float((labels >= 0.5).mean()) if labels.size else 0.0
    return {
        "filtered_index": int(filtered_index),
        "source_row_index": _row_int(row, "source_row_index", int(filtered_index)),
        "part_name": _row_value(row, "part_name", ""),
        "scenario_id": _row_value(row, "scenario_id", ""),
        "decision_step": _row_int(row, "decision_step", -1),
        "candidate_index": _row_int(row, "candidate_index", -1),
        "is_chosen": _row_int(row, "is_chosen", 0),
        "macro_class_name": _row_value(row, "macro_class_name", ""),
        "tool_choice_name": _row_value(row, "tool_choice_name", ""),
        "tool_diameter": _row_float(row, "tool_diameter", 0.0),
        "action_face_id": _row_int(row, "action_face_id", -1),
        "state_volume": _row_float(row, "state_volume", 0.0),
        "next_state_volume": _row_float(row, "next_state_volume", 0.0),
        "out_removed_volume": _row_float(row, "out_removed_volume", 0.0),
        "out_cycle_time": _row_float(row, "out_cycle_time", 0.0),
        "octree_leaf_count": int(len(labels)),
        "octree_occupied_count": int((labels >= 0.5).sum()) if labels.size else 0,
        "octree_occ_ratio": occ_ratio,
        "octree_depth_min": int(octree["depths"].min()) if len(octree["depths"]) else -1,
        "octree_depth_max": int(octree["depths"].max()) if len(octree["depths"]) else -1,
    }


def visualize_paper_style(row: pd.Series, state: dict[str, Any], octree: dict[str, Any], target_mesh: pv.DataSet | None, screenshot: str | None, show: bool) -> None:
    """DeepMill-like rendering: surface annotations instead of octree debug points."""
    occupancy_surface = _make_occupancy_surface(octree)
    action_face = _row_int(row, "action_face_id", -1)
    action_cloud = _action_face_cloud(state, action_face)
    action_region = _extract_nearest_action_region(target_mesh, action_cloud, state["scale"])

    residual_mesh = _mesh_with_nearest_sample_scalar(
        target_mesh,
        state["xyz"],
        state["current_sdf"],
        "current_residual",
    )
    next_residual_mesh = _mesh_with_nearest_sample_scalar(
        target_mesh,
        state["xyz"],
        state["next_sdf"],
        "next_residual",
    )
    removed_mesh = _mesh_with_nearest_sample_scalar(
        target_mesh,
        state["xyz"],
        state["removed_delta"],
        "removed_delta",
    )

    removed_cloud = _make_cloud(
        state["xyz"],
        removed_delta=state["removed_delta"],
    )
    removed_cloud = _sample_cloud(removed_cloud, MAX_RESIDUAL_POINTS)
    residual_cloud = _make_cloud(
        state["xyz"],
        current_sdf=state["current_sdf"],
    )
    residual_cloud = _sample_cloud(residual_cloud, MAX_RESIDUAL_POINTS)

    off_screen = screenshot is not None and not show
    plotter = pv.Plotter(shape=(2, 2), off_screen=off_screen, window_size=(1900, 1350))
    plotter.add_text(_row_title(row), position="upper_edge", font_size=10)

    def add_target(opacity: float = 0.28, edges: bool = False) -> None:
        if target_mesh is not None:
            plotter.add_mesh(
                target_mesh,
                color="gainsboro",
                opacity=opacity,
                show_edges=edges,
                edge_color="white",
                smooth_shading=True,
            )

    def add_action_region() -> None:
        if action_region is not None:
            plotter.add_mesh(action_region, color="red", opacity=0.92, show_edges=False, smooth_shading=True)
        elif action_cloud is not None:
            plotter.add_mesh(action_cloud, color="red", point_size=ACTION_FACE_POINT_SIZE, render_points_as_spheres=True)

    def add_axis_arrow() -> None:
        axis = _array(_row_value(row, "axis_dir", [0.0, 0.0, 1.0]), np.float32, (3,))
        if action_cloud is None or np.linalg.norm(axis) <= 1e-9:
            return
        center = np.asarray(action_cloud.points, dtype=np.float32).mean(axis=0, keepdims=True)
        axis_n = axis / np.linalg.norm(axis)
        mag = 0.14 * float(state["scale"]) if DENORMALIZE else 0.35
        plotter.add_arrows(center, axis_n.reshape(1, 3), mag=mag, color="red")

    def add_stock_surface(opacity: float = 0.76, color: str = "steelblue") -> None:
        if occupancy_surface is not None:
            plotter.add_mesh(
                occupancy_surface,
                color=color,
                opacity=opacity,
                show_edges=False,
                smooth_shading=True,
            )

    plotter.subplot(0, 0)
    plotter.add_text("Input CAD and selected machining region", font_size=10)
    add_target(0.72, True)
    add_action_region()
    add_axis_arrow()
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text("Next stock geometry reconstructed from occupancy", font_size=10)
    add_target(0.10, False)
    add_stock_surface(0.82, "steelblue")
    add_action_region()
    plotter.add_axes()

    plotter.subplot(1, 0)
    plotter.add_text("Material removed by this action on target surface", font_size=10)
    if removed_mesh is not None:
        _add_scalar_mesh(plotter, removed_mesh, "removed_delta", "removed", "inferno", opacity=0.96)
    else:
        add_target(0.12, False)
        if removed_cloud is not None and removed_cloud.n_points:
            plotter.add_mesh(
                removed_cloud,
                scalars="removed_delta",
                cmap="inferno",
                point_size=FACE_POINT_SIZE + 2,
                opacity=0.85,
                render_points_as_spheres=True,
                scalar_bar_args={"title": "removed"},
            )
    add_action_region()
    plotter.add_axes()

    plotter.subplot(1, 1)
    plotter.add_text("Current residual on target surface + next stock overlay", font_size=10)
    if residual_mesh is not None:
        _add_scalar_mesh(plotter, residual_mesh, "current_residual", "current residual", "viridis", opacity=0.88)
    else:
        add_target(0.10, False)
        if residual_cloud is not None and residual_cloud.n_points:
            plotter.add_mesh(
                residual_cloud,
                scalars="current_sdf",
                cmap="viridis",
                point_size=FACE_POINT_SIZE + 1,
                opacity=0.82,
                render_points_as_spheres=True,
                scalar_bar_args={"title": "current residual"},
            )
    add_stock_surface(0.20, "white")
    add_action_region()
    plotter.add_axes()

    plotter.link_views()
    plotter.view_isometric()

    if screenshot:
        os.makedirs(os.path.dirname(os.path.abspath(screenshot)) or ".", exist_ok=True)
        plotter.screenshot(screenshot)
        print(f"[OK] screenshot saved: {os.path.abspath(screenshot)}")

    if show:
        plotter.show()
    else:
        plotter.close()


def visualize(row: pd.Series, state: dict[str, Any], octree: dict[str, Any], target_mesh: pv.DataSet | None, screenshot: str | None, show: bool) -> None:
    if str(VISUAL_STYLE).strip().lower() == "paper":
        visualize_paper_style(row, state, octree, target_mesh, screenshot, show)
        return

    state_cloud_full = _make_cloud(
        state["xyz"],
        current_sdf=state["current_sdf"],
        face_id=state["face_ids"],
    )
    state_cloud = _sample_cloud(state_cloud_full, MAX_CONTEXT_FACE_POINTS)
    occ_mask = octree["occupied_mask"]
    occ_cloud_full = _make_cloud(
        octree["centers"][occ_mask],
        occ=octree["labels"][occ_mask],
        depth=octree["depths"][occ_mask].astype(np.float32),
        cell_size=octree["cell_size"][occ_mask],
    )
    occ_cloud = _sample_cloud(occ_cloud_full, MAX_BOUNDARY_POINTS)
    cell_glyph = _make_octree_cell_glyph(octree, CELL_GLYPH_OCCUPIED_ONLY)
    occupancy_surface = _make_occupancy_surface(octree)
    boundary_cloud = _make_boundary_cloud(octree)
    boundary_wireframes = _make_boundary_cell_wireframes(octree)
    bbox_outline = _make_bbox_outline(octree)
    action_face = _row_int(row, "action_face_id", -1)
    action_cloud = _action_face_cloud(state, action_face)
    residual_cloud = _make_cloud(
        state["xyz"],
        current_sdf=state["current_sdf"],
        face_id=state["face_ids"],
    )
    residual_cloud = _sample_cloud(residual_cloud, MAX_RESIDUAL_POINTS)

    off_screen = screenshot is not None and not show
    plotter = pv.Plotter(shape=(2, 2), off_screen=off_screen, window_size=(1900, 1350))
    plotter.add_text(_row_title(row), position="upper_edge", font_size=10)

    def add_target(opacity: float = TARGET_MESH_OPACITY, edges: bool = TARGET_MESH_SHOW_EDGES) -> None:
        if target_mesh is not None:
            plotter.add_mesh(target_mesh, color="silver", opacity=opacity, show_edges=edges, edge_color="white")

    def add_stock_surface(opacity: float = RECON_SURFACE_OPACITY, color: str = "dodgerblue") -> None:
        if occupancy_surface is not None:
            plotter.add_mesh(
                occupancy_surface,
                color=color,
                opacity=opacity,
                show_edges=False,
                smooth_shading=True,
            )

    plotter.subplot(0, 0)
    plotter.add_text("Action context: target CAD + red action face", font_size=9)
    add_target(0.22, True)
    if state_cloud is not None and state_cloud.n_points:
        plotter.add_mesh(state_cloud, color="black", point_size=FACE_POINT_SIZE, opacity=0.12, render_points_as_spheres=True)
    if action_cloud is not None:
        plotter.add_mesh(action_cloud, color="red", point_size=ACTION_FACE_POINT_SIZE, render_points_as_spheres=True)
    axis = _array(_row_value(row, "axis_dir", [0.0, 0.0, 1.0]), np.float32, (3,))
    if action_cloud is not None and np.linalg.norm(axis) > 1e-9:
        center = np.asarray(action_cloud.points, dtype=np.float32).mean(axis=0, keepdims=True)
        axis = axis / np.linalg.norm(axis)
        mag = 0.14 * float(state["scale"]) if DENORMALIZE else 0.35
        plotter.add_arrows(center, axis.reshape(1, 3), mag=mag, color="red")
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text("Next stock surface + boundary octree cell wireframes", font_size=9)
    add_target(0.12, False)
    add_stock_surface(RECON_SURFACE_OPACITY, "dodgerblue")
    if boundary_wireframes is not None and boundary_wireframes.n_points:
        plotter.add_mesh(
            boundary_wireframes,
            color="orangered",
            line_width=1,
            opacity=BOUNDARY_WIREFRAME_OPACITY,
        )
    plotter.add_mesh(bbox_outline, color="gray", line_width=1, opacity=0.35)
    if action_cloud is not None:
        plotter.add_mesh(action_cloud, color="red", point_size=7, opacity=0.85, render_points_as_spheres=True)
    plotter.add_axes()

    plotter.subplot(1, 0)
    plotter.add_text("Boundary-only octree cells: centers + wireframe cubes", font_size=9)
    add_target(0.08, False)
    add_stock_surface(0.16, "white")
    if boundary_wireframes is not None and boundary_wireframes.n_points:
        plotter.add_mesh(
            boundary_wireframes,
            color="orangered",
            line_width=1,
            opacity=0.55,
        )
    if boundary_cloud is not None and boundary_cloud.n_points:
        plotter.add_mesh(
            boundary_cloud,
            color="orangered",
            point_size=BOUNDARY_POINT_SIZE,
            opacity=BOUNDARY_POINT_OPACITY,
            render_points_as_spheres=True,
        )
    elif occ_cloud is not None and occ_cloud.n_points:
        plotter.add_mesh(occ_cloud, color="orangered", point_size=OCCUPIED_POINT_SIZE, opacity=0.35, render_points_as_spheres=True)
    if cell_glyph is not None and cell_glyph.n_points:
        plotter.add_mesh(cell_glyph, color="orangered", opacity=0.12, show_scalar_bar=False)
    plotter.add_axes()

    plotter.subplot(1, 1)
    plotter.add_text("Current residual context + reconstructed next stock", font_size=9)
    add_target(0.08, False)
    add_stock_surface(0.24, "white")
    if residual_cloud is not None and residual_cloud.n_points:
        plotter.add_mesh(
            residual_cloud,
            scalars="current_sdf",
            cmap="viridis",
            point_size=FACE_POINT_SIZE,
            opacity=0.78,
            render_points_as_spheres=True,
        )
    if boundary_cloud is not None and boundary_cloud.n_points:
        plotter.add_mesh(boundary_cloud, color="orangered", point_size=3, opacity=0.35, render_points_as_spheres=True)
    if boundary_wireframes is not None and boundary_wireframes.n_points:
        plotter.add_mesh(boundary_wireframes, color="orangered", line_width=1, opacity=0.12)
    if action_cloud is not None:
        plotter.add_mesh(action_cloud, color="red", point_size=8, render_points_as_spheres=True)
    plotter.add_axes()

    plotter.link_views()
    plotter.view_isometric()

    if screenshot:
        os.makedirs(os.path.dirname(os.path.abspath(screenshot)) or ".", exist_ok=True)
        plotter.screenshot(screenshot)
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
    summary_path = os.path.join(out_dir, "octree_visualization_summary.csv")
    summaries: list[dict[str, Any]] = []
    print(f"[INFO] batch parquet={parquet_path}")
    print(f"[INFO] batch rows={len(rows)} output_dir={out_dir}")

    for filtered_index, row in rows.iterrows():
        state = _load_state(row, DENORMALIZE, meta)
        octree = _load_octree(row, DENORMALIZE, state["scale"], state["center"])
        target_mesh = _load_target_mesh(row, meta, DENORMALIZE, state["scale"], state["center"])
        summary = build_summary(row, state, octree, int(filtered_index))
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
        summaries.append(summary)
        summary_path = _write_summary_csv(summaries, summary_path)

        visualize(row, state, octree, target_mesh, screenshot=screenshot_path, show=False)
        summary["visualization_status"] = "ok"
        summary_path = _write_summary_csv(summaries, summary_path)
        print(
            f"[{int(filtered_index) + 1}/{len(rows)}] "
            f"src={summary['source_row_index']} "
            f"step={summary['decision_step']} cand={summary['candidate_index']} "
            f"{summary['macro_class_name']} face={summary['action_face_id']} "
            f"K={summary['octree_leaf_count']} occ={summary['octree_occ_ratio']:.3f}"
        )

    print(f"[OK] summary saved: {summary_path}")


def main() -> None:
    if PARQUET_PATH:
        parquet_path = os.path.abspath(PARQUET_PATH)
        run_dir = os.path.dirname(parquet_path)
    elif RUN_DIR:
        run_dir = os.path.abspath(RUN_DIR)
        parquet_path = _find_parquet(run_dir)
    else:
        raise ValueError("Set either PARQUET_PATH or RUN_DIR at the top of visualize_octree_dataset_sample.py")

    meta = _load_meta(run_dir)
    df = pd.read_parquet(parquet_path)

    if BATCH_VISUALIZE_ALL_ROWS:
        out_dir = BATCH_OUTPUT_DIR
        if not out_dir:
            base = os.path.splitext(os.path.basename(parquet_path))[0]
            out_dir = os.path.join(os.path.dirname(parquet_path), f"{base}_octree_visualizations")
        batch_visualize(df, meta, parquet_path, os.path.abspath(out_dir))
        return

    row = _select_row(df, ROW_INDEX, CHOSEN_ONLY)
    state = _load_state(row, DENORMALIZE, meta)
    octree = _load_octree(row, DENORMALIZE, state["scale"], state["center"])
    target_mesh = _load_target_mesh(row, meta, DENORMALIZE, state["scale"], state["center"])

    print(f"[INFO] parquet={parquet_path}")
    print(f"[INFO] row={ROW_INDEX} chosen_only={CHOSEN_ONLY}")
    print(f"[INFO] normalization_scale={state['scale']:.6g} center={state['center'].tolist()}")
    print_report(row, state, octree)

    screenshot = SCREENSHOT_PATH if SCREENSHOT_PATH else None
    visualize(row, state, octree, target_mesh, screenshot=screenshot, show=SHOW_WINDOW)


if __name__ == "__main__":
    main()
