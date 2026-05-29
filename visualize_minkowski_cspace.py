r"""Visualize an approximate Minkowski/C-space accessibility mask for one row.

This script is intentionally separate from the training/data collection code.
Edit the constants below and run directly from VS Code/debug mode.

What it does:
    1. Loads one parquet transition row.
    2. Interpolates before/target TSDF query samples onto a coarse regular grid.
    3. Builds an oriented tool+holder structuring element from:
       tool diameter, tool length, holder diameter, holder length, and tool kind.
    4. Computes a C-space forbidden-tip mask:
       forbidden_tip[p] = any obstacle voxel overlaps tool/holder placed at tip p.
    5. Sweeps accessible cutter placements to build an approximate synthetic
       after-IPW label.
    6. Visualizes the minimal transition input and synthetic label:
       S_t, target, action face, tool/holder/axis -> synthetic S_{t+1}.

Notes:
    - This is a validation/visualization prototype, not a replacement for NX.
    - "before" stock is used for holder collision by default.
    - "target" solid is used for cutter gouge collision by default.
    - C-space is used here as a synthetic label generator, not as a model input.
    - For ball endmills, the cutting head is approximated as a ball nose plus
      cylindrical flute. Flat endmills are approximated as a cylinder.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
PARQUET_PATH = r""
PARQUET_DIR = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\_ALL_PARQUET_FILES"
PARQUET_GLOB = "*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []

# ROW_INDEX=-1 selects the row with the largest mean |after-before TSDF|.
ROW_INDEX = -1
ROW_MACRO_FILTER = ""  # e.g. "flank_finish" to inspect flank rows first.
CHOSEN_ONLY = False

OUTPUT_DIR = r"minkowski_cspace_visualizations"
SHOW_WINDOW = True
SCREENSHOT_PATH = r""

# Regular-grid quality/cost. Start small; 96 is usually enough to judge masks.
GRID_RESOLUTION = 96
GRID_PADDING_RATIO = 0.08
K_NEIGHBORS = 16
IDW_POWER = 2.0
INTERPOLATION_CHUNK_POINTS = 250_000

# Collision fields.
# holder/shank should not collide with current stock.
# cutter should not gouge protected target geometry.
HOLDER_COLLISION_FIELD = "before"  # "before", "target", or "after"
CUTTER_COLLISION_FIELD = "target"  # "before", "target", or "after"
OCCUPIED_TSDF_THRESHOLD = 0.0
TSDF_TRUNCATION_MM = 5.0

# Synthetic label generation. "auto" uses action-face ROI for finishing rows
# and global removal for roughing rows.
ACTION_FACE_ROI_MODE = "auto"  # "auto", "all", or "action_face"
ACTION_FACE_ROI_PADDING_MM = 0.0  # 0 = auto from tool radius/grid spacing

# Tool geometry. Use row values when overrides are <= 0 or empty.
TOOL_KIND_OVERRIDE = ""       # "", "flat", or "ball"
TOOL_DIAMETER_MM_OVERRIDE = 0.0
TOOL_LENGTH_MM = 80.0
HOLDER_DIAMETER_MM = 4.0
HOLDER_LENGTH_MM = 45.0

# Axis convention: the row axis_dir is passed directly to NX ToolAxisFixed.Vector.
# In that convention, the holder/spindle side is usually +axis_dir from the tip.
AXIS_DIR_OVERRIDE: list[float] | None = None
TOOL_EXTENDS_ALONG_NEG_AXIS = False

# "end" uses tip-based C-space. "flank" uses side-contact centerline C-space.
# "auto" chooses flank for macro_class_name/operation_name containing flank/swarf.
MINKOWSKI_MODE = "auto"  # "auto", "end", or "flank"
FLANK_CONTACT_AXIAL_FRACTION = 0.5  # 0=tip end, 1=holder end along flute.
FLANK_FACE_NORMAL_FLIP = 1.0        # set -1.0 if centerline offset is reversed.
FLANK_CHECK_CUTTER_GOUGE = False    # false avoids marking intentional side contact as gouge.

# Query overlays.
CHANGE_EPS = 1e-3
REMOVED_EPS = 1e-3
MAX_QUERY_POINTS_VIS = 80_000
MAX_CSPACE_POINTS_VIS = 80_000
MAX_KERNEL_POINTS_VIS = 20_000

# Visualization.
TARGET_MESH_PATH = r""
TARGET_MESH_OPACITY = 0.12
STOCK_SURFACE_OPACITY = 0.18
CSPACE_SURFACE_OPACITY = 0.28
POINT_SIZE = 5
EXPORT_VTK_FILES = True


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _array(value: Any, dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Robustly converts parquet nested/object values into numpy arrays."""
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        chunks: list[np.ndarray] = []
        stack = [value]
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
        arr = np.concatenate(chunks).astype(dtype, copy=False) if chunks else np.asarray([], dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row.index:
        return default
    value = row[key]
    return default if _is_missing(value) else value


def _resolve_files() -> list[Path]:
    files: list[Path] = []
    if EXPLICIT_PARQUET_PATHS:
        files.extend(Path(p).expanduser().resolve() for p in EXPLICIT_PARQUET_PATHS if str(p).strip())
    elif PARQUET_PATH.strip():
        files.append(Path(PARQUET_PATH).expanduser().resolve())
    elif PARQUET_DIR.strip():
        files.extend(sorted(Path(PARQUET_DIR).expanduser().resolve().glob(PARQUET_GLOB)))
    else:
        raise ValueError("Set PARQUET_PATH, PARQUET_DIR, or EXPLICIT_PARQUET_PATHS.")

    out: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            raise FileNotFoundError(path)
        out.append(path)
    if not out:
        raise ValueError("No parquet files found.")
    return out


def _load_one_file_rows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if CHOSEN_ONLY:
        if "is_chosen" not in df.columns:
            raise KeyError("CHOSEN_ONLY=True but parquet has no is_chosen column.")
        df = df[df["is_chosen"].astype(int) == 1]
    df = df.reset_index(drop=False).rename(columns={"index": "source_row_index"})
    return df


def _filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if ROW_MACRO_FILTER.strip():
        macro = ROW_MACRO_FILTER.strip().lower()
        if "macro_class_name" not in df.columns:
            return df.iloc[0:0]
        df = df[df["macro_class_name"].astype(str).str.lower() == macro]
    required = {"sdf_query_points", "sdf_tsdf_before", "sdf_tsdf_after"}
    if not required.issubset(set(df.columns)):
        return df.iloc[0:0]
    return df.reset_index(drop=True)


def _load_rows() -> tuple[pd.DataFrame, Path]:
    errors: list[str] = []
    for path in _resolve_files():
        try:
            df = _filter_rows(_load_one_file_rows(path))
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
            continue
        if not df.empty:
            return df, path

    detail = "\n".join(errors[:8])
    filter_text = ROW_MACRO_FILTER.strip() or "<none>"
    raise ValueError(
        "No usable rows were found in the configured parquet files. "
        f"ROW_MACRO_FILTER={filter_text!r}. "
        "Check PARQUET_PATH/PARQUET_DIR, CHOSEN_ONLY, and SDF columns."
        + (f"\nRecent read errors:\n{detail}" if detail else "")
    )


def _select_row(df: pd.DataFrame) -> pd.Series:
    if ROW_INDEX >= 0:
        if ROW_INDEX >= len(df):
            raise IndexError(f"ROW_INDEX={ROW_INDEX} out of range for {len(df)} rows.")
        return df.iloc[int(ROW_INDEX)]

    best_idx = 0
    best_score = -1.0
    for idx, row in df.iterrows():
        if "sdf_tsdf_before" not in row.index or "sdf_tsdf_after" not in row.index:
            continue
        before = _array(row["sdf_tsdf_before"], np.float32)
        after = _array(row["sdf_tsdf_after"], np.float32)
        count = min(before.size, after.size)
        if count <= 0:
            continue
        score = float(np.mean(np.abs(after[:count] - before[:count])))
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return df.iloc[best_idx]


def _scale_center(row: pd.Series) -> tuple[float, np.ndarray]:
    scale = float(_row_value(row, "normalization_scale", 1.0))
    center = _array(_row_value(row, "normalization_center_xyz", [0.0, 0.0, 0.0]), np.float32, (3,))
    return max(scale, 1e-6), center


def _denormalize_points(points: np.ndarray, row: pd.Series) -> np.ndarray:
    scale, center = _scale_center(row)
    return (points.astype(np.float32) * scale + center.reshape(1, 3)).astype(np.float32)


def _resolve_target_mesh_path(row: pd.Series) -> str | None:
    candidates = [TARGET_MESH_PATH, _row_value(row, "target_body_mesh_path", "")]
    static_dir = _row_value(row, "static_feature_dir", "")
    if static_dir:
        candidates.append(str(Path(static_dir) / "target_body.obj"))
    for value in candidates:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.exists():
            return str(path.resolve())
    return None


def _load_mesh(path: str | None) -> pv.PolyData | None:
    if not path:
        return None
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    return mesh.extract_surface().triangulate().clean()


def _load_target_mesh(row: pd.Series) -> pv.PolyData | None:
    return _load_mesh(_resolve_target_mesh_path(row))


def _load_sdf_samples(row: pd.Series) -> dict[str, np.ndarray]:
    if "sdf_query_points" not in row.index or _is_missing(row["sdf_query_points"]):
        raise KeyError("This row has no sdf_query_points.")
    points_norm = _array(row["sdf_query_points"], np.float32).reshape(-1, 3)
    points = _denormalize_points(points_norm, row)

    fields: dict[str, np.ndarray] = {"points": points}
    for name, col in (
        ("before", "sdf_tsdf_before"),
        ("after", "sdf_tsdf_after"),
        ("target", "sdf_target_tsdf"),
    ):
        if col in row.index and not _is_missing(row[col]):
            values = _array(row[col], np.float32)
            count = min(points.shape[0], values.size)
            fields[name] = values[:count].astype(np.float32, copy=False)
            fields["points"] = points[:count]

    if "before" not in fields or "after" not in fields:
        raise KeyError("This row requires sdf_tsdf_before and sdf_tsdf_after.")
    if "target" not in fields:
        # Fallback: after is closer to protected final geometry than before.
        fields["target"] = fields["after"]
    count = min(*(arr.shape[0] for key, arr in fields.items() if key != "points"), fields["points"].shape[0])
    fields["points"] = fields["points"][:count]
    for key in ("before", "after", "target"):
        fields[key] = fields[key][:count]
    return fields


def _build_grid_bounds(samples_xyz: np.ndarray, target_mesh: pv.PolyData | None) -> tuple[np.ndarray, np.ndarray]:
    mins = [samples_xyz.min(axis=0)]
    maxs = [samples_xyz.max(axis=0)]
    if target_mesh is not None and target_mesh.n_points > 0:
        bounds = np.asarray(target_mesh.bounds, dtype=np.float32)
        mins.append(bounds[[0, 2, 4]])
        maxs.append(bounds[[1, 3, 5]])
    bbox_min = np.min(np.stack(mins, axis=0), axis=0)
    bbox_max = np.max(np.stack(maxs, axis=0), axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    pad = extent * float(max(GRID_PADDING_RATIO, 0.0))
    return bbox_min - pad, bbox_max + pad


def _grid_dimensions(bbox_min: np.ndarray, bbox_max: np.ndarray) -> tuple[int, int, int]:
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    max_extent = float(extent.max())
    base = max(8, int(GRID_RESOLUTION))
    dims = np.maximum(np.round(extent / max_extent * float(base)).astype(np.int32), 8)
    return int(dims[0]), int(dims[1]), int(dims[2])


def _make_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    dims: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    nx, ny, nz = dims
    xs = np.linspace(float(bbox_min[0]), float(bbox_max[0]), nx, dtype=np.float32)
    ys = np.linspace(float(bbox_min[1]), float(bbox_max[1]), ny, dtype=np.float32)
    zs = np.linspace(float(bbox_min[2]), float(bbox_max[2]), nz, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    spacing = (
        float(xs[1] - xs[0]) if nx > 1 else 1.0,
        float(ys[1] - ys[0]) if ny > 1 else 1.0,
        float(zs[1] - zs[0]) if nz > 1 else 1.0,
    )
    return grid.astype(np.float32), spacing


def _interpolate_tsdf(
    sample_xyz: np.ndarray,
    sample_tsdf: np.ndarray,
    grid_points: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("scipy is required for KNN TSDF interpolation.") from exc

    k = max(1, min(int(K_NEIGHBORS), sample_xyz.shape[0]))
    tree = cKDTree(sample_xyz.astype(np.float32, copy=False))
    out = np.empty((grid_points.shape[0],), dtype=np.float32)
    eps = 1e-8
    for start in range(0, grid_points.shape[0], int(INTERPOLATION_CHUNK_POINTS)):
        stop = min(start + int(INTERPOLATION_CHUNK_POINTS), grid_points.shape[0])
        dist, idx = tree.query(grid_points[start:stop], k=k, workers=-1)
        dist = np.asarray(dist, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.int64)
        if k == 1:
            out[start:stop] = sample_tsdf[idx].astype(np.float32)
            continue
        weights = 1.0 / np.maximum(dist, eps) ** float(IDW_POWER)
        out[start:stop] = (weights * sample_tsdf[idx]).sum(axis=1) / np.maximum(weights.sum(axis=1), eps)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _unit(v: np.ndarray, fallback: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        arr = np.asarray(fallback, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
    return (arr / max(norm, 1e-8)).astype(np.float32)


def _row_axis(row: pd.Series) -> np.ndarray:
    if AXIS_DIR_OVERRIDE is not None:
        return _unit(np.asarray(AXIS_DIR_OVERRIDE, dtype=np.float32))
    return _unit(_array(_row_value(row, "axis_dir", [0.0, 0.0, 1.0]), np.float32, (3,)))


def _tool_kind(row: pd.Series) -> str:
    if TOOL_KIND_OVERRIDE.strip():
        return TOOL_KIND_OVERRIDE.strip().lower()
    value = str(_row_value(row, "tool_type_name", _row_value(row, "tool_choice_name", "flat"))).lower()
    if "ball" in value:
        return "ball"
    return "flat"


def _tool_diameter(row: pd.Series) -> float:
    if TOOL_DIAMETER_MM_OVERRIDE > 0.0:
        return float(TOOL_DIAMETER_MM_OVERRIDE)
    return float(_row_value(row, "tool_diameter", 8.0))


def _minkowski_mode(row: pd.Series) -> str:
    mode = str(MINKOWSKI_MODE).strip().lower()
    if mode in {"end", "flank"}:
        return mode
    if mode != "auto":
        raise ValueError("MINKOWSKI_MODE must be 'auto', 'end', or 'flank'.")
    macro = str(_row_value(row, "macro_class_name", "")).lower()
    operation = str(_row_value(row, "operation_name", "")).lower()
    return "flank" if ("flank" in macro or "swarf" in operation) else "end"


def _perpendicular_unit(axis: np.ndarray) -> np.ndarray:
    axis = _unit(axis)
    candidates = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    best = min(candidates, key=lambda item: abs(float(item @ axis)))
    return _unit(best - float(best @ axis) * axis)


def _estimate_normal_from_points(points: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return _unit(fallback)
    centered = points.astype(np.float32) - points.astype(np.float32).mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1].astype(np.float32)
    except Exception:
        return _unit(fallback)
    normal = _unit(normal, fallback=tuple(float(x) for x in fallback))
    if float(normal @ fallback) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def _load_action_face_normal(row: pd.Series, action_points: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    action_face = int(_row_value(row, "action_face_id", -1))
    static_dir = _row_value(row, "static_feature_dir", "")
    if static_dir and 0 <= action_face < 512:
        normal_path = Path(str(static_dir)) / "embed_face_normal.npy"
        if normal_path.exists():
            normals = np.load(normal_path).astype(np.float32).reshape(512, 3)
            return _unit(normals[action_face], fallback=tuple(float(x) for x in fallback))
    return _estimate_normal_from_points(action_points, fallback)


def _flank_radial_dir(row: pd.Series, action_points: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    axis = _unit(axis_dir)
    normal = _load_action_face_normal(row, action_points, fallback=_perpendicular_unit(axis))
    radial = normal - float(normal @ axis) * axis
    if float(np.linalg.norm(radial)) <= 1e-8:
        radial = _perpendicular_unit(axis)
    radial = _unit(radial) * float(FLANK_FACE_NORMAL_FLIP)
    return radial.astype(np.float32)


def _radius_limit_at_s(tool_kind: str, s: np.ndarray, radius: float, tool_length: float) -> np.ndarray:
    """Returns cutter radius at axial coordinate s measured from the tip."""
    out = np.full_like(s, -1.0, dtype=np.float32)
    valid = (s >= 0.0) & (s <= float(tool_length))
    if "ball" in str(tool_kind).lower():
        ball = valid & (s < float(radius))
        out[ball] = np.sqrt(np.maximum(float(radius) ** 2 - (s[ball] - float(radius)) ** 2, 0.0))
        cyl = valid & ~ball
        out[cyl] = float(radius)
    else:
        out[valid] = float(radius)
    return out


def _make_kernel_offsets(
    axis_dir: np.ndarray,
    spacing: tuple[float, float, float],
    tool_kind: str,
    tool_radius: float,
    tool_length: float,
    holder_radius: float,
    holder_length: float,
) -> dict[str, np.ndarray]:
    """Builds asymmetric grid offsets: offset = obstacle_index - tip_index."""
    axis = _unit(axis_dir)
    shaft_dir = -axis if bool(TOOL_EXTENDS_ALONG_NEG_AXIS) else axis
    spacing_arr = np.asarray(spacing, dtype=np.float32).reshape(3)
    total_length = max(float(tool_length) + float(holder_length), 1e-6)
    max_radius = max(float(tool_radius), float(holder_radius), 1e-6)
    half_extent = np.abs(shaft_dir) * total_length + max_radius + spacing_arr * 1.5
    max_steps = np.ceil(half_extent / np.maximum(spacing_arr, 1e-9)).astype(np.int32)

    xs = np.arange(-int(max_steps[0]), int(max_steps[0]) + 1, dtype=np.int32)
    ys = np.arange(-int(max_steps[1]), int(max_steps[1]) + 1, dtype=np.int32)
    zs = np.arange(-int(max_steps[2]), int(max_steps[2]) + 1, dtype=np.int32)
    offsets = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    phys = offsets.astype(np.float32) * spacing_arr.reshape(1, 3)
    s = phys @ shaft_dir.reshape(3, 1)
    s = s.reshape(-1)
    radial_vec = phys - s.reshape(-1, 1) * shaft_dir.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)

    cutter_radius_limit = _radius_limit_at_s(tool_kind, s, float(tool_radius), float(tool_length))
    cutter_mask = cutter_radius_limit >= 0.0
    cutter_mask &= radial <= (cutter_radius_limit + 0.5 * float(spacing_arr.max()))

    holder_start = float(tool_length)
    holder_stop = float(tool_length) + float(holder_length)
    holder_mask = (s >= holder_start) & (s <= holder_stop)
    holder_mask &= radial <= (float(holder_radius) + 0.5 * float(spacing_arr.max()))

    # Keep unique offsets after masks; duplicate zero/overlap does not matter.
    cutter = np.unique(offsets[cutter_mask], axis=0).astype(np.int32)
    holder = np.unique(offsets[holder_mask], axis=0).astype(np.int32)
    full = np.unique(np.vstack([cutter, holder]) if holder.size else cutter, axis=0).astype(np.int32)
    return {"cutter": cutter, "holder": holder, "full": full, "shaft_dir": shaft_dir.reshape(1, 3)}


def _make_flank_kernel_offsets(
    axis_dir: np.ndarray,
    spacing: tuple[float, float, float],
    tool_radius: float,
    tool_length: float,
    holder_radius: float,
    holder_length: float,
    contact_fraction: float,
) -> dict[str, np.ndarray]:
    """Builds offsets around a flank centerline contact configuration point.

    The configuration point is the cutter centerline point nearest the selected
    side-contact point. The flute extends both ways along the tool axis; the
    holder only extends toward the spindle side.
    """
    axis = _unit(axis_dir)
    shaft_dir = -axis if bool(TOOL_EXTENDS_ALONG_NEG_AXIS) else axis
    spacing_arr = np.asarray(spacing, dtype=np.float32).reshape(3)
    contact_fraction = float(np.clip(contact_fraction, 0.0, 1.0))
    cutter_min_s = -contact_fraction * float(tool_length)
    cutter_max_s = (1.0 - contact_fraction) * float(tool_length)
    holder_min_s = cutter_max_s
    holder_max_s = cutter_max_s + float(holder_length)

    max_back = abs(cutter_min_s)
    max_forward = max(cutter_max_s, holder_max_s)
    total_extent = max(max_back, max_forward, 1e-6)
    max_radius = max(float(tool_radius), float(holder_radius), 1e-6)
    half_extent = np.abs(shaft_dir) * total_extent + max_radius + spacing_arr * 1.5
    max_steps = np.ceil(half_extent / np.maximum(spacing_arr, 1e-9)).astype(np.int32)

    xs = np.arange(-int(max_steps[0]), int(max_steps[0]) + 1, dtype=np.int32)
    ys = np.arange(-int(max_steps[1]), int(max_steps[1]) + 1, dtype=np.int32)
    zs = np.arange(-int(max_steps[2]), int(max_steps[2]) + 1, dtype=np.int32)
    offsets = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    phys = offsets.astype(np.float32) * spacing_arr.reshape(1, 3)
    s = (phys @ shaft_dir.reshape(3, 1)).reshape(-1)
    radial_vec = phys - s.reshape(-1, 1) * shaft_dir.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)

    radius_pad = 0.5 * float(spacing_arr.max())
    cutter_mask = (s >= cutter_min_s) & (s <= cutter_max_s)
    cutter_mask &= radial <= (float(tool_radius) + radius_pad)

    holder_mask = (s >= holder_min_s) & (s <= holder_max_s)
    holder_mask &= radial <= (float(holder_radius) + radius_pad)

    cutter = np.unique(offsets[cutter_mask], axis=0).astype(np.int32)
    holder = np.unique(offsets[holder_mask], axis=0).astype(np.int32)
    full = np.unique(np.vstack([cutter, holder]) if holder.size else cutter, axis=0).astype(np.int32)
    return {"cutter": cutter, "holder": holder, "full": full, "shaft_dir": shaft_dir.reshape(1, 3)}


def _shift_or_mask(solid: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Computes forbidden[tip] = OR_offsets solid[tip + offset]."""
    out = np.zeros_like(solid, dtype=bool)
    nx, ny, nz = solid.shape
    for ox, oy, oz in offsets.astype(np.int32):
        dst_x0, dst_x1 = max(0, -int(ox)), min(nx, nx - int(ox))
        dst_y0, dst_y1 = max(0, -int(oy)), min(ny, ny - int(oy))
        dst_z0, dst_z1 = max(0, -int(oz)), min(nz, nz - int(oz))
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1 or dst_z0 >= dst_z1:
            continue
        src_x0, src_x1 = dst_x0 + int(ox), dst_x1 + int(ox)
        src_y0, src_y1 = dst_y0 + int(oy), dst_y1 + int(oy)
        src_z0, src_z1 = dst_z0 + int(oz), dst_z1 + int(oz)
        out[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] |= solid[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]
    return out


def _dilate_config_mask(config_mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Computes swept[tip + offset] = True for accessible cutter placements."""
    out = np.zeros_like(config_mask, dtype=bool)
    nx, ny, nz = config_mask.shape
    for ox, oy, oz in offsets.astype(np.int32):
        src_x0, src_x1 = max(0, -int(ox)), min(nx, nx - int(ox))
        src_y0, src_y1 = max(0, -int(oy)), min(ny, ny - int(oy))
        src_z0, src_z1 = max(0, -int(oz)), min(nz, nz - int(oz))
        if src_x0 >= src_x1 or src_y0 >= src_y1 or src_z0 >= src_z1:
            continue
        dst_x0, dst_x1 = src_x0 + int(ox), src_x1 + int(ox)
        dst_y0, dst_y1 = src_y0 + int(oy), src_y1 + int(oy)
        dst_z0, dst_z1 = src_z0 + int(oz), src_z1 + int(oz)
        out[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] |= config_mask[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]
    return out


def _tsdf_from_solid(
    solid: np.ndarray,
    spacing: tuple[float, float, float],
    truncation: float,
) -> np.ndarray:
    """Builds a normalized TSDF from a binary solid grid; inside material is negative."""
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("scipy is required for synthetic TSDF distance transforms.") from exc

    spacing_arr = tuple(float(max(value, 1e-9)) for value in spacing)
    outside = ndimage.distance_transform_edt(~solid, sampling=spacing_arr)
    inside = ndimage.distance_transform_edt(solid, sampling=spacing_arr)
    sdf = outside - inside
    tau = max(float(truncation), 1e-6)
    return np.clip(sdf / tau, -1.0, 1.0).astype(np.float32)


def _use_action_face_roi(row: pd.Series) -> bool:
    mode = str(ACTION_FACE_ROI_MODE).strip().lower()
    if mode == "all":
        return False
    if mode == "action_face":
        return True
    if mode != "auto":
        raise ValueError("ACTION_FACE_ROI_MODE must be 'auto', 'all', or 'action_face'.")
    macro = str(_row_value(row, "macro_class_name", "")).lower()
    return "rough" not in macro


def _build_action_roi_mask(
    row: pd.Series,
    grid_points: np.ndarray,
    dims: tuple[int, int, int],
    action_points: np.ndarray,
    tool_radius: float,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Limits finishing-style synthetic removal to the selected B-rep face vicinity."""
    if not _use_action_face_roi(row) or action_points.size == 0:
        return np.ones(dims, dtype=bool)

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("scipy is required for action-face ROI generation.") from exc

    spacing_max = float(np.max(np.asarray(spacing, dtype=np.float32)))
    padding = float(ACTION_FACE_ROI_PADDING_MM)
    if padding <= 0.0:
        padding = max(float(tool_radius) * 2.5, spacing_max * 3.0)

    tree = cKDTree(action_points.astype(np.float32, copy=False))
    dist, _ = tree.query(grid_points.astype(np.float32, copy=False), k=1, workers=-1)
    return (np.asarray(dist, dtype=np.float32).reshape(dims) <= padding)


def _make_image_data(
    values: np.ndarray,
    name: str,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    dims: tuple[int, int, int],
) -> pv.ImageData:
    image = pv.ImageData(dimensions=dims, spacing=spacing, origin=tuple(float(x) for x in bbox_min))
    image.point_data[name] = values.reshape(dims, order="C").ravel(order="F").astype(np.float32)
    return image


def _surface_from_mask(
    mask: np.ndarray,
    name: str,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    dims: tuple[int, int, int],
) -> pv.PolyData:
    if not bool(mask.any()):
        return pv.PolyData()
    image = _make_image_data(mask.astype(np.float32), name, bbox_min, spacing, dims)
    try:
        return image.contour([0.5], scalars=name).extract_surface().triangulate().clean()
    except Exception:
        return pv.PolyData()


def _poly(points: np.ndarray, scalars: np.ndarray | None = None, name: str = "values") -> pv.PolyData:
    cloud = pv.PolyData(points.astype(np.float32).reshape(-1, 3))
    if scalars is not None:
        cloud[name] = scalars.astype(np.float32).reshape(-1)
    return cloud


def _sample_indices(count: int, limit: int, seed: int = 0) -> np.ndarray:
    if limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(count, size=int(limit), replace=False).astype(np.int64)


def _grid_nearest_indices(points: np.ndarray, bbox_min: np.ndarray, spacing: tuple[float, float, float], dims: tuple[int, int, int]) -> np.ndarray:
    spacing_arr = np.asarray(spacing, dtype=np.float32).reshape(1, 3)
    idx = np.rint((points.astype(np.float32) - bbox_min.reshape(1, 3)) / spacing_arr).astype(np.int32)
    max_idx = np.asarray(dims, dtype=np.int32).reshape(1, 3) - 1
    return np.minimum(np.maximum(idx, 0), max_idx)


def _grid_points_from_mask(
    mask: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    limit: int,
    seed: int,
) -> np.ndarray:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    keep = _sample_indices(ijk.shape[0], limit, seed=seed)
    ijk = ijk[keep]
    return bbox_min.reshape(1, 3) + ijk.astype(np.float32) * np.asarray(spacing, dtype=np.float32).reshape(1, 3)


def _load_action_face_points(row: pd.Series) -> np.ndarray:
    static_dir = _row_value(row, "static_feature_dir", "")
    action_face = int(_row_value(row, "action_face_id", -1))
    if not static_dir or not (0 <= action_face < 512):
        return np.empty((0, 3), dtype=np.float32)
    face_pc_path = Path(str(static_dir)) / "embed_face_pc.npy"
    if not face_pc_path.exists():
        return np.empty((0, 3), dtype=np.float32)
    face_pc = np.load(face_pc_path).astype(np.float32).reshape(512, 100, 3)
    scale, center = _scale_center(row)
    pts = face_pc[action_face].reshape(-1, 3) * scale + center.reshape(1, 3)
    pts = pts[np.any(np.abs(pts) > 1e-12, axis=1)]
    return pts.astype(np.float32)


def _make_tool_mesh(
    tip: np.ndarray,
    axis_dir: np.ndarray,
    tool_kind: str,
    tool_radius: float,
    tool_length: float,
    holder_radius: float,
    holder_length: float,
) -> pv.MultiBlock:
    axis = _unit(axis_dir)
    shaft_dir = -axis if bool(TOOL_EXTENDS_ALONG_NEG_AXIS) else axis
    tip = np.asarray(tip, dtype=np.float32).reshape(3)
    parts = pv.MultiBlock()

    if "ball" in tool_kind:
        ball_center = tip + shaft_dir * float(tool_radius)
        parts.append(pv.Sphere(radius=float(tool_radius), center=tuple(float(x) for x in ball_center)))
        cyl_len = max(float(tool_length) - float(tool_radius), 1e-6)
        cyl_center = tip + shaft_dir * (float(tool_radius) + 0.5 * cyl_len)
    else:
        cyl_len = max(float(tool_length), 1e-6)
        cyl_center = tip + shaft_dir * (0.5 * cyl_len)

    parts.append(
        pv.Cylinder(
            center=tuple(float(x) for x in cyl_center),
            direction=tuple(float(x) for x in shaft_dir),
            radius=float(tool_radius),
            height=float(cyl_len),
            resolution=36,
        )
    )
    if holder_length > 0.0 and holder_radius > 0.0:
        holder_center = tip + shaft_dir * (float(tool_length) + 0.5 * float(holder_length))
        parts.append(
            pv.Cylinder(
                center=tuple(float(x) for x in holder_center),
                direction=tuple(float(x) for x in shaft_dir),
                radius=float(holder_radius),
                height=float(holder_length),
                resolution=48,
            )
        )
    return parts


def _make_flank_tool_mesh(
    contact_point: np.ndarray,
    centerline_point: np.ndarray,
    axis_dir: np.ndarray,
    tool_radius: float,
    tool_length: float,
    holder_radius: float,
    holder_length: float,
    contact_fraction: float,
) -> pv.MultiBlock:
    shaft_dir = -_unit(axis_dir) if bool(TOOL_EXTENDS_ALONG_NEG_AXIS) else _unit(axis_dir)
    centerline_point = np.asarray(centerline_point, dtype=np.float32).reshape(3)
    contact_point = np.asarray(contact_point, dtype=np.float32).reshape(3)
    contact_fraction = float(np.clip(contact_fraction, 0.0, 1.0))

    cutter_min_s = -contact_fraction * float(tool_length)
    cutter_max_s = (1.0 - contact_fraction) * float(tool_length)
    cutter_center = centerline_point + shaft_dir * (0.5 * (cutter_min_s + cutter_max_s))
    holder_center = centerline_point + shaft_dir * (cutter_max_s + 0.5 * float(holder_length))

    parts = pv.MultiBlock()
    parts.append(
        pv.Cylinder(
            center=tuple(float(x) for x in cutter_center),
            direction=tuple(float(x) for x in shaft_dir),
            radius=float(tool_radius),
            height=max(float(tool_length), 1e-6),
            resolution=48,
        )
    )
    if holder_length > 0.0 and holder_radius > 0.0:
        parts.append(
            pv.Cylinder(
                center=tuple(float(x) for x in holder_center),
                direction=tuple(float(x) for x in shaft_dir),
                radius=float(holder_radius),
                height=float(holder_length),
                resolution=48,
            )
        )
    parts.append(pv.Sphere(radius=max(float(tool_radius) * 0.12, 1e-6), center=tuple(float(x) for x in contact_point)))
    return parts


def _add_multiblock(plotter: pv.Plotter, block: pv.MultiBlock, **kwargs: Any) -> None:
    for item in block:
        if item is not None and item.n_points > 0:
            plotter.add_mesh(item, **kwargs)


def main() -> None:
    df, parquet_path = _load_rows()
    row = _select_row(df)
    source_row_index = int(_row_value(row, "source_row_index", ROW_INDEX))
    print(
        f"[Rows] parquet={parquet_path} filtered_rows={len(df)} "
        f"row_index={ROW_INDEX} source_row_index={source_row_index} "
        f"macro={_row_value(row, 'macro_class_name', '')}",
        flush=True,
    )

    fields = _load_sdf_samples(row)
    sample_xyz = fields["points"]
    target_mesh = _load_target_mesh(row)
    bbox_min, bbox_max = _build_grid_bounds(sample_xyz, target_mesh)
    dims = _grid_dimensions(bbox_min, bbox_max)
    grid_points, spacing = _make_grid_points(bbox_min, bbox_max, dims)

    print(f"[Grid] dims={dims} points={grid_points.shape[0]} spacing={spacing}", flush=True)
    tsdf_grids: dict[str, np.ndarray] = {}
    for name in ("before", "after", "target"):
        print(f"[Interpolate] {name}", flush=True)
        tsdf_grids[name] = _interpolate_tsdf(sample_xyz, fields[name], grid_points).reshape(dims)

    before_solid = tsdf_grids["before"] <= float(OCCUPIED_TSDF_THRESHOLD)
    target_solid = tsdf_grids["target"] <= float(OCCUPIED_TSDF_THRESHOLD)
    after_solid = tsdf_grids["after"] <= float(OCCUPIED_TSDF_THRESHOLD)
    solid_by_name = {"before": before_solid, "target": target_solid, "after": after_solid}
    holder_obstacle = solid_by_name[str(HOLDER_COLLISION_FIELD).strip().lower()]
    cutter_obstacle = solid_by_name[str(CUTTER_COLLISION_FIELD).strip().lower()]

    axis = _row_axis(row)
    mode = _minkowski_mode(row)
    tool_kind = _tool_kind(row)
    tool_diameter = _tool_diameter(row)
    tool_radius = 0.5 * float(tool_diameter)
    holder_radius = 0.5 * float(HOLDER_DIAMETER_MM)
    changed = np.abs(fields["after"] - fields["before"]) > float(CHANGE_EPS)
    removed = (fields["after"] - fields["before"]) > float(REMOVED_EPS)
    changed_or_removed = changed | removed
    action_points = _load_action_face_points(row)
    if action_points.size > 0:
        contact_ref = action_points.mean(axis=0)
    elif bool(changed_or_removed.any()):
        contact_ref = sample_xyz[changed_or_removed].mean(axis=0)
    else:
        contact_ref = sample_xyz.mean(axis=0)

    flank_radial = np.zeros((3,), dtype=np.float32)
    if mode == "flank":
        flank_radial = _flank_radial_dir(row, action_points, axis)
        config_ref = contact_ref + flank_radial * float(tool_radius)
        kernels = _make_flank_kernel_offsets(
            axis_dir=axis,
            spacing=spacing,
            tool_radius=tool_radius,
            tool_length=float(TOOL_LENGTH_MM),
            holder_radius=holder_radius,
            holder_length=float(HOLDER_LENGTH_MM),
            contact_fraction=float(FLANK_CONTACT_AXIAL_FRACTION),
        )
    else:
        config_ref = contact_ref
        kernels = _make_kernel_offsets(
            axis_dir=axis,
            spacing=spacing,
            tool_kind=tool_kind,
            tool_radius=tool_radius,
            tool_length=float(TOOL_LENGTH_MM),
            holder_radius=holder_radius,
            holder_length=float(HOLDER_LENGTH_MM),
        )
    print(
        "[Kernel] "
        f"mode={mode} kind={tool_kind} tool_dia={tool_diameter:.3f} tool_len={TOOL_LENGTH_MM:.3f} "
        f"holder_dia={HOLDER_DIAMETER_MM:.3f} holder_len={HOLDER_LENGTH_MM:.3f} "
        f"cutter_offsets={len(kernels['cutter'])} holder_offsets={len(kernels['holder'])}",
        flush=True,
    )

    print("[C-space] cutter gouge forbidden", flush=True)
    if mode == "flank" and not bool(FLANK_CHECK_CUTTER_GOUGE):
        cutter_forbidden = np.zeros_like(cutter_obstacle, dtype=bool)
    else:
        cutter_forbidden = _shift_or_mask(cutter_obstacle, kernels["cutter"])
    print("[C-space] holder collision forbidden", flush=True)
    holder_forbidden = _shift_or_mask(holder_obstacle, kernels["holder"])
    cspace_forbidden = cutter_forbidden | holder_forbidden

    print("[Synthetic] swept cutter and after-IPW label", flush=True)
    accessible_config = ~cspace_forbidden
    swept_cutter = _dilate_config_mask(accessible_config, kernels["cutter"])
    action_roi_mask = _build_action_roi_mask(
        row=row,
        grid_points=grid_points,
        dims=dims,
        action_points=action_points,
        tool_radius=tool_radius,
        spacing=spacing,
    )
    ideal_removal = before_solid & ~target_solid
    synthetic_removed = ideal_removal & swept_cutter & action_roi_mask
    synthetic_after_solid = before_solid & ~synthetic_removed
    synthetic_after_tsdf = _tsdf_from_solid(
        synthetic_after_solid,
        spacing=spacing,
        truncation=float(TSDF_TRUNCATION_MM),
    )

    cam_removed = before_solid & ~after_solid
    removal_agreement = synthetic_removed & cam_removed
    missed_removal = cam_removed & ~synthetic_removed
    extra_removal = synthetic_removed & ~cam_removed
    removal_union = synthetic_removed | cam_removed

    query_config_points = sample_xyz + flank_radial.reshape(1, 3) * float(tool_radius) if mode == "flank" else sample_xyz
    grid_idx = _grid_nearest_indices(query_config_points, bbox_min, spacing, dims)
    query_blocked = cspace_forbidden[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]]
    sample_grid_idx = _grid_nearest_indices(sample_xyz, bbox_min, spacing, dims)
    synthetic_query_tsdf = synthetic_after_tsdf[
        sample_grid_idx[:, 0],
        sample_grid_idx[:, 1],
        sample_grid_idx[:, 2],
    ]
    synthetic_query_removed = (synthetic_query_tsdf - fields["before"]) > float(REMOVED_EPS)
    cam_query_removed = removed

    stock_surface = _surface_from_mask(before_solid, "before_solid", bbox_min, spacing, dims)
    target_surface = _surface_from_mask(target_solid, "target_solid", bbox_min, spacing, dims)
    cam_after_surface = _surface_from_mask(after_solid, "cam_after_solid", bbox_min, spacing, dims)
    synthetic_after_surface = _surface_from_mask(synthetic_after_solid, "synthetic_after_solid", bbox_min, spacing, dims)
    cutter_cspace_surface = _surface_from_mask(cutter_forbidden, "cutter_forbidden", bbox_min, spacing, dims)
    holder_cspace_surface = _surface_from_mask(holder_forbidden, "holder_forbidden", bbox_min, spacing, dims)
    full_cspace_surface = _surface_from_mask(cspace_forbidden, "cspace_forbidden", bbox_min, spacing, dims)
    swept_cutter_surface = _surface_from_mask(swept_cutter, "swept_cutter", bbox_min, spacing, dims)
    action_roi_surface = (
        _surface_from_mask(action_roi_mask, "action_face_roi", bbox_min, spacing, dims)
        if not bool(action_roi_mask.all())
        else pv.PolyData()
    )
    agreement_surface = _surface_from_mask(removal_agreement, "removal_agreement", bbox_min, spacing, dims)
    missed_surface = _surface_from_mask(missed_removal, "missed_removal", bbox_min, spacing, dims)
    extra_surface = _surface_from_mask(extra_removal, "extra_removal", bbox_min, spacing, dims)

    out_dir = Path(OUTPUT_DIR).expanduser().resolve() / f"minkowski_cspace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_row{source_row_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cspace_points = _grid_points_from_mask(cspace_forbidden, bbox_min, spacing, MAX_CSPACE_POINTS_VIS, seed=1)
    holder_points = _grid_points_from_mask(holder_forbidden, bbox_min, spacing, MAX_CSPACE_POINTS_VIS, seed=2)
    cutter_points = _grid_points_from_mask(cutter_forbidden, bbox_min, spacing, MAX_CSPACE_POINTS_VIS, seed=3)
    kernel_offsets = kernels["full"]
    if kernel_offsets.shape[0] > MAX_KERNEL_POINTS_VIS > 0:
        kernel_offsets = kernel_offsets[_sample_indices(kernel_offsets.shape[0], MAX_KERNEL_POINTS_VIS, seed=4)]
    kernel_points = config_ref.reshape(1, 3) + kernel_offsets.astype(np.float32) * np.asarray(spacing, dtype=np.float32).reshape(1, 3)

    query_keep = _sample_indices(sample_xyz.shape[0], MAX_QUERY_POINTS_VIS, seed=5)
    changed_keep_all = np.where(changed_or_removed)[0]
    changed_keep = changed_keep_all[_sample_indices(changed_keep_all.size, MAX_QUERY_POINTS_VIS, seed=6)] if changed_keep_all.size else np.asarray([], dtype=np.int64)

    summary = {
        "parquet_path": str(parquet_path),
        "source_row_index": source_row_index,
        "part_name": str(_row_value(row, "part_name", "")),
        "decision_step": int(_row_value(row, "decision_step", -1)),
        "candidate_index": int(_row_value(row, "candidate_index", -1)),
        "action_face_id": int(_row_value(row, "action_face_id", -1)),
        "macro_class_name": str(_row_value(row, "macro_class_name", "")),
        "macro_class_id": int(_row_value(row, "macro_class_id", -1)),
        "tool_choice_name": str(_row_value(row, "tool_choice_name", "")),
        "tool_choice_id": int(_row_value(row, "tool_choice_id", -1)),
        "minkowski_mode": mode,
        "tool_kind": tool_kind,
        "tool_diameter_mm": float(tool_diameter),
        "tool_radius_mm": float(tool_radius),
        "tool_length_mm": float(TOOL_LENGTH_MM),
        "holder_diameter_mm": float(HOLDER_DIAMETER_MM),
        "holder_radius_mm": float(holder_radius),
        "holder_length_mm": float(HOLDER_LENGTH_MM),
        "axis_dir": [float(x) for x in axis],
        "flank_radial_dir": [float(x) for x in flank_radial],
        "flank_contact_axial_fraction": float(FLANK_CONTACT_AXIAL_FRACTION),
        "flank_check_cutter_gouge": bool(FLANK_CHECK_CUTTER_GOUGE),
        "reference_contact_point": [float(x) for x in contact_ref],
        "reference_config_point": [float(x) for x in config_ref],
        "tool_extends_along_neg_axis": bool(TOOL_EXTENDS_ALONG_NEG_AXIS),
        "holder_collision_field": str(HOLDER_COLLISION_FIELD),
        "cutter_collision_field": str(CUTTER_COLLISION_FIELD),
        "grid_dimensions": list(map(int, dims)),
        "grid_points": int(grid_points.shape[0]),
        "kernel_cutter_offsets": int(kernels["cutter"].shape[0]),
        "kernel_holder_offsets": int(kernels["holder"].shape[0]),
        "before_solid_ratio": float(before_solid.mean()),
        "target_solid_ratio": float(target_solid.mean()),
        "cam_after_solid_ratio": float(after_solid.mean()),
        "synthetic_after_solid_ratio": float(synthetic_after_solid.mean()),
        "ideal_removal_ratio": float(ideal_removal.mean()),
        "swept_cutter_ratio": float(swept_cutter.mean()),
        "action_face_roi_enabled": bool(not action_roi_mask.all()),
        "action_face_roi_ratio": float(action_roi_mask.mean()),
        "synthetic_removed_ratio": float(synthetic_removed.mean()),
        "cam_removed_ratio": float(cam_removed.mean()),
        "removal_iou": (
            float(removal_agreement.sum() / max(float(removal_union.sum()), 1.0))
        ),
        "removal_precision": (
            float(removal_agreement.sum() / max(float(synthetic_removed.sum()), 1.0))
        ),
        "removal_recall": (
            float(removal_agreement.sum() / max(float(cam_removed.sum()), 1.0))
        ),
        "grid_synthetic_vs_cam_tsdf_mae": float(np.mean(np.abs(synthetic_after_tsdf - tsdf_grids["after"]))),
        "cutter_forbidden_ratio": float(cutter_forbidden.mean()),
        "holder_forbidden_ratio": float(holder_forbidden.mean()),
        "cspace_forbidden_ratio": float(cspace_forbidden.mean()),
        "query_points": int(sample_xyz.shape[0]),
        "changed_or_removed_query_points": int(changed_or_removed.sum()),
        "synthetic_query_removed_points": int(synthetic_query_removed.sum()),
        "cam_query_removed_points": int(cam_query_removed.sum()),
        "query_synthetic_vs_cam_tsdf_mae": float(np.mean(np.abs(synthetic_query_tsdf - fields["after"]))),
        "blocked_query_ratio": float(query_blocked.mean()) if query_blocked.size else 0.0,
        "blocked_changed_or_removed_ratio": (
            float(query_blocked[changed_or_removed].mean()) if bool(changed_or_removed.any()) else 0.0
        ),
        "accessible_changed_or_removed_ratio": (
            float((~query_blocked[changed_or_removed]).mean()) if bool(changed_or_removed.any()) else 0.0
        ),
        "target_mesh_path": _resolve_target_mesh_path(row) or "",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if EXPORT_VTK_FILES:
        if stock_surface.n_points > 0:
            stock_surface.save(str(out_dir / "before_solid_surface.vtp"))
        if target_surface.n_points > 0:
            target_surface.save(str(out_dir / "target_solid_surface.vtp"))
        if cam_after_surface.n_points > 0:
            cam_after_surface.save(str(out_dir / "cam_after_solid_surface.vtp"))
        if synthetic_after_surface.n_points > 0:
            synthetic_after_surface.save(str(out_dir / "synthetic_after_solid_surface.vtp"))
        if cutter_cspace_surface.n_points > 0:
            cutter_cspace_surface.save(str(out_dir / "cutter_forbidden_cspace.vtp"))
        if holder_cspace_surface.n_points > 0:
            holder_cspace_surface.save(str(out_dir / "holder_forbidden_cspace.vtp"))
        if full_cspace_surface.n_points > 0:
            full_cspace_surface.save(str(out_dir / "full_forbidden_cspace.vtp"))
        if swept_cutter_surface.n_points > 0:
            swept_cutter_surface.save(str(out_dir / "swept_accessible_cutter_volume.vtp"))
        if action_roi_surface.n_points > 0:
            action_roi_surface.save(str(out_dir / "action_face_roi.vtp"))
        if agreement_surface.n_points > 0:
            agreement_surface.save(str(out_dir / "removed_agreement.vtp"))
        if missed_surface.n_points > 0:
            missed_surface.save(str(out_dir / "removed_missed_by_synthetic.vtp"))
        if extra_surface.n_points > 0:
            extra_surface.save(str(out_dir / "removed_extra_by_synthetic.vtp"))
        _make_image_data(
            synthetic_after_tsdf,
            "synthetic_after_tsdf",
            bbox_min,
            spacing,
            dims,
        ).save(str(out_dir / "synthetic_after_tsdf.vti"))
        if cspace_points.size > 0:
            _poly(cspace_points).save(str(out_dir / "sampled_cspace_forbidden_points.vtp"))
        if kernel_points.size > 0:
            _poly(kernel_points).save(str(out_dir / "tool_holder_kernel_at_reference_config.vtp"))
        query_cloud = _poly(sample_xyz[query_keep], query_blocked[query_keep].astype(np.float32), "blocked")
        query_cloud.save(str(out_dir / "query_points_blocked_flag.vtp"))

    plotter = pv.Plotter(shape=(2, 2), window_size=(1900, 1400), off_screen=not SHOW_WINDOW)
    if mode == "flank":
        tool_mesh = _make_flank_tool_mesh(
            contact_point=contact_ref,
            centerline_point=config_ref,
            axis_dir=axis,
            tool_radius=tool_radius,
            tool_length=float(TOOL_LENGTH_MM),
            holder_radius=holder_radius,
            holder_length=float(HOLDER_LENGTH_MM),
            contact_fraction=float(FLANK_CONTACT_AXIAL_FRACTION),
        )
    else:
        tool_mesh = _make_tool_mesh(
            tip=config_ref,
            axis_dir=axis,
            tool_kind=tool_kind,
            tool_radius=tool_radius,
            tool_length=float(TOOL_LENGTH_MM),
            holder_radius=holder_radius,
            holder_length=float(HOLDER_LENGTH_MM),
        )

    plotter.subplot(0, 0)
    plotter.add_text(
        "Minimal transition input: S_t, target, action face, tool/holder/axis\n"
        f"macro={_row_value(row, 'macro_class_name', '')} mode={mode}",
        font_size=9,
    )
    if stock_surface.n_points > 0:
        plotter.add_mesh(stock_surface, color="sienna", opacity=STOCK_SURFACE_OPACITY, show_edges=False)
    if target_mesh is not None and target_mesh.n_points > 0:
        plotter.add_mesh(target_mesh, color="silver", opacity=TARGET_MESH_OPACITY, show_edges=False)
    elif target_surface.n_points > 0:
        plotter.add_mesh(target_surface, color="silver", opacity=TARGET_MESH_OPACITY, show_edges=False)
    _add_multiblock(plotter, tool_mesh, color="deepskyblue", opacity=0.55, show_edges=True)
    plotter.add_arrows(config_ref.reshape(1, 3), axis.reshape(1, 3), mag=max(float(np.linalg.norm(bbox_max - bbox_min)) * 0.12, 1e-6), color="red")
    if mode == "flank":
        plotter.add_arrows(contact_ref.reshape(1, 3), flank_radial.reshape(1, 3), mag=max(float(tool_radius), 1e-6), color="yellow")
    if action_points.size > 0:
        plotter.add_mesh(_poly(action_points), color="black", point_size=7, render_points_as_spheres=True)
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text(
        "Minkowski/C-space label generator: forbidden configs + swept cutter",
        font_size=9,
    )
    if cutter_cspace_surface.n_points > 0:
        plotter.add_mesh(cutter_cspace_surface, color="orange", opacity=CSPACE_SURFACE_OPACITY, show_edges=False, label="cutter")
    if holder_cspace_surface.n_points > 0:
        plotter.add_mesh(holder_cspace_surface, color="dodgerblue", opacity=CSPACE_SURFACE_OPACITY, show_edges=False, label="holder")
    if swept_cutter_surface.n_points > 0:
        plotter.add_mesh(swept_cutter_surface, color="limegreen", opacity=0.12, show_edges=False, label="swept cutter")
    if action_roi_surface.n_points > 0:
        plotter.add_mesh(action_roi_surface, color="yellow", opacity=0.10, show_edges=False, label="action ROI")
    if target_surface.n_points > 0:
        plotter.add_mesh(target_surface, color="gray", opacity=0.06, show_edges=False)
    plotter.add_axes()

    plotter.subplot(1, 0)
    plotter.add_text("Synthetic S_{t+1} label vs CAM after", font_size=10)
    if synthetic_after_surface.n_points > 0:
        plotter.add_mesh(synthetic_after_surface, color="mediumseagreen", opacity=0.42, show_edges=False, label="synthetic")
    if cam_after_surface.n_points > 0:
        plotter.add_mesh(cam_after_surface, color="gold", opacity=0.28, show_edges=False, label="CAM after")
    if target_surface.n_points > 0:
        plotter.add_mesh(target_surface, color="gray", opacity=0.06, show_edges=False, label="target")
    plotter.add_axes()

    plotter.subplot(1, 1)
    plotter.add_text(
        "Removed volume comparison: green=match, red=missed, blue=extra",
        font_size=9,
    )
    if target_surface.n_points > 0:
        plotter.add_mesh(target_surface, color="lightgray", opacity=0.05, show_edges=False)
    if agreement_surface.n_points > 0:
        plotter.add_mesh(agreement_surface, color="lime", opacity=0.48, show_edges=False)
    if missed_surface.n_points > 0:
        plotter.add_mesh(missed_surface, color="red", opacity=0.58, show_edges=False)
    if extra_surface.n_points > 0:
        plotter.add_mesh(extra_surface, color="dodgerblue", opacity=0.52, show_edges=False)
    if changed_keep.size > 0:
        changed_points = sample_xyz[changed_keep]
        blocked = query_blocked[changed_keep]
        if bool(blocked.any()):
            plotter.add_mesh(
                _poly(changed_points[blocked]),
                color="black",
                point_size=max(POINT_SIZE - 1, 2),
                render_points_as_spheres=True,
            )
    plotter.add_axes()

    screenshot_path = Path(SCREENSHOT_PATH).expanduser().resolve() if SCREENSHOT_PATH.strip() else out_dir / "minkowski_cspace.png"
    plotter.link_views()
    plotter.view_isometric()
    if SHOW_WINDOW:
        plotter.show(screenshot=str(screenshot_path))
    else:
        plotter.screenshot(str(screenshot_path))
        plotter.close()

    print(f"[Done] output_dir={out_dir}")
    print(f"[Done] screenshot={screenshot_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
