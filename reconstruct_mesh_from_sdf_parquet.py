r"""Reconstruct a mesh from one parquet row's SDF-query TSDF samples.

This script does not require NX. It interpolates irregular SDF query samples
onto a regular grid, extracts the TSDF=0 iso-surface, and saves OBJ/VTP files.

Edit the constants below and run directly from VS Code/debug mode.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
PARQUET_PATH = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\_ALL_PARQUET_FILES\3dDataset2264_seed0_20260510_184822.parquet"
PARQUET_DIR = r""
PARQUET_GLOB = "*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []

# Set ROW_INDEX=-1 to pick the row with the largest mean |after-before TSDF|.
ROW_INDEX = -1
CHOSEN_ONLY = False

# Which TSDF field to reconstruct: "after", "before", or "target".
RECONSTRUCT_FIELD = "after"

OUTPUT_DIR = r"sdf_mesh_reconstructions"

# Regular grid quality/cost. 128 is usually a good first high-quality check.
GRID_RESOLUTION = 512
GRID_PADDING_RATIO = 0.04

# Interpolation from irregular query samples to the regular grid.
K_NEIGHBORS = 16
IDW_POWER = 2.0
INTERPOLATION_CHUNK_POINTS = 250_000

# Use final target CAD OBJ only as a weak geometric prior. This can stabilize
# regions with sparse query samples, but set False if you want parquet samples
# only.
USE_TARGET_SURFACE_PRIOR = True
TARGET_SURFACE_PRIOR_WEIGHT = 0.20
MAX_TARGET_PRIOR_POINTS = 80_000
TARGET_MESH_PATH = r""  # optional override; otherwise uses row["target_body_mesh_path"]

# Mesh post-processing.
EXTRACT_LARGEST_COMPONENT = True
SMOOTH_ITERATIONS = 50

# Debug exports.
EXPORT_GRID_VTI = False
EXPORT_QUERY_POINT_CLOUD = True


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


def _load_rows() -> tuple[pd.DataFrame, Path]:
    path = _resolve_files()[0]
    df = pd.read_parquet(path)
    if CHOSEN_ONLY:
        if "is_chosen" not in df.columns:
            raise KeyError("CHOSEN_ONLY=True but parquet has no is_chosen column.")
        df = df[df["is_chosen"].astype(int) == 1].reset_index(drop=False).rename(columns={"index": "source_row_index"})
    else:
        df = df.reset_index(drop=False).rename(columns={"index": "source_row_index"})
    if df.empty:
        raise ValueError("No rows selected.")
    return df, path


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
    candidates = [
        TARGET_MESH_PATH,
        _row_value(row, "target_body_mesh_path", ""),
    ]
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


def _load_target_mesh(row: pd.Series) -> pv.PolyData | None:
    mesh_path = _resolve_target_mesh_path(row)
    if mesh_path is None:
        return None
    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    return mesh.extract_surface().triangulate()


def _load_sdf_samples(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    points_norm = _array(row["sdf_query_points"], np.float32).reshape(-1, 3)
    if RECONSTRUCT_FIELD == "after":
        values = _array(row["sdf_tsdf_after"], np.float32)
    elif RECONSTRUCT_FIELD == "before":
        values = _array(row["sdf_tsdf_before"], np.float32)
    elif RECONSTRUCT_FIELD == "target":
        if "sdf_target_tsdf" not in row.index or _is_missing(row["sdf_target_tsdf"]):
            raise KeyError("RECONSTRUCT_FIELD='target' requires sdf_target_tsdf.")
        values = _array(row["sdf_target_tsdf"], np.float32)
    else:
        raise ValueError("RECONSTRUCT_FIELD must be one of: after, before, target.")
    count = min(points_norm.shape[0], values.size)
    return _denormalize_points(points_norm[:count], row), values[:count].astype(np.float32, copy=False)


def _sample_target_prior(mesh: pv.PolyData) -> np.ndarray:
    points = np.asarray(mesh.points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] <= MAX_TARGET_PRIOR_POINTS or MAX_TARGET_PRIOR_POINTS <= 0:
        return points
    rng = np.random.default_rng(0)
    keep = rng.choice(points.shape[0], size=int(MAX_TARGET_PRIOR_POINTS), replace=False)
    return points[keep]


def _build_grid_bounds(samples_xyz: np.ndarray, target_mesh: pv.PolyData | None) -> tuple[np.ndarray, np.ndarray]:
    mins = [samples_xyz.min(axis=0)]
    maxs = [samples_xyz.max(axis=0)]
    if target_mesh is not None:
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


def _make_grid_points(bbox_min: np.ndarray, bbox_max: np.ndarray, dims: tuple[int, int, int]) -> tuple[np.ndarray, tuple[float, float, float]]:
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
    sample_weight: np.ndarray,
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
        weights = (1.0 / np.maximum(dist, eps) ** float(IDW_POWER)) * sample_weight[idx]
        out[start:stop] = (weights * sample_tsdf[idx]).sum(axis=1) / np.maximum(weights.sum(axis=1), eps)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _make_image_data(
    tsdf_flat: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    dims: tuple[int, int, int],
) -> pv.ImageData:
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=tuple(float(x) for x in bbox_min))
    grid.point_data["tsdf"] = tsdf_flat.reshape(dims, order="C").ravel(order="F")
    return grid


def _postprocess_surface(surface: pv.PolyData) -> pv.PolyData:
    if surface.n_points <= 0:
        return surface
    out = surface.extract_surface().triangulate().clean()
    if EXTRACT_LARGEST_COMPONENT and out.n_points > 0:
        try:
            out = out.connectivity(largest=True).extract_surface().triangulate().clean()
        except Exception:
            pass
    if SMOOTH_ITERATIONS > 0 and out.n_points > 0:
        try:
            out = out.smooth(n_iter=int(SMOOTH_ITERATIONS), relaxation_factor=0.05)
        except Exception:
            pass
    return out


def _save_polydata_obj(mesh: pv.PolyData, path: Path) -> None:
    mesh = mesh.extract_surface().triangulate()
    try:
        mesh.save(str(path))
        return
    except Exception:
        pass

    faces = mesh.faces.reshape(-1, 4)
    with path.open("w", encoding="utf-8") as f:
        f.write("# reconstructed from SDF parquet\n")
        for p in np.asarray(mesh.points):
            f.write(f"v {float(p[0])} {float(p[1])} {float(p[2])}\n")
        for face in faces:
            if int(face[0]) != 3:
                continue
            a, b, c = (int(face[1]) + 1, int(face[2]) + 1, int(face[3]) + 1)
            f.write(f"f {a} {b} {c}\n")


def main() -> None:
    df, parquet_path = _load_rows()
    row = _select_row(df)
    source_row_index = int(_row_value(row, "source_row_index", ROW_INDEX))

    sample_xyz, sample_tsdf = _load_sdf_samples(row)
    target_mesh = _load_target_mesh(row)
    sample_weight = np.ones((sample_xyz.shape[0],), dtype=np.float32)

    prior_count = 0
    if USE_TARGET_SURFACE_PRIOR and target_mesh is not None and RECONSTRUCT_FIELD in {"after", "target"}:
        prior_xyz = _sample_target_prior(target_mesh)
        if prior_xyz.size > 0:
            sample_xyz = np.vstack([sample_xyz, prior_xyz.astype(np.float32)])
            sample_tsdf = np.concatenate([sample_tsdf, np.zeros((prior_xyz.shape[0],), dtype=np.float32)])
            sample_weight = np.concatenate([
                sample_weight,
                np.full((prior_xyz.shape[0],), float(TARGET_SURFACE_PRIOR_WEIGHT), dtype=np.float32),
            ])
            prior_count = int(prior_xyz.shape[0])

    bbox_min, bbox_max = _build_grid_bounds(sample_xyz, target_mesh)
    dims = _grid_dimensions(bbox_min, bbox_max)
    grid_points, spacing = _make_grid_points(bbox_min, bbox_max, dims)
    tsdf_grid = _interpolate_tsdf(sample_xyz, sample_tsdf, sample_weight, grid_points)
    image = _make_image_data(tsdf_grid, bbox_min, spacing, dims)
    surface = _postprocess_surface(image.contour([0.0], scalars="tsdf"))

    out_dir = Path(OUTPUT_DIR).expanduser().resolve() / f"sdf_mesh_recon_{datetime.now().strftime('%Y%m%d_%H%M%S')}_row{source_row_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / f"reconstructed_{RECONSTRUCT_FIELD}_row{source_row_index}.obj"
    vtp_path = out_dir / f"reconstructed_{RECONSTRUCT_FIELD}_row{source_row_index}.vtp"
    _save_polydata_obj(surface, obj_path)
    surface.save(str(vtp_path))

    if EXPORT_GRID_VTI:
        image.save(str(out_dir / f"interpolated_tsdf_grid_{RECONSTRUCT_FIELD}.vti"))
    if EXPORT_QUERY_POINT_CLOUD:
        cloud = pv.PolyData(sample_xyz.astype(np.float32))
        cloud["tsdf"] = sample_tsdf.astype(np.float32)
        cloud["sample_weight"] = sample_weight.astype(np.float32)
        cloud.save(str(out_dir / "sdf_samples_used.vtp"))

    summary = {
        "parquet_path": str(parquet_path),
        "source_row_index": source_row_index,
        "part_name": str(_row_value(row, "part_name", "")),
        "decision_step": int(_row_value(row, "decision_step", -1)),
        "candidate_index": int(_row_value(row, "candidate_index", -1)),
        "macro_class_name": str(_row_value(row, "macro_class_name", "")),
        "tool_choice_name": str(_row_value(row, "tool_choice_name", "")),
        "reconstruct_field": str(RECONSTRUCT_FIELD),
        "grid_dimensions": list(map(int, dims)),
        "grid_points": int(grid_points.shape[0]),
        "sdf_query_samples": int(sample_xyz.shape[0] - prior_count),
        "target_prior_samples": int(prior_count),
        "target_prior_weight": float(TARGET_SURFACE_PRIOR_WEIGHT if prior_count > 0 else 0.0),
        "bbox_min": [float(x) for x in bbox_min],
        "bbox_max": [float(x) for x in bbox_max],
        "surface_points": int(surface.n_points),
        "surface_cells": int(surface.n_cells),
        "obj_path": str(obj_path),
        "vtp_path": str(vtp_path),
        "target_mesh_path": _resolve_target_mesh_path(row) or "",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Done] output_dir={out_dir}")
    print(f"[Done] obj={obj_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
