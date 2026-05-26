r"""Visualize SDF-query transition parquet rows over the target OBJ.

This script does not require NX. Edit the constants below and run directly from
VS Code/debug mode.

What it shows:
    - translucent target CAD OBJ
    - query points colored by TSDF delta: after_tsdf - before_tsdf
    - changed/removed query points
    - optional affected-face sample points from affected_face_mask_512
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
PARQUET_PATH = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\_ALL_PARQUET_FILES\3dDataset2771_seed0_20260522_084415.parquet"
PARQUET_DIR = r""
PARQUET_GLOB = "*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []

# Set ROW_INDEX=-1 to automatically pick the row with the largest mean |delta|.
ROW_INDEX = -1
CHOSEN_ONLY = False

OUTPUT_DIR = r"sdf_query_visualizations"
SHOW_WINDOW = True
SCREENSHOT_PATH = r""  # optional explicit .png path

DENORMALIZE = True
TARGET_MESH_PATH = r""  # optional override; otherwise uses row["target_body_mesh_path"]
TARGET_MESH_OPACITY = 0.16
TARGET_MESH_SHOW_EDGES = True

CHANGE_EPS = 1e-3
REMOVED_EPS = 1e-3       # removed material should have after_tsdf - before_tsdf > 0
MONOTONICITY_EPS = 1e-3  # violation if after_tsdf < before_tsdf - eps

MAX_CONTEXT_QUERY_POINTS = 50000
MAX_CHANGED_QUERY_POINTS = 60000
MAX_VIOLATION_QUERY_POINTS = 60000
MAX_AFFECTED_FACE_POINTS = 30000
CONTEXT_POINT_SIZE = 3
CHANGED_POINT_SIZE = 7
VIOLATION_POINT_SIZE = 11
AFFECTED_POINT_SIZE = 8
ACTION_FACE_POINT_SIZE = 12

EXPORT_POINT_CLOUDS = True


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
    if not DENORMALIZE:
        return points.astype(np.float32, copy=False)
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


def _load_target_mesh(row: pd.Series) -> pv.DataSet | None:
    mesh_path = _resolve_target_mesh_path(row)
    if mesh_path is None:
        return None
    mesh = pv.read(mesh_path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not DENORMALIZE:
        scale, center = _scale_center(row)
        mesh = mesh.copy()
        mesh.points = (np.asarray(mesh.points, dtype=np.float32) - center.reshape(1, 3)) / scale
    return mesh


def _sample_indices(count: int, limit: int, seed: int = 0) -> np.ndarray:
    if limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(count, size=limit, replace=False).astype(np.int64)


def _load_sdf_query(row: pd.Series) -> dict[str, np.ndarray]:
    points = _array(row["sdf_query_points"], np.float32).reshape(-1, 3)
    before = _array(row["sdf_tsdf_before"], np.float32)
    after = _array(row["sdf_tsdf_after"], np.float32)
    count = min(points.shape[0], before.size, after.size)
    points = _denormalize_points(points[:count], row)
    before = before[:count]
    after = after[:count]
    delta = after - before
    target = None
    if "sdf_target_tsdf" in row.index and not _is_missing(row["sdf_target_tsdf"]):
        target = _array(row["sdf_target_tsdf"], np.float32)[:count]
    return {
        "points": points,
        "before": before,
        "after": after,
        "delta": delta,
        "target": target if target is not None else np.zeros((count,), dtype=np.float32),
    }


def _load_affected_face_points(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if "affected_face_mask_512" not in row.index or _is_missing(row["affected_face_mask_512"]):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    static_dir = _row_value(row, "static_feature_dir", "")
    if not static_dir:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    face_pc_path = Path(str(static_dir)) / "embed_face_pc.npy"
    if not face_pc_path.exists():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    face_pc = np.load(face_pc_path).astype(np.float32).reshape(512, 100, 3)
    if DENORMALIZE:
        scale, center = _scale_center(row)
        face_pc = face_pc * scale + center.reshape(1, 1, 3)
    affected = _array(row["affected_face_mask_512"], np.float32).reshape(512) > 0.5
    affected_points = face_pc[affected].reshape(-1, 3) if bool(affected.any()) else np.empty((0, 3), dtype=np.float32)
    affected_points = affected_points[np.any(np.abs(affected_points) > 1e-12, axis=1)]

    action_face = int(_row_value(row, "action_face_id", -1))
    action_points = np.empty((0, 3), dtype=np.float32)
    if 0 <= action_face < 512:
        action_points = face_pc[action_face].reshape(-1, 3)
        action_points = action_points[np.any(np.abs(action_points) > 1e-12, axis=1)]
    return affected_points.astype(np.float32), action_points.astype(np.float32)


def _poly(points: np.ndarray, scalars: np.ndarray | None = None, name: str = "values") -> pv.PolyData:
    cloud = pv.PolyData(points.astype(np.float32).reshape(-1, 3))
    if scalars is not None:
        cloud[name] = scalars.astype(np.float32).reshape(-1)
    return cloud


def _save_point_cloud(path: Path, points: np.ndarray, scalars: np.ndarray, scalar_name: str) -> None:
    if points.size == 0:
        return
    cloud = _poly(points, scalars, scalar_name)
    cloud.save(str(path))


def main() -> None:
    df, parquet_path = _load_rows()
    row = _select_row(df)
    query = _load_sdf_query(row)
    points = query["points"]
    before = query["before"]
    after = query["after"]
    delta = query["delta"]

    source_row_index = int(_row_value(row, "source_row_index", ROW_INDEX))
    out_dir = Path(OUTPUT_DIR).expanduser().resolve() / f"sdf_query_vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_row{source_row_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    changed = np.abs(delta) > float(CHANGE_EPS)
    removed = delta > float(REMOVED_EPS)
    mono_violation = after < (before - float(MONOTONICITY_EPS))

    context_idx = _sample_indices(points.shape[0], MAX_CONTEXT_QUERY_POINTS, seed=1)
    changed_idx_all = np.where(changed | removed | mono_violation)[0]
    changed_idx = changed_idx_all[_sample_indices(changed_idx_all.size, MAX_CHANGED_QUERY_POINTS, seed=2)] if changed_idx_all.size else np.asarray([], dtype=np.int64)

    affected_points, action_points = _load_affected_face_points(row)
    if affected_points.shape[0] > MAX_AFFECTED_FACE_POINTS > 0:
        affected_points = affected_points[_sample_indices(affected_points.shape[0], MAX_AFFECTED_FACE_POINTS, seed=3)]

    summary = {
        "parquet_path": str(parquet_path),
        "source_row_index": source_row_index,
        "part_name": str(_row_value(row, "part_name", "")),
        "decision_step": int(_row_value(row, "decision_step", -1)),
        "candidate_index": int(_row_value(row, "candidate_index", -1)),
        "macro_class_name": str(_row_value(row, "macro_class_name", "")),
        "tool_choice_name": str(_row_value(row, "tool_choice_name", "")),
        "action_face_id": int(_row_value(row, "action_face_id", -1)),
        "num_query_points": int(points.shape[0]),
        "changed_points": int(changed.sum()),
        "removed_points": int(removed.sum()),
        "monotonicity_violation_points": int(mono_violation.sum()),
        "changed_ratio": float(changed.mean()) if changed.size else 0.0,
        "removed_ratio": float(removed.mean()) if removed.size else 0.0,
        "monotonicity_violation_ratio": float(mono_violation.mean()) if mono_violation.size else 0.0,
        "mean_delta": float(delta.mean()) if delta.size else 0.0,
        "mean_abs_delta": float(np.abs(delta).mean()) if delta.size else 0.0,
        "max_abs_delta": float(np.abs(delta).max()) if delta.size else 0.0,
        "affected_face_points": int(affected_points.shape[0]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if EXPORT_POINT_CLOUDS:
        _save_point_cloud(out_dir / "query_all_delta.vtp", points[context_idx], delta[context_idx], "delta_tsdf")
        if changed_idx.size > 0:
            _save_point_cloud(out_dir / "query_changed_delta.vtp", points[changed_idx], delta[changed_idx], "delta_tsdf")
        violation_idx_all = np.where(mono_violation)[0]
        if violation_idx_all.size > 0:
            violation_idx = violation_idx_all[
                _sample_indices(violation_idx_all.size, MAX_VIOLATION_QUERY_POINTS, seed=4)
            ]
            _save_point_cloud(
                out_dir / "query_monotonicity_violation.vtp",
                points[violation_idx],
                before[violation_idx] - after[violation_idx],
                "violation_amount",
            )
        if affected_points.size > 0:
            _poly(affected_points).save(str(out_dir / "affected_face_points.vtp"))
        if action_points.size > 0:
            _poly(action_points).save(str(out_dir / "action_face_points.vtp"))

    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 850), off_screen=not SHOW_WINDOW)
    mesh = _load_target_mesh(row)

    for panel in range(2):
        plotter.subplot(0, panel)
        if mesh is not None:
            plotter.add_mesh(
                mesh,
                color="lightgray",
                opacity=TARGET_MESH_OPACITY,
                show_edges=TARGET_MESH_SHOW_EDGES,
                edge_color="gray",
            )
        plotter.add_axes()

    plotter.subplot(0, 0)
    plotter.add_text("All SDF query points colored by after-before TSDF", font_size=10)
    context_cloud = _poly(points[context_idx], delta[context_idx], "delta_tsdf")
    plotter.add_mesh(
        context_cloud,
        scalars="delta_tsdf",
        cmap="coolwarm",
        point_size=CONTEXT_POINT_SIZE,
        render_points_as_spheres=True,
        clim=[-1.0, 1.0],
        scalar_bar_args={"title": "delta TSDF"},
    )

    plotter.subplot(0, 1)
    plotter.add_text("Changed/removed query points + affected faces", font_size=10)
    if changed_idx.size > 0:
        changed_cloud = _poly(points[changed_idx], delta[changed_idx], "delta_tsdf")
        plotter.add_mesh(
            changed_cloud,
            scalars="delta_tsdf",
            cmap="coolwarm",
            point_size=CHANGED_POINT_SIZE,
            render_points_as_spheres=True,
            clim=[-1.0, 1.0],
            scalar_bar_args={"title": "delta TSDF"},
        )
    if affected_points.size > 0:
        plotter.add_mesh(
            _poly(affected_points),
            color="lime",
            point_size=AFFECTED_POINT_SIZE,
            render_points_as_spheres=True,
            label="affected face samples",
        )
    violation_idx_all = np.where(mono_violation)[0]
    if violation_idx_all.size > 0:
        violation_idx = violation_idx_all[
            _sample_indices(violation_idx_all.size, MAX_VIOLATION_QUERY_POINTS, seed=4)
        ]
        violation_amount = before[violation_idx] - after[violation_idx]
        plotter.add_mesh(
            _poly(points[violation_idx], violation_amount, "violation_amount"),
            scalars="violation_amount",
            cmap="plasma",
            point_size=VIOLATION_POINT_SIZE,
            render_points_as_spheres=True,
            scalar_bar_args={"title": "mono violation"},
        )
    if action_points.size > 0:
        plotter.add_mesh(
            _poly(action_points),
            color="black",
            point_size=ACTION_FACE_POINT_SIZE,
            render_points_as_spheres=True,
            label="action face samples",
        )

    screenshot_path = Path(SCREENSHOT_PATH).expanduser().resolve() if SCREENSHOT_PATH.strip() else out_dir / "sdf_query_overlay.png"
    plotter.link_views()
    plotter.camera_position = "iso"
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
