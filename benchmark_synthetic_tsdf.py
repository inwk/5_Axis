"""Benchmark and visualize target mesh TSDF generation for synthetic labels.

Edit constants below and run directly from VS Code/debug mode.

This compares the new fast path used by the synthetic dataset generator:

    target mesh -> voxelized target_solid -> EDT TSDF

Optionally, it can also run the old exact mesh signed-distance path for a
small resolution, but that is intentionally disabled by default because it can
take many minutes per part.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from collect_axis_dataset_synthetic_v2 import (
    _action_roi_mask,
    _axis_for_macro,
    _dilate_config_mask,
    _grid_from_mesh,
    _holder_forbidden_mask,
    _load_static_meta,
    _load_target_mesh,
    _load_valid_faces,
    _make_kernel_offsets,
    _mesh_signed_distance_negative_inside,
    _target_solid_from_mesh_voxelized,
    _tool_configs,
    _tsdf_from_sdf,
    _tsdf_from_solid,
)


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
STATIC_DIR = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_static_embeddings\3dDataset0144"

GRID_RESOLUTION = 160
GRID_PADDING_RATIO = 0.04
GRID_PADDING_MM = 2.0
TRUNCATION = 5.0

# Exact mesh SDF is the old slow path. Keep this off unless testing a small grid.
RUN_EXACT_MESH_SDF = False
EXACT_MAX_GRID_POINTS = 500_000

OUTPUT_DIR = r"synthetic_tsdf_benchmarks"
EXPORT_VTK = False
SAVE_SCREENSHOT = False
SHOW_WINDOW = True

# Operation-mask UI: visualizes why rough/finish removes or does not remove voxels.
RUN_OPERATION_VISUALIZATION_UI = True
OPERATION_VISUALIZE_MACROS = ["indexed_rough", "point_finish", "flank_finish"]
OPERATION_FACE_ID = -1  # -1 = first valid face.
OPERATION_PRE_ROUGH_STEPS_FOR_FINISH = 4
OPERATION_ROUGH_TOOL_DIAMETER = 12.0
OPERATION_FINISH_TOOL_DIAMETER = 6.0
OPERATION_POINT_LIMIT = 80_000
OPERATION_TSDF_POINT_LIMIT = 120_000
OPERATION_FACE_SEARCH_LIMIT = 24
OPERATION_FACE_MIN_CANDIDATES = 64
OPERATION_SEED = 0


def _time_call(label: str, fn):
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    print(f"[{label}] elapsed={elapsed:.6f}s", flush=True)
    return elapsed, out


def _make_image_data(values: np.ndarray, name: str, bbox_min: np.ndarray, spacing: tuple[float, float, float]):
    import pyvista as pv

    dims = tuple(int(v) for v in values.shape)
    image = pv.ImageData(
        dimensions=dims,
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in bbox_min),
    )
    image.point_data[name] = values.reshape(dims, order="C").ravel(order="F").astype(np.float32)
    return image


def _surface_from_tsdf(tsdf: np.ndarray, bbox_min: np.ndarray, spacing: tuple[float, float, float]):
    image = _make_image_data(tsdf, "tsdf", bbox_min, spacing)
    try:
        return image.contour([0.0], scalars="tsdf").extract_surface().triangulate().clean()
    except Exception:
        import pyvista as pv

        return pv.PolyData()


def _surface_from_mask(mask: np.ndarray, bbox_min: np.ndarray, spacing: tuple[float, float, float]):
    image = _make_image_data(mask.astype(np.float32), "solid", bbox_min, spacing)
    try:
        return image.contour([0.5], scalars="solid").extract_surface().triangulate().clean()
    except Exception:
        import pyvista as pv

        return pv.PolyData()


def _static_normalization(static_dir: Path) -> tuple[np.ndarray, float]:
    meta = _load_static_meta(static_dir)
    normalization = meta.get("normalization", {}) if isinstance(meta.get("normalization", {}), dict) else {}
    center = normalization.get("center_xyz", meta.get("normalization_center_xyz", [0.0, 0.0, 0.0]))
    scale = float(normalization.get("reference_scale", meta.get("normalization_scale", 1.0)))
    return np.asarray(center, dtype=np.float32).reshape(3), max(float(scale), 1e-6)


def _stock_solid_from_target(target_solid: np.ndarray) -> np.ndarray:
    stock_solid = np.ones(tuple(int(v) for v in target_solid.shape), dtype=bool)
    stock_solid[0, :, :] = False
    stock_solid[-1, :, :] = False
    stock_solid[:, 0, :] = False
    stock_solid[:, -1, :] = False
    stock_solid[:, :, 0] = False
    stock_solid[:, :, -1] = False
    stock_solid |= target_solid
    return stock_solid


def _select_tool_config(macro_name: str) -> dict[str, float | str]:
    configs = _tool_configs()
    if macro_name in {"indexed_rough", "flank_finish"}:
        tool_kind = "flat"
        target_diameter = float(OPERATION_ROUGH_TOOL_DIAMETER if macro_name == "indexed_rough" else OPERATION_FINISH_TOOL_DIAMETER)
    else:
        tool_kind = "ball"
        target_diameter = float(OPERATION_FINISH_TOOL_DIAMETER)
    candidates = [cfg for cfg in configs if str(cfg["tool_kind"]) == tool_kind]
    if not candidates:
        raise ValueError(f"No synthetic tool config for kind={tool_kind!r}.")
    return min(
        candidates,
        key=lambda cfg: (
            abs(float(cfg["tool_diameter"]) - target_diameter),
            float(cfg["tool_length"]),
            float(cfg["holder_diameter"]),
        ),
    )


def _mask_points(
    mask: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    limit: int,
    seed: int,
) -> np.ndarray:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if limit > 0 and ijk.shape[0] > limit:
        rng = np.random.default_rng(seed)
        keep = rng.choice(ijk.shape[0], size=int(limit), replace=False)
        ijk = ijk[keep]
    return bbox_min.reshape(1, 3) + ijk.astype(np.float32) * np.asarray(spacing, dtype=np.float32).reshape(1, 3)


def _poly_points(points: np.ndarray, scalars: np.ndarray | None = None, name: str = "value"):
    import pyvista as pv

    cloud = pv.PolyData(points.astype(np.float32).reshape(-1, 3))
    if scalars is not None:
        cloud[name] = np.asarray(scalars, dtype=np.float32).reshape(-1)
    return cloud


def _grid_indices_for_points(
    points: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    dims: tuple[int, int, int],
) -> np.ndarray:
    spacing_arr = np.asarray(spacing, dtype=np.float32).reshape(1, 3)
    idx = np.rint((points.astype(np.float32) - bbox_min.reshape(1, 3)) / spacing_arr).astype(np.int32)
    max_idx = np.asarray(dims, dtype=np.int32).reshape(1, 3) - 1
    return np.minimum(np.maximum(idx, 0), max_idx)


def _face_points(static_dir: Path, face_id: int, center: np.ndarray, scale: float) -> np.ndarray:
    path = static_dir / "embed_face_pc.npy"
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    face_pc = np.load(path).astype(np.float32).reshape(512, 100, 3)
    points = face_pc[int(face_id)].reshape(-1, 3) * float(scale) + center.reshape(1, 3)
    points = points[np.any(np.abs(points) > 1e-12, axis=1)]
    return points.astype(np.float32, copy=False)


def _reference_point(face_points: np.ndarray, fallback_mask: np.ndarray, bbox_min: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    if face_points.size:
        return face_points.mean(axis=0).astype(np.float32)
    points = _mask_points(fallback_mask, bbox_min, spacing, 4096, seed=int(OPERATION_SEED) + 700)
    if points.size:
        return points.mean(axis=0).astype(np.float32)
    return np.asarray(bbox_min, dtype=np.float32)


def _bbox_slices_for_points(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: tuple[float, float, float],
    dims: tuple[int, int, int],
    pad: float,
) -> tuple[slice, slice, slice]:
    if points.size:
        local_min = np.maximum(points.min(axis=0).astype(np.float32) - float(pad), bbox_min)
        local_max = np.minimum(points.max(axis=0).astype(np.float32) + float(pad), bbox_max)
    else:
        local_min = np.asarray(bbox_min, dtype=np.float32)
        local_max = np.asarray(bbox_max, dtype=np.float32)
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    lo = np.floor((local_min - bbox_min) / spacing_arr).astype(np.int32)
    hi = np.ceil((local_max - bbox_min) / spacing_arr).astype(np.int32) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(dims, dtype=np.int32))
    hi = np.maximum(hi, lo + 1)
    return slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2]))


def _add_tool_holder_debug_mesh(
    plotter,
    ref_point: np.ndarray,
    axis_dir: np.ndarray,
    config: dict[str, float | str],
) -> None:
    import pyvista as pv

    axis = np.asarray(axis_dir, dtype=np.float32).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    ref = np.asarray(ref_point, dtype=np.float32).reshape(3)
    tool_radius = float(config["tool_radius"])
    tool_length = float(config["tool_length"])
    holder_radius = float(config["holder_radius"])
    holder_length = float(config["holder_length"])

    if "ball" in str(config["tool_kind"]):
        ball_center = ref + axis * tool_radius
        plotter.add_mesh(pv.Sphere(radius=tool_radius, center=tuple(float(x) for x in ball_center)), color="orange", opacity=0.45)
        cutter_len = max(tool_length - tool_radius, 1e-6)
        cutter_center = ref + axis * (tool_radius + 0.5 * cutter_len)
    else:
        cutter_len = max(tool_length, 1e-6)
        cutter_center = ref + axis * (0.5 * cutter_len)

    plotter.add_mesh(
        pv.Cylinder(
            center=tuple(float(x) for x in cutter_center),
            direction=tuple(float(x) for x in axis),
            radius=tool_radius,
            height=cutter_len,
            resolution=36,
        ),
        color="orange",
        opacity=0.40,
        show_edges=True,
    )
    plotter.add_mesh(
        pv.Cylinder(
            center=tuple(float(x) for x in ref + axis * (tool_length + 0.5 * holder_length)),
            direction=tuple(float(x) for x in axis),
            radius=holder_radius,
            height=max(holder_length, 1e-6),
            resolution=48,
        ),
        color="dodgerblue",
        opacity=0.32,
        show_edges=True,
    )
    plotter.add_mesh(pv.Sphere(radius=max(tool_radius * 0.35, 1e-6), center=tuple(float(x) for x in ref)), color="yellow")


def _sample_tsdf_points(
    tsdf: np.ndarray,
    mask: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    if limit > 0 and ijk.shape[0] > limit:
        rng = np.random.default_rng(seed)
        keep = rng.choice(ijk.shape[0], size=int(limit), replace=False)
        ijk = ijk[keep]
    points = bbox_min.reshape(1, 3) + ijk.astype(np.float32) * np.asarray(spacing, dtype=np.float32).reshape(1, 3)
    values = tsdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float32, copy=False)
    return points.astype(np.float32, copy=False), values


def _operation_masks(
    *,
    macro_name: str,
    current_solid: np.ndarray,
    target_solid: np.ndarray,
    face_id: int,
    axis_dir: np.ndarray,
    config: dict[str, float | str],
    grid_points: np.ndarray,
    static_dir: Path,
    center: np.ndarray,
    scale: float,
    dims: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, np.ndarray | int | float | str]:
    holder_obstacle = current_solid if macro_name == "indexed_rough" else target_solid
    holder_forbidden = _holder_forbidden_mask(
        obstacle=holder_obstacle,
        axis_dir=axis_dir,
        tool_kind=str(config["tool_kind"]),
        tool_radius=float(config["tool_radius"]),
        tool_length=float(config["tool_length"]),
        holder_radius=float(config["holder_radius"]),
        holder_length=float(config["holder_length"]),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
    ideal_removal = current_solid & ~target_solid
    roi = _action_roi_mask(
        macro_name,
        int(face_id),
        grid_points,
        static_dir,
        center,
        float(scale),
        dims,
        float(config["tool_radius"]),
        spacing,
    )
    config_candidates = ideal_removal & roi & ~holder_forbidden
    ideal_roi = ideal_removal & roi
    holder_blocked_ideal_roi = ideal_roi & holder_forbidden
    if macro_name == "indexed_rough":
        swept_volume = np.zeros_like(config_candidates, dtype=bool)
        removed = config_candidates
        skipped_swept = True
    elif not bool(config_candidates.any()):
        swept_volume = np.zeros_like(config_candidates, dtype=bool)
        removed = np.zeros_like(config_candidates, dtype=bool)
        skipped_swept = False
    else:
        kernels = _make_kernel_offsets(
            axis_dir=axis_dir,
            spacing=spacing,
            tool_kind=str(config["tool_kind"]),
            tool_radius=float(config["tool_radius"]),
            tool_length=float(config["tool_length"]),
            holder_radius=float(config["holder_radius"]),
            holder_length=float(config["holder_length"]),
            max_abs_steps=tuple(max(int(v) - 1, 0) for v in dims),
        )
        swept_volume = _dilate_config_mask(config_candidates, kernels["cutter"])
        removed = ideal_removal & roi & swept_volume
        skipped_swept = False
    return {
        "current_solid": current_solid,
        "target_solid": target_solid,
        "holder_obstacle": holder_obstacle,
        "ideal_removal": ideal_removal,
        "roi": roi,
        "ideal_roi": ideal_roi,
        "holder_blocked_ideal_roi": holder_blocked_ideal_roi,
        "holder_forbidden": holder_forbidden,
        "config_candidates": config_candidates,
        "swept_volume": swept_volume,
        "removed": removed,
        "skipped_swept": skipped_swept,
    }


def _face_candidate_score(
    *,
    macro_name: str,
    current_solid: np.ndarray,
    target_solid: np.ndarray,
    face_id: int,
    axis_dir: np.ndarray,
    config: dict[str, float | str],
    static_dir: Path,
    center: np.ndarray,
    scale: float,
    dims: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[int, int]:
    points = _face_points(static_dir, int(face_id), center, scale)
    pad = max(float(config["tool_radius"]) * 3.0, float(max(spacing)) * 3.0, 5.0)
    xs, ys, zs = _bbox_slices_for_points(points, bbox_min, bbox_max, spacing, dims, pad)
    ideal_roi = current_solid[xs, ys, zs] & ~target_solid[xs, ys, zs]
    ideal_count = int(ideal_roi.sum())
    if ideal_count <= 0:
        return 0, 0
    holder_forbidden = _holder_forbidden_mask(
        obstacle=current_solid if macro_name == "indexed_rough" else target_solid,
        axis_dir=axis_dir,
        tool_kind=str(config["tool_kind"]),
        tool_radius=float(config["tool_radius"]),
        tool_length=float(config["tool_length"]),
        holder_radius=float(config["holder_radius"]),
        holder_length=float(config["holder_length"]),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
    return int((ideal_roi & ~holder_forbidden[xs, ys, zs]).sum()), ideal_count


def _select_operation_face(
    *,
    macro_name: str,
    current_solid: np.ndarray,
    target_solid: np.ndarray,
    valid_faces: np.ndarray,
    rough_faces: list[int],
    normals: np.ndarray,
    config: dict[str, float | str],
    static_dir: Path,
    center: np.ndarray,
    scale: float,
    dims: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[int, np.ndarray]:
    py_rng = random.Random(int(OPERATION_SEED))
    if int(OPERATION_FACE_ID) >= 0:
        face_id = int(OPERATION_FACE_ID)
        return face_id, _axis_for_macro(macro_name, face_id, normals, py_rng)
    default_face = int(valid_faces[0])
    if macro_name == "indexed_rough":
        return default_face, _axis_for_macro(macro_name, default_face, normals, py_rng)

    rough_set = set(int(face_id) for face_id in rough_faces)
    search_faces = list(rough_faces) + [int(face_id) for face_id in valid_faces if int(face_id) not in rough_set]
    best_face = default_face
    best_axis = _axis_for_macro(macro_name, default_face, normals, py_rng)
    best_candidates = -1
    best_ideal = 0
    for face_id in search_faces[:max(1, int(OPERATION_FACE_SEARCH_LIMIT))]:
        axis_dir = _axis_for_macro(macro_name, int(face_id), normals, py_rng)
        candidate_count, ideal_count = _face_candidate_score(
            macro_name=macro_name,
            current_solid=current_solid,
            target_solid=target_solid,
            face_id=int(face_id),
            axis_dir=axis_dir,
            config=config,
            static_dir=static_dir,
            center=center,
            scale=scale,
            dims=dims,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            spacing=spacing,
        )
        if candidate_count > best_candidates or (candidate_count == best_candidates and ideal_count > best_ideal):
            best_face = int(face_id)
            best_axis = axis_dir
            best_candidates = candidate_count
            best_ideal = ideal_count
        if candidate_count >= int(OPERATION_FACE_MIN_CANDIDATES):
            break
    print(
        f"[Operation UI Face Select] macro={macro_name} selected_face={best_face} "
        f"candidates={max(best_candidates, 0)} ideal_roi={best_ideal} "
        f"rough_faces={rough_faces} fallback={best_candidates <= 0}",
        flush=True,
    )
    return best_face, best_axis


def _advance_with_pre_rough(
    *,
    current_solid: np.ndarray,
    target_solid: np.ndarray,
    valid_faces: np.ndarray,
    normals: np.ndarray,
    grid_points: np.ndarray,
    static_dir: Path,
    center: np.ndarray,
    scale: float,
    dims: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, list[int]]:
    py_rng = random.Random(int(OPERATION_SEED))
    rough_config = _select_tool_config("indexed_rough")
    out = current_solid.copy()
    rough_faces: list[int] = []
    for step in range(max(0, int(OPERATION_PRE_ROUGH_STEPS_FOR_FINISH))):
        face_id = int(valid_faces[step % valid_faces.size])
        axis_dir = _axis_for_macro("indexed_rough", face_id, normals, py_rng)
        masks = _operation_masks(
            macro_name="indexed_rough",
            current_solid=out,
            target_solid=target_solid,
            face_id=face_id,
            axis_dir=axis_dir,
            config=rough_config,
            grid_points=grid_points,
            static_dir=static_dir,
            center=center,
            scale=scale,
            dims=dims,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            spacing=spacing,
        )
        removed = np.asarray(masks["removed"], dtype=bool)
        print(
            f"[Operation UI Pre-Rough] step={step} face={face_id} "
            f"removed={int(removed.sum())}",
            flush=True,
        )
        out = out & ~removed
        out |= target_solid
        rough_faces.append(face_id)
    return out, rough_faces


def _show_operation_visualization(
    static_dir: Path,
    mesh,
    grid_points: np.ndarray,
    dims: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: tuple[float, float, float],
    target_solid: np.ndarray,
    target_tsdf: np.ndarray,
) -> None:
    import pyvista as pv

    print("\n[Operation UI] using precomputed grid and target TSDF", flush=True)
    stock_solid = _stock_solid_from_target(target_solid)
    center, scale = _static_normalization(static_dir)
    valid_faces, normals = _load_valid_faces(static_dir)

    finish_current_solid, rough_faces = _advance_with_pre_rough(
        current_solid=stock_solid,
        target_solid=target_solid,
        valid_faces=valid_faces,
        normals=normals,
        grid_points=grid_points,
        static_dir=static_dir,
        center=center,
        scale=scale,
        dims=dims,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        spacing=spacing,
    )

    macros = [str(item) for item in OPERATION_VISUALIZE_MACROS]
    target_surface = _surface_from_tsdf(target_tsdf, bbox_min, spacing)

    for col, macro_name in enumerate(macros):
        current_solid = stock_solid if macro_name == "indexed_rough" else finish_current_solid
        config = _select_tool_config(macro_name)
        face_id, axis_dir = _select_operation_face(
            macro_name=macro_name,
            current_solid=current_solid,
            target_solid=target_solid,
            valid_faces=valid_faces,
            rough_faces=rough_faces,
            normals=normals,
            config=config,
            static_dir=static_dir,
            center=center,
            scale=scale,
            dims=dims,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            spacing=spacing,
        )
        start = time.perf_counter()
        masks = _operation_masks(
            macro_name=macro_name,
            current_solid=current_solid,
            target_solid=target_solid,
            face_id=face_id,
            axis_dir=axis_dir,
            config=config,
            grid_points=grid_points,
            static_dir=static_dir,
            center=center,
            scale=scale,
            dims=dims,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            spacing=spacing,
        )
        elapsed = time.perf_counter() - start
        removed = np.asarray(masks["removed"], dtype=bool)
        after_solid = current_solid & ~removed
        after_solid |= target_solid
        before_tsdf = _tsdf_from_solid(current_solid, spacing, float(TRUNCATION))
        after_tsdf = _tsdf_from_solid(after_solid, spacing, float(TRUNCATION))
        delta_tsdf = (after_tsdf - before_tsdf).astype(np.float32, copy=False)
        before_surface = _surface_from_tsdf(before_tsdf, bbox_min, spacing)
        after_surface = _surface_from_tsdf(after_tsdf, bbox_min, spacing)
        counts = {key: int(np.asarray(masks[key], dtype=bool).sum()) for key in (
            "holder_obstacle",
            "ideal_removal",
            "roi",
            "ideal_roi",
            "holder_forbidden",
            "holder_blocked_ideal_roi",
            "config_candidates",
            "removed",
        )}
        face_points = _face_points(static_dir, face_id, center, scale)
        ref_point = _reference_point(face_points, np.asarray(masks["ideal_roi"], dtype=bool), bbox_min, spacing)
        face_blocked = np.zeros((face_points.shape[0],), dtype=bool)
        if face_points.size:
            face_idx = _grid_indices_for_points(face_points, bbox_min, spacing, dims)
            face_blocked = np.asarray(masks["holder_forbidden"], dtype=bool)[face_idx[:, 0], face_idx[:, 1], face_idx[:, 2]]
        total = int(np.prod(dims))
        ideal_roi_count = max(counts["ideal_roi"], 1)
        blocked_ratio = counts["holder_blocked_ideal_roi"] / float(ideal_roi_count)
        face_blocked_ratio = float(face_blocked.mean()) if face_blocked.size else 0.0
        print(
            f"[Operation UI] macro={macro_name} face={face_id} elapsed={elapsed:.2f}s "
            f"counts={counts} blocked_ideal_roi={blocked_ratio:.4f} "
            f"face_blocked={face_blocked_ratio:.4f} tool={config['tool_kind']} dia={float(config['tool_diameter']):.3f}",
            flush=True,
        )

        plotter = pv.Plotter(shape=(3, 3), window_size=(2100, 1500), off_screen=False)
        plotter.subplot(0, 0)
        plotter.add_text(
            f"1. Stock Transition: {macro_name}\n"
            f"face={face_id} tool={config['tool_kind']} D={float(config['tool_diameter']):.1f}\n"
            "brown=before/current, blue=after, silver=target",
            font_size=9,
        )
        if target_surface.n_points > 0:
            plotter.add_mesh(target_surface, color="silver", opacity=0.10, show_edges=False)
        if before_surface.n_points > 0:
            plotter.add_mesh(before_surface, color="sienna", opacity=0.25, show_edges=False)
        if after_surface.n_points > 0:
            plotter.add_mesh(after_surface, color="deepskyblue", opacity=0.62, show_edges=False)
        removed_points = _mask_points(
            np.asarray(masks["removed"], dtype=bool),
            bbox_min,
            spacing,
            int(OPERATION_POINT_LIMIT),
            seed=int(OPERATION_SEED) + 99,
        )
        if removed_points.size:
            plotter.add_mesh(
                _poly_points(removed_points),
                color="red",
                opacity=0.95,
                point_size=6,
                render_points_as_spheres=False,
            )
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(0, 1)
        plotter.add_text(
            "2. Mask Breakdown\n"
            "cyan=ROI, blue=holder forbidden, lime=candidates, red=removed",
            font_size=9,
        )
        if target_surface.n_points > 0:
            plotter.add_mesh(target_surface, color="silver", opacity=0.08, show_edges=False)
        overlays = [
            ("roi", "cyan", 0.32, 3),
            ("holder_forbidden", "dodgerblue", 0.18, 2),
            ("config_candidates", "lime", 0.72, 5),
            ("removed", "red", 0.95, 6),
        ]
        for idx, (key, color, opacity, point_size) in enumerate(overlays):
            points = _mask_points(
                np.asarray(masks[key], dtype=bool),
                bbox_min,
                spacing,
                int(OPERATION_POINT_LIMIT),
                seed=int(OPERATION_SEED) + idx,
            )
            if points.size:
                plotter.add_mesh(
                    _poly_points(points),
                    color=color,
                    opacity=opacity,
                    point_size=point_size,
                    render_points_as_spheres=False,
                )
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(0, 2)
        plotter.add_text(
            "3. Action Face / Axis Check\n"
            "black=action face points, red=face normal, blue=tool/holder axis",
            font_size=9,
        )
        if target_surface.n_points > 0:
            plotter.add_mesh(target_surface, color="silver", opacity=0.10, show_edges=False)
        if face_points.size:
            plotter.add_mesh(
                _poly_points(face_points),
                color="black",
                point_size=8,
                render_points_as_spheres=True,
            )
        face_normal = normals[int(face_id)].astype(np.float32)
        arrow_mag = max(float(np.linalg.norm(bbox_max - bbox_min)) * 0.12, 1e-6)
        plotter.add_arrows(ref_point.reshape(1, 3), face_normal.reshape(1, 3), mag=arrow_mag, color="red")
        plotter.add_arrows(ref_point.reshape(1, 3), axis_dir.reshape(1, 3).astype(np.float32), mag=arrow_mag, color="blue")
        plotter.add_mesh(pv.Sphere(radius=max(float(config["tool_radius"]) * 0.45, 1e-6), center=tuple(float(x) for x in ref_point)), color="yellow")
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(1, 0)
        plotter.add_text(
            "4. Delta TSDF Map\n"
            "brown=before surface, blue=after surface\n"
            "points colored by after_tsdf - before_tsdf",
            font_size=9,
        )
        if before_surface.n_points > 0:
            plotter.add_mesh(before_surface, color="sienna", opacity=0.18, show_edges=False)
        if after_surface.n_points > 0:
            plotter.add_mesh(after_surface, color="deepskyblue", opacity=0.34, show_edges=False)
        tsdf_mask = (
            np.asarray(masks["roi"], dtype=bool)
            | np.asarray(masks["config_candidates"], dtype=bool)
            | np.asarray(masks["removed"], dtype=bool)
        )
        if not bool(tsdf_mask.any()):
            tsdf_mask = np.asarray(masks["ideal_removal"], dtype=bool)
        points, values = _sample_tsdf_points(
            delta_tsdf,
            tsdf_mask,
            bbox_min,
            spacing,
            int(OPERATION_TSDF_POINT_LIMIT),
            seed=int(OPERATION_SEED) + 100 + col,
        )
        if points.size:
            plotter.add_mesh(
                _poly_points(points, values, "delta_tsdf"),
                scalars="delta_tsdf",
                cmap="coolwarm",
                clim=(-1.0, 1.0),
                point_size=4,
                render_points_as_spheres=False,
                show_scalar_bar=True,
            )
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(1, 1)
        plotter.add_text(
            "5. Holder Blocking Inside Action ROI\n"
            "green=ideal_removal in ROI and not blocked\n"
            "red=ideal_removal in ROI but blocked by holder",
            font_size=9,
        )
        if target_surface.n_points > 0:
            plotter.add_mesh(target_surface, color="silver", opacity=0.08, show_edges=False)
        open_ideal_roi = np.asarray(masks["ideal_roi"], dtype=bool) & ~np.asarray(masks["holder_forbidden"], dtype=bool)
        blocked_ideal_roi = np.asarray(masks["holder_blocked_ideal_roi"], dtype=bool)
        for seed_offset, mask, color, size in (
            (200, open_ideal_roi, "lime", 6),
            (201, blocked_ideal_roi, "red", 6),
        ):
            points = _mask_points(mask, bbox_min, spacing, int(OPERATION_POINT_LIMIT), seed=int(OPERATION_SEED) + seed_offset)
            if points.size:
                plotter.add_mesh(
                    _poly_points(points),
                    color=color,
                    opacity=0.85,
                    point_size=size,
                    render_points_as_spheres=False,
                )
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(1, 2)
        plotter.add_text(
            "6. Holder Obstacle vs C-space Forbidden\n"
            "gray=collision obstacle, blue=forbidden TCP positions",
            font_size=9,
        )
        obstacle_points = _mask_points(
            np.asarray(masks["holder_obstacle"], dtype=bool),
            bbox_min,
            spacing,
            int(OPERATION_POINT_LIMIT),
            seed=int(OPERATION_SEED) + 220,
        )
        forbidden_points = _mask_points(
            np.asarray(masks["holder_forbidden"], dtype=bool),
            bbox_min,
            spacing,
            int(OPERATION_POINT_LIMIT),
            seed=int(OPERATION_SEED) + 221,
        )
        if obstacle_points.size:
            plotter.add_mesh(_poly_points(obstacle_points), color="gray", opacity=0.22, point_size=2, render_points_as_spheres=False)
        if forbidden_points.size:
            plotter.add_mesh(_poly_points(forbidden_points), color="dodgerblue", opacity=0.25, point_size=2, render_points_as_spheres=False)
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(2, 0)
        plotter.add_text(
            "7. Action Face Blocked Test\n"
            "green=face sample not forbidden, red=face sample forbidden",
            font_size=9,
        )
        if target_surface.n_points > 0:
            plotter.add_mesh(target_surface, color="silver", opacity=0.08, show_edges=False)
        if face_points.size:
            if bool((~face_blocked).any()):
                plotter.add_mesh(
                    _poly_points(face_points[~face_blocked]),
                    color="lime",
                    point_size=9,
                    render_points_as_spheres=True,
                )
            if bool(face_blocked.any()):
                plotter.add_mesh(
                    _poly_points(face_points[face_blocked]),
                    color="red",
                    point_size=9,
                    render_points_as_spheres=True,
                )
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(2, 1)
        plotter.add_text(
            "8. Tool / Holder Direction Check\n"
            "yellow=TCP/ref point, orange=cutter, blue=holder\n"
            "If blue cylinder goes into stock, axis side is wrong or too conservative.",
            font_size=9,
        )
        if before_surface.n_points > 0:
            plotter.add_mesh(before_surface, color="sienna", opacity=0.16, show_edges=False)
        if target_surface.n_points > 0:
            plotter.add_mesh(target_surface, color="silver", opacity=0.08, show_edges=False)
        _add_tool_holder_debug_mesh(plotter, ref_point, axis_dir, config)
        plotter.add_arrows(ref_point.reshape(1, 3), axis_dir.reshape(1, 3).astype(np.float32), mag=arrow_mag, color="blue")
        plotter.add_axes()
        plotter.view_isometric()

        plotter.subplot(2, 2)
        plotter.add_text(
            "9. Counts / Interpretation\n"
            f"macro={macro_name}\n"
            f"face={face_id}\n"
            f"tool={config['tool_kind']} D={float(config['tool_diameter']):.1f}\n"
            f"axis=({axis_dir[0]:.2f}, {axis_dir[1]:.2f}, {axis_dir[2]:.2f})\n"
            f"face_blocked_ratio={face_blocked_ratio:.3f}\n"
            f"blocked_ideal_roi={blocked_ratio:.3f}\n"
            f"holder_obstacle={counts['holder_obstacle']} / {total}\n"
            f"ideal_removal={counts['ideal_removal']}\n"
            f"roi={counts['roi']}\n"
            f"ideal_roi={counts['ideal_roi']}\n"
            f"holder_forbidden={counts['holder_forbidden']}\n"
            f"holder_blocked_ideal_roi={counts['holder_blocked_ideal_roi']}\n"
            f"config_candidates={counts['config_candidates']}\n"
            f"removed={counts['removed']}\n"
            f"operation_time={elapsed:.2f}s",
            font_size=10,
        )
        if target_surface.n_points > 0:
            plotter.add_mesh(target_surface, color="silver", opacity=0.14, show_edges=False)
        if removed_points.size:
            plotter.add_mesh(
                _poly_points(removed_points),
                color="red",
                opacity=0.95,
                point_size=7,
                render_points_as_spheres=True,
            )
        plotter.add_arrows(
            np.asarray([(bbox_min + bbox_max) * 0.5], dtype=np.float32),
            axis_dir.reshape(1, 3).astype(np.float32),
            mag=max(float(np.linalg.norm(bbox_max - bbox_min)) * 0.18, 1e-6),
            color="black",
        )
        plotter.add_axes()
        plotter.view_isometric()

        plotter.link_views()
        plotter.show(title=f"Synthetic operation visualization - {macro_name}")


def _save_visuals(
    out_dir: Path,
    resolution: int,
    mesh,
    target_solid: np.ndarray,
    target_tsdf: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, str]:
    import pyvista as pv

    outputs: dict[str, str] = {}
    tsdf_surface = _surface_from_tsdf(target_tsdf, bbox_min, spacing)
    solid_surface = _surface_from_mask(target_solid, bbox_min, spacing)

    if EXPORT_VTK:
        if tsdf_surface.n_points > 0:
            path = out_dir / f"target_voxel_edt_tsdf_res{resolution}.vtp"
            tsdf_surface.save(str(path))
            outputs["tsdf_surface_vtp"] = str(path)
        if solid_surface.n_points > 0:
            path = out_dir / f"target_voxel_solid_res{resolution}.vtp"
            solid_surface.save(str(path))
            outputs["solid_surface_vtp"] = str(path)

    if SAVE_SCREENSHOT:
        plotter = pv.Plotter(window_size=(1400, 900), off_screen=not SHOW_WINDOW)
        plotter.add_text(f"Voxelized target TSDF, resolution={resolution}", font_size=10)
        if not isinstance(mesh, dict):
            try:
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                faces_vtk = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
                mesh_poly = pv.PolyData(vertices, faces_vtk)
                plotter.add_mesh(mesh_poly, color="silver", opacity=0.18, show_edges=False)
            except Exception:
                pass
        elif mesh.get("polydata") is not None:
            plotter.add_mesh(mesh["polydata"], color="silver", opacity=0.18, show_edges=False)
        if tsdf_surface.n_points > 0:
            plotter.add_mesh(tsdf_surface, color="deepskyblue", opacity=0.65, show_edges=False)
        if solid_surface.n_points > 0:
            plotter.add_mesh(solid_surface, color="orange", opacity=0.22, show_edges=False)
        plotter.add_axes()
        plotter.view_isometric()
        path = out_dir / f"target_voxel_edt_tsdf_res{resolution}.png"
        if SHOW_WINDOW:
            plotter.show(screenshot=str(path))
        else:
            plotter.screenshot(str(path))
            plotter.close()
        outputs["screenshot_png"] = str(path)

    return outputs


def _run_resolution(mesh, resolution: int, out_dir: Path) -> tuple[dict[str, object], tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray, np.ndarray]]:
    print(f"\n[Resolution] {resolution}", flush=True)
    grid_time, grid_payload = _time_call(
        "grid_from_mesh",
        lambda: _grid_from_mesh(
            mesh,
            resolution=int(resolution),
            padding_ratio=float(GRID_PADDING_RATIO),
            padding_mm=float(GRID_PADDING_MM),
        ),
    )
    grid_points, dims, bbox_min, bbox_max, spacing = grid_payload
    voxels = int(np.prod(dims))
    print(
        f"[Grid] resolution={resolution} dims={dims} points={grid_points.shape[0]:,} "
        f"voxels={voxels:,} spacing={spacing}",
        flush=True,
    )

    voxel_time, target_solid = _time_call(
        "target_mesh_voxelize",
        lambda: _target_solid_from_mesh_voxelized(mesh, bbox_min, spacing, dims),
    )
    edt_time, target_tsdf = _time_call(
        "target_voxel_tsdf_edt",
        lambda: _tsdf_from_solid(target_solid, spacing, float(TRUNCATION)),
    )
    print(
        f"[Voxel EDT] solid_ratio={float(target_solid.mean()):.6f} "
        f"tsdf_min={float(target_tsdf.min()):.4f} tsdf_max={float(target_tsdf.max()):.4f}",
        flush=True,
    )

    exact: dict[str, object] = {"enabled": bool(RUN_EXACT_MESH_SDF)}
    if RUN_EXACT_MESH_SDF:
        if grid_points.shape[0] > int(EXACT_MAX_GRID_POINTS):
            print(
                f"[Exact Mesh SDF] skipped: grid_points={grid_points.shape[0]:,} "
                f"> EXACT_MAX_GRID_POINTS={EXACT_MAX_GRID_POINTS:,}",
                flush=True,
            )
            exact["skipped"] = True
        else:
            exact_time, target_sdf = _time_call(
                "target_mesh_sdf_exact",
                lambda: _mesh_signed_distance_negative_inside(mesh, grid_points),
            )
            exact_tsdf = _tsdf_from_sdf(target_sdf, float(TRUNCATION)).reshape(dims)
            mae = float(np.mean(np.abs(exact_tsdf - target_tsdf)))
            max_abs = float(np.max(np.abs(exact_tsdf - target_tsdf)))
            print(f"[Exact Compare] mae={mae:.6f} max_abs={max_abs:.6f}", flush=True)
            exact.update({"time_sec": exact_time, "mae": mae, "max_abs": max_abs})

    visuals: dict[str, str] = {}
    if EXPORT_VTK or SAVE_SCREENSHOT:
        visual_time, visuals = _time_call(
            "visualize_export",
            lambda: _save_visuals(out_dir, resolution, mesh, target_solid, target_tsdf, bbox_min, spacing),
        )
    else:
        visual_time = 0.0

    summary = {
        "resolution": int(resolution),
        "dims": list(map(int, dims)),
        "voxels": voxels,
        "spacing": [float(v) for v in spacing],
        "grid_time_sec": float(grid_time),
        "voxelize_time_sec": float(voxel_time),
        "edt_time_sec": float(edt_time),
        "visualize_time_sec": float(visual_time),
        "solid_ratio": float(target_solid.mean()),
        "tsdf_min": float(target_tsdf.min()),
        "tsdf_max": float(target_tsdf.max()),
        "exact_mesh_sdf": exact,
        "visual_outputs": visuals,
    }
    payload = (grid_points, dims, bbox_min, bbox_max, spacing, target_solid, target_tsdf)
    return summary, payload


def main() -> None:
    if not STATIC_DIR.strip():
        raise ValueError("Set STATIC_DIR to a static embedding folder containing target_body.obj.")

    static_dir = Path(STATIC_DIR).expanduser().resolve()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve() / f"target_tsdf_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] static_dir={static_dir}", flush=True)
    print(
        f"[Config] resolution={GRID_RESOLUTION} padding_ratio={GRID_PADDING_RATIO} "
        f"padding_mm={GRID_PADDING_MM} truncation={TRUNCATION}",
        flush=True,
    )
    print(f"[Config] run_exact_mesh_sdf={RUN_EXACT_MESH_SDF}", flush=True)

    load_time, mesh = _time_call("load_target_mesh", lambda: _load_target_mesh(static_dir))
    results = {
        "static_dir": str(static_dir),
        "target_body_mesh_path": str(static_dir / "target_body.obj"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "load_target_mesh_time_sec": float(load_time),
        "resolutions": [],
    }

    resolution_summary, grid_payload = _run_resolution(mesh, int(GRID_RESOLUTION), out_dir)
    results["resolutions"].append(resolution_summary)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Done] output_dir={out_dir}", flush=True)
    print(f"[Done] summary={summary_path}", flush=True)

    if RUN_OPERATION_VISUALIZATION_UI:
        grid_points, dims, bbox_min, bbox_max, spacing, target_solid, target_tsdf = grid_payload
        _show_operation_visualization(
            static_dir,
            mesh,
            grid_points,
            dims,
            bbox_min,
            bbox_max,
            spacing,
            target_solid,
            target_tsdf,
        )


if __name__ == "__main__":
    main()
