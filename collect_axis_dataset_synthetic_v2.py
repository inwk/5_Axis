"""Shared v2 helpers for NX static extraction and synthetic TSDF labels.

Stage A opens the PRT with NX only to extract static CAD features.  Stage B reads
those static files without NX and generates Minkowski/C-space transition parquet.
No CAM operations, toolpaths, IPW simulations, or NX rollouts are created.  The
static files are:

    target_body.obj
    graph_nx.json
    embed_centrality.npy
    embed_spatial_pos.npy
    embed_face_area.npy
    embed_face_type.npy
    embed_face_pc.npy
    embed_face_normal.npy
    embed_node_mask.npy
    embed_point_mask.npy

Set SYNTHETIC_EXTRACT_STATIC_WITH_NX=0 only for the legacy combined worker when
you want to reuse an existing static feature directory instead of opening the PRT.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STATIC_FILES = (
    "target_body.obj",
    "graph_nx.json",
    "embed_centrality.npy",
    "embed_spatial_pos.npy",
    "embed_face_area.npy",
    "embed_face_type.npy",
    "embed_face_pc.npy",
    "embed_face_normal.npy",
    "embed_node_mask.npy",
    "embed_point_mask.npy",
    "face_index_manifest.json",
)

DEFAULT_OUTPUT_BASENAME = "process_skeleton_dataset.parquet"
GRID_DEPTH_VALUE = 6
_GPU_CSPACE_FALLBACK_WARNED = False


def _load_schema_symbols() -> tuple[dict[str, int], dict[str, int], Any]:
    schema_path = Path(__file__).resolve().parent / "graph_sdf" / "schema.py"
    spec = importlib.util.spec_from_file_location("_graph_sdf_schema_for_synthetic_v2", schema_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load schema module: {schema_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MACRO_CLASS_TO_ID, module.TOOL_CHOICE_TO_ID, module.tool_choice_key


MACRO_CLASS_TO_ID, TOOL_CHOICE_TO_ID, tool_choice_key = _load_schema_symbols()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return bool(int(os.getenv(name, "1" if default else "0")))


def _parse_float_list(name: str, default: str) -> list[float]:
    raw = os.getenv(name, default)
    values: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if text:
            values.append(float(text))
    if not values:
        raise ValueError(f"{name} produced an empty list.")
    return values


def _parse_int_list(name: str, default: str) -> list[int]:
    return [int(round(value)) for value in _parse_float_list(name, default)]


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("._-")


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _create_run_output_dir(out_root: str, part_name: str, seed: int, pc_name: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pc_slug = _safe_filename(pc_name)
    run_name = f"{part_name}_seed{int(seed)}_{ts}"
    if pc_slug:
        run_name = f"{run_name}_{pc_slug}"
    run_dir = Path(out_root).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _strip_pc_suffix(name: str, pc_name: str = "") -> str:
    pc_slug = _safe_filename(pc_name)
    if pc_slug and name.endswith(f"_{pc_slug}"):
        return name[: -(len(pc_slug) + 1)]
    return name


def _part_name_from_prt(prt_path: str) -> str:
    return Path(prt_path).stem


def _static_dir_is_complete(path: Path) -> bool:
    return all((path / name).exists() for name in STATIC_FILES)


def _find_static_feature_dir(prt_path: str, out_root: str) -> Path:
    part_name = _part_name_from_prt(prt_path)
    explicit_root = os.getenv("SYNTHETIC_STATIC_FEATURE_ROOT", "").strip()
    candidates: list[Path] = []

    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        candidates.extend([
            root / part_name,
            root,
        ])

    out_root_path = Path(out_root).expanduser().resolve()
    patterns = [
        str(out_root_path / f"{part_name}_seed*"),
        str(out_root_path / "*" / f"{part_name}_seed*"),
    ]
    for pattern in patterns:
        for value in glob.glob(pattern):
            candidates.append(Path(value).resolve())

    complete = [path for path in candidates if path.is_dir() and _static_dir_is_complete(path)]
    if complete:
        # Prefer newest previous run when multiple static exports exist.
        return max(complete, key=lambda path: path.stat().st_mtime)

    searched = "\n".join(str(path) for path in candidates[:12])
    raise FileNotFoundError(
        "No complete static feature directory was found for "
        f"{part_name!r}. v2 does not parse raw .prt without NX.\n"
        "Provide SYNTHETIC_STATIC_FEATURE_ROOT or reuse an existing dataset run "
        f"containing: {', '.join(STATIC_FILES)}"
        + (f"\nSearched:\n{searched}" if searched else "")
    )


def _copy_static_features(src_dir: Path, dst_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name in STATIC_FILES:
        src = src_dir / name
        dst = dst_dir / name
        if name == "target_body.obj":
            shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        copied[name] = str(dst.resolve())
    return copied


def _extract_static_features_with_nx(prt_path: str, out_dir: Path, seed: int) -> dict[str, Any]:
    """Opens a PRT in NX and writes only static graph/face/mesh features."""
    import collect_axis_dataset as cad

    session = None
    work_part = None
    prt_abs = str(Path(prt_path).expanduser().resolve())
    part_name = _part_name_from_prt(prt_abs)
    try:
        session, work_part = cad.cam_session.create_session(input_file_dir=prt_abs)
        origin_body = max(work_part.Bodies, key=cad.get_body_volume)
        origin_faces = origin_body.GetFaces()
        origin_faces_tag = [face.Tag for face in origin_faces]

        graph, _, face_areas, face_types = cad.cam_utils.get_encoder_input_data(origin_faces, origin_faces_tag)
        face_areas = np.asarray(face_areas, dtype=np.float32)
        _, points_array_origin = cad.cam_utils.get_face_point_cloud(origin_faces)
        raw_face_count = int(len(origin_faces))
        if raw_face_count > 512:
            raise ValueError(
                f"Raw-face schema requires <=512 faces, but {prt_abs} has {raw_face_count} faces."
            )

        graph_dense = cad.nxg.relabel_nodes(
            graph,
            {tag: idx for idx, tag in enumerate(origin_faces_tag)},
            copy=True,
        )
        graph_json = cad.serialize_graph_to_node_link(graph_dense)
        groups_face = [[idx] for idx in range(raw_face_count)]

        centrality = cad.pad_1d_int(
            np.asarray([graph_dense.degree(n) for n in range(raw_face_count)], dtype=np.int16),
            512,
        )
        spatial_pos = cad.pad_2d_int(cad.build_graph_distance_matrix(graph_dense).astype(np.int16), 512)
        face_area_raw_512 = cad.pad_face_area(face_areas.astype(np.float32), 512)
        face_type_512 = cad.pad_1d_int(np.asarray(face_types, dtype=np.int16), 512)
        face_pc_raw_512 = cad.pad_face_pc(np.asarray(points_array_origin, dtype=np.float32), 512)
        node_mask_512 = cad.build_node_mask_512(raw_face_count)
        point_mask_512x100 = cad.build_point_mask_512x100(face_pc_raw_512, raw_face_count)
        normalization_center_xyz, normalization_scale, bbox_extent_xyz = cad.compute_reference_frame(
            face_pc_raw_512,
            node_mask_512,
        )

        points_array, norm_vecs_array, lines_array = [], [], []
        for face in origin_faces:
            if face.SolidFaceType.value == 10:
                pts, norms, lines = cad.sample_convergent_face_points(face)
            else:
                pts, norms, lines = cad.sample_face_points(face)
            points_array.append(pts)
            norm_vecs_array.append(norms)
            lines_array.append(lines)

        cad.orient_sample_directions_and_lines_outward(
            work_part,
            points_array,
            norm_vecs_array,
            lines_array,
            normalization_center_xyz,
        )
        measurement_point_coords = cad.extract_point_coordinates(points_array)

        face_normals_list = []
        for i, current_norms_raw in enumerate(norm_vecs_array):
            if current_norms_raw is None or len(current_norms_raw) == 0:
                mean_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                clean_vectors = []
                for vec in current_norms_raw:
                    if hasattr(vec, "Vector"):
                        clean_vectors.append([vec.Vector.X, vec.Vector.Y, vec.Vector.Z])
                    elif hasattr(vec, "X") and hasattr(vec, "Y") and hasattr(vec, "Z"):
                        clean_vectors.append([vec.X, vec.Y, vec.Z])
                    else:
                        clean_vectors.append(vec)
                mean_normal = np.mean(clean_vectors, axis=0).astype(np.float32)
                n_norm = float(np.linalg.norm(mean_normal))
                if n_norm > 1e-9:
                    mean_normal = mean_normal / n_norm
                else:
                    mean_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            face_points = (
                measurement_point_coords[i]
                if i < len(measurement_point_coords)
                else np.empty((0, 3), dtype=np.float32)
            )
            face_normals_list.append(
                cad.orient_normal_outward_from_center(
                    mean_normal,
                    face_points,
                    normalization_center_xyz,
                )
            )

        face_normal_512 = cad.build_group_face_normals_512(groups_face, face_normals_list, max_nodes=512)
        face_pc = cad.normalize_face_points(face_pc_raw_512, normalization_center_xyz, normalization_scale)
        face_area = cad.normalize_face_area(face_area_raw_512, normalization_scale)

        target_body_mesh_path = cad.export_body_to_obj(
            session,
            work_part,
            origin_body,
            str(out_dir / "target_body.obj"),
        )
        cad._json_dump(str(out_dir / "face_index_manifest.json"), {
            "schema_version": 1,
            "face_count": raw_face_count,
            "face_index_rule": "NX origin_body.GetFaces() order at static extraction time",
            "graph_node_id_equals_face_index": True,
            "nx_face_tags": [int(tag) for tag in origin_faces_tag],
        })
        cad._json_dump(str(out_dir / "graph_nx.json"), graph_json)
        cad._np_save(str(out_dir / "embed_centrality.npy"), centrality)
        cad._np_save(str(out_dir / "embed_spatial_pos.npy"), spatial_pos)
        cad._np_save(str(out_dir / "embed_face_area.npy"), face_area)
        cad._np_save(str(out_dir / "embed_face_type.npy"), face_type_512)
        cad._np_save(str(out_dir / "embed_face_pc.npy"), face_pc)
        cad._np_save(str(out_dir / "embed_face_normal.npy"), face_normal_512)
        cad._np_save(str(out_dir / "embed_node_mask.npy"), node_mask_512)
        cad._np_save(str(out_dir / "embed_point_mask.npy"), point_mask_512x100)

        meta = {
            "mode": "synthetic_static_feature_extraction_only",
            "prt_file_path": prt_abs,
            "target_body_mesh_path": str(Path(target_body_mesh_path).resolve()),
            "part_name": part_name,
            "seed": int(seed),
            "K_raw_faces": raw_face_count,
            "num_faces": int(len(origin_faces)),
            "normalization": {
                "center_xyz": normalization_center_xyz.astype(float).tolist(),
                "reference_scale": float(normalization_scale),
                "bbox_extent_xyz": bbox_extent_xyz.astype(float).tolist(),
            },
            "note": "NX was used only for PRT parsing/static B-rep features; no CAM operation was generated.",
        }
        cad._json_dump(str(out_dir / "meta.json"), meta)
        return {
            "normalization_center_xyz": normalization_center_xyz.astype(np.float32),
            "normalization_scale": float(normalization_scale),
            "bbox_extent_xyz": bbox_extent_xyz.astype(np.float32),
            "target_body_mesh_path": str(Path(target_body_mesh_path).resolve()),
            "raw_face_count": raw_face_count,
        }
    finally:
        try:
            import collect_axis_dataset as cad

            cad.force_close_part_by_path(prt_abs)
        except Exception:
            pass


def _load_static_meta(static_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for name in ("meta.json", "static_manifest.json"):
        path = static_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            meta.update(payload)
    return meta


def _load_valid_faces(static_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    node_mask = np.load(static_dir / "embed_node_mask.npy").reshape(-1)
    valid = np.where(node_mask[:512].astype(np.int16) == 0)[0].astype(np.int64)
    normals = np.load(static_dir / "embed_face_normal.npy").astype(np.float32).reshape(512, 3)
    if valid.size == 0:
        raise ValueError(f"No valid faces found in {static_dir}")
    return valid, normals


def _unit(vector: np.ndarray, fallback: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        arr = np.asarray(fallback, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
    return (arr / max(norm, 1e-8)).astype(np.float32)


def _load_target_mesh(static_dir: Path):
    mesh_path = static_dir / "target_body.obj"
    try:
        import trimesh
    except Exception:
        trimesh = None

    if trimesh is not None:
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"target_body.obj did not load as Trimesh: {mesh_path}")
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False,
        )
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            raise ValueError(f"Empty target mesh: {mesh_path}")
        return mesh

    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError("Stage B requires trimesh or pyvista to read target_body.obj.") from exc
    poly = pv.read(str(mesh_path))
    if isinstance(poly, pv.MultiBlock):
        poly = poly.combine()
    poly = poly.extract_surface().triangulate().clean()
    faces = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    vertices = np.asarray(poly.points, dtype=np.float64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Empty target mesh: {mesh_path}")
    bounds_raw = np.asarray(poly.bounds, dtype=np.float32)
    bounds = np.stack([bounds_raw[[0, 2, 4]], bounds_raw[[1, 3, 5]]], axis=0)
    return {
        "vertices": vertices,
        "faces": faces,
        "bounds": bounds,
        "polydata": poly,
    }


def _grid_from_mesh(mesh, resolution: int, padding_ratio: float, padding_mm: float) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray, tuple[float, float, float]]:
    bounds_value = mesh["bounds"] if isinstance(mesh, dict) else mesh.bounds
    bounds = np.asarray(bounds_value, dtype=np.float32)
    bbox_min = bounds[0].astype(np.float32)
    bbox_max = bounds[1].astype(np.float32)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    pad = extent * float(max(padding_ratio, 0.0)) + float(max(padding_mm, 0.0))
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    base = max(16, int(resolution))
    dims_np = np.maximum(np.round(extent / float(extent.max()) * float(base)).astype(np.int32), 16)
    dims = (int(dims_np[0]), int(dims_np[1]), int(dims_np[2]))
    xs = np.linspace(float(bbox_min[0]), float(bbox_max[0]), dims[0], dtype=np.float32)
    ys = np.linspace(float(bbox_min[1]), float(bbox_max[1]), dims[1], dtype=np.float32)
    zs = np.linspace(float(bbox_min[2]), float(bbox_max[2]), dims[2], dtype=np.float32)
    points = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    spacing = (
        float(xs[1] - xs[0]) if dims[0] > 1 else 1.0,
        float(ys[1] - ys[0]) if dims[1] > 1 else 1.0,
        float(zs[1] - zs[0]) if dims[2] > 1 else 1.0,
    )
    return points.astype(np.float32), dims, bbox_min.astype(np.float32), bbox_max.astype(np.float32), spacing


def _mesh_signed_distance_negative_inside(mesh, points: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    if isinstance(mesh, dict):
        try:
            import pyvista as pv
            from scipy.spatial import cKDTree
        except Exception as exc:
            raise RuntimeError("pyvista and scipy are required for OBJ fallback SDF generation.") from exc
        vertices = np.asarray(mesh["vertices"], dtype=np.float32)
        tree = cKDTree(vertices)
        poly = mesh["polydata"]
        out = np.empty((points.shape[0],), dtype=np.float32)
        for start in range(0, points.shape[0], max(1, int(chunk))):
            stop = min(start + max(1, int(chunk)), points.shape[0])
            pts = points[start:stop].astype(np.float32, copy=False)
            dist, _ = tree.query(pts, k=1, workers=-1)
            cloud = pv.PolyData(pts)
            selected = cloud.select_enclosed_points(poly, tolerance=1e-6, check_surface=False)
            inside = np.asarray(selected.point_data["SelectedPoints"], dtype=bool)
            out[start:stop] = np.where(inside, -dist, dist).astype(np.float32)
        return out

    prox = None
    out = np.empty((points.shape[0],), dtype=np.float32)
    try:
        prox = mesh.nearest
    except Exception:
        prox = None
    for start in range(0, points.shape[0], max(1, int(chunk))):
        stop = min(start + max(1, int(chunk)), points.shape[0])
        pts = points[start:stop].astype(np.float64, copy=False)
        try:
            raw_signed = np.asarray(mesh.nearest.signed_distance(pts), dtype=np.float64)
            dist = np.abs(raw_signed)
        except Exception:
            if prox is None:
                raise
            closest, dist, _ = prox.on_surface(pts)
            dist = np.asarray(dist, dtype=np.float64)
        inside = np.asarray(mesh.contains(pts), dtype=bool)
        out[start:stop] = np.where(inside, -dist, dist).astype(np.float32)
    return out


def _tsdf_from_sdf(sdf: np.ndarray, truncation: float) -> np.ndarray:
    tau = max(float(truncation), 1e-6)
    return np.clip(np.asarray(sdf, dtype=np.float32) / tau, -1.0, 1.0).astype(np.float32)


def _tsdf_from_solid(solid: np.ndarray, spacing: tuple[float, float, float], truncation: float) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception as exc:
        raise RuntimeError("scipy is required for synthetic TSDF generation.") from exc

    spacing_tuple = tuple(float(max(v, 1e-9)) for v in spacing)
    outside = ndimage.distance_transform_edt(~solid, sampling=spacing_tuple)
    inside = ndimage.distance_transform_edt(solid, sampling=spacing_tuple)
    return _tsdf_from_sdf(outside - inside, truncation).reshape(solid.shape)


def _grid_nearest_indices(points: np.ndarray, bbox_min: np.ndarray, spacing: tuple[float, float, float], dims: tuple[int, int, int]) -> np.ndarray:
    spacing_arr = np.asarray(spacing, dtype=np.float32).reshape(1, 3)
    idx = np.rint((points.astype(np.float32) - bbox_min.reshape(1, 3)) / spacing_arr).astype(np.int32)
    max_idx = np.asarray(dims, dtype=np.int32).reshape(1, 3) - 1
    return np.minimum(np.maximum(idx, 0), max_idx)


def _sample_grid_values(values: np.ndarray, points: np.ndarray, bbox_min: np.ndarray, spacing: tuple[float, float, float], dims: tuple[int, int, int]) -> np.ndarray:
    idx = _grid_nearest_indices(points, bbox_min, spacing, dims)
    return values[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32, copy=False)


def _radius_limit_at_s(tool_kind: str, s: np.ndarray, radius: float, tool_length: float) -> np.ndarray:
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
    axis = _unit(axis_dir)
    spacing_arr = np.asarray(spacing, dtype=np.float32).reshape(3)
    total_length = max(float(tool_length) + float(holder_length), 1e-6)
    max_radius = max(float(tool_radius), float(holder_radius), 1e-6)
    half_extent = np.abs(axis) * total_length + max_radius + spacing_arr * 1.5
    max_steps = np.ceil(half_extent / np.maximum(spacing_arr, 1e-9)).astype(np.int32)
    xs = np.arange(-int(max_steps[0]), int(max_steps[0]) + 1, dtype=np.int32)
    ys = np.arange(-int(max_steps[1]), int(max_steps[1]) + 1, dtype=np.int32)
    zs = np.arange(-int(max_steps[2]), int(max_steps[2]) + 1, dtype=np.int32)
    offsets = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    phys = offsets.astype(np.float32) * spacing_arr.reshape(1, 3)
    s = (phys @ axis.reshape(3, 1)).reshape(-1)
    radial_vec = phys - s.reshape(-1, 1) * axis.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)
    pad = 0.5 * float(spacing_arr.max())
    cutter_radius_limit = _radius_limit_at_s(tool_kind, s, float(tool_radius), float(tool_length))
    cutter_mask = cutter_radius_limit >= 0.0
    cutter_mask &= radial <= (cutter_radius_limit + pad)
    holder_mask = (s >= float(tool_length)) & (s <= float(tool_length) + float(holder_length))
    holder_mask &= radial <= (float(holder_radius) + pad)
    return {
        "cutter": np.unique(offsets[cutter_mask], axis=0).astype(np.int32),
        "holder": np.unique(offsets[holder_mask], axis=0).astype(np.int32),
    }


def _shift_or_mask_cpu(solid: np.ndarray, offsets: np.ndarray) -> np.ndarray:
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


def _dilate_config_mask_cpu(config_mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
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


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _use_gpu_cspace() -> bool:
    mode = os.getenv("SYNTHETIC_CSPACE_DEVICE", "auto").strip().lower()
    if mode in {"0", "false", "no", "cpu"}:
        return False
    if mode not in {"auto", "cuda", "gpu"}:
        raise ValueError("SYNTHETIC_CSPACE_DEVICE must be 'auto', 'cpu', or 'cuda'.")
    return _torch_cuda_available()


def _scatter_offsets_gpu(mask: np.ndarray, offsets: np.ndarray, *, add_offsets: bool) -> np.ndarray:
    """GPU sparse OR over offsets.

    add_offsets=True computes out[p + offset] |= mask[p].
    add_offsets=False computes out[p - offset] |= mask[p].
    """
    import torch

    active = np.argwhere(mask)
    if active.size == 0 or offsets.size == 0:
        return np.zeros_like(mask, dtype=bool)

    device = torch.device("cuda")
    active_t = torch.as_tensor(active.astype(np.int32, copy=False), device=device)
    offsets_t = torch.as_tensor(offsets.astype(np.int32, copy=False), device=device)
    nx, ny, nz = (int(v) for v in mask.shape)
    out = torch.zeros(nx * ny * nz, dtype=torch.bool, device=device)

    max_pairs = max(1, _env_int("SYNTHETIC_CSPACE_GPU_MAX_PAIRS", 4_000_000))
    chunk = max(1, max_pairs // max(int(active_t.shape[0]), 1))
    for start in range(0, int(offsets_t.shape[0]), chunk):
        off = offsets_t[start:start + chunk]
        points = active_t[:, None, :] + off[None, :, :] if add_offsets else active_t[:, None, :] - off[None, :, :]
        valid = (
            (points[..., 0] >= 0)
            & (points[..., 0] < nx)
            & (points[..., 1] >= 0)
            & (points[..., 1] < ny)
            & (points[..., 2] >= 0)
            & (points[..., 2] < nz)
        )
        if bool(valid.any().item()):
            valid_points = points[valid].to(torch.int64)
            linear = valid_points[:, 0] * (ny * nz) + valid_points[:, 1] * nz + valid_points[:, 2]
            out[linear] = True

    torch.cuda.synchronize()
    return out.reshape(mask.shape).cpu().numpy().astype(bool, copy=False)


def _warn_gpu_cspace_fallback(exc: Exception) -> None:
    global _GPU_CSPACE_FALLBACK_WARNED
    if _GPU_CSPACE_FALLBACK_WARNED:
        return
    _GPU_CSPACE_FALLBACK_WARNED = True
    print(f"[Synthetic C-space] GPU path failed; falling back to CPU. reason={exc}", flush=True)


def _shift_or_mask_gpu(solid: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return _scatter_offsets_gpu(solid, offsets, add_offsets=False)


def _dilate_config_mask_gpu(config_mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return _scatter_offsets_gpu(config_mask, offsets, add_offsets=True)


def _shift_or_mask(solid: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    if _use_gpu_cspace():
        try:
            return _shift_or_mask_gpu(solid, offsets)
        except Exception as exc:
            _warn_gpu_cspace_fallback(exc)
    return _shift_or_mask_cpu(solid, offsets)


def _dilate_config_mask(config_mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    if _use_gpu_cspace():
        try:
            return _dilate_config_mask_gpu(config_mask, offsets)
        except Exception as exc:
            _warn_gpu_cspace_fallback(exc)
    return _dilate_config_mask_cpu(config_mask, offsets)


def _scaled_dims(dims: tuple[int, int, int], max_resolution: int) -> tuple[int, int, int]:
    max_dim = max(int(v) for v in dims)
    target = max(1, int(max_resolution))
    if max_dim <= target:
        return tuple(int(v) for v in dims)
    scale = float(target) / float(max_dim)
    return tuple(max(4, int(round(float(v) * scale))) for v in dims)


def _resample_mask_nearest(mask: np.ndarray, out_dims: tuple[int, int, int]) -> np.ndarray:
    in_dims = tuple(int(v) for v in mask.shape)
    out_dims = tuple(int(v) for v in out_dims)
    if in_dims == out_dims:
        return mask.astype(bool, copy=True)
    axes = [
        np.rint(np.linspace(0, in_dims[axis] - 1, out_dims[axis])).astype(np.int64)
        for axis in range(3)
    ]
    return mask[np.ix_(axes[0], axes[1], axes[2])].astype(bool, copy=False)


def _downsample_mask_any(mask: np.ndarray, out_dims: tuple[int, int, int]) -> np.ndarray:
    in_dims = tuple(int(v) for v in mask.shape)
    out_dims = tuple(int(v) for v in out_dims)
    if in_dims == out_dims:
        return mask.astype(bool, copy=True)
    if all(in_dims[axis] % out_dims[axis] == 0 for axis in range(3)):
        factors = tuple(in_dims[axis] // out_dims[axis] for axis in range(3))
        return mask.reshape(
            out_dims[0], factors[0],
            out_dims[1], factors[1],
            out_dims[2], factors[2],
        ).any(axis=(1, 3, 5))

    x_edges = np.linspace(0, in_dims[0], out_dims[0] + 1).astype(np.int32)
    y_edges = np.linspace(0, in_dims[1], out_dims[1] + 1).astype(np.int32)
    z_edges = np.linspace(0, in_dims[2], out_dims[2] + 1).astype(np.int32)
    out = np.zeros(out_dims, dtype=bool)
    for ix in range(out_dims[0]):
        xs = slice(int(x_edges[ix]), max(int(x_edges[ix + 1]), int(x_edges[ix]) + 1))
        for iy in range(out_dims[1]):
            ys = slice(int(y_edges[iy]), max(int(y_edges[iy + 1]), int(y_edges[iy]) + 1))
            for iz in range(out_dims[2]):
                zs = slice(int(z_edges[iz]), max(int(z_edges[iz + 1]), int(z_edges[iz]) + 1))
                out[ix, iy, iz] = bool(mask[xs, ys, zs].any())
    return out


def _spacing_for_dims(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    dims: tuple[int, int, int],
) -> tuple[float, float, float]:
    extent = np.maximum(np.asarray(bbox_max, dtype=np.float32) - np.asarray(bbox_min, dtype=np.float32), 1e-6)
    return (
        float(extent[0] / max(int(dims[0]) - 1, 1)),
        float(extent[1] / max(int(dims[1]) - 1, 1)),
        float(extent[2] / max(int(dims[2]) - 1, 1)),
    )


def _holder_forbidden_mask(
    obstacle: np.ndarray,
    axis_dir: np.ndarray,
    tool_kind: str,
    tool_radius: float,
    tool_length: float,
    holder_radius: float,
    holder_length: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    coarse_resolution = _env_int("SYNTHETIC_HOLDER_CSPACE_RESOLUTION", 64)
    full_dims = tuple(int(v) for v in obstacle.shape)
    coarse_dims = _scaled_dims(full_dims, coarse_resolution)
    if coarse_dims == full_dims:
        kernels = _make_kernel_offsets(
            axis_dir=axis_dir,
            spacing=_spacing_for_dims(bbox_min, bbox_max, full_dims),
            tool_kind=tool_kind,
            tool_radius=tool_radius,
            tool_length=tool_length,
            holder_radius=holder_radius,
            holder_length=holder_length,
        )
        return _shift_or_mask(obstacle, kernels["holder"])

    coarse_obstacle = _downsample_mask_any(obstacle, coarse_dims)
    coarse_spacing = _spacing_for_dims(bbox_min, bbox_max, coarse_dims)
    coarse_kernels = _make_kernel_offsets(
        axis_dir=axis_dir,
        spacing=coarse_spacing,
        tool_kind=tool_kind,
        tool_radius=tool_radius,
        tool_length=tool_length,
        holder_radius=holder_radius,
        holder_length=holder_length,
    )
    coarse_forbidden = _shift_or_mask_cpu(coarse_obstacle, coarse_kernels["holder"])
    return _resample_mask_nearest(coarse_forbidden, full_dims)


def _axis_clearance_features(
    solid: np.ndarray,
    flat_indices: np.ndarray,
    axis_dir: np.ndarray,
    spacing: tuple[float, float, float],
    scale: float,
    tool_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns approach-side stock depth and a simple blocked flag per grid index."""
    indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    dims = tuple(int(v) for v in solid.shape)
    ijk = np.stack(np.unravel_index(indices, dims), axis=1).astype(np.float32)
    inside = solid.reshape(-1)[indices].astype(bool, copy=False)
    clearance = np.zeros((indices.shape[0],), dtype=np.float32)
    if not bool(inside.any()):
        return clearance, clearance

    axis = _unit(axis_dir)
    spacing_arr = np.asarray(spacing, dtype=np.float32).reshape(3)
    index_dir = axis / np.maximum(spacing_arr, 1e-9)
    max_component = float(np.max(np.abs(index_dir)))
    if max_component <= 1e-9:
        return clearance, clearance

    step_index = (index_dir / max_component).astype(np.float32)
    step_distance = float(np.linalg.norm(step_index * spacing_arr))
    max_steps = int(os.getenv("SYNTHETIC_AXIS_CLEARANCE_MAX_STEPS", str(max(dims) * 2)))
    max_steps = max(1, max_steps)

    pos = ijk.copy()
    active = inside.copy()
    for step in range(1, max_steps + 1):
        if not bool(active.any()):
            break
        pos[active] += step_index.reshape(1, 3)
        rounded = np.rint(pos[active]).astype(np.int32)
        in_bounds = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < dims[0])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < dims[1])
            & (rounded[:, 2] >= 0)
            & (rounded[:, 2] < dims[2])
        )
        active_indices = np.flatnonzero(active)
        exited = active_indices[~in_bounds]
        if exited.size:
            clearance[exited] = float(step) * step_distance
            active[exited] = False
        if bool(in_bounds.any()):
            bounded = rounded[in_bounds]
            bounded_active_indices = active_indices[in_bounds]
            empty = ~solid[bounded[:, 0], bounded[:, 1], bounded[:, 2]]
            reached_empty = bounded_active_indices[empty]
            if reached_empty.size:
                clearance[reached_empty] = float(step) * step_distance
                active[reached_empty] = False
    if bool(active.any()):
        clearance[active] = float(max_steps) * step_distance

    scale = max(float(scale), 1e-6)
    clearance_norm = (clearance / scale).astype(np.float32, copy=False)
    blocked = (inside & (clearance > float(tool_radius))).astype(np.float32, copy=False)
    return clearance_norm, blocked


def _action_roi_mask(
    row_macro: str,
    face_id: int,
    grid_points: np.ndarray,
    static_dir: Path,
    center: np.ndarray,
    scale: float,
    dims: tuple[int, int, int],
    tool_radius: float,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    if row_macro == "indexed_rough":
        return np.ones(dims, dtype=bool)
    face_pc = np.load(static_dir / "embed_face_pc.npy").astype(np.float32).reshape(512, 100, 3)
    points = face_pc[int(face_id)].reshape(-1, 3)
    points = points[np.any(np.abs(points) > 1e-12, axis=1)]
    if points.size == 0:
        return np.ones(dims, dtype=bool)
    points_raw = points * float(scale) + np.asarray(center, dtype=np.float32).reshape(1, 3)
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("scipy is required for action-face ROI generation.") from exc
    tree = cKDTree(points_raw.astype(np.float32, copy=False))
    dist, _ = tree.query(grid_points.astype(np.float32, copy=False), k=1, workers=-1)
    pad = max(float(tool_radius) * 2.5, float(max(spacing)) * 3.0)
    return (np.asarray(dist, dtype=np.float32).reshape(dims) <= pad)


def _select_indices(rng: np.random.Generator, count: int, limit: int, priority_mask: np.ndarray | None = None, priority_fraction: float = 0.5) -> np.ndarray:
    limit = int(max(1, limit))
    if count <= limit and priority_mask is None:
        return np.arange(count, dtype=np.int64)
    if priority_mask is not None and bool(priority_mask.any()):
        priority = np.where(priority_mask.reshape(-1))[0]
        other = np.where(~priority_mask.reshape(-1))[0]
        p_count = min(int(round(limit * float(priority_fraction))), limit)
        p_count = min(p_count, priority.size)
        chosen_parts = []
        if p_count > 0:
            chosen_parts.append(rng.choice(priority, size=p_count, replace=priority.size < p_count))
        remaining = limit - p_count
        source = other if other.size > 0 else priority
        if remaining > 0:
            chosen_parts.append(rng.choice(source, size=remaining, replace=source.size < remaining))
        chosen = np.concatenate(chosen_parts).astype(np.int64)
        rng.shuffle(chosen)
        return chosen
    if count >= limit:
        return rng.choice(count, size=limit, replace=False).astype(np.int64)
    return rng.choice(count, size=limit, replace=True).astype(np.int64)


def _perp_axis(normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    basis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(n @ basis)) > 0.85:
        basis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return _unit(np.cross(n, basis))


def _axis_for_macro(macro_name: str, face_id: int, normals: np.ndarray, rng: random.Random) -> np.ndarray:
    normal = _unit(normals[int(face_id)])
    if macro_name in {"point_finish", "indexed_finish"}:
        return normal
    if macro_name == "flank_finish":
        return _perp_axis(normal)
    base_axes = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    ]
    if macro_name == "indexed_rough":
        axis_library = base_axes + [normal, -normal]
    else:
        axis_library = base_axes
    return axis_library[rng.randrange(len(axis_library))]


def _tool_configs() -> list[dict[str, float | str]]:
    flat_diameters = _parse_float_list("SYNTHETIC_FLAT_TOOL_DIAMETERS", "4,6,8,10,12,16,20")
    ball_diameters = _parse_float_list("SYNTHETIC_BALL_TOOL_DIAMETERS", "4,6,8")
    length_mults = _parse_int_list("SYNTHETIC_TOOL_LENGTH_MULTIPLIERS", "3,4,5")
    holder_dia_mults = _parse_int_list("SYNTHETIC_HOLDER_DIAMETER_MULTIPLIERS", "3,4,5")
    holder_length = float(os.getenv("SYNTHETIC_HOLDER_LENGTH_MM", "100.0"))

    configs: list[dict[str, float | str]] = []
    for kind, diameters in (("flat", flat_diameters), ("ball", ball_diameters)):
        for diameter in diameters:
            for tool_len_mult in length_mults:
                for holder_dia_mult in holder_dia_mults:
                    configs.append({
                        "tool_kind": kind,
                        "tool_diameter": float(diameter),
                        "tool_radius": float(diameter) * 0.5,
                        "tool_length": float(diameter) * float(tool_len_mult),
                        "holder_diameter": float(diameter) * float(holder_dia_mult),
                        "holder_radius": float(diameter) * float(holder_dia_mult) * 0.5,
                        "holder_length": holder_length,
                        "tool_length_multiplier": float(tool_len_mult),
                        "holder_diameter_multiplier": float(holder_dia_mult),
                        "holder_length_multiplier": 0.0,
                    })
    return configs


def _compatible_macros(tool_kind: str) -> list[str]:
    if tool_kind == "flat":
        return ["indexed_rough", "flank_finish"]
    if tool_kind == "ball":
        return ["indexed_finish", "point_finish"]
    return ["indexed_rough", "indexed_finish", "point_finish", "flank_finish"]


def _serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _write_rows_to_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{key: _serialize(value) for key, value in row.items()} for row in rows])
    df.to_parquet(path, index=False)


def _generate_rows(
    *,
    prt_path: str,
    out_dir: Path,
    static_dir: Path,
    seed: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    part_name = _part_name_from_prt(prt_path)
    valid_faces, normals = _load_valid_faces(static_dir)
    max_action_faces = _env_int("SYNTHETIC_MAX_ACTION_FACES", 0)
    part_hash = int(hashlib.sha1(part_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(int(seed) + part_hash)
    np_rng = np.random.default_rng(int(seed) + part_hash)

    face_ids = valid_faces.tolist()
    rng.shuffle(face_ids)
    if max_action_faces > 0:
        face_ids = face_ids[:max_action_faces]
    if not face_ids:
        raise ValueError(f"No action faces available for {static_dir}")

    configs = _tool_configs()
    rng.shuffle(configs)

    grid_resolution = _env_int("SYNTHETIC_GRID_RESOLUTION", 256)
    grid_padding_ratio = float(os.getenv("SYNTHETIC_GRID_PADDING_RATIO", "0.10"))
    grid_padding_mm = float(os.getenv("SYNTHETIC_GRID_PADDING_MM", "5.0"))
    steps_per_rollout = max(1, _env_int("SYNTHETIC_STEPS_PER_ROLLOUT", 4))
    rough_steps = max(1, _env_int("SYNTHETIC_ROUGH_STEPS", _env_int("EARLY_ROUGH_ONLY_STEPS", 2)))
    rough_steps = min(rough_steps, steps_per_rollout)
    sdf_query_count = max(256, _env_int("SYNTHETIC_SDF_QUERY_COUNT", _env_int("SDF_QUERY_COUNT", 32768)))
    octree_query_count = max(256, _env_int("SYNTHETIC_OCTREE_QUERY_NODES", 4096))
    tsdf_truncation = float(os.getenv("SYNTHETIC_TSDF_TRUNCATION", os.getenv("SDF_QUERY_TRUNCATION", "5.0")))
    tsdf_truncation = max(tsdf_truncation, 1e-6)

    shared_info = {
        "row_unit": "one_synthetic_minkowski_transition",
        "label_status": "synthetic_minkowski_cspace",
        "static_feature_source": "nx_prt_static_extraction_or_reused_static_dir",
        "synthetic_generator": "dense_grid_minkowski_cspace_v2",
        "grid_resolution": int(grid_resolution),
        "steps_per_rollout": int(steps_per_rollout),
        "rough_steps": int(rough_steps),
        "sdf_query_count": int(sdf_query_count),
        "octree_query_count": int(octree_query_count),
        "tsdf_truncation": float(tsdf_truncation),
        "tool_diameters": {
            "flat": _parse_float_list("SYNTHETIC_FLAT_TOOL_DIAMETERS", "4,6,8,10,12,16,20"),
            "ball": _parse_float_list("SYNTHETIC_BALL_TOOL_DIAMETERS", "4,6,8"),
        },
        "tool_length_rule": "tool_length = diameter * integer_multiplier",
        "holder_diameter_rule": "holder_diameter = diameter * integer_multiplier",
        "holder_length_rule": "holder_length = fixed_mm",
        "holder_length_mm": float(os.getenv("SYNTHETIC_HOLDER_LENGTH_MM", "100.0")),
    }
    meta = _load_static_meta(static_dir)
    normalization = meta.get("normalization", {}) if isinstance(meta.get("normalization", {}), dict) else {}
    center = normalization.get("center_xyz", meta.get("normalization_center_xyz", [0.0, 0.0, 0.0]))
    scale = float(normalization.get("reference_scale", meta.get("normalization_scale", 1.0)))
    extent = normalization.get("bbox_extent_xyz", meta.get("bbox_extent_xyz", [1.0, 1.0, 1.0]))
    center_np = np.asarray(center, dtype=np.float32).reshape(3)
    scale = max(float(scale), 1e-6)
    extent_np = np.asarray(extent, dtype=np.float32).reshape(3)
    common = {
        "part_name": part_name,
        "prt_file_path": str(Path(prt_path).expanduser().resolve()),
        "target_body_mesh_path": str((static_dir / "target_body.obj").resolve()),
        "seed": int(seed),
        "static_feature_dir": str(static_dir.resolve()),
        "graph_nx_json_path": str((static_dir / "graph_nx.json").resolve()),
        "normalization_center_xyz": center_np,
        "normalization_scale": float(scale),
        "bbox_extent_xyz": extent_np,
        "info_json": json.dumps(shared_info, ensure_ascii=False),
    }

    mesh = _load_target_mesh(static_dir)
    grid_points, dims, bbox_min, bbox_max, spacing = _grid_from_mesh(
        mesh,
        resolution=grid_resolution,
        padding_ratio=grid_padding_ratio,
        padding_mm=grid_padding_mm,
    )
    target_sdf = _mesh_signed_distance_negative_inside(mesh, grid_points)
    target_tsdf = _tsdf_from_sdf(target_sdf, tsdf_truncation).reshape(dims)
    target_solid = target_tsdf <= 0.0

    stock_solid = np.ones(dims, dtype=bool)
    stock_solid[0, :, :] = False
    stock_solid[-1, :, :] = False
    stock_solid[:, 0, :] = False
    stock_solid[:, -1, :] = False
    stock_solid[:, :, 0] = False
    stock_solid[:, :, -1] = False
    stock_solid |= target_solid
    stock_tsdf = _tsdf_from_solid(stock_solid, spacing, tsdf_truncation)

    face_pc_norm = np.load(static_dir / "embed_face_pc.npy").astype(np.float32).reshape(512, 100, 3)
    face_pc_raw = face_pc_norm * float(scale) + center_np.reshape(1, 1, 3)
    node_mask = np.load(static_dir / "embed_node_mask.npy").astype(np.int16).reshape(512)
    valid_node_mask = node_mask == 0
    grid_points_norm = ((grid_points - center_np.reshape(1, 3)) / float(scale)).astype(np.float32)
    flat_grid_count = int(grid_points.shape[0])
    grid_depths = np.full((flat_grid_count,), GRID_DEPTH_VALUE, dtype=np.int16)

    flat_configs = [config for config in configs if str(config["tool_kind"]) == "flat"]
    ball_configs = [config for config in configs if str(config["tool_kind"]) == "ball"]
    if not flat_configs or not ball_configs:
        raise ValueError("Synthetic tool config requires at least one flat and one ball tool.")

    def _face_point_sdf_raw(tsdf_grid: np.ndarray) -> np.ndarray:
        sampled = _sample_grid_values(
            tsdf_grid,
            face_pc_raw.reshape(-1, 3),
            bbox_min,
            spacing,
            dims,
        ).reshape(512, 100)
        return (sampled * float(tsdf_truncation)).astype(np.float32, copy=False)

    def _node_mean_sdf(point_sdf_raw: np.ndarray) -> np.ndarray:
        return point_sdf_raw.reshape(512, 100).mean(axis=1, dtype=np.float32)

    def _sample_transition_payload(
        before_tsdf: np.ndarray,
        after_tsdf: np.ndarray,
        before_solid: np.ndarray,
        removed_mask: np.ndarray,
        axis_dir: np.ndarray,
        tool_radius: float,
    ) -> dict[str, np.ndarray]:
        before_flat = before_tsdf.reshape(-1)
        after_flat = after_tsdf.reshape(-1)
        target_flat = target_tsdf.reshape(-1)
        changed_flat = np.abs(after_flat - before_flat) > 1e-4
        removed_flat = removed_mask.reshape(-1)
        priority = changed_flat | removed_flat

        sdf_idx = _select_indices(
            np_rng,
            flat_grid_count,
            sdf_query_count,
            priority_mask=priority,
            priority_fraction=0.55,
        )
        oct_idx = _select_indices(
            np_rng,
            flat_grid_count,
            octree_query_count,
            priority_mask=priority,
            priority_fraction=0.55,
        )

        before_occ = (before_flat[oct_idx] <= 0.0).astype(np.float32)
        after_occ = (after_flat[oct_idx] <= 0.0).astype(np.float32)
        target_occ = (target_flat[oct_idx] <= 0.0).astype(np.float32)
        removed_occ = removed_flat[oct_idx].astype(np.float32)
        sdf_clearance, sdf_blocked = _axis_clearance_features(
            before_solid,
            sdf_idx,
            axis_dir,
            spacing,
            float(scale),
            float(tool_radius),
        )
        oct_clearance, oct_blocked = _axis_clearance_features(
            before_solid,
            oct_idx,
            axis_dir,
            spacing,
            float(scale),
            float(tool_radius),
        )
        return {
            "sdf_query_points": grid_points_norm[sdf_idx].astype(np.float32, copy=False),
            "sdf_tsdf_before": before_flat[sdf_idx].astype(np.float32, copy=False),
            "sdf_tsdf_after": after_flat[sdf_idx].astype(np.float32, copy=False),
            "sdf_delta_tsdf": (after_flat[sdf_idx] - before_flat[sdf_idx]).astype(np.float32, copy=False),
            "sdf_target_tsdf": target_flat[sdf_idx].astype(np.float32, copy=False),
            "sdf_axis_clearance_before": sdf_clearance,
            "sdf_axis_blocked_before": sdf_blocked,
            "octree_centers": grid_points_norm[oct_idx].astype(np.float32, copy=False),
            "octree_depths": grid_depths[oct_idx].astype(np.int16, copy=False),
            "octree_occ_labels_before": before_occ,
            "octree_occ_labels": after_occ,
            "octree_fill_before": before_occ,
            "octree_fill_after": after_occ,
            "octree_removed_fraction": removed_occ,
            "octree_tsdf_before": before_flat[oct_idx].astype(np.float32, copy=False),
            "octree_tsdf_after": after_flat[oct_idx].astype(np.float32, copy=False),
            "octree_delta_tsdf": (after_flat[oct_idx] - before_flat[oct_idx]).astype(np.float32, copy=False),
            "octree_target_tsdf": target_flat[oct_idx].astype(np.float32, copy=False),
            "octree_axis_clearance_before": oct_clearance,
            "octree_axis_blocked_before": oct_blocked,
            "octree_bbox_min": ((bbox_min - center_np) / float(scale)).astype(np.float32),
            "octree_bbox_max": ((bbox_max - center_np) / float(scale)).astype(np.float32),
        }

    def _apply_synthetic_operation(
        current_solid: np.ndarray,
        macro_name: str,
        face_id: int,
        config: dict[str, float | str],
        axis_dir: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kernels = _make_kernel_offsets(
            axis_dir=axis_dir,
            spacing=spacing,
            tool_kind=str(config["tool_kind"]),
            tool_radius=float(config["tool_radius"]),
            tool_length=float(config["tool_length"]),
            holder_radius=float(config["holder_radius"]),
            holder_length=float(config["holder_length"]),
        )
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
            center_np,
            float(scale),
            dims,
            float(config["tool_radius"]),
            spacing,
        )
        config_candidates = ideal_removal & roi & ~holder_forbidden
        swept_volume = _dilate_config_mask(config_candidates, kernels["cutter"])
        removed = ideal_removal & roi & swept_volume
        next_solid = current_solid & ~removed
        next_solid |= target_solid
        return next_solid, removed, holder_forbidden

    rows: list[dict[str, Any]] = []
    candidate_index = 0
    rollout_index = 0
    finish_macros = ("indexed_finish", "point_finish", "flank_finish")
    while len(rows) < max_rows:
        scenario_id = f"synth_rollout_{rollout_index:05d}"
        current_solid = stock_solid.copy()
        current_tsdf = stock_tsdf.copy()
        rough_done = np.zeros((512,), dtype=np.float32)
        rng.shuffle(face_ids)

        for step in range(steps_per_rollout):
            if len(rows) >= max_rows:
                break
            if step < rough_steps:
                macro_name = "indexed_rough"
                config = flat_configs[rng.randrange(len(flat_configs))]
            else:
                macro_name = finish_macros[rng.randrange(len(finish_macros))]
                config_pool = flat_configs if macro_name == "flank_finish" else ball_configs
                config = config_pool[rng.randrange(len(config_pool))]

            face_id = int(face_ids[(rollout_index * steps_per_rollout + step) % len(face_ids)])
            macro_id = int(MACRO_CLASS_TO_ID[macro_name])
            axis_dir = _axis_for_macro(macro_name, face_id, normals, rng)
            before_tsdf = current_tsdf
            before_point_sdf_raw = _face_point_sdf_raw(before_tsdf)
            before_node_sdf_raw = _node_mean_sdf(before_point_sdf_raw)

            next_solid, removed_mask, holder_forbidden = _apply_synthetic_operation(
                current_solid=current_solid,
                macro_name=macro_name,
                face_id=face_id,
                config=config,
                axis_dir=axis_dir,
            )
            after_tsdf = _tsdf_from_solid(next_solid, spacing, tsdf_truncation)
            after_point_sdf_raw = _face_point_sdf_raw(after_tsdf)
            after_node_sdf_raw = _node_mean_sdf(after_point_sdf_raw)
            affected_delta = np.maximum(after_node_sdf_raw - before_node_sdf_raw, 0.0).astype(np.float32)
            affected_mask = ((affected_delta > 1e-4) & valid_node_mask).astype(np.float32)
            finish_ready = ((rough_done > 0.5) & valid_node_mask).astype(np.float32)
            node_process_state = np.stack([rough_done, finish_ready], axis=1).astype(np.float32)
            payload = _sample_transition_payload(
                before_tsdf=before_tsdf,
                after_tsdf=after_tsdf,
                before_solid=current_solid,
                removed_mask=removed_mask,
                axis_dir=axis_dir,
                tool_radius=float(config["tool_radius"]),
            )

            tool_kind = str(config["tool_kind"])
            tool_choice_name = tool_choice_key(tool_kind, float(config["tool_diameter"]))
            row = dict(common)
            row.update({
                "scenario_id": scenario_id,
                "parent_scenario_id": "",
                "decision_step": int(step),
                "candidate_index": int(candidate_index),
                "is_chosen": 1,
                "synthetic_label_status": "synthetic_minkowski_cspace",
                "synthetic_grid_dims": np.asarray(dims, dtype=np.int16),
                "synthetic_grid_spacing": np.asarray(spacing, dtype=np.float32),
                "synthetic_tsdf_truncation": float(tsdf_truncation),
                "synthetic_removed_voxels": int(removed_mask.sum()),
                "synthetic_holder_forbidden_ratio": float(holder_forbidden.mean()),
                "macro_class_id": macro_id,
                "macro_class_name": macro_name,
                "action_face_id": int(face_id),
                "action_face_valid": 1,
                "anchor_face_id": int(face_id) if macro_name == "indexed_rough" else -1,
                "target_face_id": -1 if macro_name == "indexed_rough" else int(face_id),
                "tool_kind": tool_kind,
                "tool_type_name": tool_kind,
                "tool_choice_name": tool_choice_name,
                "tool_choice_id": int(TOOL_CHOICE_TO_ID.get(tool_choice_name, -1)),
                "tool_choice_valid": int(tool_choice_name in TOOL_CHOICE_TO_ID),
                "tool_diameter": float(config["tool_diameter"]),
                "tool_radius": float(config["tool_radius"]),
                "tool_length": float(config["tool_length"]),
                "holder_diameter": float(config["holder_diameter"]),
                "holder_radius": float(config["holder_radius"]),
                "holder_length": float(config["holder_length"]),
                "tool_length_multiplier": float(config["tool_length_multiplier"]),
                "holder_diameter_multiplier": float(config["holder_diameter_multiplier"]),
                "holder_length_multiplier": float(config["holder_length_multiplier"]),
                "axis_dir": axis_dir.astype(np.float32),
                "axis_visible": np.ones((512,), dtype=np.int16),
                "axis_visible_512": np.ones((512,), dtype=np.int16),
                "axis_visible_ratio": 1.0,
                "node_process_state": node_process_state,
                "state_point_sdf_raw_512x100": before_point_sdf_raw,
                "next_point_sdf_raw_512x100": after_point_sdf_raw,
                "next_node_sdf_raw_512": after_node_sdf_raw.astype(np.float32, copy=False),
                "affected_face_delta_512": affected_delta,
                "affected_face_mask_512": affected_mask,
            })
            row.update(payload)
            rows.append(row)
            candidate_index += 1

            current_solid = next_solid
            current_tsdf = after_tsdf
            if macro_name == "indexed_rough":
                rough_done[valid_node_mask] = 1.0
        rollout_index += 1

    return rows


def collect_synthetic_episode(prt_path: str, out_root: str, seed: int, pc_name: str = "") -> dict[str, Any]:
    part_name = _part_name_from_prt(prt_path)
    out_dir = _create_run_output_dir(out_root, part_name, seed, pc_name=pc_name)
    extract_with_nx = _env_bool("SYNTHETIC_EXTRACT_STATIC_WITH_NX", True)
    reuse_static = _env_bool("SYNTHETIC_REUSE_STATIC_FEATURES", False)
    static_src: Path | None = None
    if extract_with_nx:
        _extract_static_features_with_nx(prt_path, out_dir, seed)
    elif reuse_static:
        static_src = _find_static_feature_dir(prt_path, out_root)
        _copy_static_features(static_src, out_dir)
    else:
        raise ValueError(
            "SYNTHETIC_EXTRACT_STATIC_WITH_NX=0 requires SYNTHETIC_REUSE_STATIC_FEATURES=1 "
            "and a complete static feature directory."
        )

    max_rows = max(1, _env_int("SYNTHETIC_SCENARIOS_PER_PART", 512))
    rows = _generate_rows(
        prt_path=prt_path,
        out_dir=out_dir,
        static_dir=out_dir,
        seed=seed,
        max_rows=max_rows,
    )
    if not rows:
        raise ValueError(f"No synthetic rows generated for {prt_path}")

    parquet_path = out_dir / f"{part_name}_seed{int(seed)}_{DEFAULT_OUTPUT_BASENAME}"
    chosen_path = out_dir / f"{part_name}_seed{int(seed)}_process_skeleton_dataset_chosen_only.parquet"
    _write_rows_to_parquet(rows, parquet_path)
    _write_rows_to_parquet(rows, chosen_path)

    pc_slug = _safe_filename(pc_name)
    parquet_run_name = _strip_pc_suffix(out_dir.name, pc_slug)
    global_dir = Path(out_root).expanduser().resolve() / "_ALL_PARQUET_FILES"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_parquet = global_dir / f"{parquet_run_name}.parquet"
    global_chosen = global_dir / f"{parquet_run_name}_chosen_only.parquet"
    shutil.copy2(parquet_path, global_parquet)
    shutil.copy2(chosen_path, global_chosen)

    episode_record = {
        "part_name": part_name,
        "prt_file_path": str(Path(prt_path).expanduser().resolve()),
        "seed": int(seed),
        "row_unit": "one_synthetic_minkowski_transition",
        "static_feature_source": str(static_src) if static_src is not None else "nx_prt_static_extraction",
        "nx_usage": "prt_open_static_brep_feature_extraction_only",
        "num_rows": int(len(rows)),
        "label_status": "synthetic_minkowski_cspace",
        "termination": {"reason": "synthetic_scenario_generation_complete"},
        "outputs": {
            "parquet_path": str(parquet_path),
            "chosen_parquet_path": str(chosen_path),
            "global_parquet_path": str(global_parquet),
            "global_chosen_parquet_path": str(global_chosen),
        },
    }
    (out_dir / "episode_record.json").write_text(
        json.dumps(episode_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return episode_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic v2 scenario dataset worker")
    parser.add_argument("--input", type=str, required=True, help="Path to input .prt file")
    parser.add_argument("--output", type=str, required=True, help="Output root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pc-name", type=str, default=os.getenv("PC_NAME", ""))
    args = parser.parse_args()

    record = collect_synthetic_episode(
        prt_path=args.input,
        out_root=args.output,
        seed=int(args.seed),
        pc_name=args.pc_name,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
