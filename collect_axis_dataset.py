"""Collects axis-action training rows by simulating stepwise CAM rollouts in NX."""
import os
import json
import math
import gc
import sys
from typing import Any, Dict, Tuple, List
from datetime import datetime
import argparse

import numpy as np
import networkx as nxg
from networkx.readwrite import json_graph
import pandas as pd
import NXOpen

from graph_face_compression import reduce_visibility_by_groups
from graph_sdf.schema import (
    MACRO_CLASS_TO_ID,
    TOOL_CHOICE_TO_ID,
    TOOL_LIBRARY,
    build_tool_choice_mask_for_macro_class,
    tool_choice_key,
)

from CAM import session as cam_session
from CAM import geometry, operations
from CAM import utils as cam_utils

from CAM.measurements import get_body_volume, sample_convergent_face_points, sample_face_points


def save_part(session, work_part, directory):
    work_part = session.Parts.Work
    savePart = work_part.SaveAs(directory)
    savePart.Dispose()

def serialize_graph_to_node_link(G: nxg.Graph) -> dict:
    """Serializes a NetworkX graph into node-link JSON format."""
    return json_graph.node_link_data(G)

def _ensure_dir(p: str) -> None:
    """Performs: ensure dir."""
    os.makedirs(p, exist_ok=True)

def _json_dump(path: str, obj: Any) -> None:
    """Performs: json dump."""
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _np_save(path: str, arr: np.ndarray) -> None:
    """Performs: np save."""
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr)

def _vector_norm(v) -> float:
    """Performs: norm."""
    return float(math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))

def _normalize_vector(v):
    """Performs: normalize."""
    n = _vector_norm(v)
    if n <= 1e-12: return (0.0, 0.0, 0.0)
    return (float(v[0]/n), float(v[1]/n), float(v[2]/n))

def _dot_product(a, b) -> float:
    """Performs: dot."""
    return float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

def _angle_between_vectors_deg(a, b) -> float:
    """Performs: angle deg."""
    aa = _normalize_vector(a)
    bb = _normalize_vector(b)
    d = max(-1.0, min(1.0, _dot_product(aa, bb)))
    return float(math.degrees(math.acos(d)))

def _deduplicate_directions(dirs, angle_tol_deg: float = 1.0):
    """Performs: dedup dirs."""
    kept = []
    for d in dirs:
        dn = _normalize_vector(d)
        if dn == (0.0, 0.0, 0.0): continue
        if any(_angle_between_vectors_deg(dn, k) <= angle_tol_deg for k in kept): continue
        kept.append(dn)
    return kept

def _is_direction_used(d, used_dirs, angle_tol_deg: float) -> bool:
    """Performs: is dir used."""
    dn = _normalize_vector(d)
    return any(_angle_between_vectors_deg(dn, u) <= angle_tol_deg for u in used_dirs)

def get_predefined_axis_directions():
    """Performs: get predefined axis directions."""
    base = [( 1, 0, 0), (-1, 0, 0), ( 0, 1, 0), ( 0,-1, 0), ( 0, 0, 1)]
    out = []
    for d in base:
        dn = _normalize_vector(d)
        out.append(dn if dn[2] >= 0 else (-dn[0], -dn[1], -dn[2]))
    return _deduplicate_directions(out, angle_tol_deg=0.1)

def build_graph_distance_matrix(G: nxg.Graph) -> np.ndarray:
    """Builds an all-pairs shortest-path matrix for graph nodes."""
    nodes = list(G.nodes())
    n = len(nodes)
    dist = np.full((n, n), -1, dtype=np.int16)
    for i, src in enumerate(nodes):
        lengths = nxg.shortest_path_length(G, source=src)
        for dst, d in lengths.items():
            dist[i, nodes.index(dst)] = int(d)
    return dist

def pad_1d_int(arr: np.ndarray, n: int, pad_val: int = 0) -> np.ndarray:
    """Performs: pad 1d int."""
    out = np.full((n,), pad_val, dtype=np.int16)
    m = min(len(arr), n)
    out[:m] = arr[:m].astype(np.int16)
    return out

def pad_2d_int(mat: np.ndarray, n: int, pad_val: int = 0) -> np.ndarray:
    """Performs: pad 2d int."""
    out = np.full((n, n), pad_val, dtype=np.int16)
    h, w = mat.shape
    out[:min(h, n), :min(w, n)] = mat[:min(h, n), :min(w, n)].astype(np.int16)
    return out

def pad_face_area(face_area: np.ndarray, n: int) -> np.ndarray:
    """Pads raw face area values without normalization."""
    out = np.zeros((n, 1), dtype=np.float32)
    m = min(len(face_area), n)
    out[:m, 0] = face_area[:m].astype(np.float32)
    return out

def pad_face_pc(points_group: np.ndarray, n: int = 512) -> np.ndarray:
    """Pads raw grouped point clouds without normalization."""
    K = points_group.shape[0]
    out = np.zeros((n, points_group.shape[1], 3), dtype=np.float32)
    m = min(K, n)
    out[:m] = points_group[:m].astype(np.float32)
    return out

def pad_visibility(vis: np.ndarray, n: int = 512) -> np.ndarray:
    """Performs: pad visibility."""
    out = np.zeros((n,), dtype=np.int16)
    m = min(len(vis), n)
    out[:m] = vis[:m].astype(np.int16)
    return out

def pad_1d_float(arr: np.ndarray, n: int, pad_val: float = 0.0) -> np.ndarray:
    """Performs: pad 1d float."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    out = np.full((n,), pad_val, dtype=np.float32)
    m = min(len(arr), n)
    out[:m] = arr[:m]
    return out

def reduce_scalar_by_groups(values: np.ndarray, groups, weights: np.ndarray) -> np.ndarray:
    """Performs: reduce scalar by groups."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    out = np.zeros((len(groups),), dtype=np.float32)
    for k, idxs in enumerate(groups):
        if not idxs:
            out[k] = 0.0
            continue
        v = values[idxs]
        w = weights[idxs]
        s = float(w.sum())
        out[k] = float((v*w).sum()/s) if s > 1e-12 else float(v.mean())
    return out

def compute_done_mask_from_dev_red(
    dev_red_512: np.ndarray,
    tol: float,
    node_mask_512: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    """Computes done-mask from residual thickness (optionally excluding padded nodes)."""
    dev = np.asarray(dev_red_512, dtype=np.float32).reshape(-1)
    valid = np.isfinite(dev)
    if node_mask_512 is not None:
        valid = valid & (np.asarray(node_mask_512, dtype=np.int16).reshape(-1) == 0)
    done = np.zeros_like(dev, dtype=np.int16)
    if valid.any():
        done[valid] = (dev[valid] <= float(tol)).astype(np.int16)
        ratio = float(done[valid].mean())
    else:
        ratio = 0.0
    return done, ratio

def compute_done_mask_from_dev_red_with_K(dev_red_512: np.ndarray, tol: float, K: int) -> Tuple[np.ndarray, bool]:
    """Performs: compute done mask from dev red with k."""
    dev = np.asarray(dev_red_512, dtype=np.float32).reshape(-1)
    done = np.zeros_like(dev, dtype=np.int16)
    K = int(max(0, min(K, dev.shape[0])))
    if K > 0:
        v = dev[:K]
        v = np.where(np.isfinite(v), v, 0.0)
        done[:K] = (v <= float(tol)).astype(np.int16)
        all_done = bool(np.all(done[:K] == 1))
    else:
        all_done = True
    return done, all_done

def visible_all_done(done_mask_512: np.ndarray, visible_512: np.ndarray, K: int) -> bool:
    """Performs: visible all done."""
    K = int(max(0, min(K, len(done_mask_512), len(visible_512))))
    if K == 0: return True
    vis = (np.asarray(visible_512[:K], dtype=np.int16) == 1)
    if not vis.any(): return True
    done = (np.asarray(done_mask_512[:K], dtype=np.int16) == 1)
    return bool(np.all(done[vis]))

def select_single_axis_direction(
    origin_faces,
    groups_face,
    face_areas: np.ndarray,
    state_dev_red_512: np.ndarray,
    theUfSession,
    used_axis_dirs: List[Tuple[float, float, float]],
    used_tol_deg: float = 3.0,
) -> Dict[str, Any] | None:
    """Selects one executable axis directly from remaining geometry (no candidate rollout ranking)."""
    dev = np.asarray(state_dev_red_512, dtype=np.float32).reshape(-1)
    best_axis: Dict[str, Any] | None = None

    for gi, idxs in enumerate(groups_face):
        if not idxs or gi >= len(dev):
            continue

        remaining = float(max(dev[gi], 0.0))
        if remaining <= 0.0:
            continue

        best_face_idx = None
        best_area = -1.0
        group_area_sum = 0.0
        for fi in idxs:
            area = float(face_areas[fi]) if fi < len(face_areas) else 0.0
            group_area_sum += max(area, 0.0)
            if area > best_area:
                best_area = area
                best_face_idx = fi

        if best_face_idx is None or best_area <= 0.0:
            continue

        _, _, direction_data, _, _, _, _ = theUfSession.Modeling.AskFaceData(origin_faces[best_face_idx].Tag)
        direction = _normalize_vector((float(direction_data[0]), float(direction_data[1]), float(direction_data[2])))

        if direction == (0.0, 0.0, 0.0):
            continue

        if direction[2] < 0:
            direction = (-direction[0], -direction[1], -direction[2])

        if _is_direction_used(direction, used_axis_dirs, used_tol_deg):
            continue

        score = float(group_area_sum * remaining)
        if best_axis is None or score > float(best_axis["score"]):
            best_axis = {"dir": direction, "source": "group_dir", "score": score}

    if best_axis is not None:
        return best_axis

    for direction in get_predefined_axis_directions():
        if not _is_direction_used(direction, used_axis_dirs, used_tol_deg):
            return {"dir": direction, "source": "predefined", "score": 0.0}

    return None

def build_default_operation_sequence():
    """Returns the default machining operation sequence used in rollouts."""
    return [
        {"optype": "3D Adaptive Roughing", "tool_kind": "flat", "tool_diameter": 12.0, "path_type": "FollowPart"},
        {"optype": "Cavity Mill",         "tool_kind": "flat", "tool_diameter": 4.0,  "path_type": "FollowPart"},
        {"optype": "Area Mill",           "tool_kind": "ball", "tool_diameter": 4.0,  "path_type": "FollowPart"},
    ]


def build_swarf_test_operation_sequence():
    """Returns a short sequence that includes one swarf finishing operation."""
    return [
        {"optype": "3D Adaptive Roughing", "tool_kind": "flat", "tool_diameter": 12.0, "path_type": "FollowPart"},
        {"optype": "Swarf Mill",           "tool_kind": "flat", "tool_diameter": 8.0,  "path_type": "FollowPart"},
    ]


def build_point_test_operation_sequence():
    """Returns a short sequence that includes one point milling finish operation."""
    return [
        {"optype": "3D Adaptive Roughing", "tool_kind": "flat", "tool_diameter": 12.0, "path_type": "FollowPart"},
        {"optype": "Point Mill",           "tool_kind": "ball", "tool_diameter": 8.0,  "path_type": "FollowPart"},
    ]


SDF_FEATURE_SCALE = 10.0
ROUGH_DONE_DELTA_EPS = 1e-5
FINISH_READY_TOL = float(os.getenv("FINISH_READY_TOL", "1.0"))


def build_node_mask_512(num_valid_nodes: int, max_nodes: int = 512) -> np.ndarray:
    """Builds a node padding mask where 1 indicates padded nodes."""
    mask = np.ones((max_nodes,), dtype=np.int16)
    mask[: min(num_valid_nodes, max_nodes)] = 0
    return mask


def build_point_mask_512x100(
    face_pc_raw_512x100x3: np.ndarray,
    num_valid_nodes: int,
    max_nodes: int = 512,
) -> np.ndarray:
    """Builds a point padding mask aligned with fixed 512x100 node-point slots."""
    points = np.asarray(face_pc_raw_512x100x3, dtype=np.float32)
    points_per_node = int(points.shape[1])
    mask = np.ones((max_nodes, points_per_node), dtype=np.int16)
    valid_nodes = min(num_valid_nodes, max_nodes)
    if valid_nodes <= 0:
        return mask
    point_valid = np.any(np.abs(points[:valid_nodes]) > 1e-12, axis=-1).astype(np.int16)
    mask[:valid_nodes, :] = (point_valid == 0).astype(np.int16)
    return mask


def build_state_points_tensor(
    face_pc_512x100x3: np.ndarray,
    face_normal_512x3: np.ndarray,
    point_sdf_512x100: np.ndarray,
    reference_scale: float,
) -> np.ndarray:
    """Builds state tensor [512, 100, 7] = normalized xyz + normal + normalized local sdf."""
    state_points = np.zeros((512, 100, 7), dtype=np.float32)
    state_points[:, :, 0:3] = np.asarray(face_pc_512x100x3, dtype=np.float32)
    state_points[:, :, 3:6] = np.broadcast_to(
        np.asarray(face_normal_512x3, dtype=np.float32)[:, None, :],
        (512, 100, 3),
    )
    scale = float(max(reference_scale, 1e-6))
    sdf = np.asarray(point_sdf_512x100, dtype=np.float32).reshape(512, 100, 1) / scale
    state_points[:, :, 6:7] = sdf
    return state_points


def build_normalized_node_sdf_512x1(node_sdf_512: np.ndarray, reference_scale: float) -> np.ndarray:
    """Converts raw residual thickness to normalized [512, 1] node supervision."""
    scale = float(max(reference_scale, 1e-6))
    return (np.asarray(node_sdf_512, dtype=np.float32).reshape(512, 1) / scale).astype(np.float32)


def build_normalized_point_sdf_512x100(point_sdf_512x100: np.ndarray, reference_scale: float) -> np.ndarray:
    """Converts raw local residual thickness to normalized [512, 100] supervision."""
    scale = float(max(reference_scale, 1e-6))
    return (np.asarray(point_sdf_512x100, dtype=np.float32).reshape(512, 100) / scale).astype(np.float32)


def compute_reference_frame(
    face_pc_512x100x3: np.ndarray,
    node_mask_512: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Computes one part-level normalization frame shared across the full rollout."""
    face_pc = np.asarray(face_pc_512x100x3, dtype=np.float32)
    valid_nodes = np.asarray(node_mask_512, dtype=np.int16).reshape(-1) == 0
    valid_points = face_pc[valid_nodes].reshape(-1, 3)
    if valid_points.size == 0:
        return np.zeros((3,), dtype=np.float32), 1.0, np.ones((3,), dtype=np.float32)

    bbox_min = valid_points.min(axis=0)
    bbox_max = valid_points.max(axis=0)
    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    extent = np.maximum(bbox_max - bbox_min, 1e-6).astype(np.float32)
    scale = float(max(np.linalg.norm(extent), 1e-6))
    return center, scale, extent


def normalize_face_points(
    face_pc_512x100x3: np.ndarray,
    center_xyz: np.ndarray,
    reference_scale: float,
) -> np.ndarray:
    """Normalizes grouped point clouds using one fixed part-level frame."""
    scale = float(max(reference_scale, 1e-6))
    return ((np.asarray(face_pc_512x100x3, dtype=np.float32) - center_xyz.reshape(1, 1, 3)) / scale).astype(np.float32)


def normalize_face_area(face_area_512x1: np.ndarray, reference_scale: float) -> np.ndarray:
    """Normalizes face area by the squared part reference scale."""
    scale_sq = float(max(reference_scale, 1e-6)) ** 2
    return (np.asarray(face_area_512x1, dtype=np.float32) / scale_sq).astype(np.float32)


def extract_point_coordinates(points_array) -> List[np.ndarray]:
    """Extracts raw xyz coordinates from NX point objects for later interpolation."""
    coords_per_face: List[np.ndarray] = []
    for face_points in points_array:
        coords = []
        for point in face_points:
            coords.append(
                [
                    float(point.Coordinates.X),
                    float(point.Coordinates.Y),
                    float(point.Coordinates.Z),
                ]
            )
        coords_per_face.append(np.asarray(coords, dtype=np.float32))
    return coords_per_face


def build_group_face_normals_512(
    groups_face: List[List[int]],
    origin_face_normals: List[np.ndarray],
    max_nodes: int = 512,
) -> np.ndarray:
    """Aggregates original per-face normals into compressed graph-node normals."""
    out = np.zeros((max_nodes, 3), dtype=np.float32)
    for group_idx, face_indices in enumerate(groups_face[:max_nodes]):
        vectors = []
        for face_idx in face_indices:
            if 0 <= int(face_idx) < len(origin_face_normals):
                vectors.append(np.asarray(origin_face_normals[int(face_idx)], dtype=np.float32))
        if not vectors:
            out[group_idx] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            continue
        mean_normal = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(mean_normal))
        out[group_idx] = (mean_normal / norm) if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return out


def build_group_point_sdf_512x100(
    group_points_512x100x3: np.ndarray,
    groups_face: List[List[int]],
    measurement_point_coords_per_face: List[np.ndarray],
    measurement_point_sdf_per_face: List[np.ndarray],
    fallback_node_sdf_512: np.ndarray,
) -> np.ndarray:
    """Interpolates face-level measurement points onto the fixed 512x100 grouped point cloud."""
    grouped_point_sdf = np.zeros((512, 100), dtype=np.float32)
    fallback_node_sdf = np.asarray(fallback_node_sdf_512, dtype=np.float32).reshape(-1)

    for group_idx, face_indices in enumerate(groups_face[:512]):
        query_points = np.asarray(group_points_512x100x3[group_idx], dtype=np.float32)
        ref_points = []
        ref_values = []
        for face_idx in face_indices:
            if 0 <= int(face_idx) < len(measurement_point_coords_per_face):
                coords = np.asarray(measurement_point_coords_per_face[int(face_idx)], dtype=np.float32)
                values = np.asarray(measurement_point_sdf_per_face[int(face_idx)], dtype=np.float32).reshape(-1)
                count = min(len(coords), len(values))
                if count > 0:
                    ref_points.append(coords[:count])
                    ref_values.append(values[:count])

        if ref_points:
            ref_points_concat = np.concatenate(ref_points, axis=0)
            ref_values_concat = np.concatenate(ref_values, axis=0)
            diff = query_points[:, None, :] - ref_points_concat[None, :, :]
            nearest = np.argmin(np.sum(diff * diff, axis=-1), axis=1)
            grouped_point_sdf[group_idx] = ref_values_concat[nearest]
        else:
            fallback = float(fallback_node_sdf[group_idx]) if group_idx < len(fallback_node_sdf) else 0.0
            grouped_point_sdf[group_idx] = fallback

    return grouped_point_sdf


def build_finish_ready_mask_512(
    node_sdf_512: np.ndarray,
    rough_done_mask_512: np.ndarray,
    node_mask_512: np.ndarray,
    done_tol: float,
    ready_tol: float,
) -> np.ndarray:
    """Builds a local-finishing readiness mask from rough history and current residual."""
    residual = np.asarray(node_sdf_512, dtype=np.float32).reshape(-1)
    rough_done = np.asarray(rough_done_mask_512, dtype=np.int16).reshape(-1)
    node_mask = np.asarray(node_mask_512, dtype=np.int16).reshape(-1)
    ready = (
        (rough_done == 1)
        & (node_mask == 0)
        & np.isfinite(residual)
        & (residual > float(done_tol))
        & (residual <= float(ready_tol))
    )
    return ready.astype(np.int16)


def is_local_operation(optype: str) -> bool:
    """Returns whether the operation targets a local finishing region."""
    return optype in {"Cavity Mill", "Area Mill", "Swarf Mill", "Point Mill"}


def is_three_axis_tool_orientation(axis_dir: Tuple[float, float, float], tol_deg: float = 5.0) -> bool:
    """Checks whether the tool axis is effectively aligned with machine +Z."""
    return _angle_between_vectors_deg(axis_dir, (0.0, 0.0, 1.0)) <= float(tol_deg)


def infer_macro_class_name(optype: str, axis_dir: Tuple[float, float, float]) -> str:
    """Maps one executed NX operation to the planner macro class vocabulary."""
    if optype == "3D Adaptive Roughing":
        return "indexed_rough"
    if optype in {"Cavity Mill", "Area Mill"}:
        return "indexed_finish"
    if optype == "Swarf Mill":
        return "flank_finish"
    if optype == "Point Mill":
        return "point_finish"
    raise ValueError(f"Unsupported optype for macro class mapping: {optype}")


def resolve_tool_name(
    tool_kind: str,
    tool_diameter: float,
    flat_tools: List[str],
    ball_tools: List[str],
) -> str:
    """Resolves a tool name from the pre-created tool lists using kind and diameter."""
    dia = float(tool_diameter)
    flat_map = {20.0: 0, 16.0: 1, 12.0: 2, 10.0: 3, 8.0: 4, 6.0: 5, 4.0: 6}
    ball_map = {8.0: 0, 6.0: 1, 4.0: 2}
    key = str(tool_kind).lower()

    if key == "ball":
        idx = ball_map.get(dia, len(ball_tools) - 1)
        return ball_tools[int(max(0, min(idx, len(ball_tools) - 1)))]

    idx = flat_map.get(dia, len(flat_tools) - 1)
    return flat_tools[int(max(0, min(idx, len(flat_tools) - 1)))]


def infer_target_node_id(
    state_node_sdf_512: np.ndarray,
    next_node_sdf_512: np.ndarray,
    valid_mask_512: np.ndarray | None = None,
) -> int:
    """Selects the primary target node as the largest positive residual drop."""
    before = np.asarray(state_node_sdf_512, dtype=np.float32).reshape(-1)
    after = np.asarray(next_node_sdf_512, dtype=np.float32).reshape(-1)
    delta = np.maximum(before - after, 0.0)
    if valid_mask_512 is not None:
        delta = delta * np.asarray(valid_mask_512, dtype=np.float32).reshape(-1)
    best_idx = int(np.argmax(delta))
    if float(delta[best_idx]) <= 1e-6:
        return -1
    return best_idx


def infer_axis_face_id_from_normal(
    face_normal_512x3: np.ndarray,
    axis_dir: Tuple[float, float, float],
    valid_mask_512: np.ndarray,
) -> int:
    """Selects a face whose normal best aligns with the selected tool-axis direction."""
    normals = np.asarray(face_normal_512x3, dtype=np.float32)
    valid = np.asarray(valid_mask_512, dtype=np.int16).reshape(-1)
    axis = np.asarray(axis_dir, dtype=np.float32).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        return -1
    axis = axis / axis_norm
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_norm = np.where(normal_norm > 1e-9, normal_norm, 1.0)
    unit_normals = normals / normal_norm
    scores = np.dot(unit_normals, axis)
    scores[valid == 0] = -1e9
    best_idx = int(np.argmax(scores))
    if float(scores[best_idx]) <= -1e8:
        return -1
    return best_idx


def build_macro_class_mask_5(
    state_done_mask_512: np.ndarray,
    rough_done_mask_512: np.ndarray,
    finish_ready_mask_512: np.ndarray,
    node_mask_512: np.ndarray,
) -> np.ndarray:
    """Builds an invalid-mask for macro classes (1=invalid, 0=valid)."""
    active = np.asarray(node_mask_512, dtype=np.int16) == 0
    state_not_done = (np.asarray(state_done_mask_512, dtype=np.int16) == 0)
    rough_done = (np.asarray(rough_done_mask_512, dtype=np.int16) == 1)
    finish_ready = (np.asarray(finish_ready_mask_512, dtype=np.int16) == 1)

    rough_possible = bool(np.any(active & state_not_done & (~rough_done)))
    finish_possible = bool(np.any(active & state_not_done & rough_done & finish_ready))
    all_done = bool(np.all(~state_not_done | (np.asarray(node_mask_512, dtype=np.int16) == 1)))

    mask = np.ones((len(MACRO_CLASS_TO_ID),), dtype=np.int16)
    if rough_possible:
        mask[MACRO_CLASS_TO_ID["indexed_rough"]] = 0
    if finish_possible:
        mask[MACRO_CLASS_TO_ID["indexed_finish"]] = 0
        mask[MACRO_CLASS_TO_ID["point_finish"]] = 0
        mask[MACRO_CLASS_TO_ID["flank_finish"]] = 0
    if all_done or (not rough_possible and not finish_possible):
        mask[MACRO_CLASS_TO_ID["stop"]] = 0

    return mask


def build_global_process_state(
    prev_macro_class_id: int,
    rough_rows_emitted: int,
    finish_rows_emitted: int,
    bbox_extent_xyz: np.ndarray,
    reference_scale: float,
) -> np.ndarray:
    """Builds a compact global context vector with process history and part scale."""
    out = np.zeros((11,), dtype=np.float32)
    if 0 <= int(prev_macro_class_id) < len(MACRO_CLASS_TO_ID):
        out[int(prev_macro_class_id)] = 1.0
    total = max(rough_rows_emitted + finish_rows_emitted, 1)
    out[5] = float(rough_rows_emitted / total)
    out[6] = float(finish_rows_emitted / total)
    ref_scale = float(max(reference_scale, 1e-6))
    out[7:10] = np.asarray(bbox_extent_xyz, dtype=np.float32).reshape(3) / ref_scale
    out[10] = float(np.log1p(ref_scale))
    return out


# ──────────────────────────────────────────────────────────────────────
# Action-first candidate generation: (macro_class, target_node, tool)
# → axis is derived deterministically from the action, not chosen first.
# ──────────────────────────────────────────────────────────────────────

MACRO_CLASS_TO_OPTYPE = {
    "indexed_rough": "3D Adaptive Roughing",
    "indexed_finish": "Area Mill",
    "point_finish": "Point Mill",
    "flank_finish": "Swarf Mill",
}


def derive_axis_direction(
    macro_class_name: str,
    target_face_normal: np.ndarray,
) -> Tuple[float, float, float]:
    """Derives tool axis from macro class and target face geometry.

    Indexed / 5-axis finishing → selected face normal (upper hemisphere)
    """
    n = _normalize_vector(tuple(float(x) for x in target_face_normal))
    if n == (0.0, 0.0, 0.0):
        return (0.0, 0.0, 1.0)
    if n[2] < 0:
        n = (-n[0], -n[1], -n[2])
    return n


def sample_valid_tools_for_macro(
    macro_class_name: str,
    max_tools: int = 3,
) -> List[Tuple[str, float]]:
    """Returns process-aware tool candidates for a macro class."""
    macro_id = MACRO_CLASS_TO_ID.get(macro_class_name, -1)
    if macro_id < 0:
        return []
    mask = build_tool_choice_mask_for_macro_class(macro_id)
    valid = [TOOL_LIBRARY[i] for i, m in enumerate(mask) if m == 0]
    if not valid:
        return []

    def _sort_tools(tools: List[Tuple[str, float]], descending: bool) -> List[Tuple[str, float]]:
        return sorted(tools, key=lambda item: float(item[1]), reverse=descending)

    def _pick_spread(ordered_tools: List[Tuple[str, float]], count: int) -> List[Tuple[str, float]]:
        count = max(1, int(count))
        if len(ordered_tools) <= count:
            return ordered_tools
        if count == 1:
            return [ordered_tools[0]]

        sampled: List[Tuple[str, float]] = []
        used_idx: set[int] = set()
        for raw_idx in np.linspace(0, len(ordered_tools) - 1, num=count):
            idx = int(round(float(raw_idx)))
            idx = max(0, min(idx, len(ordered_tools) - 1))
            while idx in used_idx and idx + 1 < len(ordered_tools):
                idx += 1
            while idx in used_idx and idx - 1 >= 0:
                idx -= 1
            if idx in used_idx:
                continue
            used_idx.add(idx)
            sampled.append(ordered_tools[idx])
        if len(sampled) < count:
            for idx, tool in enumerate(ordered_tools):
                if idx in used_idx:
                    continue
                sampled.append(tool)
                if len(sampled) >= count:
                    break
        return sampled

    if macro_class_name == "indexed_rough":
        # Roughing should bias toward large cutters, but still expose size diversity.
        ordered = _sort_tools(valid, descending=True)
    elif macro_class_name in {"indexed_finish", "point_finish"}:
        # Finishing should start from smaller tools, while keeping a spread.
        ordered = _sort_tools(valid, descending=False)
    elif macro_class_name == "flank_finish":
        # Flank milling usually prefers larger flat tools.
        ordered = _sort_tools(valid, descending=True)
    else:
        ordered = list(valid)

    return _pick_spread(ordered, max_tools)


def create_geometry_for_state(
    session,
    work_part,
    prt_file_path: str,
    workpiece_name_chain: List[str] | None,
    origin_body,
    commit_to_chain: bool,
):
    """Creates a workpiece geometry tied to an IPW source chain.

    When `commit_to_chain=True`, the provided chain is advanced by appending a
    new workpiece source. Callers may pass either the persistent global chain
    or a local copy used only for temporary candidate rollout.
    """
    if workpiece_name_chain is None:
        temp_chain: List[str] = []
        return geometry.create_geometry(
            session, work_part, prt_file_path, temp_chain, origin_body, True, False,
        )

    if len(workpiece_name_chain) == 0:
        return geometry.create_geometry(
            session, work_part, prt_file_path, workpiece_name_chain, origin_body, True, False,
        )

    return geometry.create_geometry(
        session, work_part, prt_file_path, workpiece_name_chain, origin_body, commit_to_chain, False,
    )


def restore_workpiece_chain(workpiece_name_chain: List[str], snapshot: List[str]) -> None:
    """Restores the Python-side IPW source chain after an NX undo rollback."""
    workpiece_name_chain[:] = list(snapshot)


def create_operation_geometry_for_depth(
    session,
    work_part,
    prt_file_path: str,
    workpiece_name_chain: List[str],
    origin_body,
    operation_depth: int,
    base_object_blank,
):
    """Returns the CAM geometry for the next operation in a scenario.

    The first operation must be created on the original WORKPIECE geometry,
    matching the legacy data generator. Later operations create workpiece_1,
    workpiece_2, ... sourced from the previous IPW.
    """
    if int(operation_depth) <= 0:
        if base_object_blank is None:
            obj_blank, _, _ = geometry.create_geometry(
                session, work_part, prt_file_path, workpiece_name_chain, origin_body, True, False,
            )
            return obj_blank
        return base_object_blank
    obj_blank, _, _ = geometry.create_geometry(
        session, work_part, prt_file_path, workpiece_name_chain, origin_body, True, False,
    )
    return obj_blank


def create_measure_geometry_for_depth(
    session,
    work_part,
    prt_file_path: str,
    workpiece_name_chain: List[str],
    origin_body,
    state_depth: int,
    base_object_blank,
):
    """Returns a CAM geometry for measuring the current scenario state."""
    if int(state_depth) <= 0:
        if base_object_blank is None:
            obj_blank, _, _ = geometry.create_geometry(
                session, work_part, prt_file_path, workpiece_name_chain, origin_body, True, False,
            )
            return obj_blank
        return base_object_blank
    # Special case after the first operation: create_geometry needs add_list=True
    # so the temporary geometry uses the WORKPIECE IPW instead of AutoBlock.
    add_list = len(workpiece_name_chain) == 1
    obj_blank, _, _ = geometry.create_geometry(
        session, work_part, prt_file_path, workpiece_name_chain, origin_body, add_list, False,
    )
    return obj_blank


def select_action_face_candidates(
    state_node_sdf_512: np.ndarray,
    rough_done_mask_512: np.ndarray,
    finish_ready_mask_512: np.ndarray,
    node_mask_512: np.ndarray,
    face_normal_512x3: np.ndarray,
    max_rough_targets: int = 5,
    max_finish_targets: int = 5,
    normal_dedup_tol_deg: float = 10.0,
) -> Dict[str, List[int]]:
    """Selects candidate raw faces for indexed roughing and finishing.

    Indexed roughing: residual-heavy faces that are not rough-done yet.
    Finishing: finish-ready faces with remaining material.
    """
    valid = np.asarray(node_mask_512, dtype=np.int16).reshape(-1) == 0
    sdf = np.asarray(state_node_sdf_512, dtype=np.float32).reshape(-1)
    normals = np.asarray(face_normal_512x3, dtype=np.float32)

    def _pick_diverse(eligible_mask, max_count):
        idx = np.where(eligible_mask)[0]
        if len(idx) == 0:
            return []
        order = np.argsort(-sdf[idx])
        selected, used_n = [], []
        for i in order:
            nid = int(idx[i])
            n = _normalize_vector(tuple(normals[nid]))
            if n[2] < 0:
                n = (-n[0], -n[1], -n[2])
            if any(_angle_between_vectors_deg(n, u) <= normal_dedup_tol_deg for u in used_n):
                continue
            selected.append(nid)
            used_n.append(n)
            if len(selected) >= max_count:
                break
        return selected

    rough_elig = valid & (np.asarray(rough_done_mask_512, dtype=np.int16).reshape(-1) == 0) & (sdf > 0)
    finish_elig = valid & (np.asarray(finish_ready_mask_512, dtype=np.int16).reshape(-1) == 1)
    return {
        "indexed_rough": _pick_diverse(rough_elig, max_rough_targets),
        "finish": _pick_diverse(finish_elig, max_finish_targets),
    }


def generate_action_candidates(
    state_node_sdf_512: np.ndarray,
    rough_done_mask_512: np.ndarray,
    finish_ready_mask_512: np.ndarray,
    node_mask_512: np.ndarray,
    face_normal_512x3: np.ndarray,
    max_rough_targets: int = 5,
    max_finish_targets: int = 5,
    max_tools_per_class: int = 7,
) -> List[Dict[str, Any]]:
    """Generates action candidates as (macro_class, action_face, tool).

    The selected raw face acts as:
    - anchor face for indexed roughing
    - target face for indexed/point/flank finishing
    """
    targets = select_action_face_candidates(
        state_node_sdf_512, rough_done_mask_512, finish_ready_mask_512,
        node_mask_512, face_normal_512x3,
        max_rough_targets=max_rough_targets,
        max_finish_targets=max_finish_targets,
    )
    normals = np.asarray(face_normal_512x3, dtype=np.float32)
    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(macro, face_id, tk, td, axis):
        key = (macro, face_id, tk, td)
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "macro_class_name": macro,
            "action_face_id": int(face_id),
            "anchor_face_id": int(face_id) if macro == "indexed_rough" else -1,
            "target_face_id": -1 if macro == "indexed_rough" else int(face_id),
            "tool_kind": tk,
            "tool_diameter": td,
            "optype": MACRO_CLASS_TO_OPTYPE[macro],
            "axis_dir": axis,
            "path_type": "FollowPart",
        })

    if targets["indexed_rough"]:
        for face_id in targets["indexed_rough"]:
            axis = derive_axis_direction("indexed_rough", normals[face_id])
            for tk, td in sample_valid_tools_for_macro("indexed_rough", max_tools_per_class):
                _add("indexed_rough", face_id, tk, td, axis)

    for face_id in targets["finish"]:
        axis = derive_axis_direction("indexed_finish", normals[face_id])
        for tk, td in sample_valid_tools_for_macro("indexed_finish", max_tools_per_class):
            _add("indexed_finish", face_id, tk, td, axis)

        for tk, td in sample_valid_tools_for_macro("point_finish", max_tools_per_class):
            _add("point_finish", face_id, tk, td, axis)

        for tk, td in sample_valid_tools_for_macro("flank_finish", max_tools_per_class):
            _add("flank_finish", face_id, tk, td, axis)

    return candidates


def simulate_single_action(
    session, work_part, prt_file_path,
    origin_body, origin_faces, groups_face, face_areas,
    points_array, norm_vecs_array, lines_array,
    flat_tools, ball_tools,
    action: Dict[str, Any],
    surface_finish_tol: float = 0.01,
    group_reference_points_512x100x3: np.ndarray | None = None,
    measurement_point_coords: List[np.ndarray] | None = None,
    workpiece_name_chain: List[str] | None = None,
    commit_to_chain: bool = False,
    operation_depth: int = 0,
    base_object_blank=None,
) -> Dict[str, Any]:
    """Simulates one action and returns transition data.

    The NX operation is applied permanently — the caller must manage
    undo marks if rollback is needed.  Internal measurement geometry
    is cleaned up via its own undo marks.
    """
    optype = action["optype"]
    tool_kind = action["tool_kind"]
    tool_diameter = float(action["tool_diameter"])
    axis_dir = tuple(action["axis_dir"])
    action_face_id = int(action["action_face_id"])

    # Visibility for this axis
    visible_set: set = set()
    visible_512 = np.zeros((512,), dtype=np.int16)
    view_vec = NXOpen.Vector3d(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]))
    vis_tags = cam_utils.identify_visible_faces(origin_body, view_vec)
    visible_set = set(vis_tags)
    vis_orig = np.array([1 if f.Tag in visible_set else 0 for f in origin_faces], dtype=np.int16)
    vis_red = reduce_visibility_by_groups(vis_orig, groups_face)
    visible_512 = pad_visibility(vis_red, 512)

    capture_pw = (group_reference_points_512x100x3 is not None
                  and measurement_point_coords is not None)
    op_list: list = []
    ct_list: list = []
    state_chain = workpiece_name_chain
    if state_chain is None:
        state_chain = []

    # ── Measure BEFORE ──
    m_bef = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "sim_meas_bef")
    m_bef_chain_snapshot = list(state_chain)
    pw_before = None
    try:
        obj_b = create_measure_geometry_for_depth(
            session, work_part, prt_file_path, state_chain, origin_body,
            int(operation_depth), base_object_blank,
        )
        if capture_pw:
            dev_before, pw_raw_bef, vol_before = cam_utils.measure_ipw_state_detailed(
                session, work_part, obj_b, flat_tools[0],
                points_array, norm_vecs_array, lines_array,
            )
            dev_bef_red = reduce_scalar_by_groups(np.asarray(dev_before, dtype=np.float32), groups_face, weights=face_areas)
            pw_before = build_group_point_sdf_512x100(
                group_reference_points_512x100x3, groups_face,
                measurement_point_coords, pw_raw_bef,
                pad_1d_float(dev_bef_red, 512),
            )
        else:
            dev_before, vol_before = cam_utils.measure_ipw_state(
                session, work_part, obj_b, flat_tools[0],
                points_array, norm_vecs_array, lines_array,
            )
            dev_bef_red = reduce_scalar_by_groups(np.asarray(dev_before, dtype=np.float32), groups_face, weights=face_areas)
    finally:
        session.UndoToMark(m_bef, "sim_meas_bef")
        session.DeleteUndoMark(m_bef, "sim_meas_bef")
        restore_workpiece_chain(state_chain, m_bef_chain_snapshot)

    dev_bef_red_512 = pad_1d_float(dev_bef_red, 512)
    result: Dict[str, Any] = {
        "ok": True, "error": None,
        "optype": optype, "tool_kind": tool_kind, "tool_diameter": tool_diameter,
        "axis_dir": axis_dir, "path_type": action.get("path_type", "FollowPart"),
        "volume_before": float(vol_before),
        "dev_before_red_512": dev_bef_red_512,
        "visible_512": visible_512,
    }
    if pw_before is not None:
        result["point_sdf_before_512x100"] = pw_before.astype(np.float32)

    # ── Apply NX operation ──
    def _select_local_target_faces():
        if not (0 <= action_face_id < len(origin_faces)):
            return []
        if float(dev_before[action_face_id]) <= surface_finish_tol:
            return []
        face_obj = origin_faces[action_face_id]
        if face_obj.Tag not in visible_set:
            return []
        return [face_obj]

    obj_op = create_operation_geometry_for_depth(
        session, work_part, prt_file_path, state_chain, origin_body,
        int(operation_depth), base_object_blank,
    )
    tool = resolve_tool_name(tool_kind, tool_diameter, flat_tools, ball_tools)

    if optype == "3D Adaptive Roughing":
        operations.create_3d_adaptive_roughing(
            session, work_part, tool, obj_op, op_list, ct_list,
            tool_orientation=axis_dir,
        )
    elif optype in {"Area Mill", "Cavity Mill"}:
        sel = _select_local_target_faces()
        if not sel:
            ct_list.append(0.0)
        else:
            operations.create_surface_contour(
                session, work_part, tool, obj_op, sel, op_list, ct_list,
                tool_orientation=axis_dir,
            )
    elif optype == "Swarf Mill":
        sel = _select_local_target_faces()
        if not sel:
            ct_list.append(0.0)
        else:
            operations.create_swarf_milling(
                session, work_part, tool, obj_op, sel, op_list, ct_list,
            )
    elif optype == "Point Mill":
        sel = _select_local_target_faces()
        if not sel:
            ct_list.append(0.0)
        else:
            operations.create_point_milling(
                session, work_part, tool, obj_op, sel, op_list, ct_list,
            )
    else:
        raise ValueError(f"Unknown optype: {optype}")

    # ── Measure AFTER ──
    m_aft = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "sim_meas_aft")
    m_aft_chain_snapshot = list(state_chain)
    try:
        obj_a = create_measure_geometry_for_depth(
            session, work_part, prt_file_path, state_chain, origin_body,
            int(operation_depth) + 1, base_object_blank,
        )
        if capture_pw:
            dev_after, pw_raw_aft, vol_after = cam_utils.measure_ipw_state_detailed(
                session, work_part, obj_a, flat_tools[0],
                points_array, norm_vecs_array, lines_array,
            )
            dev_aft_red = reduce_scalar_by_groups(np.asarray(dev_after, dtype=np.float32), groups_face, weights=face_areas)
            result["point_sdf_after_512x100"] = build_group_point_sdf_512x100(
                group_reference_points_512x100x3, groups_face,
                measurement_point_coords, pw_raw_aft,
                pad_1d_float(dev_aft_red, 512),
            ).astype(np.float32)
        else:
            dev_after, vol_after = cam_utils.measure_ipw_state(
                session, work_part, obj_a, flat_tools[0],
                points_array, norm_vecs_array, lines_array,
            )
            dev_aft_red = reduce_scalar_by_groups(np.asarray(dev_after, dtype=np.float32), groups_face, weights=face_areas)
    finally:
        session.UndoToMark(m_aft, "sim_meas_aft")
        session.DeleteUndoMark(m_aft, "sim_meas_aft")
        restore_workpiece_chain(state_chain, m_aft_chain_snapshot)

    result["volume_after"] = float(vol_after)
    result["removed_volume"] = max(float(vol_before) - float(vol_after), 0.0)
    result["cycle_time"] = float(ct_list[-1]) if ct_list else 0.0
    result["dev_after_red_512"] = pad_1d_float(dev_aft_red, 512)
    return result


# ── Legacy axis-first functions (kept for reference) ──────────────────

def select_axis_candidates(
    origin_faces,
    groups_face,
    face_areas: np.ndarray,
    state_dev_red_512: np.ndarray,
    theUfSession,
    max_candidates: int = 4,
) -> List[Tuple[float, float, float]]:
    """Returns top axis candidates from residual-heavy groups plus +Z fallback."""
    dev = np.asarray(state_dev_red_512, dtype=np.float32).reshape(-1)
    scored_dirs: List[Tuple[float, Tuple[float, float, float]]] = []

    for gi, idxs in enumerate(groups_face):
        if not idxs or gi >= len(dev):
            continue
        remaining = float(max(dev[gi], 0.0))
        if remaining <= 0.0:
            continue
        best_face_idx = None
        best_area = -1.0
        group_area_sum = 0.0
        for fi in idxs:
            area = float(face_areas[fi]) if fi < len(face_areas) else 0.0
            group_area_sum += max(area, 0.0)
            if area > best_area:
                best_area = area
                best_face_idx = fi
        if best_face_idx is None:
            continue
        _, _, direction_data, _, _, _, _ = theUfSession.Modeling.AskFaceData(origin_faces[best_face_idx].Tag)
        direction = _normalize_vector(
            (float(direction_data[0]), float(direction_data[1]), float(direction_data[2]))
        )
        if direction == (0.0, 0.0, 0.0):
            continue
        if direction[2] < 0:
            direction = (-direction[0], -direction[1], -direction[2])
        score = float(group_area_sum * remaining)
        scored_dirs.append((score, direction))

    scored_dirs.sort(key=lambda x: x[0], reverse=True)
    ordered_dirs = [d for _, d in scored_dirs]
    ordered_dirs = _deduplicate_directions(ordered_dirs, angle_tol_deg=2.0)

    if (0.0, 0.0, 1.0) not in ordered_dirs:
        ordered_dirs.insert(0, (0.0, 0.0, 1.0))
    return ordered_dirs[: max(1, int(max_candidates))]


def compute_axis_visible_mask_512(
    origin_body,
    origin_faces,
    groups_face,
    axis_dir: Tuple[float, float, float],
) -> np.ndarray:
    """Computes reduced/padded visibility mask for a given axis direction."""
    view_vector = NXOpen.Vector3d(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]))
    visible_tags = cam_utils.identify_visible_faces(origin_body, view_vector)
    visible_set = set(visible_tags)
    visible_orig = np.array([1 if f.Tag in visible_set else 0 for f in origin_faces], dtype=np.int16)
    visible_reduced = reduce_visibility_by_groups(visible_orig, groups_face)
    return pad_visibility(visible_reduced, 512)


def generate_limited_action_candidates(
    origin_body,
    origin_faces,
    groups_face,
    face_areas: np.ndarray,
    state_dev_red_512: np.ndarray,
    state_done_mask_512: np.ndarray,
    rough_done_mask_512: np.ndarray,
    finish_ready_mask_512: np.ndarray,
    node_mask_512: np.ndarray,
    theUfSession,
    max_axis_candidates: int = 4,
) -> List[Dict[str, Any]]:
    """Generates bounded action candidates from the current state."""
    candidates: List[Dict[str, Any]] = []
    axis_candidates = select_axis_candidates(
        origin_faces=origin_faces,
        groups_face=groups_face,
        face_areas=face_areas,
        state_dev_red_512=state_dev_red_512,
        theUfSession=theUfSession,
        max_candidates=max_axis_candidates,
    )

    for axis_dir in axis_candidates:
        axis_visible_512 = compute_axis_visible_mask_512(origin_body, origin_faces, groups_face, axis_dir)
        active = (axis_visible_512 == 1) & (node_mask_512 == 0) & (state_done_mask_512 == 0)
        rough_possible = bool(np.any(active & (rough_done_mask_512 == 0)))
        finish_possible = bool(np.any(active & (rough_done_mask_512 == 1) & (finish_ready_mask_512 == 1)))

        if rough_possible:
            candidates.append({
                "optype": "3D Adaptive Roughing",
                "tool_kind": "flat",
                "tool_diameter": 12.0,
                "path_type": "FollowPart",
                "axis_dir": axis_dir,
            })
        if finish_possible:
            candidates.append({
                "optype": "Area Mill",
                "tool_kind": "ball",
                "tool_diameter": 4.0,
                "path_type": "FollowPart",
                "axis_dir": axis_dir,
            })
            candidates.append({
                "optype": "Point Mill",
                "tool_kind": "ball",
                "tool_diameter": 8.0,
                "path_type": "FollowPart",
                "axis_dir": axis_dir,
            })
            candidates.append({
                "optype": "Swarf Mill",
                "tool_kind": "flat",
                "tool_diameter": 8.0,
                "path_type": "FollowPart",
                "axis_dir": axis_dir,
            })

    uniq: List[Dict[str, Any]] = []
    seen = set()
    for c in candidates:
        key = (
            c["optype"],
            c["tool_kind"],
            float(c["tool_diameter"]),
            tuple(round(float(v), 3) for v in c["axis_dir"]),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def score_operation_step(step_rec: Dict[str, Any], finish_tol: float = 0.01) -> float:
    """Scores one executed step using removal gain and time penalty."""
    if not bool(step_rec.get("ok", True)):
        return -1e6
    removed = float(step_rec.get("removed_volume", 0.0) or 0.0)
    volume_before = float(step_rec.get("volume_before", 0.0) or 0.0)
    cycle_time = float(step_rec.get("cycle_time", 0.0) or 0.0)
    if volume_before <= 1e-9:
        return -1e6
    removed_ratio = removed / volume_before

    dev_before = np.asarray(step_rec.get("dev_before_red_512", np.zeros((512,), dtype=np.float32)), dtype=np.float32)
    dev_after = np.asarray(step_rec.get("dev_after_red_512", dev_before), dtype=np.float32)
    done_before, ratio_before = compute_done_mask_from_dev_red(dev_before, finish_tol)
    done_after, ratio_after = compute_done_mask_from_dev_red(dev_after, finish_tol)
    done_gain = float(max(ratio_after - ratio_before, 0.0))
    if removed <= 1e-8:
        return -100.0
    return float(4.0 * removed_ratio + 2.0 * done_gain - 0.002 * cycle_time)


def evaluate_action_with_lookahead(
    session,
    work_part,
    prt_file_path,
    origin_body,
    origin_faces,
    groups_face,
    face_areas,
    points_array,
    norm_vecs_array,
    lines_array,
    flat_tools,
    ball_tools,
    action: Dict[str, Any],
    rough_done_mask_512: np.ndarray,
    node_mask_512: np.ndarray,
    theUfSession,
    finish_tol: float = 0.01,
    lookahead_depth: int = 2,
    max_next_candidates: int = 4,
) -> float:
    """Evaluates one first action with bounded one-step lookahead using undo rollback."""
    eval_mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "eval_first_action")
    try:
        first_summary = simulate_rollout_for_axis(
            session=session,
            work_part=work_part,
            prt_file_path=prt_file_path,
            origin_body=origin_body,
            origin_faces=origin_faces,
            groups_face=groups_face,
            face_areas=face_areas,
            points_array=points_array,
            norm_vecs_array=norm_vecs_array,
            lines_array=lines_array,
            flat_tools=flat_tools,
            ball_tools=ball_tools,
            axis_dir=tuple(action["axis_dir"]),
            seq=[{
                "optype": action["optype"],
                "tool_kind": action["tool_kind"],
                "tool_diameter": float(action["tool_diameter"]),
                "path_type": action["path_type"],
            }],
            out_dir_for_this_rollout=".",
            surface_finish_tol=finish_tol,
        )
        steps = first_summary.get("steps", [])
        if not steps:
            return -1e6
        first_step = steps[0]
        first_score = score_operation_step(first_step, finish_tol=finish_tol)
        if lookahead_depth <= 1:
            return float(first_score)

        next_state_volume, next_state_dev = measure_current_state(
            session,
            work_part,
            prt_file_path,
            origin_body,
            origin_faces,
            groups_face,
            face_areas,
            points_array,
            norm_vecs_array,
            lines_array,
            flat_tools,
        )
        _ = next_state_volume
        next_done_mask, _ = compute_done_mask_from_dev_red(
            next_state_dev,
            finish_tol,
            node_mask_512=node_mask_512,
        )
        rough_next = np.asarray(rough_done_mask_512, dtype=np.int16).copy()
        if action["optype"] == "3D Adaptive Roughing":
            before = np.asarray(first_step.get("dev_before_red_512", next_state_dev), dtype=np.float32)
            after = np.asarray(first_step.get("dev_after_red_512", before), dtype=np.float32)
            rough_impacted = (
                (np.maximum(before - after, 0.0) > float(ROUGH_DONE_DELTA_EPS))
                & (np.asarray(node_mask_512, dtype=np.int16) == 0)
            ).astype(np.int16)
            rough_next = np.maximum(rough_next, rough_impacted)
        finish_next = build_finish_ready_mask_512(
            node_sdf_512=next_state_dev,
            rough_done_mask_512=rough_next,
            node_mask_512=node_mask_512,
            done_tol=finish_tol,
            ready_tol=FINISH_READY_TOL,
        )

        next_candidates = generate_limited_action_candidates(
            origin_body=origin_body,
            origin_faces=origin_faces,
            groups_face=groups_face,
            face_areas=face_areas,
            state_dev_red_512=next_state_dev,
            state_done_mask_512=next_done_mask,
            rough_done_mask_512=rough_next,
            finish_ready_mask_512=finish_next,
            node_mask_512=node_mask_512,
            theUfSession=theUfSession,
            max_axis_candidates=max_next_candidates,
        )
        if not next_candidates:
            return float(first_score)

        best_next = -1e6
        next_eval_candidates = next_candidates[: max_next_candidates]
        for nc in next_eval_candidates:
            next_mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "eval_next_action")
            try:
                next_summary = simulate_rollout_for_axis(
                    session=session,
                    work_part=work_part,
                    prt_file_path=prt_file_path,
                    origin_body=origin_body,
                    origin_faces=origin_faces,
                    groups_face=groups_face,
                    face_areas=face_areas,
                    points_array=points_array,
                    norm_vecs_array=norm_vecs_array,
                    lines_array=lines_array,
                    flat_tools=flat_tools,
                    ball_tools=ball_tools,
                    axis_dir=tuple(nc["axis_dir"]),
                    seq=[{
                        "optype": nc["optype"],
                        "tool_kind": nc["tool_kind"],
                        "tool_diameter": float(nc["tool_diameter"]),
                        "path_type": nc["path_type"],
                    }],
                    out_dir_for_this_rollout=".",
                    surface_finish_tol=finish_tol,
                )
                steps2 = next_summary.get("steps", [])
                if steps2:
                    best_next = max(best_next, score_operation_step(steps2[0], finish_tol=finish_tol))
            finally:
                session.UndoToMark(next_mark, "eval_next_action")
                session.DeleteUndoMark(next_mark, "eval_next_action")
        return float(first_score + 0.6 * best_next)
    finally:
        session.UndoToMark(eval_mark, "eval_first_action")
        session.DeleteUndoMark(eval_mark, "eval_first_action")

def force_close_part_by_path(target_path: str) -> None:
    """Performs: force close part by path."""
    session = NXOpen.Session.GetSession()
    for part in list(session.Parts):
        if part.FullPath and part.FullPath.lower() == target_path.lower():
            NXOpen.UF.UFSession.GetUFSession().Part.Close(part.Tag, 0, 1)
    gc.collect()

def measure_current_state(
    session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
    face_areas, points_array, norm_vecs_array, lines_array, flat_tools,
    workpiece_name_chain: List[str] | None = None,
    state_depth: int = 0,
    base_object_blank=None,
) -> Tuple[float, np.ndarray]:
    """Measures current IPW state volume and reduced deviation features."""
    if workpiece_name_chain is None:
        workpiece_name_chain = []
    m = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "measure_state")
    chain_snapshot = list(workpiece_name_chain)
    try:
        obj_blank = create_measure_geometry_for_depth(
            session, work_part, prt_file_path, workpiece_name_chain, origin_body,
            int(state_depth), base_object_blank,
        )
        dev, vol = cam_utils.measure_ipw_state(
            session, work_part, obj_blank, flat_tools[0],
            points_array, norm_vecs_array, lines_array
        )
        dev_red = reduce_scalar_by_groups(np.asarray(dev, dtype=np.float32), groups_face, weights=face_areas)
        dev_red_512 = pad_1d_float(dev_red, 512)
    finally:
        session.UndoToMark(m, "measure_state")
        session.DeleteUndoMark(m, "measure_state")
        restore_workpiece_chain(workpiece_name_chain, chain_snapshot)
    return float(vol), dev_red_512

def simulate_rollout_for_axis(
    session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
    face_areas, points_array, norm_vecs_array, lines_array, flat_tools, ball_tools,
    axis_dir, seq, out_dir_for_this_rollout, surface_finish_tol: float = 0.01,
    pre_visible_tags=None, pre_visible_512=None,
    capture_pointwise: bool = False,
    group_reference_points_512x100x3: np.ndarray | None = None,
    measurement_point_coords: List[np.ndarray] | None = None,
):
    """Simulates one selected axis from the current state and returns rollout metrics."""
    _ensure_dir(out_dir_for_this_rollout)
    if pre_visible_tags is not None:
        visible_set = set(pre_visible_tags)
        visible_512 = np.asarray(pre_visible_512, dtype=np.int16) if pre_visible_512 is not None else np.zeros((512,), dtype=np.int16)
    else:
        visible_set = set()
        visible_512 = np.zeros((512,), dtype=np.int16)
        view_vector = NXOpen.Vector3d(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]))
        visible_tags = cam_utils.identify_visible_faces(origin_body, view_vector)
        visible_set = set(visible_tags)
        visible_orig = np.array([1 if f.Tag in visible_set else 0 for f in origin_faces], dtype=int)
        visible_reduced = reduce_visibility_by_groups(visible_orig, groups_face)
        visible_512 = pad_visibility(visible_reduced, 512)

    operation_list, cycle_time_list, stl_file_list = [], [], []
    mark0 = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "rollout_start_measure")
    obj_blank0, _, _ = geometry.create_geometry(session, work_part, prt_file_path, [], origin_body, True, False)
    dev0, vol_start = cam_utils.measure_ipw_state(session, work_part, obj_blank0, flat_tools[0], points_array, norm_vecs_array, lines_array)
    dev0_red = reduce_scalar_by_groups(np.asarray(dev0, dtype=np.float32), groups_face, weights=face_areas)
    dev_before_red_512 = pad_1d_float(dev0_red, 512)
    session.UndoToMark(mark0, "rollout_start_measure")
    session.DeleteUndoMark(mark0, "rollout_start_measure")

    vol_cur = float(vol_start)
    steps = []
    last_dev_after_red_512 = dev_before_red_512.copy()
    K_groups = len(groups_face)

    for si, st in enumerate(seq):
        cur_done_mask_512, _ = compute_done_mask_from_dev_red_with_K(last_dev_after_red_512, surface_finish_tol, K_groups)
        if visible_all_done(cur_done_mask_512, visible_512, K_groups):
            break
        step_rec = {
            "step_index": si, "optype": st["optype"], "tool_kind": st["tool_kind"],
            "tool_diameter": float(st["tool_diameter"]), "path_type": st["path_type"],
            "tool_axis": tuple(axis_dir), "volume_before": None, "volume_after": None,
            "removed_volume": None, "cycle_time": None, "ok": True, "error": None,
            "dev_before_red_512": None, "dev_after_red_512": None,
        }
        m0 = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, f"rollout_measure_before_{si}")
        obj_blank_b, _, _ = geometry.create_geometry(session, work_part, prt_file_path, [], origin_body, True, False)
        point_sdf_before_512x100 = None
        if capture_pointwise and group_reference_points_512x100x3 is not None and measurement_point_coords is not None:
            dev_before, pointwise_before, _ = cam_utils.measure_ipw_state_detailed(
                session,
                work_part,
                obj_blank_b,
                flat_tools[0],
                points_array,
                norm_vecs_array,
                lines_array,
            )
            dev_before_red_tmp = reduce_scalar_by_groups(np.asarray(dev_before, dtype=np.float32), groups_face, weights=face_areas)
            point_sdf_before_512x100 = build_group_point_sdf_512x100(
                group_points_512x100x3=group_reference_points_512x100x3,
                groups_face=groups_face,
                measurement_point_coords_per_face=measurement_point_coords,
                measurement_point_sdf_per_face=pointwise_before,
                fallback_node_sdf_512=pad_1d_float(dev_before_red_tmp, 512),
            )
        else:
            dev_before, _ = cam_utils.measure_ipw_state(session, work_part, obj_blank_b, flat_tools[0], points_array, norm_vecs_array, lines_array)
        dev_before_red = reduce_scalar_by_groups(np.asarray(dev_before, dtype=np.float32), groups_face, weights=face_areas)
        session.UndoToMark(m0, f"rollout_measure_before_{si}")
        session.DeleteUndoMark(m0, f"rollout_measure_before_{si}")
        step_rec["dev_before_red_512"] = pad_1d_float(dev_before_red, 512)
        if point_sdf_before_512x100 is not None:
            step_rec["point_sdf_before_512x100"] = point_sdf_before_512x100.astype(np.float32)

        apply_mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, f"rollout_apply_{si}")
        try:
            obj_blank, _, _ = geometry.create_geometry(session, work_part, prt_file_path, [], origin_body, True, False)
            if st["optype"] == "3D Adaptive Roughing":
                tool = resolve_tool_name(st["tool_kind"], st["tool_diameter"], flat_tools, ball_tools)
                operations.create_3d_adaptive_roughing(session, work_part, tool, obj_blank, operation_list, cycle_time_list, tool_orientation=axis_dir)
            elif st["optype"] == "Cavity Mill":
                face_idx_list = cam_utils.classify_faces_for_operation(session, work_part, None, "Cavity Mill", origin_faces, dev_before)
                selected_faces = [origin_faces[i] for i in face_idx_list]
                tool = resolve_tool_name(st["tool_kind"], st["tool_diameter"], flat_tools, ball_tools)
                stepover = float(st["tool_diameter"]) / 2.0
                operations.create_cavity_milling(session, work_part, tool, st["path_type"], obj_blank, selected_faces, operation_list, cycle_time_list, stepover, tool_orientation=axis_dir)
            elif st["optype"] == "Area Mill":
                face_idx_list = cam_utils.classify_faces_for_operation(session, work_part, None, "Area Mill", origin_faces, dev_before)
                selected_faces = []
                for i in face_idx_list:
                    if float(dev_before[i]) <= float(surface_finish_tol):
                        continue
                    if origin_faces[i].Tag not in visible_set:
                        continue
                    selected_faces.append(origin_faces[i])
                if len(selected_faces) == 0:
                    cycle_time_list.append(0.0)
                else:
                    tool = resolve_tool_name(st["tool_kind"], st["tool_diameter"], flat_tools, ball_tools)
                    operations.create_surface_contour(session, work_part, tool, obj_blank, selected_faces, operation_list, cycle_time_list, tool_orientation=axis_dir)
            elif st["optype"] == "Swarf Mill":
                face_idx_list = cam_utils.classify_faces_for_operation(session, work_part, None, "Area Mill", origin_faces, dev_before)
                selected_faces = []
                for i in face_idx_list:
                    if float(dev_before[i]) <= float(surface_finish_tol):
                        continue
                    if origin_faces[i].Tag not in visible_set:
                        continue
                    selected_faces.append(origin_faces[i])
                if len(selected_faces) == 0:
                    cycle_time_list.append(0.0)
                else:
                    tool = resolve_tool_name(st["tool_kind"], st["tool_diameter"], flat_tools, ball_tools)
                    operations.create_swarf_milling(
                        session,
                        work_part,
                        tool,
                        obj_blank,
                        selected_faces,
                        operation_list,
                        cycle_time_list,
                    )
            elif st["optype"] == "Point Mill":
                face_idx_list = cam_utils.classify_faces_for_operation(session, work_part, None, "Area Mill", origin_faces, dev_before)
                selected_faces = []
                for i in face_idx_list:
                    if float(dev_before[i]) <= float(surface_finish_tol):
                        continue
                    if origin_faces[i].Tag not in visible_set:
                        continue
                    selected_faces.append(origin_faces[i])
                if len(selected_faces) == 0:
                    cycle_time_list.append(0.0)
                else:
                    tool = resolve_tool_name(st["tool_kind"], st["tool_diameter"], flat_tools, ball_tools)
                    operations.create_point_milling(
                        session,
                        work_part,
                        tool,
                        obj_blank,
                        selected_faces,
                        operation_list,
                        cycle_time_list,
                    )
            else:
                raise ValueError(f"unknown optype: {st['optype']}")
        except Exception:
            session.UndoToMark(apply_mark, f"rollout_apply_rollback_{si}")
            session.DeleteUndoMark(apply_mark, f"rollout_apply_{si}")
            raise
        else:
            session.DeleteUndoMark(apply_mark, f"rollout_apply_{si}")

        m1 = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, f"rollout_measure_after_{si}")
        try:
            obj_blank_a, _, _ = geometry.create_geometry(session, work_part, prt_file_path, [], origin_body, True, False)
            point_sdf_after_512x100 = None
            if capture_pointwise and group_reference_points_512x100x3 is not None and measurement_point_coords is not None:
                dev_after, pointwise_after, vol_after = cam_utils.measure_ipw_state_detailed(
                    session,
                    work_part,
                    obj_blank_a,
                    flat_tools[0],
                    points_array,
                    norm_vecs_array,
                    lines_array,
                    savepath=stl_file_list,
                )
                dev_after_red_tmp = reduce_scalar_by_groups(np.asarray(dev_after, dtype=np.float32), groups_face, weights=face_areas)
                point_sdf_after_512x100 = build_group_point_sdf_512x100(
                    group_points_512x100x3=group_reference_points_512x100x3,
                    groups_face=groups_face,
                    measurement_point_coords_per_face=measurement_point_coords,
                    measurement_point_sdf_per_face=pointwise_after,
                    fallback_node_sdf_512=pad_1d_float(dev_after_red_tmp, 512),
                )
            else:
                dev_after, vol_after = cam_utils.measure_ipw_state(
                    session,
                    work_part,
                    obj_blank_a,
                    flat_tools[0],
                    points_array,
                    norm_vecs_array,
                    lines_array,
                    savepath=stl_file_list,
                )
            dev_after_red = reduce_scalar_by_groups(np.asarray(dev_after, dtype=np.float32), groups_face, weights=face_areas)
        finally:
            session.UndoToMark(m1, f"rollout_measure_after_{si}")
            session.DeleteUndoMark(m1, f"rollout_measure_after_{si}")

        removed = max(float(vol_cur) - float(vol_after), 0.0)
        cyc = float(cycle_time_list[-1]) if cycle_time_list else 0.0
        step_rec["volume_before"] = float(vol_cur)
        step_rec["volume_after"] = float(vol_after)
        step_rec["removed_volume"] = float(removed)
        step_rec["cycle_time"] = float(cyc)
        step_rec["dev_after_red_512"] = pad_1d_float(dev_after_red, 512)
        if point_sdf_after_512x100 is not None:
            step_rec["point_sdf_after_512x100"] = point_sdf_after_512x100.astype(np.float32)
        steps.append(step_rec)
        vol_cur = float(vol_after)
        last_dev_after_red_512 = np.asarray(step_rec["dev_after_red_512"], dtype=np.float32)

    total_removed = float(sum(s.get("removed_volume") or 0.0 for s in steps))
    total_time = float(sum(s.get("cycle_time") or 0.0 for s in steps))
    summary = {
        "axis_dir": tuple(axis_dir),
        "num_steps_done": int(len(steps)),
        "total_removed_volume": total_removed,
        "total_cycle_time": total_time,
        "steps": steps,
        "volume_before": float(vol_start),
        "volume_after": float(vol_cur),
        "visible_512": visible_512.astype(np.int16),
        "dev_before_red_512": dev_before_red_512.astype(np.float32),
        "dev_after_red_512": last_dev_after_red_512.astype(np.float32),
    }
    return summary

def create_run_output_dir(out_root: str, part_name: str, seed: int) -> str:
    """Creates a timestamped output directory for the current run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{part_name}_seed{int(seed)}_{ts}"
    run_dir = os.path.join(out_root, run_name)
    _ensure_dir(run_dir)
    return run_dir

def _to_serializable_list(x):
    """Performs: to list."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _to_safe_float(x, default=0.0) -> float:
    """Performs: safe float."""
    return float(x)

# ----------------------------
# ----------------------------
def collect_dataset_episode(prt_file_path: str, out_root: str, seed: int = 0, global_parquet_dir: str = None):
    """Collects one full dataset episode for a single part file."""
    session, work_part = cam_session.create_session(input_file_dir=prt_file_path)
    theUfSession = NXOpen.UF.UFSession.GetUFSession()  # noqa: kept for legacy helpers

    origin_body = max(work_part.Bodies, key=get_body_volume)
    origin_faces = origin_body.GetFaces()
    origin_faces_tag = [f.Tag for f in origin_faces]

    graph, _, face_areas, _ = cam_utils.get_encoder_input_data(origin_faces, origin_faces_tag)
    face_areas = np.asarray(face_areas, dtype=np.float32)
    _, points_array_origin = cam_utils.get_face_point_cloud(origin_faces)
    raw_face_count = int(len(origin_faces))
    if raw_face_count > 512:
        force_close_part_by_path(prt_file_path)
        raise ValueError(
            f"Raw-face schema requires <=512 faces, but {prt_file_path} has {raw_face_count} faces."
        )

    graph_dense = nxg.relabel_nodes(
        graph,
        {tag: idx for idx, tag in enumerate(origin_faces_tag)},
        copy=True,
    )
    graph_json = serialize_graph_to_node_link(graph_dense)
    groups_face = [[idx] for idx in range(raw_face_count)]

    K = raw_face_count
    centrality = pad_1d_int(np.array([graph_dense.degree(n) for n in range(raw_face_count)], dtype=np.int16), 512)
    spatial_pos = pad_2d_int(build_graph_distance_matrix(graph_dense).astype(np.int16), 512)
    face_area_raw_512 = pad_face_area(face_areas.astype(np.float32), 512)
    face_pc_raw_512 = pad_face_pc(np.asarray(points_array_origin, dtype=np.float32), 512)

    points_array, norm_vecs_array, lines_array = [], [], []
    for face in origin_faces:
        if face.SolidFaceType.value == 10:
            pts, norms, lines = sample_convergent_face_points(face)
        else:
            pts, norms, lines = sample_face_points(face)
        points_array.append(pts)
        norm_vecs_array.append(norms)
        lines_array.append(lines)

    mill_tool_types = ["MILL", "BALL_MILL"]
    flat_tools, ball_tools = [], []
    for d in [20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 4.0]:
        cam_utils.create_cam_tool(session, work_part, tool_diameter=d, tool_type=mill_tool_types[0], tool_list=flat_tools)
    for d in [8.0, 6.0, 4.0]:
        cam_utils.create_cam_tool(session, work_part, tool_diameter=d, tool_type=mill_tool_types[1], tool_list=ball_tools)

    part_name = os.path.splitext(os.path.basename(prt_file_path))[0]
    out_dir = create_run_output_dir(out_root, part_name, seed)
    run_name = os.path.basename(out_dir)

    NUM_DECISION_STEPS = int(os.getenv("NUM_DECISION_STEPS", "8"))
    SURFACE_FINISH_TOL = 0.01
    MAX_ROUGH_TARGETS = int(os.getenv("MAX_ROUGH_TARGETS", "5"))
    MAX_FINISH_TARGETS = int(os.getenv("MAX_FINISH_TARGETS", "5"))
    MAX_TOOLS_PER_CLASS = int(os.getenv("MAX_TOOLS_PER_CLASS", "7"))
    BEAM_WIDTH = max(1, int(os.getenv("BEAM_WIDTH", "3")))
    SAMPLED_BRANCHES = max(0, int(os.getenv("SAMPLED_BRANCHES", "1")))

    face_normals_list = []
    if norm_vecs_array:
        for i in range(len(norm_vecs_array)):
            current_norms_raw = norm_vecs_array[i]
            if current_norms_raw is None or len(current_norms_raw) == 0:
                mean_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                clean_vectors = []
                for vec in current_norms_raw:
                    if hasattr(vec, 'Vector'):
                        clean_vectors.append([vec.Vector.X, vec.Vector.Y, vec.Vector.Z])
                    elif hasattr(vec, 'X') and hasattr(vec, 'Y') and hasattr(vec, 'Z'):
                        clean_vectors.append([vec.X, vec.Y, vec.Z])
                    else:
                        clean_vectors.append(vec)
                mean_normal = np.mean(clean_vectors, axis=0)
                n_norm = np.linalg.norm(mean_normal)
                if n_norm > 1e-9: mean_normal = mean_normal / n_norm
                else: mean_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            face_normals_list.append(mean_normal)
    face_normal_512 = build_group_face_normals_512(groups_face, face_normals_list, max_nodes=512)

    node_mask_512 = build_node_mask_512(K)
    point_mask_512x100 = build_point_mask_512x100(face_pc_raw_512, K)
    measurement_point_coords = extract_point_coordinates(points_array)
    normalization_center_xyz, normalization_scale, bbox_extent_xyz = compute_reference_frame(
        face_pc_raw_512,
        node_mask_512,
    )
    face_pc = normalize_face_points(face_pc_raw_512, normalization_center_xyz, normalization_scale)
    face_area = normalize_face_area(face_area_raw_512, normalization_scale)
    workpiece_name_chain: List[str] = []
    base_object_blank, _, _ = geometry.create_geometry(
        session, work_part, prt_file_path, workpiece_name_chain, origin_body, True, False,
    )

    _json_dump(os.path.join(out_dir, "meta.json"), {
        "prt_file_path": os.path.abspath(prt_file_path),
        "part_name": part_name, "seed": int(seed), "K_raw_faces": int(K),
        "num_faces": int(len(origin_faces)),
        "note": {
            "mode": "process_skeleton_dataset",
            "row_unit": "one_executed_nx_operation",
            "planner_schema": "graph_sdf_process_planner",
            "candidate_strategy": "beam_sampled_action_transition",
            "decision_order": "macro_face_tool_then_axis_derived",
            "rollout": {
                "beam_width": int(BEAM_WIDTH),
                "sampled_branches": int(SAMPLED_BRANCHES),
            },
            "node_process_state_channels": ["rough_done", "finish_ready"],
            "global_process_state_channels": [
                "prev_macro_onehot_5",
                "rough_ratio",
                "finish_ratio",
                "bbox_extent_over_scale_xyz",
                "log_ref_scale",
            ],
            "macro_classes_present": list(MACRO_CLASS_TO_ID.keys()),
            "normalization": {
                "center_xyz": normalization_center_xyz.tolist(),
                "reference_scale": float(normalization_scale),
                "bbox_extent_xyz": bbox_extent_xyz.tolist(),
            },
        },
    })
    _np_save(os.path.join(out_dir, "embed_centrality.npy"), centrality)
    _np_save(os.path.join(out_dir, "embed_spatial_pos.npy"), spatial_pos)
    _np_save(os.path.join(out_dir, "embed_face_area.npy"), face_area)
    _np_save(os.path.join(out_dir, "embed_face_pc.npy"), face_pc)

    rough_done_cumulative_512 = np.zeros((512,), dtype=np.int16)
    prev_macro_class_id = -1
    rough_rows_emitted = 0
    finish_rows_emitted = 0
    parquet_rows: List[Dict[str, Any]] = []
    episode_record = {
        "part_name": part_name, "seed": int(seed),
        "num_decision_steps": int(NUM_DECISION_STEPS),
        "surface_finish_tol": float(SURFACE_FINISH_TOL),
        "row_unit": "one_executed_nx_operation",
        "steps": [],
    }

    def _build_parquet_row(
        action: Dict[str, Any],
        result: Dict[str, Any],
        state_node_sdf_raw: np.ndarray,
        rough_done_mask_for_row: np.ndarray,
        prev_macro_class_id_for_row: int,
        rough_rows_emitted_for_row: int,
        finish_rows_emitted_for_row: int,
        target_node_id: int,
        decision_step: int,
        candidate_index: int,
        is_chosen: int,
        scenario_id: str = "root",
        parent_scenario_id: str = "",
    ) -> Dict[str, Any] | None:
        """Assembles one parquet row from a simulated action result."""
        macro_class_name = action["macro_class_name"]
        macro_class_id = int(MACRO_CLASS_TO_ID[macro_class_name])
        tool_kind = action["tool_kind"]
        tool_diameter = float(action["tool_diameter"])
        tool_choice_name = tool_choice_key(tool_kind, tool_diameter)
        tool_choice_id = int(TOOL_CHOICE_TO_ID.get(tool_choice_name, -1))
        tool_choice_valid = int(tool_choice_id >= 0)
        axis_visible_512 = np.asarray(result["visible_512"], dtype=np.int16)

        next_node_sdf_raw = np.asarray(result["dev_after_red_512"], dtype=np.float32)
        delta_node_sdf = np.maximum(state_node_sdf_raw - next_node_sdf_raw, 0.0)

        state_done_mask_512, state_done_ratio = compute_done_mask_from_dev_red(
            state_node_sdf_raw, SURFACE_FINISH_TOL, node_mask_512=node_mask_512,
        )
        next_done_mask_512, next_done_ratio = compute_done_mask_from_dev_red(
            next_node_sdf_raw, SURFACE_FINISH_TOL, node_mask_512=node_mask_512,
        )
        rough_done_mask = np.asarray(rough_done_mask_for_row, dtype=np.int16).copy()
        finish_ready_mask = build_finish_ready_mask_512(
            node_sdf_512=state_node_sdf_raw,
            rough_done_mask_512=rough_done_mask,
            node_mask_512=node_mask_512,
            done_tol=SURFACE_FINISH_TOL,
            ready_tol=FINISH_READY_TOL,
        )

        # action_face_mask: 0 = valid candidate, 1 = invalid
        if is_local_operation(action["optype"]):
            valid_target_mask = (
                (axis_visible_512 == 1)
                & (state_done_mask_512 == 0)
                & (rough_done_mask == 1)
                & (finish_ready_mask == 1)
                & (node_mask_512 == 0)
            ).astype(np.int16)
        else:
            valid_target_mask = (
                (state_done_mask_512 == 0) & (node_mask_512 == 0)
            ).astype(np.int16)
        action_face_mask_512 = (valid_target_mask == 0).astype(np.int16)

        # macro class mask
        macro_class_mask = build_macro_class_mask_5(
            state_done_mask_512=state_done_mask_512,
            rough_done_mask_512=rough_done_mask,
            finish_ready_mask_512=finish_ready_mask,
            node_mask_512=node_mask_512,
        )
        macro_class_mask[macro_class_id] = 0

        # tool choice mask
        tool_choice_mask = np.asarray(
            build_tool_choice_mask_for_macro_class(macro_class_id), dtype=np.int16,
        )
        if 0 <= tool_choice_id < len(TOOL_LIBRARY):
            tool_choice_mask[tool_choice_id] = 0

        node_process_state = np.stack(
            [rough_done_mask.astype(np.float32), finish_ready_mask.astype(np.float32)],
            axis=-1,
        )

        state_point_sdf_raw = np.asarray(
            result.get(
                "point_sdf_before_512x100",
                np.broadcast_to(state_node_sdf_raw.reshape(512, 1), (512, 100)),
            ),
            dtype=np.float32,
        )
        next_point_sdf_raw = np.asarray(
            result.get(
                "point_sdf_after_512x100",
                np.broadcast_to(next_node_sdf_raw.reshape(512, 1), (512, 100)),
            ),
            dtype=np.float32,
        )

        state_points_tensor = build_state_points_tensor(
            face_pc, face_normal_512, state_point_sdf_raw, normalization_scale,
        )
        # Note: shape must be [512] (not [512, 1]) to match dataset.py expectations.
        next_node_sdf_norm = build_normalized_node_sdf_512x1(next_node_sdf_raw, normalization_scale).reshape(512)
        next_point_sdf_norm = build_normalized_point_sdf_512x100(next_point_sdf_raw, normalization_scale)

        global_process_state = build_global_process_state(
            prev_macro_class_id=prev_macro_class_id_for_row,
            rough_rows_emitted=rough_rows_emitted_for_row,
            finish_rows_emitted=finish_rows_emitted_for_row,
            bbox_extent_xyz=bbox_extent_xyz,
            reference_scale=normalization_scale,
        )

        removed_volume = float(result.get("removed_volume", 0.0) or 0.0)
        vol_before = float(result.get("volume_before", 0.0) or 0.0)
        axis_dir = tuple(action["axis_dir"])

        return {
            "part_name": part_name,
            "prt_file_path": os.path.abspath(prt_file_path),
            "graph_nx_json": json.dumps(graph_json),
            "seed": int(seed),
            "decision_step": int(decision_step),
            "candidate_index": int(candidate_index),
            "is_chosen": int(is_chosen),
            "scenario_id": str(scenario_id),
            "parent_scenario_id": str(parent_scenario_id),

            "macro_class_id": int(macro_class_id),
            "macro_class_name": macro_class_name,
            "tool_choice_id": int(tool_choice_id if tool_choice_id >= 0 else 0),
            "tool_choice_name": tool_choice_name,
            "action_face_id": int(target_node_id),
            "action_face_valid": int(target_node_id >= 0),
            "anchor_face_id": int(target_node_id) if macro_class_name == "indexed_rough" else -1,
            "target_face_id": -1 if macro_class_name == "indexed_rough" else int(target_node_id),
            "tool_choice_valid": int(tool_choice_valid),

            "state_points": _to_serializable_list(state_points_tensor.astype(np.float32)),
            "node_process_state": _to_serializable_list(node_process_state.astype(np.float32)),
            "global_process_state": _to_serializable_list(global_process_state.astype(np.float32)),
            "next_node_sdf": _to_serializable_list(next_node_sdf_norm.astype(np.float32)),
            "next_point_sdf": _to_serializable_list(next_point_sdf_norm.astype(np.float32)),
            "node_mask": _to_serializable_list(node_mask_512.astype(np.int16)),
            "point_mask": _to_serializable_list(point_mask_512x100.astype(np.int16)),
            "macro_class_mask": _to_serializable_list(macro_class_mask.astype(np.int16)),
            "tool_choice_mask": _to_serializable_list(tool_choice_mask.astype(np.int16)),
            "action_face_mask": _to_serializable_list(action_face_mask_512.astype(np.int16)),

            "centrality_512": _to_serializable_list(np.asarray(centrality, dtype=np.int16)),
            "spatial_pos_512x512": _to_serializable_list(np.asarray(spatial_pos, dtype=np.int16)),
            "face_area_512x1": _to_serializable_list(np.asarray(face_area, dtype=np.float32)),

            "axis_visible_512": _to_serializable_list(axis_visible_512.astype(np.int16)),
            "state_node_sdf_raw_512": _to_serializable_list(state_node_sdf_raw.astype(np.float32)),
            "next_node_sdf_raw_512": _to_serializable_list(next_node_sdf_raw.astype(np.float32)),
            "state_point_sdf_raw_512x100": _to_serializable_list(state_point_sdf_raw.astype(np.float32)),
            "next_point_sdf_raw_512x100": _to_serializable_list(next_point_sdf_raw.astype(np.float32)),
            "state_done_mask_512": _to_serializable_list(state_done_mask_512.astype(np.int16)),
            "next_done_mask_512": _to_serializable_list(next_done_mask_512.astype(np.int16)),
            "rough_done_mask_512": _to_serializable_list(rough_done_mask.astype(np.int16)),
            "finish_ready_mask_512": _to_serializable_list(finish_ready_mask.astype(np.int16)),
            "state_done_ratio": _to_safe_float(state_done_ratio),
            "next_done_ratio": _to_safe_float(next_done_ratio),

            "axis_dir": list(map(float, axis_dir)),
            "operation_name": str(action["optype"]),
            "path_type": str(action.get("path_type", "FollowPart")),
            "tool_type_name": tool_kind,
            "tool_diameter": _to_safe_float(tool_diameter),
            "state_volume": _to_safe_float(vol_before),
            "next_state_volume": _to_safe_float(result.get("volume_after", vol_before)),
            "out_removed_volume": _to_safe_float(removed_volume),
            "out_removed_ratio": _to_safe_float(removed_volume / max(vol_before, 1e-9)),
            "out_cycle_time": _to_safe_float(result.get("cycle_time", 0.0)),
            "out_ok": bool(result.get("ok", True)),
            "normalization_center_xyz": _to_serializable_list(normalization_center_xyz.astype(np.float32)),
            "normalization_scale": _to_safe_float(normalization_scale),
            "bbox_extent_xyz": _to_serializable_list(bbox_extent_xyz.astype(np.float32)),

            "info_json": json.dumps({
                "surface_finish_tol": float(SURFACE_FINISH_TOL),
                "candidate_strategy": "beam_sampled_action_transition",
                "normalization": "part_bbox_center_and_diagonal",
                "rough_done_delta_eps": float(ROUGH_DONE_DELTA_EPS),
                "finish_ready_tol": float(FINISH_READY_TOL),
                "row_unit": "one_executed_nx_operation",
            }, ensure_ascii=False),
        }

    rng = np.random.default_rng(int(seed))

    root_chain_snapshot = list(workpiece_name_chain)
    beam_root_mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "beam_root")

    def _restore_to_root() -> None:
        """Restores NX and the Python-side IPW chain to the initial branch root."""
        session.UndoToMark(beam_root_mark, "beam_root")
        restore_workpiece_chain(workpiece_name_chain, root_chain_snapshot)

    def _replay_history(history: List[Dict[str, Any]]) -> Tuple[bool, str | None]:
        """Replays a scenario action sequence from the current root state."""
        for hi, hist_action in enumerate(history):
            result = simulate_single_action(
                session=session, work_part=work_part,
                prt_file_path=prt_file_path,
                origin_body=origin_body, origin_faces=origin_faces,
                groups_face=groups_face, face_areas=face_areas,
                points_array=points_array, norm_vecs_array=norm_vecs_array,
                lines_array=lines_array, flat_tools=flat_tools, ball_tools=ball_tools,
                action=hist_action,
                surface_finish_tol=SURFACE_FINISH_TOL,
                workpiece_name_chain=workpiece_name_chain,
                commit_to_chain=True,
                operation_depth=hi,
                base_object_blank=base_object_blank,
            )
        return True, None

    def _make_child_branch(
        parent_branch: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any],
        state_sdf: np.ndarray,
        child_score: float,
        child_id: str,
    ) -> Dict[str, Any]:
        rough_done_next = np.asarray(parent_branch["rough_done"], dtype=np.int16).copy()
        rough_rows_next = int(parent_branch["rough_rows"])
        finish_rows_next = int(parent_branch["finish_rows"])
        macro_class_name = str(action["macro_class_name"])
        macro_class_id = int(MACRO_CLASS_TO_ID[macro_class_name])
        next_sdf = np.asarray(result["dev_after_red_512"], dtype=np.float32)
        delta = np.maximum(np.asarray(state_sdf, dtype=np.float32) - next_sdf, 0.0)

        if macro_class_name == "indexed_rough":
            rough_rows_next += 1
            rough_impacted = (
                (delta > float(ROUGH_DONE_DELTA_EPS))
                & (node_mask_512 == 0)
            ).astype(np.int16)
            rough_done_next = np.maximum(rough_done_next, rough_impacted)
        else:
            finish_rows_next += 1

        return {
            "id": child_id,
            "parent_id": str(parent_branch["id"]),
            "history": list(parent_branch["history"]) + [dict(action)],
            "rough_done": rough_done_next,
            "prev_macro": macro_class_id,
            "rough_rows": rough_rows_next,
            "finish_rows": finish_rows_next,
            "score": float(child_score),
        }

    def _select_next_children(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not children:
            return []
        ordered = sorted(children, key=lambda item: float(item["score"]), reverse=True)
        selected_positions: List[int] = list(range(min(BEAM_WIDTH, len(ordered))))
        remaining = [i for i in range(len(ordered)) if i not in set(selected_positions)]
        if SAMPLED_BRANCHES > 0 and remaining:
            sample_count = min(SAMPLED_BRANCHES, len(remaining))
            sampled = rng.choice(np.asarray(remaining, dtype=np.int32), size=sample_count, replace=False)
            selected_positions.extend(int(x) for x in sampled.tolist())
        selected_positions = sorted(set(selected_positions), key=lambda idx: float(ordered[idx]["score"]), reverse=True)
        return [ordered[idx] for idx in selected_positions]

    try:
        branches: List[Dict[str, Any]] = [{
            "id": "s0",
            "parent_id": "",
            "history": [],
            "rough_done": rough_done_cumulative_512.copy(),
            "prev_macro": prev_macro_class_id,
            "rough_rows": rough_rows_emitted,
            "finish_rows": finish_rows_emitted,
            "score": 0.0,
        }]

        for t in range(NUM_DECISION_STEPS):
            depth_children: List[Dict[str, Any]] = []
            depth_record: Dict[str, Any] = {
                "t": int(t),
                "num_input_branches": int(len(branches)),
                "branch_records": [],
            }

            for bi, branch in enumerate(branches):
                _restore_to_root()
                replay_ok, replay_error = _replay_history(branch["history"])
                if not replay_ok:
                    depth_record["branch_records"].append({
                        "scenario_id": str(branch["id"]),
                        "stopped": True,
                        "reason": "replay_failed",
                        "error": replay_error,
                    })
                    continue

                K_groups = len(groups_face)
                state_volume, state_dev_red_512 = measure_current_state(
                    session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
                    face_areas, points_array, norm_vecs_array, lines_array, flat_tools,
                    workpiece_name_chain=workpiece_name_chain,
                    state_depth=len(branch["history"]),
                    base_object_blank=base_object_blank,
                )
                state_done_mask_512, all_done = compute_done_mask_from_dev_red_with_K(
                    state_dev_red_512, SURFACE_FINISH_TOL, K_groups,
                )
                state_done_ratio = float(state_done_mask_512[:K_groups].mean()) if K_groups > 0 else 1.0
                if all_done:
                    depth_record["branch_records"].append({
                        "scenario_id": str(branch["id"]),
                        "stopped": True,
                        "reason": "all_faces_done",
                        "state_volume": float(state_volume),
                        "state_done_ratio": float(state_done_ratio),
                    })
                    continue

                finish_ready_current_512 = build_finish_ready_mask_512(
                    node_sdf_512=state_dev_red_512,
                    rough_done_mask_512=np.asarray(branch["rough_done"], dtype=np.int16),
                    node_mask_512=node_mask_512,
                    done_tol=SURFACE_FINISH_TOL,
                    ready_tol=FINISH_READY_TOL,
                )

                candidates = generate_action_candidates(
                    state_node_sdf_512=state_dev_red_512,
                    rough_done_mask_512=np.asarray(branch["rough_done"], dtype=np.int16),
                    finish_ready_mask_512=finish_ready_current_512,
                    node_mask_512=node_mask_512,
                    face_normal_512x3=face_normal_512,
                    max_rough_targets=MAX_ROUGH_TARGETS,
                    max_finish_targets=MAX_FINISH_TARGETS,
                    max_tools_per_class=MAX_TOOLS_PER_CLASS,
                )
                candidate_tools = sorted({
                    f"{c['tool_kind']}_{float(c['tool_diameter']):.1f}" for c in candidates
                })
                candidate_macros = sorted({str(c["macro_class_name"]) for c in candidates})
                print(
                    f"[INFO] t={t} branch={branch['id']}: {len(candidates)} candidates "
                    f"(done={state_done_ratio:.2f}, macros={candidate_macros}, tools={candidate_tools})",
                    flush=True,
                )

                if not candidates:
                    depth_record["branch_records"].append({
                        "scenario_id": str(branch["id"]),
                        "stopped": True,
                        "reason": "no_candidates",
                        "state_volume": float(state_volume),
                        "state_done_ratio": float(state_done_ratio),
                    })
                    continue

                scored: List[Tuple[float, int, Dict[str, Any], Dict[str, Any], int, Dict[str, Any]]] = []
                reject_stats = {"no_effect": 0}
                reject_examples: List[str] = []

                for ci, action in enumerate(candidates):
                    cand_chain_snapshot = list(workpiece_name_chain)
                    cand_mark = session.SetUndoMark(
                        NXOpen.Session.MarkVisibility.Invisible,
                        f"beam_t{t:02d}_b{bi:03d}_c{ci:03d}",
                    )
                    try:
                        result = simulate_single_action(
                            session=session, work_part=work_part,
                            prt_file_path=prt_file_path,
                            origin_body=origin_body, origin_faces=origin_faces,
                            groups_face=groups_face, face_areas=face_areas,
                            points_array=points_array, norm_vecs_array=norm_vecs_array,
                            lines_array=lines_array, flat_tools=flat_tools, ball_tools=ball_tools,
                            action=action,
                            surface_finish_tol=SURFACE_FINISH_TOL,
                            workpiece_name_chain=workpiece_name_chain,
                            commit_to_chain=True,
                            operation_depth=len(branch["history"]),
                            base_object_blank=base_object_blank,
                        )
                    finally:
                        session.UndoToMark(cand_mark, f"beam_t{t:02d}_b{bi:03d}_c{ci:03d}")
                        session.DeleteUndoMark(cand_mark, f"beam_t{t:02d}_b{bi:03d}_c{ci:03d}")
                        restore_workpiece_chain(workpiece_name_chain, cand_chain_snapshot)

                    removed = float(result.get("removed_volume", 0.0) or 0.0)
                    after_sdf = np.asarray(result["dev_after_red_512"], dtype=np.float32)
                    delta = np.maximum(state_dev_red_512 - after_sdf, 0.0)
                    max_delta = float(np.nanmax(delta)) if delta.size else 0.0
                    max_abs_delta = float(np.nanmax(np.abs(after_sdf - state_dev_red_512))) if after_sdf.size else 0.0
                    if removed <= 1e-6 and max_delta <= 1e-6:
                        reject_stats["no_effect"] += 1
                        if len(reject_examples) < 5:
                            reject_examples.append(
                                f"c{ci}:{action['macro_class_name']} "
                                f"{action['tool_kind']}_{float(action['tool_diameter']):.1f} "
                                f"face={action['action_face_id']} no_effect "
                                f"vol={float(result.get('volume_before', 0.0) or 0.0):.6f}"
                                f"->{float(result.get('volume_after', 0.0) or 0.0):.6f} "
                                f"removed={removed:.6g} max_delta={max_delta:.6g} "
                                f"max_abs_delta={max_abs_delta:.6g} "
                                f"ct={float(result.get('cycle_time', 0.0) or 0.0):.6g}"
                            )
                        continue

                    vol_before = float(result.get("volume_before", 1.0) or 1.0)
                    removed_ratio = removed / max(vol_before, 1e-9)
                    next_done_mask, next_done_ratio = compute_done_mask_from_dev_red(
                        np.asarray(result["dev_after_red_512"], dtype=np.float32),
                        SURFACE_FINISH_TOL, node_mask_512=node_mask_512,
                    )
                    cur_done_ratio = float(
                        compute_done_mask_from_dev_red(
                            state_dev_red_512, SURFACE_FINISH_TOL, node_mask_512=node_mask_512,
                        )[1]
                    )
                    done_gain = max(float(next_done_ratio) - cur_done_ratio, 0.0)
                    cycle_time = float(result.get("cycle_time", 0.0) or 0.0)
                    action_score = 4.0 * removed_ratio + 2.0 * done_gain - 0.002 * cycle_time
                    child_score = float(branch["score"]) + float(action_score)
                    child_id = f"{branch['id']}.{t}.{ci}"
                    child_branch = _make_child_branch(
                        branch, action, result, state_dev_red_512, child_score, child_id,
                    )

                    row = _build_parquet_row(
                        action=action,
                        result=result,
                        state_node_sdf_raw=state_dev_red_512,
                        rough_done_mask_for_row=np.asarray(branch["rough_done"], dtype=np.int16),
                        prev_macro_class_id_for_row=int(branch["prev_macro"]),
                        rough_rows_emitted_for_row=int(branch["rough_rows"]),
                        finish_rows_emitted_for_row=int(branch["finish_rows"]),
                        target_node_id=int(action["action_face_id"]),
                        decision_step=t,
                        candidate_index=ci,
                        is_chosen=0,
                        scenario_id=child_id,
                        parent_scenario_id=str(branch["id"]),
                    )
                    row_index = len(parquet_rows)
                    parquet_rows.append(row)
                    scored.append((child_score, ci, action, result, row_index, child_branch))

                branch_record: Dict[str, Any] = {
                    "scenario_id": str(branch["id"]),
                    "state_volume_before": float(state_volume),
                    "state_done_ratio_before": float(state_done_ratio),
                    "num_candidates": int(len(candidates)),
                    "num_effective": int(len(scored)),
                    "candidate_tools": candidate_tools,
                    "candidate_macros": candidate_macros,
                    "reject_stats": reject_stats,
                }
                if reject_examples:
                    branch_record["reject_examples"] = reject_examples
                if not scored:
                    print(
                        f"[WARN] t={t} branch={branch['id']}: no effective candidates. "
                        f"reject_stats={reject_stats} examples={reject_examples}",
                        flush=True,
                    )
                    branch_record["stopped"] = True
                    branch_record["reason"] = "no_effective_candidates"
                else:
                    best_for_branch = max(scored, key=lambda x: x[0])
                    branch_record.update({
                        "best_candidate_index": int(best_for_branch[1]),
                        "best_macro": str(best_for_branch[2]["macro_class_name"]),
                        "best_tool": f"{best_for_branch[2]['tool_kind']}_{best_for_branch[2]['tool_diameter']}",
                        "best_action_face": int(best_for_branch[2]["action_face_id"]),
                        "best_score": float(best_for_branch[0]),
                    })
                    for child_score, ci, action, result, row_index, child_branch in scored:
                        depth_children.append({
                            "score": float(child_score),
                            "row_index": int(row_index),
                            "branch": child_branch,
                            "candidate_index": int(ci),
                            "macro": str(action["macro_class_name"]),
                            "tool": f"{action['tool_kind']}_{action['tool_diameter']}",
                            "action_face": int(action["action_face_id"]),
                            "state_volume_after": float(result.get("volume_after", state_volume)),
                            "state_done_ratio_after": float(
                                compute_done_mask_from_dev_red(
                                    np.asarray(result["dev_after_red_512"], dtype=np.float32),
                                    SURFACE_FINISH_TOL, node_mask_512=node_mask_512,
                                )[1]
                            ),
                        })

                depth_record["branch_records"].append(branch_record)

            if not depth_children:
                depth_record.update({
                    "stopped": True,
                    "reason": "no_next_branches",
                    "num_next_branches": 0,
                })
                episode_record["steps"].append(depth_record)
                break

            selected_children = _select_next_children(depth_children)
            next_branches: List[Dict[str, Any]] = []
            selected_records: List[Dict[str, Any]] = []
            for child in selected_children:
                row_index = int(child["row_index"])
                if 0 <= row_index < len(parquet_rows):
                    parquet_rows[row_index]["is_chosen"] = 1
                next_branch = child["branch"]
                next_branches.append(next_branch)
                selected_records.append({
                    "scenario_id": str(next_branch["id"]),
                    "parent_scenario_id": str(next_branch["parent_id"]),
                    "candidate_index": int(child["candidate_index"]),
                    "macro": str(child["macro"]),
                    "tool": str(child["tool"]),
                    "action_face": int(child["action_face"]),
                    "score": float(child["score"]),
                    "state_volume_after": float(child["state_volume_after"]),
                    "state_done_ratio_after": float(child["state_done_ratio_after"]),
                })

            print(
                f"[INFO] t={t}: selected {len(next_branches)} next branches "
                f"from {len(depth_children)} effective transitions",
                flush=True,
            )
            depth_record.update({
                "num_effective_transitions": int(len(depth_children)),
                "num_next_branches": int(len(next_branches)),
                "beam_width": int(BEAM_WIDTH),
                "sampled_branches": int(SAMPLED_BRANCHES),
                "selected_branches": selected_records,
            })
            episode_record["steps"].append(depth_record)
            branches = next_branches

    finally:
        _restore_to_root()
        session.DeleteUndoMark(beam_root_mark, "beam_root")

        parquet_path = os.path.join(out_dir, f"{part_name}_seed{int(seed)}_process_skeleton_dataset.parquet")
        if parquet_rows:
            df = pd.DataFrame(parquet_rows)
            _ensure_dir(os.path.dirname(parquet_path))
            df.to_parquet(parquet_path, index=False)
            
            if global_parquet_dir:
                _ensure_dir(global_parquet_dir)
                global_filename = f"{run_name}.parquet"
                global_path = os.path.join(global_parquet_dir, global_filename)
                df.to_parquet(global_path, index=False)
                print(f"[INFO] Copied parquet to: {global_path}")

        _json_dump(os.path.join(out_dir, "episode_record.json"), episode_record)
        force_close_part_by_path(prt_file_path)

    return {
        "out_dir": out_dir, "parquet_path": parquet_path,
        "num_rows": int(len(parquet_rows)),
        "num_operation_rows": int(len(parquet_rows)),
        "num_decision_steps": int(NUM_DECISION_STEPS),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Axis Dataset from PRT")
    parser.add_argument("--input", type=str, required=False, help="Path to input .prt file")
    parser.add_argument("--output", type=str, required=False, help="Path to output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if args.input and args.output:
        prt_file_path = args.input
        out_root_dir = args.output
        current_seed = args.seed
    else:
        print("[Warning] No arguments provided. Using hardcoded default paths.")
        prt_file_path = r"C:\Users\inwoo\Desktop\3+2_Variable_Axis\test_bracket_step.prt"
        out_root_dir = r"D:\axis_dataset_out"
        current_seed = 0

    prt_path = os.path.abspath(prt_file_path)
    out_root = os.path.abspath(out_root_dir)
    
    GLOBAL_PARQUET_DIR = os.path.join(out_root, "_ALL_PARQUET_FILES")

    if not os.path.isfile(prt_path):
        print(f"[Error] PRT file not found: {prt_path}", file=sys.stderr)
        sys.exit(1)
    _ensure_dir(out_root)

    try:
        ret = collect_dataset_episode(prt_path, out_root, seed=current_seed, global_parquet_dir=GLOBAL_PARQUET_DIR)
        print(json.dumps(ret, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[Critical Error] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
