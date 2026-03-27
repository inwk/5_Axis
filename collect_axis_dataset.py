"""Collects axis-action training rows by simulating stepwise CAM rollouts in NX."""
import os
import json
import math
import gc
import random
import sys
from typing import Any, Dict, Tuple, List
from datetime import datetime
import argparse

import numpy as np
import networkx as nxg
from networkx.readwrite import json_graph
import pandas as pd
import NXOpen

from graph_face_compression import compress_graph_by_area, reduce_visibility_by_groups
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
    """Performs: pad face area."""
    out = np.zeros((n, 1), dtype=np.float32)
    m = min(len(face_area), n)
    out[:m, 0] = face_area[:m].astype(np.float32) / 1000.0
    return out

def pad_face_pc(points_group: np.ndarray, n: int = 512) -> np.ndarray:
    """Performs: pad face pc."""
    K = points_group.shape[0]
    out = np.zeros((n, points_group.shape[1], 3), dtype=np.float32)
    m = min(K, n)
    out[:m] = points_group[:m].astype(np.float32) / 10000.0
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

def compute_done_mask_from_dev_red(dev_red_512: np.ndarray, tol: float) -> Tuple[np.ndarray, float]:
    """Performs: compute done mask from dev red."""
    dev = np.asarray(dev_red_512, dtype=np.float32).reshape(-1)
    valid = np.isfinite(dev) & (dev != 0.0)
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

        try:
            _, _, direction_data, _, _, _, _ = theUfSession.Modeling.AskFaceData(origin_faces[best_face_idx].Tag)
            direction = _normalize_vector((float(direction_data[0]), float(direction_data[1]), float(direction_data[2])))
        except Exception:
            continue

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
FINISH_READY_TOL = 0.05


def build_node_mask_512(num_valid_nodes: int, max_nodes: int = 512) -> np.ndarray:
    """Builds a node padding mask where 1 indicates padded nodes."""
    mask = np.ones((max_nodes,), dtype=np.int16)
    mask[: min(num_valid_nodes, max_nodes)] = 0
    return mask


def build_point_mask_512x100(num_valid_nodes: int, points_per_node: int = 100, max_nodes: int = 512) -> np.ndarray:
    """Builds a point padding mask aligned with fixed 512x100 node-point slots."""
    mask = np.ones((max_nodes, points_per_node), dtype=np.int16)
    mask[: min(num_valid_nodes, max_nodes), :] = 0
    return mask


def build_state_points_tensor(
    face_pc_512x100x3: np.ndarray,
    face_normal_512x3: np.ndarray,
    node_sdf_512: np.ndarray,
) -> np.ndarray:
    """Builds state tensor [512, 100, 7] = xyz + normal + normalized residual sdf."""
    state_points = np.zeros((512, 100, 7), dtype=np.float32)
    state_points[:, :, 0:3] = np.asarray(face_pc_512x100x3, dtype=np.float32)
    state_points[:, :, 3:6] = np.broadcast_to(
        np.asarray(face_normal_512x3, dtype=np.float32)[:, None, :],
        (512, 100, 3),
    )
    sdf = np.asarray(node_sdf_512, dtype=np.float32).reshape(512, 1, 1) / float(SDF_FEATURE_SCALE)
    state_points[:, :, 6:7] = sdf
    return state_points


def build_normalized_node_sdf_512x1(node_sdf_512: np.ndarray) -> np.ndarray:
    """Converts raw residual thickness to normalized [512, 1] node supervision."""
    return (np.asarray(node_sdf_512, dtype=np.float32).reshape(512, 1) / float(SDF_FEATURE_SCALE)).astype(np.float32)


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
    is_three_axis = is_three_axis_tool_orientation(axis_dir)
    if optype == "3D Adaptive Roughing":
        return "3_axis_rough" if is_three_axis else "3p2_axis_rough"
    if optype in {"Cavity Mill", "Area Mill"}:
        return "3_axis_finish" if is_three_axis else "3p2_axis_finish"
    if optype == "Swarf Mill":
        return "5_axis_flank_finish"
    if optype == "Point Mill":
        return "5_axis_point_finish"
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


def build_macro_class_mask_7(
    state_done_mask_512: np.ndarray,
    rough_done_mask_512: np.ndarray,
    finish_ready_mask_512: np.ndarray,
    axis_visible_512: np.ndarray,
    node_mask_512: np.ndarray,
) -> np.ndarray:
    """Builds an invalid-mask for macro classes (1=invalid, 0=valid)."""
    active = (
        (np.asarray(axis_visible_512, dtype=np.int16) == 1)
        & (np.asarray(node_mask_512, dtype=np.int16) == 0)
    )
    state_not_done = (np.asarray(state_done_mask_512, dtype=np.int16) == 0)
    rough_done = (np.asarray(rough_done_mask_512, dtype=np.int16) == 1)
    finish_ready = (np.asarray(finish_ready_mask_512, dtype=np.int16) == 1)

    rough_possible = bool(np.any(active & state_not_done & (~rough_done)))
    finish_possible = bool(np.any(active & state_not_done & rough_done & finish_ready))
    all_done = bool(np.all(~state_not_done | (np.asarray(node_mask_512, dtype=np.int16) == 1)))

    mask = np.ones((len(MACRO_CLASS_TO_ID),), dtype=np.int16)
    if rough_possible:
        mask[MACRO_CLASS_TO_ID["3_axis_rough"]] = 0
        mask[MACRO_CLASS_TO_ID["3p2_axis_rough"]] = 0
    if finish_possible:
        mask[MACRO_CLASS_TO_ID["3_axis_finish"]] = 0
        mask[MACRO_CLASS_TO_ID["3p2_axis_finish"]] = 0
        mask[MACRO_CLASS_TO_ID["5_axis_point_finish"]] = 0
        mask[MACRO_CLASS_TO_ID["5_axis_flank_finish"]] = 0
    if all_done or (not rough_possible and not finish_possible):
        mask[MACRO_CLASS_TO_ID["stop"]] = 0

    return mask


def build_global_process_state(
    prev_macro_class_id: int,
    rough_rows_emitted: int,
    finish_rows_emitted: int,
) -> np.ndarray:
    """Builds a compact global process-history vector."""
    out = np.zeros((9,), dtype=np.float32)
    if 0 <= int(prev_macro_class_id) < 7:
        out[int(prev_macro_class_id)] = 1.0
    total = max(rough_rows_emitted + finish_rows_emitted, 1)
    out[7] = float(rough_rows_emitted / total)
    out[8] = float(finish_rows_emitted / total)
    return out


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
        try:
            _, _, direction_data, _, _, _, _ = theUfSession.Modeling.AskFaceData(origin_faces[best_face_idx].Tag)
            direction = _normalize_vector(
                (float(direction_data[0]), float(direction_data[1]), float(direction_data[2]))
            )
        except Exception:
            continue
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
    try:
        view_vector = NXOpen.Vector3d(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]))
        visible_tags = cam_utils.identify_visible_faces(origin_body, view_vector)
        visible_set = set(visible_tags)
        visible_orig = np.array([1 if f.Tag in visible_set else 0 for f in origin_faces], dtype=np.int16)
        visible_reduced = reduce_visibility_by_groups(visible_orig, groups_face)
        return pad_visibility(visible_reduced, 512)
    except Exception:
        return np.zeros((512,), dtype=np.int16)


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
        next_done_mask, _ = compute_done_mask_from_dev_red(next_state_dev, finish_tol)
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
        try:
            if part.FullPath and part.FullPath.lower() == target_path.lower():
                NXOpen.UF.UFSession.GetUFSession().Part.Close(part.Tag, 0, 1)
        except Exception:
            pass
    gc.collect()

def measure_current_state(
    session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
    face_areas, points_array, norm_vecs_array, lines_array, flat_tools,
) -> Tuple[float, np.ndarray]:
    """Measures current IPW state volume and reduced deviation features."""
    m = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "measure_state")
    try:
        obj_blank, _, _ = geometry.create_geometry(session, work_part, prt_file_path, [], origin_body, True, False)
        dev, vol = cam_utils.measure_ipw_state(
            session, work_part, obj_blank, flat_tools[0],
            points_array, norm_vecs_array, lines_array
        )
        dev_red = reduce_scalar_by_groups(np.asarray(dev, dtype=np.float32), groups_face, weights=face_areas)
        dev_red_512 = pad_1d_float(dev_red, 512)
    except Exception:
        vol = 0.0
        dev_red_512 = np.zeros((512,), dtype=np.float32)
    finally:
        session.UndoToMark(m, "measure_state")
        session.DeleteUndoMark(m, "measure_state")
    return float(vol), dev_red_512

def simulate_rollout_for_axis(
    session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
    face_areas, points_array, norm_vecs_array, lines_array, flat_tools, ball_tools,
    axis_dir, seq, out_dir_for_this_rollout, surface_finish_tol: float = 0.01,
    pre_visible_tags=None, pre_visible_512=None,
):
    """Simulates one selected axis from the current state and returns rollout metrics."""
    _ensure_dir(out_dir_for_this_rollout)
    if pre_visible_tags is not None:
        visible_set = set(pre_visible_tags)
        visible_512 = np.asarray(pre_visible_512, dtype=np.int16) if pre_visible_512 is not None else np.zeros((512,), dtype=np.int16)
    else:
        visible_set = set()
        visible_512 = np.zeros((512,), dtype=np.int16)
        try:
            view_vector = NXOpen.Vector3d(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]))
            visible_tags = cam_utils.identify_visible_faces(origin_body, view_vector)
            visible_set = set(visible_tags)
            visible_orig = np.array([1 if f.Tag in visible_set else 0 for f in origin_faces], dtype=int)
            visible_reduced = reduce_visibility_by_groups(visible_orig, groups_face)
            visible_512 = pad_visibility(visible_reduced, 512)
        except Exception:
            pass

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
        dev_before, _ = cam_utils.measure_ipw_state(session, work_part, obj_blank_b, flat_tools[0], points_array, norm_vecs_array, lines_array)
        dev_before_red = reduce_scalar_by_groups(np.asarray(dev_before, dtype=np.float32), groups_face, weights=face_areas)
        session.UndoToMark(m0, f"rollout_measure_before_{si}")
        session.DeleteUndoMark(m0, f"rollout_measure_before_{si}")
        step_rec["dev_before_red_512"] = pad_1d_float(dev_before_red, 512)

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
                    try:
                        if float(dev_before[i]) <= float(surface_finish_tol): continue
                    except: pass
                    try:
                        if origin_faces[i].Tag not in visible_set: continue
                    except: pass
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
                    try:
                        if float(dev_before[i]) <= float(surface_finish_tol):
                            continue
                    except Exception:
                        pass
                    try:
                        if origin_faces[i].Tag not in visible_set:
                            continue
                    except Exception:
                        pass
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
                    try:
                        if float(dev_before[i]) <= float(surface_finish_tol):
                            continue
                    except Exception:
                        pass
                    try:
                        if origin_faces[i].Tag not in visible_set:
                            continue
                    except Exception:
                        pass
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
        except Exception as e:
            step_rec["ok"] = False
            step_rec["error"] = f"apply_error: {repr(e)}"
            session.UndoToMark(apply_mark, f"rollout_apply_rollback_{si}")
            session.DeleteUndoMark(apply_mark, f"rollout_apply_{si}")
            step_rec["volume_before"] = float(vol_cur)
            step_rec["volume_after"] = float(vol_cur)
            step_rec["removed_volume"] = 0.0
            step_rec["cycle_time"] = 0.0
            step_rec["dev_after_red_512"] = np.asarray(step_rec["dev_before_red_512"], dtype=np.float32)
            steps.append(step_rec)
            break
        else:
            session.DeleteUndoMark(apply_mark, f"rollout_apply_{si}")

        m1 = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, f"rollout_measure_after_{si}")
        obj_blank_a, _, _ = geometry.create_geometry(session, work_part, prt_file_path, [], origin_body, True, False)
        try:
            dev_after, vol_after = cam_utils.measure_ipw_state(session, work_part, obj_blank_a, flat_tools[0], points_array, norm_vecs_array, lines_array, savepath=stl_file_list)
            dev_after_red = reduce_scalar_by_groups(np.asarray(dev_after, dtype=np.float32), groups_face, weights=face_areas)
        except Exception as e:
            vol_after = vol_cur
            dev_after_red = reduce_scalar_by_groups(np.asarray(dev_before, dtype=np.float32), groups_face, weights=face_areas)
            step_rec["ok"] = False
            step_rec["error"] = f"measure_error: {repr(e)}"
        session.UndoToMark(m1, f"rollout_measure_after_{si}")
        session.DeleteUndoMark(m1, f"rollout_measure_after_{si}")

        removed = max(float(vol_cur) - float(vol_after), 0.0)
        cyc = float(cycle_time_list[-1]) if cycle_time_list else 0.0
        step_rec["volume_before"] = float(vol_cur)
        step_rec["volume_after"] = float(vol_after)
        step_rec["removed_volume"] = float(removed)
        step_rec["cycle_time"] = float(cyc)
        step_rec["dev_after_red_512"] = pad_1d_float(dev_after_red, 512)
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
    try:
        return float(x)
    except Exception:
        return float(default)

# ----------------------------
# ----------------------------
def collect_dataset_episode(prt_file_path: str, out_root: str, seed: int = 0, global_parquet_dir: str = None):
    """Collects one full dataset episode for a single part file."""
    rng = random.Random(seed)
    session, work_part = cam_session.create_session(input_file_dir=prt_file_path)
    theUfSession = NXOpen.UF.UFSession.GetUFSession()

    origin_body = max(work_part.Bodies, key=get_body_volume)
    origin_faces = origin_body.GetFaces()
    origin_faces_tag = [f.Tag for f in origin_faces]

    graph, _, face_areas, _ = cam_utils.get_encoder_input_data(origin_faces, origin_faces_tag)
    face_areas = np.asarray(face_areas, dtype=np.float32)
    _, points_array_origin = cam_utils.get_face_point_cloud(origin_faces)
    visible_init = np.zeros(len(origin_faces), dtype=int)

    tag_to_face_idx = {tag: i for i, tag in enumerate(origin_faces_tag)}
    G_new, areas_new, points_new, _, _, groups_internal, node_labels = compress_graph_by_area(
        graph, face_areas=face_areas, face_points=[np.asarray(p, dtype=np.float32) for p in points_array_origin],
        face_visible=visible_init, max_nodes=512, area_threshold=100.0, target_points_per_node=100, seed=42,
    )
    graph_json = serialize_graph_to_node_link(G_new)

    groups_face = []
    for grp in groups_internal:
        idxs = []
        for internal_idx in grp:
            tag = node_labels[internal_idx]
            idxs.append(tag_to_face_idx[tag])
        groups_face.append(idxs)

    K = G_new.number_of_nodes()
    centrality = pad_1d_int(np.array([G_new.degree(n) for n in G_new.nodes()], dtype=np.int16), 512)
    spatial_pos = pad_2d_int(build_graph_distance_matrix(G_new).astype(np.int16), 512)
    face_area = pad_face_area(np.asarray(areas_new, dtype=np.float32), 512)
    face_pc = pad_face_pc(np.asarray(points_new, dtype=np.float32), 512)

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

    use_swarf_test_sequence = os.getenv("USE_SWARF_TEST_SEQUENCE", "0") == "1"
    use_point_test_sequence = os.getenv("USE_POINT_TEST_SEQUENCE", "0") == "1"
    if use_point_test_sequence:
        seq = build_point_test_operation_sequence()
    elif use_swarf_test_sequence:
        seq = build_swarf_test_operation_sequence()
    else:
        seq = build_default_operation_sequence()
    part_name = os.path.splitext(os.path.basename(prt_file_path))[0]
    out_dir = create_run_output_dir(out_root, part_name, seed)

    run_name = os.path.basename(out_dir)

    _json_dump(os.path.join(out_dir, "meta.json"), {
        "prt_file_path": os.path.abspath(prt_file_path),
        "part_name": part_name, "seed": int(seed), "K_nodes_compressed": int(K),
        "num_faces": int(len(origin_faces)), "sequence": seq,
        "note": {
            "mode": "process_skeleton_dataset",
            "row_unit": "one_executed_nx_operation",
            "planner_schema": "graph_sdf_process_planner",
            "node_process_state_channels": ["rough_done", "finish_ready"],
            "global_process_state_channels": ["prev_macro_onehot_7", "rough_ratio", "finish_ratio"],
            "action_selection": "limited_action_lookahead",
            "default_lookahead_depth": int(os.getenv("ACTION_LOOKAHEAD_DEPTH", "2")),
            "current_macro_classes_present": [
                "3_axis_rough",
                "3_axis_finish",
                "3p2_axis_rough",
                "3p2_axis_finish",
                "5_axis_flank_finish",
                "5_axis_point_finish",
            ],
        }
    })
    _np_save(os.path.join(out_dir, "embed_centrality.npy"), centrality)
    _np_save(os.path.join(out_dir, "embed_spatial_pos.npy"), spatial_pos)
    _np_save(os.path.join(out_dir, "embed_face_area.npy"), face_area)
    _np_save(os.path.join(out_dir, "embed_face_pc.npy"), face_pc)

    NUM_DECISION_STEPS = 5
    ACTIONS_PER_STEP = 1
    SURFACE_FINISH_TOL = 0.01

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
    face_normal_512 = np.zeros((512, 3), dtype=np.float32)
    m = min(len(face_normals_list), 512)
    if m > 0:
        face_normal_512[:m] = np.array(face_normals_list[:m], dtype=np.float32)

    node_mask_512 = build_node_mask_512(K)
    point_mask_512x100 = build_point_mask_512x100(K, points_per_node=face_pc.shape[1] if face_pc.ndim == 3 else 100)

    used_axis_dirs: List[Tuple[float, float, float]] = []
    USED_AXIS_ANGLE_TOL_DEG = 3.0
    rough_done_cumulative_512 = np.zeros((512,), dtype=np.int16)
    prev_macro_class_id = -1
    rough_rows_emitted = 0
    finish_rows_emitted = 0
    parquet_rows: List[Dict[str, Any]] = []
    episode_record = {
        "part_name": part_name, "seed": int(seed), "num_decision_steps": int(NUM_DECISION_STEPS),
        "actions_per_step": int(ACTIONS_PER_STEP), "surface_finish_tol": float(SURFACE_FINISH_TOL),
        "row_unit": "one_executed_nx_operation", "steps": [],
    }

    try:
        for t in range(NUM_DECISION_STEPS):
            base_mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, f"BASE_STATE_t{t:02d}")
            K_groups = len(groups_face)
            state_volume, state_dev_red_512 = measure_current_state(
                session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
                face_areas, points_array, norm_vecs_array, lines_array, flat_tools,
            )
            state_done_mask_512, all_done = compute_done_mask_from_dev_red_with_K(state_dev_red_512, SURFACE_FINISH_TOL, K_groups)

            if all_done:
                episode_record["steps"].append({"t": int(t), "stopped": True, "reason": "all_faces_done_before_action", "state_volume": float(state_volume)})
                session.UndoToMark(base_mark, f"ROLLBACK_STOP_t{t:02d}")
                try: session.DeleteUndoMark(base_mark, f"BASE_STATE_t{t:02d}")
                except Exception: pass
                break

            state_done_ratio = float(state_done_mask_512[:K_groups].mean()) if K_groups > 0 else 1.0
            finish_ready_current_512 = build_finish_ready_mask_512(
                node_sdf_512=state_dev_red_512,
                rough_done_mask_512=rough_done_cumulative_512,
                node_mask_512=node_mask_512,
                done_tol=SURFACE_FINISH_TOL,
                ready_tol=FINISH_READY_TOL,
            )
            candidate_actions = generate_limited_action_candidates(
                origin_body=origin_body,
                origin_faces=origin_faces,
                groups_face=groups_face,
                face_areas=face_areas,
                state_dev_red_512=state_dev_red_512,
                state_done_mask_512=state_done_mask_512,
                rough_done_mask_512=rough_done_cumulative_512,
                finish_ready_mask_512=finish_ready_current_512,
                node_mask_512=node_mask_512,
                theUfSession=theUfSession,
                max_axis_candidates=int(os.getenv("MAX_AXIS_CANDIDATES", "4")),
            )

            if not candidate_actions:
                episode_record["steps"].append({
                    "t": int(t),
                    "stopped": True,
                    "reason": "no_action_candidates",
                    "used_axis_dirs": [tuple(map(float, d)) for d in used_axis_dirs],
                })
                session.UndoToMark(base_mark, f"ROLLBACK_STOP_t{t:02d}")
                try: session.DeleteUndoMark(base_mark, f"BASE_STATE_t{t:02d}")
                except Exception: pass
                break

            lookahead_depth = int(os.getenv("ACTION_LOOKAHEAD_DEPTH", "2"))
            max_action_candidates = int(os.getenv("MAX_ACTION_CANDIDATES", "12"))
            scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
            for action in candidate_actions[: max(1, max_action_candidates)]:
                score = evaluate_action_with_lookahead(
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
                    action=action,
                    rough_done_mask_512=rough_done_cumulative_512,
                    node_mask_512=node_mask_512,
                    theUfSession=theUfSession,
                    finish_tol=SURFACE_FINISH_TOL,
                    lookahead_depth=lookahead_depth,
                    max_next_candidates=int(os.getenv("MAX_NEXT_ACTION_CANDIDATES", "4")),
                )
                scored_candidates.append((float(score), action))

            if not scored_candidates:
                episode_record["steps"].append({
                    "t": int(t),
                    "stopped": True,
                    "reason": "no_scored_candidates",
                    "used_axis_dirs": [tuple(map(float, d)) for d in used_axis_dirs],
                })
                session.UndoToMark(base_mark, f"ROLLBACK_STOP_t{t:02d}")
                try: session.DeleteUndoMark(base_mark, f"BASE_STATE_t{t:02d}")
                except Exception: pass
                break

            best_score, chosen_action = max(scored_candidates, key=lambda x: x[0])
            chosen_axis = {
                "dir": tuple(chosen_action["axis_dir"]),
                "source": "lookahead_candidate",
                "score": float(best_score),
                "optype": chosen_action["optype"],
            }
            chosen_axis_dir = tuple(chosen_action["axis_dir"])
            chosen_summary = simulate_rollout_for_axis(
                session=session, work_part=work_part, prt_file_path=prt_file_path, origin_body=origin_body,
                origin_faces=origin_faces, groups_face=groups_face, face_areas=face_areas,
                points_array=points_array, norm_vecs_array=norm_vecs_array, lines_array=lines_array,
                flat_tools=flat_tools, ball_tools=ball_tools, axis_dir=chosen_axis_dir, seq=[{
                    "optype": chosen_action["optype"],
                    "tool_kind": chosen_action["tool_kind"],
                    "tool_diameter": float(chosen_action["tool_diameter"]),
                    "path_type": chosen_action["path_type"],
                }],
                out_dir_for_this_rollout=os.path.join(out_dir, "chosen_apply", f"t{t:02d}"),
                surface_finish_tol=SURFACE_FINISH_TOL,
            )
            used_axis_dirs.append(_normalize_vector(chosen_axis_dir))
            try: session.DeleteUndoMark(base_mark, f"BASE_STATE_t{t:02d}")
            except Exception: pass

            next_volume, next_dev_red_512 = measure_current_state(
                session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
                face_areas, points_array, norm_vecs_array, lines_array, flat_tools,
            )
            done_after_mask_512, done_after_ratio = compute_done_mask_from_dev_red(next_dev_red_512, SURFACE_FINISH_TOL)
            new_done = (done_after_mask_512.astype(np.int16) - state_done_mask_512.astype(np.int16))
            new_done = np.clip(new_done, 0, 1).astype(np.int16)
            new_done_count = int(new_done.sum())

            out_removed_volume = float(chosen_summary.get("total_removed_volume", max(float(state_volume) - float(next_volume), 0.0)))
            out_cycle_time = float(chosen_summary.get("total_cycle_time", 0.0))
            out_removed_ratio = float(out_removed_volume / max(float(state_volume), 1e-9))
            axis_visible_512 = np.asarray(chosen_summary.get("visible_512", np.zeros((512,), dtype=np.int16)), dtype=np.int16)
            axis_source_id = int(0 if chosen_axis["source"] == "predefined" else 1)

            print(
                f"[DEBUG] t={t}, Axis: {tuple(f'{x:.2f}' for x in chosen_axis_dir)} "
                f"| Vol: {out_removed_volume:.2f}, Done: {new_done_count}, Time: {out_cycle_time:.1f}s",
                flush=True,
            )

            for seq_op_index, step_rec in enumerate(chosen_summary.get("steps", [])):
                state_node_sdf_raw = np.asarray(
                    step_rec.get("dev_before_red_512", state_dev_red_512),
                    dtype=np.float32,
                )
                next_node_sdf_raw = np.asarray(
                    step_rec.get("dev_after_red_512", state_node_sdf_raw),
                    dtype=np.float32,
                )
                removed_volume_step = float(step_rec.get("removed_volume", 0.0) or 0.0)
                delta_node_sdf = np.maximum(state_node_sdf_raw - next_node_sdf_raw, 0.0)
                has_effect = bool(removed_volume_step > 1e-6 or float(delta_node_sdf.max()) > 1e-6)
                if not has_effect:
                    continue

                macro_class_name = infer_macro_class_name(step_rec["optype"], chosen_axis_dir)
                macro_class_id = int(MACRO_CLASS_TO_ID[macro_class_name])
                tool_kind = str(step_rec["tool_kind"])
                tool_diameter = float(step_rec["tool_diameter"])
                tool_choice_name = tool_choice_key(tool_kind, tool_diameter)
                tool_choice_id = int(TOOL_CHOICE_TO_ID.get(tool_choice_name, -1))
                tool_choice_valid = int(tool_choice_id >= 0)

                state_done_mask_step_512, state_done_ratio_step = compute_done_mask_from_dev_red(
                    state_node_sdf_raw,
                    SURFACE_FINISH_TOL,
                )
                next_done_mask_step_512, next_done_ratio_step = compute_done_mask_from_dev_red(
                    next_node_sdf_raw,
                    SURFACE_FINISH_TOL,
                )
                rough_done_mask_step_512 = rough_done_cumulative_512.copy()
                finish_ready_mask_step_512 = build_finish_ready_mask_512(
                    node_sdf_512=state_node_sdf_raw,
                    rough_done_mask_512=rough_done_mask_step_512,
                    node_mask_512=node_mask_512,
                    done_tol=SURFACE_FINISH_TOL,
                    ready_tol=FINISH_READY_TOL,
                )
                node_process_state = np.stack(
                    [
                        rough_done_mask_step_512.astype(np.float32),
                        finish_ready_mask_step_512.astype(np.float32),
                    ],
                    axis=-1,
                )
                state_points_tensor = build_state_points_tensor(face_pc, face_normal_512, state_node_sdf_raw)
                next_node_sdf = build_normalized_node_sdf_512x1(next_node_sdf_raw)

                target_node_valid = 1
                if is_local_operation(step_rec["optype"]):
                    valid_target_mask_512 = (
                        (axis_visible_512 == 1)
                        & (state_done_mask_step_512 == 0)
                        & (rough_done_mask_step_512 == 1)
                        & (finish_ready_mask_step_512 == 1)
                        & (node_mask_512 == 0)
                    ).astype(np.int16)
                    target_node_id = infer_target_node_id(
                        state_node_sdf_512=state_node_sdf_raw,
                        next_node_sdf_512=next_node_sdf_raw,
                        valid_mask_512=valid_target_mask_512,
                    )
                    if target_node_id < 0:
                        continue
                else:
                    valid_target_mask_512 = (
                        (axis_visible_512 == 1)
                        & (node_mask_512 == 0)
                    ).astype(np.int16)
                    target_node_id = infer_axis_face_id_from_normal(
                        face_normal_512x3=face_normal_512,
                        axis_dir=chosen_axis_dir,
                        valid_mask_512=valid_target_mask_512,
                    )
                    if target_node_id < 0:
                        continue

                target_node_mask_512 = (valid_target_mask_512 == 0).astype(np.int16)
                macro_class_mask_7 = build_macro_class_mask_7(
                    state_done_mask_512=state_done_mask_step_512,
                    rough_done_mask_512=rough_done_mask_step_512,
                    finish_ready_mask_512=finish_ready_mask_step_512,
                    axis_visible_512=axis_visible_512,
                    node_mask_512=node_mask_512,
                )
                macro_class_mask_7[macro_class_id] = 0

                tool_choice_mask = np.asarray(
                    build_tool_choice_mask_for_macro_class(macro_class_id),
                    dtype=np.int16,
                )
                if 0 <= tool_choice_id < len(TOOL_LIBRARY):
                    tool_choice_mask[tool_choice_id] = 0

                global_process_state = build_global_process_state(
                    prev_macro_class_id=prev_macro_class_id,
                    rough_rows_emitted=rough_rows_emitted,
                    finish_rows_emitted=finish_rows_emitted,
                )

                parquet_rows.append({
                    "part_name": part_name,
                    "prt_file_path": os.path.abspath(prt_file_path),
                    "graph_nx_json": json.dumps(graph_json),
                    "seed": int(seed),
                    "decision_step": int(t),
                    "sequence_op_index": int(seq_op_index),

                    "macro_class_id": int(macro_class_id),
                    "macro_class_name": macro_class_name,
                    "tool_choice_id": int(tool_choice_id if tool_choice_id >= 0 else 0),
                    "tool_choice_name": tool_choice_name,
                    "target_node_id": int(target_node_id),
                    "target_node_valid": int(target_node_valid),
                    "tool_choice_valid": int(tool_choice_valid),

                    "state_points": _to_serializable_list(state_points_tensor.astype(np.float32)),
                    "node_process_state": _to_serializable_list(node_process_state.astype(np.float32)),
                    "global_process_state": _to_serializable_list(global_process_state.astype(np.float32)),
                    "next_node_sdf": _to_serializable_list(next_node_sdf.astype(np.float32)),
                    "node_mask": _to_serializable_list(node_mask_512.astype(np.int16)),
                    "point_mask": _to_serializable_list(point_mask_512x100.astype(np.int16)),
                    "macro_class_mask": _to_serializable_list(macro_class_mask_7.astype(np.int16)),
                    "tool_choice_mask": _to_serializable_list(tool_choice_mask.astype(np.int16)),
                    "target_node_mask": _to_serializable_list(target_node_mask_512.astype(np.int16)),

                    "centrality_512": _to_serializable_list(np.asarray(centrality, dtype=np.int16)),
                    "spatial_pos_512x512": _to_serializable_list(np.asarray(spatial_pos, dtype=np.int16)),
                    "face_area_512x1": _to_serializable_list(np.asarray(face_area, dtype=np.float32)),

                    "axis_visible_512": _to_serializable_list(axis_visible_512.astype(np.int16)),
                    "state_node_sdf_raw_512": _to_serializable_list(state_node_sdf_raw.astype(np.float32)),
                    "next_node_sdf_raw_512": _to_serializable_list(next_node_sdf_raw.astype(np.float32)),
                    "state_done_mask_512": _to_serializable_list(state_done_mask_step_512.astype(np.int16)),
                    "next_done_mask_512": _to_serializable_list(next_done_mask_step_512.astype(np.int16)),
                    "rough_done_mask_512": _to_serializable_list(rough_done_mask_step_512.astype(np.int16)),
                    "finish_ready_mask_512": _to_serializable_list(finish_ready_mask_step_512.astype(np.int16)),
                    "state_done_ratio": _to_safe_float(state_done_ratio_step),
                    "next_done_ratio": _to_safe_float(next_done_ratio_step),

                    "axis_dir": list(map(float, chosen_axis_dir)),
                    "axis_source": int(axis_source_id),
                    "axis_select_score": _to_safe_float(chosen_axis.get("score", 0.0)),
                    "operation_name": str(step_rec["optype"]),
                    "path_type": str(step_rec["path_type"]),
                    "tool_type_name": tool_kind,
                    "tool_diameter": _to_safe_float(tool_diameter),
                    "state_volume": _to_safe_float(step_rec.get("volume_before", state_volume)),
                    "next_state_volume": _to_safe_float(step_rec.get("volume_after", next_volume)),
                    "out_removed_volume": _to_safe_float(removed_volume_step),
                    "out_removed_ratio": _to_safe_float(
                        removed_volume_step / max(float(step_rec.get("volume_before", state_volume) or 0.0), 1e-9)
                    ),
                    "out_cycle_time": _to_safe_float(step_rec.get("cycle_time", 0.0)),
                    "out_ok": bool(step_rec.get("ok", True)),

                    "info_json": json.dumps({
                        "surface_finish_tol": float(SURFACE_FINISH_TOL),
                        "axis_policy": "limited_action_lookahead",
                        "used_axis_exclusion_tol_deg": float(USED_AXIS_ANGLE_TOL_DEG),
                        "sdf_feature_scale": float(SDF_FEATURE_SCALE),
                        "rough_done_delta_eps": float(ROUGH_DONE_DELTA_EPS),
                        "finish_ready_tol": float(FINISH_READY_TOL),
                        "row_unit": "one_executed_nx_operation",
                    }, ensure_ascii=False),
                })

                prev_macro_class_id = macro_class_id
                if macro_class_name in {"3_axis_rough", "3p2_axis_rough"}:
                    rough_rows_emitted += 1
                elif macro_class_name in {
                    "3_axis_finish",
                    "3p2_axis_finish",
                    "5_axis_point_finish",
                    "5_axis_flank_finish",
                }:
                    finish_rows_emitted += 1

                if macro_class_name in {"3_axis_rough", "3p2_axis_rough"}:
                    rough_impacted = (
                        (delta_node_sdf > float(ROUGH_DONE_DELTA_EPS))
                        & (node_mask_512 == 0)
                    ).astype(np.int16)
                    rough_done_cumulative_512 = np.maximum(
                        rough_done_cumulative_512,
                        rough_impacted,
                    )

            episode_record["steps"].append({
                "t": int(t),
                "state_volume_before": float(state_volume),
                "state_done_ratio_before": float(state_done_ratio),
                "chosen_axis": {
                    "dir": tuple(map(float, chosen_axis_dir)),
                    "source": chosen_axis["source"],
                    "score": float(chosen_axis.get("score", 0.0)),
                },
                "chosen_apply_summary": {
                    "total_removed_volume": float(chosen_summary.get("total_removed_volume", 0.0)),
                    "total_cycle_time": float(chosen_summary.get("total_cycle_time", 0.0)),
                    "num_steps_done": int(chosen_summary.get("num_steps_done", 0)),
                },
                "state_volume_after": float(next_volume),
                "state_done_ratio_after": float(done_after_ratio),
                "used_axis_dirs": [tuple(map(float, d)) for d in used_axis_dirs],
            })

    finally:
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
        try: force_close_part_by_path(prt_file_path)
        except Exception: pass

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
