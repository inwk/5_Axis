"""Collects axis-action training rows by simulating stepwise CAM rollouts in NX."""
import os
import json
import math
import gc
import sys
import shutil
import ctypes
from typing import Any, Dict, Tuple, List
from datetime import datetime
import argparse


# ── NX stdout/stderr safety wrapper ──────────────────────────────────────────
# NX replaces sys.stdout/stderr with a custom stream whose underlying Windows
# HANDLE does not support flush().  Calling print(..., flush=True) triggers
# sys.stdout.flush() → Windows EINVAL (OSError 22).  Wrapping the streams
# silences that error so the script continues safely inside NX.
class _NXSafeStream:
    """Wraps a stream so that OSError on write() or flush() is swallowed."""

    def __init__(self, stream):
        self._s = stream

    def write(self, text):
        try:
            return self._s.write(text)
        except OSError:
            return 0

    def flush(self):
        try:
            self._s.flush()
        except OSError:
            pass

    def __getattr__(self, name):
        return getattr(self._s, name)


def _is_nx_stream(stream) -> bool:
    """Returns True when the stream is NX's non-file stdout (fileno() raises)."""
    try:
        stream.fileno()
        return False
    except Exception:
        return True


# Wrap unconditionally — harmless for real file streams, fixes NX streams.
if not isinstance(sys.stdout, _NXSafeStream):
    sys.stdout = _NXSafeStream(sys.stdout)
if not isinstance(sys.stderr, _NXSafeStream):
    sys.stderr = _NXSafeStream(sys.stderr)
# ─────────────────────────────────────────────────────────────────────────────

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

from CAM.measurements import (
    create_line_with_point_and_vector,
    get_body_volume,
    sample_convergent_face_points,
    sample_face_points,
)


def save_part(session, work_part, directory):
    work_part = session.Parts.Work
    savePart = work_part.SaveAs(directory)
    savePart.Dispose()

def export_body_to_obj(session, work_part, body, output_path: str) -> str:
    """Exports an NX body as an OBJ mesh for non-NX visualization overlays."""
    output_path = os.path.abspath(output_path)
    _ensure_dir(os.path.dirname(output_path))
    obj_creator = session.DexManager.CreateWavefrontObjCreator()
    obj_creator.ExportSelectionBlock.SelectionScope = NXOpen.ObjectSelector.Scope.SelectedObjects
    obj_creator.AngularTolerance = 17.999999999999996
    obj_creator.FlattenAssemblyStructure = True
    obj_creator.ExportSelectionBlock.SelectionComp.Add(body)
    obj_creator.OutputFile = output_path
    obj_creator.FileSaveFlag = False
    obj_creator.Commit()
    obj_creator.Destroy()
    return output_path

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

def _safe_filename(text: str) -> str:
    """Returns a filesystem-safe filename fragment."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))

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
    base = [( 1, 0, 0), (-1, 0, 0), ( 0, 1, 0), ( 0,-1, 0), ( 0, 0, 1), (0, 0, -1)]
    out = []
    for d in base:
        dn = _normalize_vector(d)
        out.append(dn)
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

# ── Adaptive octree occupancy sampling ────────────────────────────────────
# The octree target is the only geometric transition supervision used by the
# new model.  Current face SDF remains an encoder input, but next-state SDF is
# no longer required as a training target.
OCTREE_ENABLED: bool = os.getenv("OCTREE_ENABLED", "1") != "0"
OCTREE_COARSE_DEPTH: int = int(os.getenv("OCTREE_COARSE_DEPTH", "3"))
OCTREE_FINE_DEPTH: int = int(os.getenv("OCTREE_FINE_DEPTH", "5"))
OCTREE_MAX_NODES: int = int(os.getenv("OCTREE_MAX_NODES", "4096"))
OCTREE_BBOX_PADDING: float = float(os.getenv("OCTREE_BBOX_PADDING", "0.05"))
MEMORY_DEBUG: bool = bool(int(os.getenv("MEMORY_DEBUG", "1")))
MEMORY_DEBUG_CANDIDATE_EVERY: int = max(1, int(os.getenv("MEMORY_DEBUG_CANDIDATE_EVERY", "1")))


if os.name == "nt":
    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]


    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]


    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
        ctypes.c_ulong,
    ]
    _psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    _kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    _kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int


def _format_gb(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{float(num_bytes) / (1024.0 ** 3):.2f}GB"


def _get_memory_snapshot() -> Dict[str, Any] | None:
    """Returns a lightweight process/system memory snapshot for debug logging."""
    if os.name != "nt":
        return None
    try:
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
        ok = _psapi.GetProcessMemoryInfo(
            _kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None

        mem_status = _MEMORYSTATUSEX()
        mem_status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
            return None

        return {
            "pid": int(os.getpid()),
            "working_set": int(counters.WorkingSetSize),
            "peak_working_set": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PrivateUsage),
            "commit_bytes": int(counters.PagefileUsage),
            "peak_commit_bytes": int(counters.PeakPagefileUsage),
            "avail_phys": int(mem_status.ullAvailPhys),
            "total_phys": int(mem_status.ullTotalPhys),
            "system_commit_total": int(mem_status.ullTotalPageFile),
            "system_commit_avail": int(mem_status.ullAvailPageFile),
            "memory_load_pct": int(mem_status.dwMemoryLoad),
        }
    except Exception:
        return None


def _log_memory(stage: str, **extra: Any) -> None:
    """Prints current process memory so log files reveal peak phases."""
    if not MEMORY_DEBUG:
        return
    snap = _get_memory_snapshot()
    if snap is None:
        print(f"[MEM] stage={stage} pid={os.getpid()} snapshot=unavailable", flush=True)
        return
    extra_text = " ".join(
        f"{key}={value}"
        for key, value in extra.items()
        if value is not None
    )
    sys_commit_used = snap["system_commit_total"] - snap["system_commit_avail"]
    parts = [
        f"stage={stage}",
        f"pid={snap['pid']}",
        f"rss={_format_gb(snap['working_set'])}",
        f"peak_rss={_format_gb(snap['peak_working_set'])}",
        f"private={_format_gb(snap['private_bytes'])}",
        f"commit={_format_gb(snap['commit_bytes'])}",
        f"peak_commit={_format_gb(snap['peak_commit_bytes'])}",
        f"avail_phys={_format_gb(snap['avail_phys'])}",
        f"sys_commit={_format_gb(sys_commit_used)}/{_format_gb(snap['system_commit_total'])}",
        f"mem_load={snap['memory_load_pct']}%",
    ]
    if extra_text:
        parts.append(extra_text)
    print("[MEM] " + " ".join(parts), flush=True)


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


def orient_normal_outward_from_center(
    normal: np.ndarray,
    face_points_xyz: np.ndarray,
    part_center_xyz: np.ndarray,
) -> np.ndarray:
    """Orients a face normal away from the part-level center."""
    n = np.asarray(normal, dtype=np.float32).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n = n / n_norm

    pts = np.asarray(face_points_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return n.astype(np.float32)

    face_center = np.mean(pts, axis=0)
    outward_hint = face_center - np.asarray(part_center_xyz, dtype=np.float32).reshape(3)
    if float(np.linalg.norm(outward_hint)) > 1e-9 and float(np.dot(n, outward_hint)) < 0.0:
        n = -n
    return n.astype(np.float32)


def _nx_point_xyz(point_obj) -> np.ndarray:
    c = point_obj.Coordinates
    return np.array([float(c.X), float(c.Y), float(c.Z)], dtype=np.float32)


def _nx_direction_xyz(direction_obj) -> np.ndarray:
    v = direction_obj.Vector
    return np.array([float(v.X), float(v.Y), float(v.Z)], dtype=np.float32)


def orient_sample_directions_and_lines_outward(
    work_part,
    points_array,
    norm_vecs_array,
    lines_array,
    part_center_xyz: np.ndarray,
) -> None:
    """Rebuilds probe directions/lines so each face points away from the part center."""
    center = np.asarray(part_center_xyz, dtype=np.float32).reshape(3)

    for face_idx, face_points in enumerate(points_array):
        if face_idx >= len(norm_vecs_array):
            continue
        face_norms = norm_vecs_array[face_idx]
        count = min(len(face_points), len(face_norms))
        if count <= 0:
            continue

        coords = np.stack([_nx_point_xyz(face_points[i]) for i in range(count)], axis=0)
        vectors = np.stack([_nx_direction_xyz(face_norms[i]) for i in range(count)], axis=0)
        mean_normal = vectors.mean(axis=0)
        n_norm = float(np.linalg.norm(mean_normal))
        outward_hint = coords.mean(axis=0) - center
        if n_norm <= 1e-9 or float(np.linalg.norm(outward_hint)) <= 1e-9:
            continue
        if float(np.dot(mean_normal / n_norm, outward_hint)) >= 0.0:
            continue

        new_norms = []
        new_lines = []
        for i in range(count):
            v = -vectors[i]
            v_norm = float(np.linalg.norm(v))
            if v_norm <= 1e-9:
                v = -mean_normal
                v_norm = float(np.linalg.norm(v))
            if v_norm <= 1e-9:
                v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                v_norm = 1.0
            v = v / v_norm
            direction = work_part.Directions.CreateDirection(
                NXOpen.Point3d(0.0, 0.0, 0.0),
                NXOpen.Vector3d(float(v[0]), float(v[1]), float(v[2])),
                NXOpen.SmartObject.UpdateOption.DontUpdate,
            )
            point_xyz = coords[i]
            new_norms.append(direction)
            new_lines.append(
                create_line_with_point_and_vector(
                    work_part,
                    [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])],
                    direction,
                    length=1000.0,
                )
            )

        norm_vecs_array[face_idx] = new_norms
        lines_array[face_idx] = new_lines


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
    current_volume: float,
    initial_volume: float,
    bbox_extent_xyz: np.ndarray,
    reference_scale: float,
) -> np.ndarray:
    """Builds a compact global context vector with coarse progress and part scale."""
    out = np.zeros((11,), dtype=np.float32)
    if 0 <= int(prev_macro_class_id) < len(MACRO_CLASS_TO_ID):
        out[int(prev_macro_class_id)] = 1.0
    init_vol = float(max(initial_volume, 1e-9))
    cur_vol = float(max(min(current_volume, init_vol), 0.0))
    out[5] = float(max(min((init_vol - cur_vol) / init_vol, 1.0), 0.0))  # cumulative removed ratio
    out[6] = float(max(min(cur_vol / init_vol, 1.0), 0.0))                # remaining volume ratio
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

    Indexed / 5-axis finishing -> selected outward face normal.
    """
    n = _normalize_vector(tuple(float(x) for x in target_face_normal))
    if n == (0.0, 0.0, 0.0):
        return (0.0, 0.0, 1.0)
    return n


def sample_valid_tools_for_macro(
    macro_class_name: str,
    max_tools: int = 3,
    rng: np.random.Generator | None = None,
) -> List[Tuple[str, float]]:
    """Randomly samples valid tool candidates for a macro class."""
    macro_id = MACRO_CLASS_TO_ID.get(macro_class_name, -1)
    if macro_id < 0:
        return []
    mask = build_tool_choice_mask_for_macro_class(macro_id)
    valid = [TOOL_LIBRARY[i] for i, m in enumerate(mask) if m == 0]
    if not valid:
        return []

    count = max(1, min(int(max_tools), len(valid)))
    if count >= len(valid):
        return list(valid)

    local_rng = rng if rng is not None else np.random.default_rng()
    sampled_idx = local_rng.choice(len(valid), size=count, replace=False)
    return [valid[int(i)] for i in sampled_idx]


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
    allow_finish: bool = True,
    rng: np.random.Generator | None = None,
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
            for tk, td in sample_valid_tools_for_macro("indexed_rough", max_tools_per_class, rng=rng):
                _add("indexed_rough", face_id, tk, td, axis)

    if allow_finish:
        for face_id in targets["finish"]:
            axis = derive_axis_direction("indexed_finish", normals[face_id])
            for tk, td in sample_valid_tools_for_macro("indexed_finish", max_tools_per_class, rng=rng):
                _add("indexed_finish", face_id, tk, td, axis)

            for tk, td in sample_valid_tools_for_macro("point_finish", max_tools_per_class, rng=rng):
                _add("point_finish", face_id, tk, td, axis)

            for tk, td in sample_valid_tools_for_macro("flank_finish", max_tools_per_class, rng=rng):
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
    precomputed_before_state: Dict[str, Any] | None = None,
    # ── Adaptive octree occupancy sampling ────────────────────────────────
    # Leaf centers are sampled from obj_a (post-op body) while it is still live
    # in the NX undo stack.
    octree_bbox_min_raw: "np.ndarray | None" = None,  # [3] world bbox lower corner
    octree_bbox_max_raw: "np.ndarray | None" = None,  # [3] world bbox upper corner
    octree_coarse_depth: int = 3,
    octree_fine_depth: int = 5,
    octree_max_nodes: int = 4096,
    octree_bbox_padding: float = 0.05,
    octree_enabled: bool = False,
    memory_log_label: str | None = None,
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
    if memory_log_label is not None:
        _log_memory(
            f"{memory_log_label}:start",
            macro=action.get("macro_class_name"),
            optype=optype,
            face=action_face_id,
            depth=operation_depth,
        )

    # ── Measure BEFORE ──
    pw_before = None
    if precomputed_before_state is not None:
        dev_before = np.asarray(precomputed_before_state["dev_raw"], dtype=np.float32).reshape(-1)
        vol_before = float(precomputed_before_state["volume"])
        dev_bef_red_512 = np.asarray(
            precomputed_before_state.get(
                "dev_red_512",
                pad_1d_float(
                    reduce_scalar_by_groups(dev_before, groups_face, weights=face_areas),
                    512,
                ),
            ),
            dtype=np.float32,
        ).reshape(512)
        if "point_sdf_before_512x100" in precomputed_before_state:
            pw_before = np.asarray(
                precomputed_before_state["point_sdf_before_512x100"],
                dtype=np.float32,
            ).reshape(512, 100)
    else:
        m_bef = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "sim_meas_bef")
        m_bef_chain_snapshot = list(state_chain)
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
    if memory_log_label is not None:
        _log_memory(f"{memory_log_label}:after_before_measure", volume_before=float(vol_before))
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

    local_target_faces = None
    if optype in {"Area Mill", "Cavity Mill", "Swarf Mill", "Point Mill"}:
        local_target_faces = _select_local_target_faces()
        if not local_target_faces:
            result["volume_after"] = float(vol_before)
            result["removed_volume"] = 0.0
            result["cycle_time"] = 0.0
            result["dev_after_red_512"] = dev_bef_red_512.copy()
            return result

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
        operations.create_surface_contour(
            session, work_part, tool, obj_op, local_target_faces, op_list, ct_list,
            tool_orientation=axis_dir,
        )
    elif optype == "Swarf Mill":
        operations.create_swarf_milling(
            session, work_part, tool, obj_op, local_target_faces, op_list, ct_list,
        )
    elif optype == "Point Mill":
        operations.create_point_milling(
            session, work_part, tool, obj_op, local_target_faces, op_list, ct_list,
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
        if memory_log_label is not None:
            _log_memory(f"{memory_log_label}:after_after_measure", volume_after=float(vol_after))

        # ── Adaptive octree occupancy target (inside try while obj_a is live) ──
        if octree_enabled and octree_bbox_min_raw is not None and octree_bbox_max_raw is not None:
            centers_raw, depths, labels, bbox_min_for_octree, bbox_max_for_octree = cam_utils.sample_ipw_octree_state(
                session=session,
                work_part=work_part,
                object_blank=obj_a,
                tool_name=flat_tools[0],
                bbox_min=np.asarray(octree_bbox_min_raw, dtype=np.float32).reshape(3),
                bbox_max=np.asarray(octree_bbox_max_raw, dtype=np.float32).reshape(3),
                coarse_depth=octree_coarse_depth,
                fine_depth=octree_fine_depth,
                max_nodes=octree_max_nodes,
                bbox_padding=octree_bbox_padding,
            )
            result["octree_centers_raw"] = centers_raw      # [K, 3] world mm
            result["octree_depths"] = depths                # [K] int16
            result["octree_occ_labels"] = labels            # [K] float32  (AFTER labels)
            result["octree_bbox_min_raw"] = bbox_min_for_octree.astype(np.float32)
            result["octree_bbox_max_raw"] = bbox_max_for_octree.astype(np.float32)
            if memory_log_label is not None:
                _log_memory(
                    f"{memory_log_label}:after_octree",
                    octree_nodes=int(labels.size),
                )
            if labels.size > 0:
                occ_ratio = float(np.mean(labels >= 0.5))
                if occ_ratio <= 0.001 or occ_ratio >= 0.999:
                    print(
                        "[WARN] simulate_single_action: octree occupancy is nearly constant "
                        f"(occ_ratio={occ_ratio:.4f}, K={int(labels.size)}). "
                        "Check IPW membership and bbox.",
                        flush=True,
                    )

            # ── Query BEFORE-state occupancy at the same cell positions ──────
            # Creates the before-operation IPW in a nested undo mark so it does
            # not interfere with obj_a (the after-state body still live here).
            # The before-labels enable the monotonicity training constraint.
            if centers_raw is not None and len(centers_raw) > 0:
                m_bef2 = session.SetUndoMark(
                    NXOpen.Session.MarkVisibility.Invisible, "sim_bef_occ"
                )
                bef2_chain_snap = list(state_chain)
                try:
                    obj_b_inner = create_measure_geometry_for_depth(
                        session, work_part, prt_file_path, state_chain, origin_body,
                        int(operation_depth), base_object_blank,
                    )
                    before_labels = cam_utils.query_ipw_occupancy_at_positions(
                        session=session,
                        work_part=work_part,
                        object_blank=obj_b_inner,
                        tool_name=flat_tools[0],
                        centers_xyz=centers_raw,
                    )
                    result["octree_occ_labels_before"] = before_labels  # [K] float32
                    if memory_log_label is not None:
                        _log_memory(
                            f"{memory_log_label}:after_before_occ",
                            before_nodes=int(before_labels.size),
                        )
                    before_ratio = float(np.mean(before_labels >= 0.5)) if before_labels.size > 0 else float("nan")
                    after_ratio  = float(np.mean(labels >= 0.5)) if labels.size > 0 else float("nan")
                    if before_ratio < after_ratio - 0.01:
                        print(
                            "[WARN] simulate_single_action: before-occ < after-occ "
                            f"(before={before_ratio:.4f}, after={after_ratio:.4f}). "
                            "Monotonicity violated in ground-truth — check IPW chain.",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"[WARN] simulate_single_action: before-occupancy query failed: {exc}",
                        flush=True,
                    )
                finally:
                    session.UndoToMark(m_bef2, "sim_bef_occ")
                    session.DeleteUndoMark(m_bef2, "sim_bef_occ")
                    restore_workpiece_chain(state_chain, bef2_chain_snap)

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
    """Returns top axis candidates from residual-heavy groups plus predefined fallback directions."""
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
        score = float(group_area_sum * remaining)
        scored_dirs.append((score, direction))

    scored_dirs.sort(key=lambda x: x[0], reverse=True)
    ordered_dirs = [d for _, d in scored_dirs]
    ordered_dirs = _deduplicate_directions(ordered_dirs, angle_tol_deg=2.0)

    for fallback_dir in get_predefined_axis_directions():
        if not any(_angle_between_vectors_deg(fallback_dir, d) <= 2.0 for d in ordered_dirs):
            ordered_dirs.append(fallback_dir)
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

def measure_current_state_for_branch(
    session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
    face_areas, points_array, norm_vecs_array, lines_array, flat_tools,
    workpiece_name_chain: List[str] | None = None,
    state_depth: int = 0,
    base_object_blank=None,
    capture_pointwise: bool = False,
    group_reference_points_512x100x3: np.ndarray | None = None,
    measurement_point_coords: List[np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Measures a branch state once and reuses it for all candidate actions.

    Candidate actions from the same branch share the same current IPW.  This
    cache avoids repeated GetInputIpw/deviation queries inside every candidate
    simulation while preserving the same raw and reduced state tensors.
    """
    _log_memory("branch_measure:start", depth=state_depth)
    if workpiece_name_chain is None:
        workpiece_name_chain = []
    m = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "measure_branch_state")
    chain_snapshot = list(workpiece_name_chain)
    try:
        obj_blank = create_measure_geometry_for_depth(
            session, work_part, prt_file_path, workpiece_name_chain, origin_body,
            int(state_depth), base_object_blank,
        )
        out: Dict[str, Any] = {}
        if capture_pointwise and group_reference_points_512x100x3 is not None and measurement_point_coords is not None:
            dev_raw, pointwise_raw, vol = cam_utils.measure_ipw_state_detailed(
                session,
                work_part,
                obj_blank,
                flat_tools[0],
                points_array,
                norm_vecs_array,
                lines_array,
            )
            dev_red = reduce_scalar_by_groups(np.asarray(dev_raw, dtype=np.float32), groups_face, weights=face_areas)
            dev_red_512 = pad_1d_float(dev_red, 512)
            out["point_sdf_before_512x100"] = build_group_point_sdf_512x100(
                group_reference_points_512x100x3,
                groups_face,
                measurement_point_coords,
                pointwise_raw,
                dev_red_512,
            ).astype(np.float32)
        else:
            dev_raw, vol = cam_utils.measure_ipw_state(
                session,
                work_part,
                obj_blank,
                flat_tools[0],
                points_array,
                norm_vecs_array,
                lines_array,
            )
            dev_red = reduce_scalar_by_groups(np.asarray(dev_raw, dtype=np.float32), groups_face, weights=face_areas)
            dev_red_512 = pad_1d_float(dev_red, 512)

        out.update({
            "volume": float(vol),
            "dev_raw": np.asarray(dev_raw, dtype=np.float32).reshape(-1),
            "dev_red_512": np.asarray(dev_red_512, dtype=np.float32).reshape(512),
        })
        _log_memory("branch_measure:end", depth=state_depth, volume=float(vol))
        return out
    finally:
        session.UndoToMark(m, "measure_branch_state")
        session.DeleteUndoMark(m, "measure_branch_state")
        restore_workpiece_chain(workpiece_name_chain, chain_snapshot)

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

def _compute_removed_ratio(current_volume: float, reference_volume: float) -> float:
    """Returns cumulative removed-material ratio w.r.t. the episode's initial stock."""
    ref = float(max(reference_volume, 1e-9))
    cur = float(current_volume)
    return float(max(min((ref - cur) / ref, 1.0), 0.0))

def _iter_row_batches(rows: List[Dict[str, Any]], batch_size: int):
    """Yields fixed-size row batches to cap peak memory during parquet writes."""
    batch_size = max(1, int(batch_size))
    total = len(rows)
    for start in range(0, total, batch_size):
        yield rows[start:start + batch_size]

def _serialize_for_parquet(value: Any) -> Any:
    """Converts numpy-heavy cells into parquet-friendly Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return [_serialize_for_parquet(v) for v in value]
    if isinstance(value, list):
        return [_serialize_for_parquet(v) for v in value]
    return value

def _make_parquet_stream_state(
    parquet_path: str,
    chosen_only_path: str | None = None,
) -> Dict[str, Any]:
    """Creates append-only parquet writer state for one episode."""
    _ensure_dir(os.path.dirname(parquet_path))
    if chosen_only_path:
        _ensure_dir(os.path.dirname(chosen_only_path))
    return {
        "parquet_path": parquet_path,
        "chosen_only_path": chosen_only_path,
        "writer": None,
        "chosen_writer": None,
        "schema": None,
        "engine": None,
        "pa": None,
        "pq": None,
        "fallback_rows": [],
        "fallback_chosen_rows": [],
        "num_rows": 0,
        "num_chosen_rows": 0,
    }

def _append_serialized_rows_to_parquet_stream(
    state: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    """Appends already-serialized row dicts to the episode parquet stream."""
    if not rows:
        return

    state["num_rows"] = int(state["num_rows"]) + int(len(rows))
    chosen_rows = [row for row in rows if int(row.get("is_chosen", 0)) == 1]
    state["num_chosen_rows"] = int(state["num_chosen_rows"]) + int(len(chosen_rows))

    if state["engine"] is None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            state["engine"] = "pyarrow"
            state["pa"] = pa
            state["pq"] = pq
        except Exception:
            state["engine"] = "fallback"

    if state["engine"] == "fallback":
        state["fallback_rows"].extend(rows)
        if state["chosen_only_path"]:
            state["fallback_chosen_rows"].extend(chosen_rows)
        return

    pa = state["pa"]
    pq = state["pq"]
    table = pa.Table.from_pylist(rows)
    if state["writer"] is None:
        state["schema"] = table.schema
        state["writer"] = pq.ParquetWriter(state["parquet_path"], state["schema"], compression="snappy")
    state["writer"].write_table(table, row_group_size=len(rows))
    del table

    if state["chosen_only_path"] and chosen_rows:
        chosen_table = pa.Table.from_pylist(chosen_rows, schema=state["schema"])
        if state["chosen_writer"] is None:
            state["chosen_writer"] = pq.ParquetWriter(state["chosen_only_path"], state["schema"], compression="snappy")
        state["chosen_writer"].write_table(chosen_table, row_group_size=len(chosen_rows))
        del chosen_table

    gc.collect()

def _close_parquet_stream(state: Dict[str, Any]) -> Tuple[int, int]:
    """Closes the parquet stream and writes any fallback buffers if needed."""
    try:
        if state["engine"] == "fallback":
            if state["fallback_rows"]:
                df = pd.DataFrame(state["fallback_rows"])
                df.to_parquet(state["parquet_path"], index=False)
            if state["chosen_only_path"]:
                chosen_df = pd.DataFrame(state["fallback_chosen_rows"])
                chosen_df.to_parquet(state["chosen_only_path"], index=False)
        elif state["engine"] == "pyarrow":
            if state["chosen_only_path"] and state["chosen_writer"] is None and state["schema"] is not None:
                empty_table = state["pa"].Table.from_batches([], schema=state["schema"])
                state["pq"].write_table(empty_table, state["chosen_only_path"], compression="snappy")
    finally:
        if state["writer"] is not None:
            state["writer"].close()
        if state["chosen_writer"] is not None:
            state["chosen_writer"].close()

    return int(state["num_rows"]), int(state["num_chosen_rows"])

# ----------------------------
# ----------------------------
def collect_dataset_episode(prt_file_path: str, out_root: str, seed: int = 0, global_parquet_dir: str = None):
    """Collects one full dataset episode for a single part file."""
    session, work_part = cam_session.create_session(input_file_dir=prt_file_path)
    theUfSession = NXOpen.UF.UFSession.GetUFSession()  # noqa: kept for legacy helpers
    _log_memory("episode:start", part=os.path.basename(prt_file_path), seed=int(seed))

    origin_body = max(work_part.Bodies, key=get_body_volume)
    origin_faces = origin_body.GetFaces()
    origin_faces_tag = [f.Tag for f in origin_faces]

    graph, _, face_areas, face_types = cam_utils.get_encoder_input_data(origin_faces, origin_faces_tag)
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
    face_type_512 = pad_1d_int(np.asarray(face_types, dtype=np.int16), 512)
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
    _log_memory("episode:after_static_prep", raw_faces=raw_face_count)

    mill_tool_types = ["MILL", "BALL_MILL"]
    flat_tools, ball_tools = [], []
    for d in [20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 4.0]:
        cam_utils.create_cam_tool(session, work_part, tool_diameter=d, tool_type=mill_tool_types[0], tool_list=flat_tools)
    for d in [8.0, 6.0, 4.0]:
        cam_utils.create_cam_tool(session, work_part, tool_diameter=d, tool_type=mill_tool_types[1], tool_list=ball_tools)

    part_name = os.path.splitext(os.path.basename(prt_file_path))[0]
    out_dir = create_run_output_dir(out_root, part_name, seed)
    run_name = os.path.basename(out_dir)
    parquet_path = ""
    chosen_parquet_path = ""
    num_rows = 0
    num_chosen_rows = 0
    episode_completed = False
    target_body_mesh_path = export_body_to_obj(
        session,
        work_part,
        origin_body,
        os.path.join(out_dir, "target_body.obj"),
    )

    # Rollout settings:
    # - fixed_steps: bounded horizon for quick data collection.
    # - until_done: continue until all branches are done, with safety caps.
    # - fixed_steps: bounded horizon; preferred for shape-transition dataset
    #   collection where full process completion is unnecessary.
    ROLLOUT_MODE = os.getenv("ROLLOUT_MODE", "fixed_steps").strip().lower()
    if ROLLOUT_MODE not in {"until_done", "fixed_steps"}:
        raise ValueError(f"Unsupported ROLLOUT_MODE: {ROLLOUT_MODE!r}")
    FIXED_DECISION_STEPS = int(os.getenv("FIXED_DECISION_STEPS", "3"))
    MAX_TOTAL_DECISION_STEPS = int(os.getenv("MAX_TOTAL_DECISION_STEPS", "64"))
    if FIXED_DECISION_STEPS <= 0 or MAX_TOTAL_DECISION_STEPS <= 0:
        raise ValueError("FIXED_DECISION_STEPS and MAX_TOTAL_DECISION_STEPS must be positive")
    EARLY_ROUGH_ONLY_STEPS = max(0, int(os.getenv("EARLY_ROUGH_ONLY_STEPS", "3")))
    MAX_NO_EFFECT_STREAK = max(0, int(os.getenv("MAX_NO_EFFECT_STREAK", "4")))
    NO_EFFECT_REMOVED_VOLUME_EPS = float(os.getenv("NO_EFFECT_REMOVED_VOLUME_EPS", "1e-4"))
    NO_EFFECT_REMOVED_RATIO_GAIN_EPS = float(
        os.getenv("NO_EFFECT_REMOVED_RATIO_GAIN_EPS", os.getenv("NO_EFFECT_DONE_GAIN_EPS", "1e-4"))
    )
    planned_decision_steps = (
        FIXED_DECISION_STEPS
        if ROLLOUT_MODE == "fixed_steps"
        else MAX_TOTAL_DECISION_STEPS
    )

    # Candidate generation / beam settings.
    # With FIXED_DECISION_STEPS=3 and BEAM_WIDTH=3 the expected row count is:
    #   step0: 1 branch × candidates  +
    #   step1: 3 branches × candidates  +
    #   step2: 3 branches × candidates
    # ≈ (1+3+3) × (MAX_ROUGH_TARGETS × MAX_TOOLS_PER_CLASS) ≈ 7 × 18 = 126 rows,
    # giving a comfortable buffer above the 100-row target even after NX failures.
    MAX_ROUGH_TARGETS = int(os.getenv("MAX_ROUGH_TARGETS", "6"))
    MAX_FINISH_TARGETS = int(os.getenv("MAX_FINISH_TARGETS", "5"))
    MAX_TOOLS_PER_CLASS = int(os.getenv("MAX_TOOLS_PER_CLASS", "3"))
    BEAM_WIDTH = max(1, int(os.getenv("BEAM_WIDTH", "3")))
    SAMPLED_BRANCHES = max(0, int(os.getenv("SAMPLED_BRANCHES", "1")))
    finish_ready_tol_default = float(globals().get("FINISH_READY_TOL", 1.0))
    FINISH_READY_TOL = float(os.getenv("FINISH_READY_TOL", str(finish_ready_tol_default)))
    SURFACE_FINISH_TOL = 0.01
    MIN_EFFECTIVE_REMOVED_VOLUME = float(os.getenv("MIN_EFFECTIVE_REMOVED_VOLUME", "0.1"))
    MIN_EFFECTIVE_REMOVED_RATIO = float(os.getenv("MIN_EFFECTIVE_REMOVED_RATIO", "0.0"))
    MIN_EFFECTIVE_MAX_NODE_DELTA = float(os.getenv("MIN_EFFECTIVE_MAX_NODE_DELTA", "0.0"))
    MIN_EFFECTIVE_REMOVED_RATIO_GAIN = float(
        os.getenv("MIN_EFFECTIVE_REMOVED_RATIO_GAIN", os.getenv("MIN_EFFECTIVE_DONE_GAIN", "0.0"))
    )
    FAIL_ON_CANDIDATE_ERROR = bool(int(os.getenv("FAIL_ON_CANDIDATE_ERROR", "0")))
    SAVE_PART_ON_CANDIDATE_ERROR = bool(int(os.getenv("SAVE_PART_ON_CANDIDATE_ERROR", "0")))
    # Save all simulated candidates (not just the chosen one) so that every
    # (before, operation, after) triple becomes a training sample, maximising
    # data yield per NX simulation run.
    SAVE_CHOSEN_ONLY_PARQUET = bool(int(os.getenv("SAVE_CHOSEN_ONLY_PARQUET", "0")))
    # Rows are appended in small finalized batches so parquet export never
    # needs the full episode table in memory. Keep the default conservative.
    PARQUET_WRITE_BATCH_ROWS = max(1, int(os.getenv("PARQUET_WRITE_BATCH_ROWS", "20")))
    ERROR_PART_SNAPSHOT_DIR = os.getenv("ERROR_PART_SNAPSHOT_DIR", "")

    def _is_effective_transition(
        removed: float,
        removed_ratio: float,
        max_delta: float,
        removed_ratio_gain: float,
    ) -> bool:
        if removed <= 1e-6 and max_delta <= 1e-6 and removed_ratio_gain <= 1e-9:
            return False
        checks = []
        if MIN_EFFECTIVE_REMOVED_VOLUME > 0.0:
            checks.append(removed >= MIN_EFFECTIVE_REMOVED_VOLUME)
        if MIN_EFFECTIVE_REMOVED_RATIO > 0.0:
            checks.append(removed_ratio >= MIN_EFFECTIVE_REMOVED_RATIO)
        if MIN_EFFECTIVE_MAX_NODE_DELTA > 0.0:
            checks.append(max_delta >= MIN_EFFECTIVE_MAX_NODE_DELTA)
        if MIN_EFFECTIVE_REMOVED_RATIO_GAIN > 0.0:
            checks.append(removed_ratio_gain >= MIN_EFFECTIVE_REMOVED_RATIO_GAIN)
        return any(checks) if checks else True

    def _save_candidate_error_part(t: int, branch_id: str, candidate_index: int, action: Dict[str, Any], exc: Exception) -> str:
        if not bool(SAVE_PART_ON_CANDIDATE_ERROR):
            return ""
        snapshot_dir = ERROR_PART_SNAPSHOT_DIR or os.path.join(out_dir, "error_prt_snapshots")
        _ensure_dir(snapshot_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tool_name = f"{action.get('tool_kind', 'tool')}_{action.get('tool_diameter', 'dia')}"
        filename = _safe_filename(
            f"{part_name}_seed{int(seed)}_t{int(t):02d}_b{branch_id}_c{int(candidate_index):03d}_"
            f"{action.get('macro_class_name', 'macro')}_{tool_name}_face{action.get('action_face_id', 'face')}_{stamp}.prt"
        )
        snapshot_path = os.path.abspath(os.path.join(snapshot_dir, filename))
        try:
            save_part(session, work_part, snapshot_path)
            print(f"[WARN] Saved error PRT snapshot: {snapshot_path}", flush=True)
            return snapshot_path
        except Exception as save_exc:
            print(
                f"[WARN] Failed to save error PRT snapshot for candidate error {exc!r}: {save_exc!r}",
                flush=True,
            )
            return ""

    def _save_fatal_error_part(label: str, exc: Exception) -> str:
        if not bool(SAVE_PART_ON_CANDIDATE_ERROR):
            return ""
        snapshot_dir = ERROR_PART_SNAPSHOT_DIR or os.path.join(out_dir, "error_prt_snapshots")
        _ensure_dir(snapshot_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = _safe_filename(f"{part_name}_seed{int(seed)}_fatal_{label}_{stamp}.prt")
        snapshot_path = os.path.abspath(os.path.join(snapshot_dir, filename))
        try:
            save_part(session, work_part, snapshot_path)
            print(f"[WARN] Saved fatal error PRT snapshot: {snapshot_path}", flush=True)
            return snapshot_path
        except Exception as save_exc:
            print(
                f"[WARN] Failed to save fatal error PRT snapshot for {exc!r}: {save_exc!r}",
                flush=True,
            )
            return ""

    node_mask_512 = build_node_mask_512(K)
    point_mask_512x100 = build_point_mask_512x100(face_pc_raw_512, K)
    measurement_point_coords = extract_point_coordinates(points_array)
    normalization_center_xyz, normalization_scale, bbox_extent_xyz = compute_reference_frame(
        face_pc_raw_512,
        node_mask_512,
    )
    orient_sample_directions_and_lines_outward(
        work_part,
        points_array,
        norm_vecs_array,
        lines_array,
        normalization_center_xyz,
    )

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
            face_points = measurement_point_coords[i] if i < len(measurement_point_coords) else np.empty((0, 3), dtype=np.float32)
            mean_normal = orient_normal_outward_from_center(
                mean_normal,
                face_points,
                normalization_center_xyz,
            )
            face_normals_list.append(mean_normal)
    face_normal_512 = build_group_face_normals_512(groups_face, face_normals_list, max_nodes=512)

    face_pc = normalize_face_points(face_pc_raw_512, normalization_center_xyz, normalization_scale)
    face_area = normalize_face_area(face_area_raw_512, normalization_scale)
    workpiece_name_chain: List[str] = []
    base_object_blank, _, _ = geometry.create_geometry(
        session, work_part, prt_file_path, workpiece_name_chain, origin_body, True, False,
    )

    _json_dump(os.path.join(out_dir, "meta.json"), {
        "prt_file_path": os.path.abspath(prt_file_path),
        "target_body_mesh_path": target_body_mesh_path,
        "part_name": part_name, "seed": int(seed), "K_raw_faces": int(K),
        "num_faces": int(len(origin_faces)),
        "note": {
            "mode": "process_skeleton_dataset",
            "row_unit": "one_executed_nx_operation",
            "planner_schema": "graph_sdf_process_planner",
            "candidate_strategy": "beam_sampled_action_transition",
            "decision_order": "macro_face_tool_then_axis_derived",
            "rollout": {
                "mode": str(ROLLOUT_MODE),
                "fixed_decision_steps": int(FIXED_DECISION_STEPS),
                "max_total_decision_steps": int(MAX_TOTAL_DECISION_STEPS),
                "early_rough_only_steps": int(EARLY_ROUGH_ONLY_STEPS),
                "beam_width": int(BEAM_WIDTH),
                "sampled_branches": int(SAMPLED_BRANCHES),
                "max_no_effect_streak": int(MAX_NO_EFFECT_STREAK),
                "no_effect_removed_volume_eps": float(NO_EFFECT_REMOVED_VOLUME_EPS),
                "no_effect_removed_ratio_gain_eps": float(NO_EFFECT_REMOVED_RATIO_GAIN_EPS),
            },
            "node_process_state_channels": ["rough_done", "finish_ready"],
            "face_type_schema": {
                "source": "NXOpen.Face.SolidFaceType.value with legacy blend/small-hole remapping",
                "padding": 0,
                "legacy_remap": {
                    "blend_or_type_ge_5": 5,
                    "small_circular_planar_face": 6,
                },
            },
            "global_process_state_channels": [
                "prev_macro_onehot_5",
                "cumulative_removed_ratio",
                "remaining_volume_ratio",
                "bbox_extent_over_scale_xyz",
                "log_ref_scale",
            ],
            "macro_classes_present": list(MACRO_CLASS_TO_ID.keys()),
            "normalization": {
                "center_xyz": normalization_center_xyz.tolist(),
                "reference_scale": float(normalization_scale),
                "bbox_extent_xyz": bbox_extent_xyz.tolist(),
            },
            "target_body_mesh_path": target_body_mesh_path,
            "effective_transition_filter": {
                "min_removed_volume": float(MIN_EFFECTIVE_REMOVED_VOLUME),
                "min_removed_ratio": float(MIN_EFFECTIVE_REMOVED_RATIO),
                "min_max_node_delta": float(MIN_EFFECTIVE_MAX_NODE_DELTA),
                "min_removed_ratio_gain": float(MIN_EFFECTIVE_REMOVED_RATIO_GAIN),
            },
            "fail_on_candidate_error": bool(FAIL_ON_CANDIDATE_ERROR),
            "save_part_on_candidate_error": bool(SAVE_PART_ON_CANDIDATE_ERROR),
            "error_part_snapshot_dir": os.path.abspath(ERROR_PART_SNAPSHOT_DIR or os.path.join(out_dir, "error_prt_snapshots")),
        },
    })
    _np_save(os.path.join(out_dir, "embed_centrality.npy"), centrality)
    _np_save(os.path.join(out_dir, "embed_spatial_pos.npy"), spatial_pos)
    _np_save(os.path.join(out_dir, "embed_face_area.npy"), face_area)
    _np_save(os.path.join(out_dir, "embed_face_type.npy"), face_type_512)
    _np_save(os.path.join(out_dir, "embed_face_pc.npy"), face_pc)

    rough_done_cumulative_512 = np.zeros((512,), dtype=np.int16)
    prev_macro_class_id = -1
    rough_rows_emitted = 0
    finish_rows_emitted = 0
    parquet_path = os.path.join(out_dir, f"{part_name}_seed{int(seed)}_process_skeleton_dataset.parquet")
    chosen_parquet_path = os.path.join(out_dir, f"{part_name}_seed{int(seed)}_process_skeleton_dataset_chosen_only.parquet")

    shared_info_json = json.dumps({
        "surface_finish_tol": float(SURFACE_FINISH_TOL),
        "candidate_strategy": "beam_sampled_action_transition",
        "normalization": "part_bbox_center_and_diagonal",
        "rough_done_delta_eps": float(ROUGH_DONE_DELTA_EPS),
        "finish_ready_tol": float(FINISH_READY_TOL),
        "min_effective_removed_volume": float(MIN_EFFECTIVE_REMOVED_VOLUME),
        "min_effective_removed_ratio": float(MIN_EFFECTIVE_REMOVED_RATIO),
        "min_effective_max_node_delta": float(MIN_EFFECTIVE_MAX_NODE_DELTA),
        "min_effective_removed_ratio_gain": float(MIN_EFFECTIVE_REMOVED_RATIO_GAIN),
        "fail_on_candidate_error": bool(FAIL_ON_CANDIDATE_ERROR),
        "save_part_on_candidate_error": bool(SAVE_PART_ON_CANDIDATE_ERROR),
        "error_part_snapshot_dir": os.path.abspath(ERROR_PART_SNAPSHOT_DIR or os.path.join(out_dir, "error_prt_snapshots")),
        "octree_enabled": bool(OCTREE_ENABLED),
        "octree_coarse_depth": int(OCTREE_COARSE_DEPTH),
        "octree_fine_depth": int(OCTREE_FINE_DEPTH),
        "octree_max_nodes": int(OCTREE_MAX_NODES),
        "octree_bbox_padding": float(OCTREE_BBOX_PADDING),
        "row_unit": "one_executed_nx_operation",
    }, ensure_ascii=False)
    shared_row_payload: Dict[str, Any] = {
        "part_name": part_name,
        "prt_file_path": os.path.abspath(prt_file_path),
        "target_body_mesh_path": target_body_mesh_path,
        "graph_nx_json": json.dumps(graph_json),
        "seed": int(seed),
        "node_mask": node_mask_512.astype(np.int16),
        "point_mask": point_mask_512x100.astype(np.int16),
        "centrality_512": np.asarray(centrality, dtype=np.int16),
        "spatial_pos_512x512": np.asarray(spatial_pos, dtype=np.int16),
        "face_area_512x1": np.asarray(face_area, dtype=np.float32),
        "face_type_512": np.asarray(face_type_512, dtype=np.int16),
        "normalization_center_xyz": normalization_center_xyz.astype(np.float32),
        "normalization_scale": float(normalization_scale),
        "bbox_extent_xyz": bbox_extent_xyz.astype(np.float32),
        "info_json": shared_info_json,
    }
    parquet_stream = _make_parquet_stream_state(
        parquet_path,
        chosen_only_path=chosen_parquet_path if bool(SAVE_CHOSEN_ONLY_PARQUET) else None,
    )
    pending_transition_rows: List[Dict[str, Any]] = []
    episode_record = {
        "part_name": part_name, "seed": int(seed),
        "num_decision_steps": int(planned_decision_steps),
        "rollout_mode": str(ROLLOUT_MODE),
        "fixed_decision_steps": int(FIXED_DECISION_STEPS),
        "max_total_decision_steps": int(MAX_TOTAL_DECISION_STEPS),
        "max_no_effect_streak": int(MAX_NO_EFFECT_STREAK),
        "no_effect_removed_volume_eps": float(NO_EFFECT_REMOVED_VOLUME_EPS),
        "no_effect_removed_ratio_gain_eps": float(NO_EFFECT_REMOVED_RATIO_GAIN_EPS),
        "surface_finish_tol": float(SURFACE_FINISH_TOL),
        "row_unit": "one_executed_nx_operation",
        "steps": [],
    }

    def _materialize_parquet_row(transition_row: Dict[str, Any]) -> Dict[str, Any]:
        """Builds the final parquet row only at flush time to avoid row duplication in memory."""
        state_point_sdf_raw = np.asarray(transition_row["state_point_sdf_raw_512x100"], dtype=np.float32)
        next_node_sdf_raw = np.asarray(transition_row["next_node_sdf_raw_512"], dtype=np.float32)
        next_point_sdf_raw = np.asarray(transition_row["next_point_sdf_raw_512x100"], dtype=np.float32)

        row = dict(shared_row_payload)
        row.update(transition_row)
        row["state_points"] = build_state_points_tensor(
            face_pc,
            face_normal_512,
            state_point_sdf_raw,
            normalization_scale,
        ).astype(np.float32)
        row["next_node_sdf"] = build_normalized_node_sdf_512x1(
            next_node_sdf_raw,
            normalization_scale,
        ).reshape(512).astype(np.float32)
        row["next_point_sdf"] = build_normalized_point_sdf_512x100(
            next_point_sdf_raw,
            normalization_scale,
        ).astype(np.float32)
        return {key: _serialize_for_parquet(value) for key, value in row.items()}

    def _flush_pending_transition_rows() -> None:
        """Flushes finalized transition rows and frees Python-side row memory."""
        nonlocal pending_transition_rows
        if not pending_transition_rows:
            return
        _log_memory("parquet_flush:start", pending_rows=len(pending_transition_rows))
        for batch_rows in _iter_row_batches(pending_transition_rows, PARQUET_WRITE_BATCH_ROWS):
            serialized_rows = [_materialize_parquet_row(row) for row in batch_rows]
            _append_serialized_rows_to_parquet_stream(parquet_stream, serialized_rows)
            del serialized_rows
            gc.collect()
        pending_transition_rows = []
        gc.collect()
        _log_memory("parquet_flush:end", pending_rows=0)

    def _build_transition_row(
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
        """Assembles one transition row payload; shared fields are added only on flush."""
        macro_class_name = action["macro_class_name"]
        macro_class_id = int(MACRO_CLASS_TO_ID[macro_class_name])
        tool_kind = action["tool_kind"]
        tool_diameter = float(action["tool_diameter"])
        tool_choice_name = tool_choice_key(tool_kind, tool_diameter)
        tool_choice_id = int(TOOL_CHOICE_TO_ID.get(tool_choice_name, -1))
        tool_choice_valid = int(tool_choice_id >= 0)
        axis_visible_512 = np.asarray(result["visible_512"], dtype=np.int16)

        next_node_sdf_raw = np.asarray(result["dev_after_red_512"], dtype=np.float32)

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

        removed_volume = float(result.get("removed_volume", 0.0) or 0.0)
        vol_before = float(result.get("volume_before", 0.0) or 0.0)
        axis_dir = tuple(action["axis_dir"])

        global_process_state = build_global_process_state(
            prev_macro_class_id=prev_macro_class_id_for_row,
            current_volume=vol_before,
            initial_volume=float(initial_stock_volume if initial_stock_volume is not None else max(vol_before, 1e-9)),
            bbox_extent_xyz=bbox_extent_xyz,
            reference_scale=normalization_scale,
        )

        # ── Adaptive octree occupancy ground truth ────────────────────────
        # Octree leaf centers were sampled inside simulate_single_action while
        # the post-operation NX body was still live.  Store normalized centers,
        # per-leaf depths, and binary next-state occupancy labels.
        octree_centers_norm: np.ndarray | None = None
        octree_depths: np.ndarray | None = None
        octree_occ_labels: np.ndarray | None = None
        octree_occ_labels_before: np.ndarray | None = None
        octree_bbox_min_norm: np.ndarray | None = None
        octree_bbox_max_norm: np.ndarray | None = None
        if OCTREE_ENABLED:
            raw_centers = result.get("octree_centers_raw")       # [K, 3] | None
            raw_depths = result.get("octree_depths")             # [K]    | None
            raw_labels = result.get("octree_occ_labels")         # [K]    | None  (after)
            raw_labels_before = result.get("octree_occ_labels_before")  # [K] | None (before)
            if raw_centers is not None and raw_depths is not None and raw_labels is not None:
                octree_scale = float(max(normalization_scale, 1e-6))
                raw_c = np.asarray(raw_centers, dtype=np.float32).reshape(-1, 3)
                octree_centers_norm = (raw_c - normalization_center_xyz.reshape(1, 3)) / octree_scale
                octree_depths = np.asarray(raw_depths, dtype=np.int16).reshape(-1)
                octree_occ_labels = np.asarray(raw_labels, dtype=np.float32).reshape(-1)
                if raw_labels_before is not None:
                    octree_occ_labels_before = np.asarray(raw_labels_before, dtype=np.float32).reshape(-1)
                raw_bbox_min = np.asarray(result.get("octree_bbox_min_raw", raw_c.min(axis=0)), dtype=np.float32).reshape(3)
                raw_bbox_max = np.asarray(result.get("octree_bbox_max_raw", raw_c.max(axis=0)), dtype=np.float32).reshape(3)
                octree_bbox_min_norm = (raw_bbox_min - normalization_center_xyz) / octree_scale
                octree_bbox_max_norm = (raw_bbox_max - normalization_center_xyz) / octree_scale

        return {
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

            "node_process_state": node_process_state.astype(np.float32),
            "global_process_state": global_process_state.astype(np.float32),
            "macro_class_mask": macro_class_mask.astype(np.int16),
            "tool_choice_mask": tool_choice_mask.astype(np.int16),
            "action_face_mask": action_face_mask_512.astype(np.int16),

            "axis_visible_512": axis_visible_512.astype(np.int16),
            "state_node_sdf_raw_512": np.asarray(state_node_sdf_raw, dtype=np.float32),
            "next_node_sdf_raw_512": next_node_sdf_raw.astype(np.float32),
            "state_point_sdf_raw_512x100": state_point_sdf_raw.astype(np.float32),
            "next_point_sdf_raw_512x100": next_point_sdf_raw.astype(np.float32),
            "state_done_mask_512": state_done_mask_512.astype(np.int16),
            "next_done_mask_512": next_done_mask_512.astype(np.int16),
            "rough_done_mask_512": rough_done_mask.astype(np.int16),
            "finish_ready_mask_512": finish_ready_mask.astype(np.int16),
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

            # ── Octree occupancy transition target
            "octree_centers": octree_centers_norm.reshape(-1).astype(np.float32) if octree_centers_norm is not None else None,
            "octree_depths": octree_depths.astype(np.int16) if octree_depths is not None else None,
            # After-operation occupancy (primary training target).
            "octree_occ_labels": octree_occ_labels.astype(np.float32) if octree_occ_labels is not None else None,
            # Before-operation occupancy at the same cell positions.
            # Used for the monotonicity training constraint:
            #   once a cell is empty (before=0) it must remain empty (after=0).
            "octree_occ_labels_before": octree_occ_labels_before.astype(np.float32) if octree_occ_labels_before is not None else None,
            "octree_bbox_min": octree_bbox_min_norm.astype(np.float32) if octree_bbox_min_norm is not None else None,
            "octree_bbox_max": octree_bbox_max_norm.astype(np.float32) if octree_bbox_max_norm is not None else None,
        }

    rng = np.random.default_rng(int(seed))

    root_chain_snapshot = list(workpiece_name_chain)
    beam_root_mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "beam_root")

    def _restore_to_root(strict: bool = True) -> bool:
        """Restores NX and the Python-side IPW chain to the initial branch root."""
        try:
            session.UndoToMark(beam_root_mark, "beam_root")
        except Exception as exc:
            if strict:
                raise
            print(f"[WARN] Could not restore beam_root undo mark: {exc!r}", flush=True)
            restore_workpiece_chain(workpiece_name_chain, root_chain_snapshot)
            return False
        restore_workpiece_chain(workpiece_name_chain, root_chain_snapshot)
        return True

    def _refresh_root_mark() -> None:
        """Drops the current root undo mark and recreates it at the clean root state."""
        nonlocal beam_root_mark
        try:
            session.DeleteUndoMark(beam_root_mark, "beam_root")
        except Exception as exc:
            print(f"[WARN] Could not delete beam_root undo mark during refresh: {exc!r}", flush=True)
        beam_root_mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "beam_root")

    def _reset_root_mark_to_initial_state() -> None:
        """Restores the initial state and recreates the root undo mark to cap undo growth."""
        _restore_to_root()
        _refresh_root_mark()

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
        t = 0
        no_effect_streak = 0
        initial_stock_volume = None
        last_selected_removed_ratio = None
        termination_reason = "max_total_steps"
        while t < planned_decision_steps:
            depth_children: List[Dict[str, Any]] = []
            depth_record: Dict[str, Any] = {
                "t": int(t),
                "num_input_branches": int(len(branches)),
                "branch_records": [],
            }
            done_branch_count = 0

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
                    _reset_root_mark_to_initial_state()
                    continue

                K_groups = len(groups_face)
                branch_before_state = measure_current_state_for_branch(
                    session, work_part, prt_file_path, origin_body, origin_faces, groups_face,
                    face_areas, points_array, norm_vecs_array, lines_array, flat_tools,
                    workpiece_name_chain=workpiece_name_chain,
                    state_depth=len(branch["history"]),
                    base_object_blank=base_object_blank,
                )
                state_volume = float(branch_before_state["volume"])
                if initial_stock_volume is None:
                    initial_stock_volume = float(max(state_volume, 1e-9))
                state_dev_red_512 = np.asarray(branch_before_state["dev_red_512"], dtype=np.float32).reshape(512)
                state_done_mask_512, all_done = compute_done_mask_from_dev_red_with_K(
                    state_dev_red_512, SURFACE_FINISH_TOL, K_groups,
                )
                state_done_ratio = float(state_done_mask_512[:K_groups].mean()) if K_groups > 0 else 1.0
                state_removed_ratio = _compute_removed_ratio(state_volume, initial_stock_volume)
                if all_done:
                    done_branch_count += 1
                    depth_record["branch_records"].append({
                        "scenario_id": str(branch["id"]),
                        "stopped": True,
                        "reason": "all_faces_done",
                        "state_volume": float(state_volume),
                        "state_removed_ratio": float(state_removed_ratio),
                        "state_done_ratio": float(state_done_ratio),
                    })
                    _reset_root_mark_to_initial_state()
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
                    allow_finish=bool(t >= EARLY_ROUGH_ONLY_STEPS),
                    rng=rng,
                )
                candidate_tools = sorted({
                    f"{c['tool_kind']}_{float(c['tool_diameter']):.1f}" for c in candidates
                })
                candidate_macros = sorted({str(c["macro_class_name"]) for c in candidates})
                print(
                    f"[INFO] t={t} branch={branch['id']}: {len(candidates)} candidates "
                    f"(removed={state_removed_ratio:.2f}, macros={candidate_macros}, tools={candidate_tools})",
                    flush=True,
                )
                _log_memory(
                    "branch:before_candidates",
                    t=int(t),
                    branch=str(branch["id"]),
                    candidates=int(len(candidates)),
                    removed_ratio=f"{state_removed_ratio:.4f}",
                )

                if not candidates:
                    depth_record["branch_records"].append({
                        "scenario_id": str(branch["id"]),
                        "stopped": True,
                        "reason": "no_candidates",
                        "state_volume": float(state_volume),
                        "state_removed_ratio": float(state_removed_ratio),
                        "state_done_ratio": float(state_done_ratio),
                    })
                    _reset_root_mark_to_initial_state()
                    continue

                # Branch-invariant octree bbox.  Candidate simulations can reuse
                # this instead of recomputing it for every candidate action.
                _oct_valid_pts = face_pc_raw_512[node_mask_512 == 0]  # [K_valid, P, 3]
                _oct_valid_pts_flat = _oct_valid_pts.reshape(-1, 3)
                _oct_raw_min = _oct_valid_pts_flat.min(axis=0)
                _oct_raw_max = _oct_valid_pts_flat.max(axis=0)
                _oct_pad = (_oct_raw_max - _oct_raw_min) * OCTREE_BBOX_PADDING
                _oct_raw_min = _oct_raw_min - _oct_pad
                _oct_raw_max = _oct_raw_max + _oct_pad

                scored: List[Tuple[float, int, Dict[str, Any], Dict[str, Any], int, Dict[str, Any]]] = []
                reject_stats = {"cam_error": 0, "no_effect": 0, "below_min_effect": 0}
                reject_examples: List[str] = []

                for ci, action in enumerate(candidates):
                    memory_log_label = None
                    if MEMORY_DEBUG and (ci % MEMORY_DEBUG_CANDIDATE_EVERY == 0 or ci == len(candidates) - 1):
                        memory_log_label = f"candidate:t{t}:b{branch['id']}:c{ci}"
                    cand_chain_snapshot = list(workpiece_name_chain)
                    cand_mark = session.SetUndoMark(
                        NXOpen.Session.MarkVisibility.Invisible,
                        f"beam_t{t:02d}_b{bi:03d}_c{ci:03d}",
                    )
                    cand_mark_name = f"beam_t{t:02d}_b{bi:03d}_c{ci:03d}"
                    candidate_error = None
                    candidate_error_snapshot_path = ""
                    rollback_error = None
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
                            precomputed_before_state=branch_before_state,
                            # Octree occupancy sampling
                            octree_bbox_min_raw=_oct_raw_min,
                            octree_bbox_max_raw=_oct_raw_max,
                            octree_coarse_depth=OCTREE_COARSE_DEPTH,
                            octree_fine_depth=OCTREE_FINE_DEPTH,
                            octree_max_nodes=OCTREE_MAX_NODES,
                            octree_bbox_padding=OCTREE_BBOX_PADDING,
                            octree_enabled=OCTREE_ENABLED,
                            memory_log_label=memory_log_label,
                        )
                    except Exception as exc:
                        candidate_error_snapshot_path = _save_candidate_error_part(
                            int(t), str(branch["id"]), int(ci), action, exc,
                        )
                        candidate_error = RuntimeError(
                            "simulate_single_action failed "
                            f"t={t} branch={branch['id']} candidate={ci} "
                            f"macro={action.get('macro_class_name')} "
                            f"tool={action.get('tool_kind')}_{action.get('tool_diameter')} "
                            f"face={action.get('action_face_id')} "
                            f"axis={action.get('axis_dir')} "
                            f"history_depth={len(branch['history'])} "
                            f"snapshot={candidate_error_snapshot_path or '<not_saved>'}"
                        )
                        candidate_error.__cause__ = exc
                    finally:
                        try:
                            session.UndoToMark(cand_mark, cand_mark_name)
                            session.DeleteUndoMark(cand_mark, cand_mark_name)
                        except Exception as exc:
                            rollback_error = exc
                            print(
                                f"[WARN] Candidate rollback failed: t={t} branch={branch['id']} "
                                f"candidate={ci} mark={cand_mark_name} err={exc!r}",
                                flush=True,
                            )
                        restore_workpiece_chain(workpiece_name_chain, cand_chain_snapshot)
                    if rollback_error is not None:
                        rollback_runtime_error = RuntimeError(
                            "candidate rollback failed after action simulation "
                            f"t={t} branch={branch['id']} candidate={ci} "
                            f"macro={action.get('macro_class_name')} "
                            f"tool={action.get('tool_kind')}_{action.get('tool_diameter')} "
                            f"face={action.get('action_face_id')} "
                            f"mark={cand_mark_name} "
                            f"snapshot={candidate_error_snapshot_path or '<not_saved>'}"
                        )
                        rollback_runtime_error.__cause__ = rollback_error
                        raise rollback_runtime_error
                    if candidate_error is not None:
                        if bool(FAIL_ON_CANDIDATE_ERROR):
                            raise candidate_error
                        reject_stats["cam_error"] += 1
                        if len(reject_examples) < 5:
                            reject_examples.append(
                                f"c{ci}:{action['macro_class_name']} "
                                f"{action['tool_kind']}_{float(action['tool_diameter']):.1f} "
                                f"face={action['action_face_id']} cam_error "
                                f"err={candidate_error.__cause__!r} "
                                f"snapshot={candidate_error_snapshot_path or '<not_saved>'}"
                            )
                        continue

                    removed = float(result.get("removed_volume", 0.0) or 0.0)
                    after_sdf = np.asarray(result["dev_after_red_512"], dtype=np.float32)
                    delta = np.maximum(state_dev_red_512 - after_sdf, 0.0)
                    max_delta = float(np.nanmax(delta)) if delta.size else 0.0
                    max_abs_delta = float(np.nanmax(np.abs(after_sdf - state_dev_red_512))) if after_sdf.size else 0.0
                    vol_before = float(result.get("volume_before", 1.0) or 1.0)
                    removed_ratio = removed / max(vol_before, 1e-9)
                    volume_after = float(result.get("volume_after", vol_before) or vol_before)
                    removed_ratio_after = _compute_removed_ratio(volume_after, initial_stock_volume)
                    removed_ratio_gain = max(float(removed_ratio_after - state_removed_ratio), 0.0)
                    next_done_mask, next_done_ratio = compute_done_mask_from_dev_red(
                        np.asarray(result["dev_after_red_512"], dtype=np.float32),
                        SURFACE_FINISH_TOL, node_mask_512=node_mask_512,
                    )
                    is_zero_effect = removed <= 1e-6 and max_delta <= 1e-6 and removed_ratio_gain <= 1e-9
                    is_effective = _is_effective_transition(removed, removed_ratio, max_delta, removed_ratio_gain)
                    if is_zero_effect or not is_effective:
                        stat_key = "no_effect" if is_zero_effect else "below_min_effect"
                        reject_stats[stat_key] += 1
                        if len(reject_examples) < 5:
                            reject_examples.append(
                                f"c{ci}:{action['macro_class_name']} "
                                f"{action['tool_kind']}_{float(action['tool_diameter']):.1f} "
                                f"face={action['action_face_id']} {stat_key} "
                                f"vol={float(result.get('volume_before', 0.0) or 0.0):.6f}"
                                f"->{float(result.get('volume_after', 0.0) or 0.0):.6f} "
                                f"removed={removed:.6g} ratio={removed_ratio:.6g} "
                                f"max_delta={max_delta:.6g} removed_gain={removed_ratio_gain:.6g} "
                                f"max_abs_delta={max_abs_delta:.6g} "
                                f"ct={float(result.get('cycle_time', 0.0) or 0.0):.6g}"
                            )
                        continue

                    cycle_time = float(result.get("cycle_time", 0.0) or 0.0)
                    action_score = 6.0 * removed_ratio_gain - 0.002 * cycle_time
                    child_score = float(branch["score"]) + float(action_score)
                    child_id = f"{branch['id']}.{t}.{ci}"
                    child_branch = _make_child_branch(
                        branch, action, result, state_dev_red_512, child_score, child_id,
                    )

                    row = _build_transition_row(
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
                    pending_transition_rows.append(row)
                    scored.append((child_score, ci, action, result, row, child_branch))

                branch_record: Dict[str, Any] = {
                    "scenario_id": str(branch["id"]),
                    "state_volume_before": float(state_volume),
                    "state_removed_ratio_before": float(state_removed_ratio),
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
                    for child_score, ci, action, result, row, child_branch in scored:
                        depth_children.append({
                            "score": float(child_score),
                            "row_ref": row,
                            "branch": child_branch,
                            "candidate_index": int(ci),
                            "macro": str(action["macro_class_name"]),
                            "tool": f"{action['tool_kind']}_{action['tool_diameter']}",
                            "action_face": int(action["action_face_id"]),
                            "removed_volume": float(result.get("removed_volume", 0.0) or 0.0),
                            "state_volume_after": float(result.get("volume_after", state_volume)),
                            "state_removed_ratio_after": float(
                                _compute_removed_ratio(
                                    float(result.get("volume_after", state_volume) or state_volume),
                                    initial_stock_volume,
                                )
                            ),
                            "state_done_ratio_after": float(
                                compute_done_mask_from_dev_red(
                                    np.asarray(result["dev_after_red_512"], dtype=np.float32),
                                    SURFACE_FINISH_TOL, node_mask_512=node_mask_512,
                                )[1]
                            ),
                        })

                _log_memory(
                    "branch:after_candidates",
                    t=int(t),
                    branch=str(branch["id"]),
                    effective=int(len(scored)),
                    pending_rows=int(len(pending_transition_rows)),
                )

                depth_record["branch_records"].append(branch_record)
                _reset_root_mark_to_initial_state()

            if not depth_children:
                reason = "all_faces_done" if done_branch_count >= len(branches) and len(branches) > 0 else "no_next_branches"
                depth_record.update({
                    "stopped": True,
                    "reason": reason,
                    "num_next_branches": 0,
                })
                episode_record["steps"].append(depth_record)
                termination_reason = reason
                break

            selected_children = _select_next_children(depth_children)
            next_branches: List[Dict[str, Any]] = []
            selected_records: List[Dict[str, Any]] = []
            for child in selected_children:
                child_row = child.get("row_ref")
                if child_row is not None:
                    child_row["is_chosen"] = 1
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
                    "state_removed_ratio_after": float(child["state_removed_ratio_after"]),
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
            selected_removed_mean = float(np.mean([float(x["removed_volume"]) for x in selected_children])) if selected_children else 0.0
            selected_removed_ratio_mean = float(np.mean([float(x["state_removed_ratio_after"]) for x in selected_children])) if selected_children else 0.0
            selected_done_ratio_mean = float(np.mean([float(x["state_done_ratio_after"]) for x in selected_children])) if selected_children else 0.0
            if last_selected_removed_ratio is None:
                selected_removed_ratio_gain = 0.0
            else:
                selected_removed_ratio_gain = float(selected_removed_ratio_mean - last_selected_removed_ratio)
            last_selected_removed_ratio = selected_removed_ratio_mean

            if (
                selected_removed_mean <= float(NO_EFFECT_REMOVED_VOLUME_EPS)
                and selected_removed_ratio_gain <= float(NO_EFFECT_REMOVED_RATIO_GAIN_EPS)
            ):
                no_effect_streak += 1
            else:
                no_effect_streak = 0
            depth_record["selected_removed_mean"] = float(selected_removed_mean)
            depth_record["selected_removed_ratio_mean"] = float(selected_removed_ratio_mean)
            depth_record["selected_removed_ratio_gain"] = float(selected_removed_ratio_gain)
            depth_record["selected_done_ratio_mean"] = float(selected_done_ratio_mean)
            depth_record["no_effect_streak"] = int(no_effect_streak)
            episode_record["steps"].append(depth_record)
            _flush_pending_transition_rows()

            if MAX_NO_EFFECT_STREAK > 0 and no_effect_streak >= MAX_NO_EFFECT_STREAK:
                termination_reason = "max_no_effect_streak"
                print(
                    f"[INFO] stop: t={t} no_effect_streak={no_effect_streak} "
                    f"(removed_mean={selected_removed_mean:.6g}, removed_ratio_gain={selected_removed_ratio_gain:.6g})",
                    flush=True,
                )
                break
            branches = next_branches
            t += 1

        episode_record["termination"] = {
            "reason": str(termination_reason),
            "executed_steps": int(len(episode_record["steps"])),
            "planned_max_steps": int(planned_decision_steps),
        }
        episode_completed = True

    except Exception as exc:
        _save_fatal_error_part("episode", exc)
        raise
    finally:
        _flush_pending_transition_rows()
        num_rows, num_chosen_rows = _close_parquet_stream(parquet_stream)
        if episode_completed:
            if num_rows > 0 and global_parquet_dir:
                _ensure_dir(global_parquet_dir)
                global_filename = f"{run_name}.parquet"
                global_path = os.path.join(global_parquet_dir, global_filename)
                shutil.copy2(parquet_path, global_path)
                print(f"[INFO] Copied parquet to: {global_path}")
                if bool(SAVE_CHOSEN_ONLY_PARQUET):
                    global_chosen_filename = f"{run_name}_chosen_only.parquet"
                    global_chosen_path = os.path.join(global_parquet_dir, global_chosen_filename)
                    shutil.copy2(chosen_parquet_path, global_chosen_path)
                    print(f"[INFO] Copied chosen-only parquet to: {global_chosen_path}")

            _json_dump(os.path.join(out_dir, "episode_record.json"), episode_record)
        _restore_to_root(strict=False)
        try:
            session.DeleteUndoMark(beam_root_mark, "beam_root")
        except Exception as exc:
            print(f"[WARN] Could not delete beam_root undo mark: {exc!r}", flush=True)
        force_close_part_by_path(prt_file_path)

    return {
        "out_dir": out_dir, "parquet_path": parquet_path,
        "chosen_parquet_path": chosen_parquet_path if bool(SAVE_CHOSEN_ONLY_PARQUET) else "",
        "num_rows": int(num_rows),
        "num_chosen_rows": int(num_chosen_rows),
        "num_operation_rows": int(num_rows),
        "num_decision_steps": int(len(episode_record["steps"])),
        "planned_max_steps": int(planned_decision_steps),
        "rollout_mode": str(ROLLOUT_MODE),
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
        default_base_dir = r"\\165.132.180.130\04_개별폴더\22. 통합과정 오인욱"
        prt_file_path = os.path.join(default_base_dir, "prt_dataset", "test_bracket_step.prt")
        out_root_dir = os.path.join(default_base_dir, "sdf_dataset_out")
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
