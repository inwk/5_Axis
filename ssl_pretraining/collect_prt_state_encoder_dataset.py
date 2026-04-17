"""Build a prt-only StateEncoder SSL parquet dataset without CAM simulation.

This extractor opens each .prt in NX, reads only CAD face-graph features, and
writes one SSL row per part.  It does not create CAM operations, IPW bodies,
toolpaths, octree occupancy labels, or SDF transition labels.

Edit the constants below and run from NX/Open Python or VSCode attached to NX.
No CLI arguments are required.
"""

from __future__ import annotations

from datetime import datetime
import glob
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import pandas as pd
import NXOpen
import NXOpen.UF


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CAM.measurements import getFaceArea, get_body_volume


# NX stdout/stderr can raise OSError(22) on flush in debug sessions.
class _NXSafeStream:
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


if not isinstance(sys.stdout, _NXSafeStream):
    sys.stdout = _NXSafeStream(sys.stdout)
if not isinstance(sys.stderr, _NXSafeStream):
    sys.stderr = _NXSafeStream(sys.stderr)


# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode.
# ---------------------------------------------------------------------------
PRT_DIR = r""
PRT_GLOB = "*.prt"
EXPLICIT_PRT_PATHS: list[str] = []

OUTPUT_DIR = r"C:\Users\inwoo\Desktop\5_Axis\ssl_state_encoder_dataset"
OUTPUT_PARQUET = r""  # If empty, a timestamped parquet is written under OUTPUT_DIR.
MAX_PARTS = 0  # 0 means all resolved parts.
FAIL_ON_ERROR = False

MAX_NODES = 512
POINTS_PER_FACE = 100


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_serializable_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _resolve_prt_files() -> list[str]:
    paths: list[str] = []
    if EXPLICIT_PRT_PATHS:
        paths.extend(str(Path(p).expanduser().resolve()) for p in EXPLICIT_PRT_PATHS if str(p).strip())
    elif PRT_DIR:
        paths.extend(str(Path(p).resolve()) for p in sorted(glob.glob(os.path.join(PRT_DIR, PRT_GLOB))))
    else:
        raise ValueError("Set either EXPLICIT_PRT_PATHS or PRT_DIR at the top of collect_prt_state_encoder_dataset.py")

    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PRT file not found: {path}")
        unique.append(path)
    if MAX_PARTS > 0:
        unique = unique[: int(MAX_PARTS)]
    if not unique:
        raise ValueError("No .prt files matched the configured paths.")
    return unique


def _open_part(prt_path: str):
    session = NXOpen.Session.GetSession()
    _, load_status = session.Parts.OpenActiveDisplay(prt_path, NXOpen.DisplayPartOption.AllowAdditional)
    load_status.Dispose()
    return session, session.Parts.Work


def _force_close_part_by_path(target_path: str) -> None:
    session = NXOpen.Session.GetSession()
    uf = NXOpen.UF.UFSession.GetUFSession()
    for part in list(session.Parts):
        if part.FullPath and part.FullPath.lower() == target_path.lower():
            uf.Part.Close(part.Tag, 0, 1)


def _resample_points(points: np.ndarray, target_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] <= 0:
        return np.zeros((target_count, 3), dtype=np.float32)
    if points.shape[0] == target_count:
        return points.astype(np.float32)
    if points.shape[0] > target_count:
        idx = np.linspace(0, points.shape[0] - 1, target_count, dtype=np.int64)
        return points[idx].astype(np.float32)
    reps = int(math.ceil(target_count / max(points.shape[0], 1)))
    tiled = np.tile(points, (reps, 1))
    return tiled[:target_count].astype(np.float32)


def _unit(v: np.ndarray, fallback: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.asarray(fallback, dtype=np.float32)
    return (v / n).astype(np.float32)


def _sample_regular_face_xyz_and_normal(face, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    uf = NXOpen.UF.UFSession.GetUFSession()
    points = []
    normals = []

    try:
        for edge in face.GetEdges():
            for vertex in edge.GetVertices():
                points.append([float(vertex.X), float(vertex.Y), float(vertex.Z)])
    except Exception:
        pass

    try:
        uv = uf.Modeling.AskFaceUvMinmax(face.Tag)
        side = max(2, int(math.ceil(math.sqrt(sample_count))))
        us = np.linspace(float(uv[0]), float(uv[1]), side)
        vs = np.linspace(float(uv[2]), float(uv[3]), side)
        for u in us:
            for v in vs:
                point, _, _, _, _, unit_norm, _ = uf.Modeling.AskFaceProps(face.Tag, [float(u), float(v)])
                points.append([float(point[0]), float(point[1]), float(point[2])])
                normals.append([float(unit_norm[0]), float(unit_norm[1]), float(unit_norm[2])])
    except Exception:
        pass

    points_arr = _resample_points(np.asarray(points, dtype=np.float32), sample_count)
    if normals:
        normal = _unit(np.mean(np.asarray(normals, dtype=np.float32), axis=0))
    else:
        try:
            _, _, direction_data, _, _, _, _ = uf.Modeling.AskFaceData(face.Tag)
            normal = _unit(np.asarray(direction_data, dtype=np.float32))
        except Exception:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return points_arr, normal


def _sample_convergent_face_xyz_and_normal(face, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    points = []
    normals = []
    try:
        facet = face.GetFirstFacetOnFace()
        for _ in range(int(face.GetNumberOfFacets())):
            if facet is None:
                break
            vertices = facet.GetVertices()
            for vertex in vertices:
                points.append([float(vertex.X), float(vertex.Y), float(vertex.Z)])
            vec = NXOpen.ConvergentFacet.GetUnitNormal(facet)
            normals.append([float(vec.X), float(vec.Y), float(vec.Z)])
            facet = face.GetNextFacet(facet)
    except Exception:
        pass
    points_arr = _resample_points(np.asarray(points, dtype=np.float32), sample_count)
    normal = _unit(np.mean(np.asarray(normals, dtype=np.float32), axis=0)) if normals else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return points_arr, normal


def _sample_face_xyz_and_normal(face, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    if int(face.SolidFaceType.value) == 10:
        return _sample_convergent_face_xyz_and_normal(face, sample_count)
    return _sample_regular_face_xyz_and_normal(face, sample_count)


def _legacy_face_type(face, face_area: float) -> int:
    face_type = int(face.SolidFaceType.value)
    try:
        _, is_blended = face.GetBlendData()
    except Exception:
        is_blended = False
    if face_type == 10:
        return face_type
    if is_blended or face_type >= 5:
        return 5
    if face_type == 1:
        try:
            if len(face.GetEdges()) == 1 and float(face_area) <= math.pi * math.pow(5.1, 2):
                return 6
        except Exception:
            pass
    return face_type


def _build_graph(faces, face_tags: list[int]) -> nx.Graph:
    uf = NXOpen.UF.UFSession.GetUFSession()
    tag_set = set(face_tags)
    graph = nx.Graph()
    graph.add_nodes_from(face_tags)
    for face_tag in face_tags:
        try:
            for adj_tag in uf.Modeling.AskAdjacFaces(face_tag):
                if adj_tag in tag_set:
                    graph.add_edge(face_tag, adj_tag)
        except Exception:
            pass
    return nx.relabel_nodes(graph, {tag: idx for idx, tag in enumerate(face_tags)}, copy=True)


def _build_graph_distance_matrix(graph: nx.Graph, max_nodes: int) -> np.ndarray:
    nodes = list(graph.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    dist = np.full((max_nodes, max_nodes), -1, dtype=np.int16)
    for src in nodes:
        for dst, d in nx.shortest_path_length(graph, source=src).items():
            dist[index[src], index[dst]] = int(d)
    return dist


def _node_mask(num_valid_nodes: int, max_nodes: int) -> np.ndarray:
    mask = np.ones((max_nodes,), dtype=np.int16)
    mask[: min(num_valid_nodes, max_nodes)] = 0
    return mask


def _point_mask(face_pc_raw: np.ndarray, num_valid_nodes: int) -> np.ndarray:
    mask = np.ones((MAX_NODES, POINTS_PER_FACE), dtype=np.int16)
    valid_nodes = min(num_valid_nodes, MAX_NODES)
    if valid_nodes <= 0:
        return mask
    point_valid = np.any(np.abs(face_pc_raw[:valid_nodes]) > 1e-12, axis=-1)
    mask[:valid_nodes, :] = (~point_valid).astype(np.int16)
    return mask


def _compute_reference_frame(face_pc_raw: np.ndarray, node_mask: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    valid_nodes = np.asarray(node_mask, dtype=np.int16) == 0
    valid_points = np.asarray(face_pc_raw, dtype=np.float32)[valid_nodes].reshape(-1, 3)
    if valid_points.size == 0:
        return np.zeros((3,), dtype=np.float32), 1.0, np.ones((3,), dtype=np.float32)
    bbox_min = valid_points.min(axis=0)
    bbox_max = valid_points.max(axis=0)
    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    extent = np.maximum(bbox_max - bbox_min, 1e-6).astype(np.float32)
    scale = float(max(np.linalg.norm(extent), 1e-6))
    return center, scale, extent


def _orient_normals_outward(normals: np.ndarray, face_pc_raw: np.ndarray, center: np.ndarray, node_count: int) -> np.ndarray:
    out = np.asarray(normals, dtype=np.float32).copy()
    for idx in range(min(node_count, out.shape[0])):
        n = _unit(out[idx])
        face_center = np.mean(face_pc_raw[idx], axis=0)
        hint = face_center - center
        if float(np.linalg.norm(hint)) > 1e-9 and float(np.dot(n, hint)) < 0.0:
            n = -n
        out[idx] = n
    return out


def _build_state_points(face_pc_norm: np.ndarray, normals: np.ndarray) -> np.ndarray:
    state = np.zeros((MAX_NODES, POINTS_PER_FACE, 7), dtype=np.float32)
    state[:, :, 0:3] = face_pc_norm.astype(np.float32)
    state[:, :, 3:6] = np.broadcast_to(normals.astype(np.float32)[:, None, :], (MAX_NODES, POINTS_PER_FACE, 3))
    # state[..., 6] is current SDF/residual.  PRT-only SSL has no IPW state, so it is zero-filled.
    return state


def _collect_one_part(prt_path: str) -> dict[str, Any]:
    session, work_part = _open_part(prt_path)
    mark = session.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "ssl_prt_extract")
    try:
        origin_body = max(work_part.Bodies, key=get_body_volume)
        origin_faces = list(origin_body.GetFaces())
        raw_face_count = int(len(origin_faces))
        if raw_face_count > MAX_NODES:
            raise ValueError(f"Raw-face schema requires <= {MAX_NODES} faces, got {raw_face_count}: {prt_path}")

        face_tags = [face.Tag for face in origin_faces]
        graph = _build_graph(origin_faces, face_tags)
        graph_json = json_graph.node_link_data(graph)

        face_pc_raw = np.zeros((MAX_NODES, POINTS_PER_FACE, 3), dtype=np.float32)
        face_normals = np.zeros((MAX_NODES, 3), dtype=np.float32)
        face_area = np.zeros((MAX_NODES, 1), dtype=np.float32)
        face_type = np.zeros((MAX_NODES,), dtype=np.int16)

        for idx, face in enumerate(origin_faces):
            area = float(getFaceArea(face))
            points, normal = _sample_face_xyz_and_normal(face, POINTS_PER_FACE)
            face_pc_raw[idx] = points
            face_normals[idx] = normal
            face_area[idx, 0] = area
            face_type[idx] = int(_legacy_face_type(face, area))

        node_mask = _node_mask(raw_face_count, MAX_NODES)
        point_mask = _point_mask(face_pc_raw, raw_face_count)
        center, scale, extent = _compute_reference_frame(face_pc_raw, node_mask)
        face_normals = _orient_normals_outward(face_normals, face_pc_raw, center, raw_face_count)
        face_pc_norm = ((face_pc_raw - center.reshape(1, 1, 3)) / float(max(scale, 1e-6))).astype(np.float32)
        face_area_norm = (face_area / (float(max(scale, 1e-6)) ** 2)).astype(np.float32)
        state_points = _build_state_points(face_pc_norm, face_normals)

        centrality = np.zeros((MAX_NODES,), dtype=np.int16)
        for node in graph.nodes():
            if int(node) < MAX_NODES:
                centrality[int(node)] = int(graph.degree(node))
        spatial_pos = _build_graph_distance_matrix(graph, MAX_NODES)

        part_name = os.path.splitext(os.path.basename(prt_path))[0]
        return {
            "part_name": part_name,
            "prt_file_path": os.path.abspath(prt_path),
            "graph_nx_json": json.dumps(graph_json),
            "state_points": _to_serializable_list(state_points),
            "node_process_state": _to_serializable_list(np.zeros((MAX_NODES, 2), dtype=np.float32)),
            "node_mask": _to_serializable_list(node_mask),
            "point_mask": _to_serializable_list(point_mask),
            "centrality_512": _to_serializable_list(centrality),
            "spatial_pos_512x512": _to_serializable_list(spatial_pos),
            "face_area_512x1": _to_serializable_list(face_area_norm),
            "face_type_512": _to_serializable_list(face_type),
            "normalization_center_xyz": _to_serializable_list(center.astype(np.float32)),
            "normalization_scale": float(scale),
            "bbox_extent_xyz": _to_serializable_list(extent.astype(np.float32)),
            "info_json": json.dumps(
                {
                    "mode": "prt_only_state_encoder_ssl",
                    "row_unit": "one_part",
                    "state_points_channels": ["x", "y", "z", "nx", "ny", "nz", "current_sdf_zero_filled"],
                    "face_type_schema": {
                        "source": "NXOpen.Face.SolidFaceType.value with legacy blend/small-hole remapping",
                        "padding": 0,
                        "legacy_remap": {
                            "blend_or_type_ge_5": 5,
                            "small_circular_planar_face": 6,
                        },
                    },
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
            ),
        }
    finally:
        try:
            session.UndoToMark(mark, "ssl_prt_extract")
            session.DeleteUndoMark(mark, "ssl_prt_extract")
        except Exception:
            pass
        _force_close_part_by_path(os.path.abspath(prt_path))


def main() -> None:
    prt_files = _resolve_prt_files()
    _ensure_dir(OUTPUT_DIR)
    output_path = OUTPUT_PARQUET.strip()
    if not output_path:
        output_path = os.path.join(
            OUTPUT_DIR,
            f"state_encoder_ssl_prt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
        )
    output_path = os.path.abspath(output_path)
    _ensure_dir(os.path.dirname(output_path))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    print(f"[INFO] PRT files={len(prt_files)}")
    print(f"[INFO] output={output_path}")
    for idx, prt_path in enumerate(prt_files, start=1):
        try:
            row = _collect_one_part(prt_path)
            rows.append(row)
            print(f"[{idx}/{len(prt_files)}] ok part={row['part_name']} faces={int(np.sum(np.asarray(row['node_mask']) == 0))}")
        except Exception as exc:
            errors.append({"prt_file_path": prt_path, "error": repr(exc)})
            print(f"[WARN] failed {prt_path}: {exc!r}")
            if FAIL_ON_ERROR:
                raise

    if not rows:
        raise RuntimeError(f"No rows collected. errors={errors[:5]}")

    pd.DataFrame(rows).to_parquet(output_path, index=False)
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "output_parquet": output_path,
                "num_rows": len(rows),
                "num_errors": len(errors),
                "errors": errors,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] parquet saved: {output_path}")
    print(f"[OK] summary saved: {summary_path}")


if __name__ == "__main__":
    main()

