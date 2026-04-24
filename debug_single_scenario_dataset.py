"""Runs a single-scenario rollout for one PRT and saves debug artifacts."""
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import networkx as nxg
import numpy as np
import pandas as pd
import NXOpen

import collect_axis_dataset as cad

from CAM import geometry
from CAM import session as cam_session
from CAM import utils as cam_utils
from CAM.measurements import (
    get_body_volume,
    sample_convergent_face_points,
    sample_face_points,
)


def _create_run_output_dir(out_root: str, part_name: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"{part_name}_seed{int(seed)}_single_debug_{ts}")
    cad._ensure_dir(run_dir)
    return run_dir


def _bool_int(value: bool) -> int:
    return int(bool(value))


def _compute_effective_transition(
    action: Dict[str, Any],
    state_sdf_512: np.ndarray,
    next_sdf_512: np.ndarray,
    removed: float,
    removed_ratio: float,
    max_delta: float,
    removed_ratio_gain: float,
    min_removed_volume: float,
    min_removed_ratio: float,
    min_max_node_delta: float,
    min_removed_ratio_gain: float,
    min_finish_local_drop_ratio: float,
) -> Dict[str, Any]:
    return cad.evaluate_transition_effectiveness(
        action=action,
        before_sdf_512=state_sdf_512,
        after_sdf_512=next_sdf_512,
        removed=removed,
        removed_ratio=removed_ratio,
        max_delta=max_delta,
        removed_ratio_gain=removed_ratio_gain,
        min_removed_volume=min_removed_volume,
        min_removed_ratio=min_removed_ratio,
        min_max_node_delta=min_max_node_delta,
        min_removed_ratio_gain=min_removed_ratio_gain,
        min_finish_local_drop_ratio=min_finish_local_drop_ratio,
    )


def _candidate_priority_key(action: Dict[str, Any], state_node_sdf_raw: np.ndarray) -> tuple:
    macro_priority = {
        "indexed_rough": 0,
        "indexed_finish": 1,
        "point_finish": 2,
        "flank_finish": 3,
    }
    macro_name = str(action["macro_class_name"])
    face_id = int(action["action_face_id"])
    residual = 0.0
    state_sdf = np.asarray(state_node_sdf_raw, dtype=np.float32).reshape(-1)
    if 0 <= face_id < len(state_sdf):
        residual = float(state_sdf[face_id])
    tool_diameter = float(action["tool_diameter"])
    if macro_name == "indexed_rough":
        tool_rank = -tool_diameter
    else:
        tool_rank = tool_diameter
    return (
        int(macro_priority.get(macro_name, 99)),
        -residual,
        tool_rank,
        face_id,
        str(action["tool_kind"]),
    )


def _build_shared_row_payload(
    part_name: str,
    prt_file_path: str,
    target_body_mesh_path: str,
    graph_json: Dict[str, Any],
    seed: int,
    node_mask_512: np.ndarray,
    point_mask_512x100: np.ndarray,
    centrality: np.ndarray,
    spatial_pos: np.ndarray,
    face_area: np.ndarray,
    face_type_512: np.ndarray,
    normalization_center_xyz: np.ndarray,
    normalization_scale: float,
    bbox_extent_xyz: np.ndarray,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    shared_info_json = json.dumps(
        {
            "surface_finish_tol": float(settings["surface_finish_tol"]),
            "candidate_strategy": "single_scenario_best_only_debug",
            "normalization": "part_bbox_center_and_diagonal",
            "rough_done_delta_eps": float(cad.ROUGH_DONE_DELTA_EPS),
            "finish_ready_tol": float(settings["finish_ready_tol"]),
            "min_effective_removed_volume": float(settings["min_effective_removed_volume"]),
            "min_effective_removed_ratio": float(settings["min_effective_removed_ratio"]),
            "min_effective_max_node_delta": float(settings["min_effective_max_node_delta"]),
            "min_effective_removed_ratio_gain": float(settings["min_effective_removed_ratio_gain"]),
            "min_effective_finish_local_drop_ratio": float(settings["min_effective_finish_local_drop_ratio"]),
            "beam_width": 1,
            "sampled_branches": 0,
            "octree_enabled": bool(settings["octree_enabled"]),
            "octree_coarse_depth": int(settings["octree_coarse_depth"]),
            "octree_fine_depth": int(settings["octree_fine_depth"]),
            "octree_max_nodes": int(settings["octree_max_nodes"]),
            "octree_bbox_padding": float(settings["octree_bbox_padding"]),
            "row_unit": "one_executed_nx_operation",
        },
        ensure_ascii=False,
    )
    return {
        "part_name": part_name,
        "prt_file_path": os.path.abspath(prt_file_path),
        "target_body_mesh_path": target_body_mesh_path,
        "seed": int(seed),
        "static_feature_dir": os.path.abspath(os.path.dirname(target_body_mesh_path)),
        "graph_nx_json_path": os.path.abspath(os.path.join(os.path.dirname(target_body_mesh_path), "graph_nx.json")),
        "normalization_center_xyz": np.asarray(normalization_center_xyz, dtype=np.float32),
        "normalization_scale": float(normalization_scale),
        "bbox_extent_xyz": np.asarray(bbox_extent_xyz, dtype=np.float32),
        "info_json": shared_info_json,
    }


def _build_transition_row(
    shared_row_payload: Dict[str, Any],
    action: Dict[str, Any],
    result: Dict[str, Any],
    state_node_sdf_raw: np.ndarray,
    rough_done_mask_for_row: np.ndarray,
    prev_macro_class_id_for_row: int,
    target_node_id: int,
    decision_step: int,
    candidate_index: int,
    scenario_id: str,
    parent_scenario_id: str,
    node_mask_512: np.ndarray,
    normalization_center_xyz: np.ndarray,
    normalization_scale: float,
    bbox_extent_xyz: np.ndarray,
    surface_finish_tol: float,
    finish_ready_tol: float,
    initial_stock_volume: float,
    octree_enabled: bool,
) -> Dict[str, Any]:
    macro_class_name = str(action["macro_class_name"])
    macro_class_id = int(cad.MACRO_CLASS_TO_ID[macro_class_name])
    tool_kind = str(action["tool_kind"])
    tool_diameter = float(action["tool_diameter"])
    tool_choice_name = cad.tool_choice_key(tool_kind, tool_diameter)
    tool_choice_id = int(cad.TOOL_CHOICE_TO_ID.get(tool_choice_name, -1))
    axis_visible_512 = cad._as_i16(result["visible_512"])

    next_node_sdf_raw = cad._as_f32(result["dev_after_red_512"])
    state_done_mask_512, state_done_ratio = cad.compute_done_mask_from_dev_red(
        state_node_sdf_raw,
        surface_finish_tol,
        node_mask_512=node_mask_512,
    )
    next_done_mask_512, next_done_ratio = cad.compute_done_mask_from_dev_red(
        next_node_sdf_raw,
        surface_finish_tol,
        node_mask_512=node_mask_512,
    )
    rough_done_mask = cad._as_i16(rough_done_mask_for_row).copy()
    finish_ready_mask = cad.build_finish_ready_mask_512(
        node_sdf_512=state_node_sdf_raw,
        rough_done_mask_512=rough_done_mask,
        node_mask_512=node_mask_512,
        done_tol=surface_finish_tol,
        ready_tol=finish_ready_tol,
    )

    if cad.is_local_operation(action["optype"]):
        valid_target_mask = (
            (axis_visible_512 == 1)
            & (state_done_mask_512 == 0)
            & (rough_done_mask == 1)
            & (finish_ready_mask == 1)
            & (node_mask_512 == 0)
        ).astype(np.int16)
    else:
        valid_target_mask = (
            (state_done_mask_512 == 0)
            & (node_mask_512 == 0)
        ).astype(np.int16)
    action_face_mask_512 = (valid_target_mask == 0).astype(np.int16)

    macro_class_mask = cad.build_macro_class_mask_5(
        state_done_mask_512=state_done_mask_512,
        rough_done_mask_512=rough_done_mask,
        finish_ready_mask_512=finish_ready_mask,
        node_mask_512=node_mask_512,
    )
    macro_class_mask[macro_class_id] = 0

    tool_choice_mask = np.asarray(
        cad.build_tool_choice_mask_for_macro_class(macro_class_id),
        dtype=np.int16,
    )
    if 0 <= tool_choice_id < len(cad.TOOL_LIBRARY):
        tool_choice_mask[tool_choice_id] = 0

    node_process_state = np.stack(
        [rough_done_mask.astype(np.float32), finish_ready_mask.astype(np.float32)],
        axis=-1,
    )
    state_point_sdf_raw = cad._as_f32(
        result.get(
            "point_sdf_before_512x100",
            np.broadcast_to(cad._as_f32(state_node_sdf_raw).reshape(512, 1), (512, 100)),
        )
    )
    next_point_sdf_raw = cad._as_f32(
        result.get(
            "point_sdf_after_512x100",
            np.broadcast_to(next_node_sdf_raw.reshape(512, 1), (512, 100)),
        )
    )

    removed_volume = float(result.get("removed_volume", 0.0) or 0.0)
    vol_before = float(result.get("volume_before", 0.0) or 0.0)
    axis_dir = tuple(action["axis_dir"])
    global_process_state = cad.build_global_process_state(
        prev_macro_class_id=prev_macro_class_id_for_row,
        current_volume=vol_before,
        initial_volume=float(max(initial_stock_volume, 1e-9)),
        bbox_extent_xyz=bbox_extent_xyz,
        reference_scale=normalization_scale,
    )

    octree_centers_norm = None
    octree_depths = None
    octree_occ_labels = None
    octree_occ_labels_before = None
    octree_bbox_min_norm = None
    octree_bbox_max_norm = None
    if octree_enabled:
        raw_centers = result.get("octree_centers_raw")
        raw_depths = result.get("octree_depths")
        raw_labels = result.get("octree_occ_labels")
        raw_labels_before = result.get("octree_occ_labels_before")
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

    row = dict(shared_row_payload)
    row.update(
        {
            "decision_step": int(decision_step),
            "candidate_index": int(candidate_index),
            "is_chosen": 1,
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
            "tool_choice_valid": int(tool_choice_id >= 0),
            "node_process_state": cad._as_f32(node_process_state),
            "global_process_state": cad._as_f32(global_process_state),
            "macro_class_mask": cad._as_i16(macro_class_mask),
            "tool_choice_mask": cad._as_i16(tool_choice_mask),
            "action_face_mask": cad._as_i16(action_face_mask_512),
            "axis_visible_512": cad._as_i16(axis_visible_512),
            "state_node_sdf_raw_512": cad._as_f32(state_node_sdf_raw),
            "next_node_sdf_raw_512": cad._as_f32(next_node_sdf_raw),
            "state_point_sdf_raw_512x100": cad._as_f32(state_point_sdf_raw),
            "next_point_sdf_raw_512x100": cad._as_f32(next_point_sdf_raw),
            "state_done_mask_512": cad._as_i16(state_done_mask_512),
            "next_done_mask_512": cad._as_i16(next_done_mask_512),
            "rough_done_mask_512": cad._as_i16(rough_done_mask),
            "finish_ready_mask_512": cad._as_i16(finish_ready_mask),
            "state_done_ratio": cad._to_safe_float(state_done_ratio),
            "next_done_ratio": cad._to_safe_float(next_done_ratio),
            "axis_dir": list(map(float, axis_dir)),
            "operation_name": str(action["optype"]),
            "path_type": str(action.get("path_type", "FollowPart")),
            "tool_type_name": tool_kind,
            "tool_diameter": cad._to_safe_float(tool_diameter),
            "state_volume": cad._to_safe_float(vol_before),
            "next_state_volume": cad._to_safe_float(result.get("volume_after", vol_before)),
            "out_removed_volume": cad._to_safe_float(removed_volume),
            "out_removed_ratio": cad._to_safe_float(removed_volume / max(vol_before, 1e-9)),
            "out_cycle_time": cad._to_safe_float(result.get("cycle_time", 0.0)),
            "out_ok": bool(result.get("ok", True)),
            "octree_centers": cad._as_f32(octree_centers_norm).reshape(-1) if octree_centers_norm is not None else None,
            "octree_depths": cad._as_i16(octree_depths) if octree_depths is not None else None,
            "octree_occ_labels": cad._as_f32(octree_occ_labels) if octree_occ_labels is not None else None,
            "octree_occ_labels_before": cad._as_f32(octree_occ_labels_before) if octree_occ_labels_before is not None else None,
            "octree_bbox_min": cad._as_f32(octree_bbox_min_norm) if octree_bbox_min_norm is not None else None,
            "octree_bbox_max": cad._as_f32(octree_bbox_max_norm) if octree_bbox_max_norm is not None else None,
        }
    )
    return {key: cad._serialize_for_parquet(value) for key, value in row.items()}


def _append_finish_reason_rows(
    rows: List[Dict[str, Any]],
    decision_step: int,
    scenario_id: str,
    state_volume: float,
    state_removed_ratio: float,
    state_done_ratio: float,
    state_node_sdf_raw: np.ndarray,
    state_done_mask_512: np.ndarray,
    rough_done_mask_512: np.ndarray,
    finish_ready_mask_512: np.ndarray,
    node_mask_512: np.ndarray,
    face_normal_512: np.ndarray,
    surface_finish_tol: float,
    finish_ready_tol: float,
    raw_face_count: int,
) -> None:
    state_node_sdf_raw = np.asarray(state_node_sdf_raw, dtype=np.float32).reshape(-1)
    state_done_mask_512 = np.asarray(state_done_mask_512, dtype=np.int16).reshape(-1)
    rough_done_mask_512 = np.asarray(rough_done_mask_512, dtype=np.int16).reshape(-1)
    finish_ready_mask_512 = np.asarray(finish_ready_mask_512, dtype=np.int16).reshape(-1)
    node_mask_512 = np.asarray(node_mask_512, dtype=np.int16).reshape(-1)
    face_normal_512 = np.asarray(face_normal_512, dtype=np.float32).reshape(-1, 3)

    for face_id in range(int(raw_face_count)):
        residual = float(state_node_sdf_raw[face_id])
        node_valid = bool(node_mask_512[face_id] == 0)
        state_done = bool(state_done_mask_512[face_id] == 1)
        rough_done = bool(rough_done_mask_512[face_id] == 1)
        finish_ready = bool(finish_ready_mask_512[face_id] == 1)
        residual_gt_done_tol = bool(residual > float(surface_finish_tol))
        residual_le_ready_tol = bool(residual <= float(finish_ready_tol))
        rough_candidate = bool(node_valid and (not rough_done) and residual > 0.0)
        finish_candidate = bool(node_valid and (not state_done) and rough_done and finish_ready)
        reason_parts: List[str] = []
        if not node_valid:
            reason_parts.append("padded")
        if rough_done:
            reason_parts.append("rough_done")
        else:
            reason_parts.append("rough_pending")
        if state_done:
            reason_parts.append("state_done")
        else:
            reason_parts.append("state_not_done")
        if residual_gt_done_tol:
            reason_parts.append("residual_gt_done_tol")
        else:
            reason_parts.append("residual_le_done_tol")
        if residual_le_ready_tol:
            reason_parts.append("residual_le_finish_ready_tol")
        else:
            reason_parts.append("residual_gt_finish_ready_tol")
        if finish_ready:
            reason_parts.append("finish_ready")
        if finish_candidate:
            reason_parts.append("finish_candidate")
        rows.append(
            {
                "decision_step": int(decision_step),
                "scenario_id": str(scenario_id),
                "face_id": int(face_id),
                "state_volume": float(state_volume),
                "state_removed_ratio": float(state_removed_ratio),
                "state_done_ratio": float(state_done_ratio),
                "state_node_sdf_raw": residual,
                "node_valid": _bool_int(node_valid),
                "state_done": _bool_int(state_done),
                "rough_done": _bool_int(rough_done),
                "finish_ready": _bool_int(finish_ready),
                "rough_candidate": _bool_int(rough_candidate),
                "finish_candidate": _bool_int(finish_candidate),
                "residual_gt_done_tol": _bool_int(residual_gt_done_tol),
                "residual_le_finish_ready_tol": _bool_int(residual_le_ready_tol),
                "normal_x": float(face_normal_512[face_id, 0]),
                "normal_y": float(face_normal_512[face_id, 1]),
                "normal_z": float(face_normal_512[face_id, 2]),
                "reason": "|".join(reason_parts),
            }
        )


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    cad._ensure_dir(os.path.dirname(path))
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")


def collect_single_scenario_debug(prt_file_path: str, out_root: str, seed: int, settings: Dict[str, Any]) -> Dict[str, Any]:
    session = None
    parquet_stream = None
    work_part = None
    out_dir = ""
    parquet_path = ""
    candidate_csv_path = ""
    finish_csv_path = ""
    part_name = os.path.splitext(os.path.basename(prt_file_path))[0]
    candidate_debug_rows: List[Dict[str, Any]] = []
    finish_reason_rows: List[Dict[str, Any]] = []
    episode_record: Dict[str, Any] = {}
    num_rows = 0

    try:
        session, work_part = cam_session.create_session(input_file_dir=prt_file_path)
        cad._log_memory("single_debug:start", part=os.path.basename(prt_file_path), seed=int(seed))

        origin_body = max(work_part.Bodies, key=get_body_volume)
        origin_faces = origin_body.GetFaces()
        origin_faces_tag = [f.Tag for f in origin_faces]
        raw_face_count = int(len(origin_faces))
        if raw_face_count > 512:
            raise ValueError(f"Raw-face schema requires <=512 faces, but {prt_file_path} has {raw_face_count} faces.")

        graph, _, face_areas, face_types = cam_utils.get_encoder_input_data(origin_faces, origin_faces_tag)
        face_areas = np.asarray(face_areas, dtype=np.float32)
        _, points_array_origin = cam_utils.get_face_point_cloud(origin_faces)
        graph_dense = nxg.relabel_nodes(
            graph,
            {tag: idx for idx, tag in enumerate(origin_faces_tag)},
            copy=True,
        )
        graph_json = cad.serialize_graph_to_node_link(graph_dense)
        groups_face = [[idx] for idx in range(raw_face_count)]

        centrality = cad.pad_1d_int(np.array([graph_dense.degree(n) for n in range(raw_face_count)], dtype=np.int16), 512)
        spatial_pos = cad.pad_2d_int(cad.build_graph_distance_matrix(graph_dense).astype(np.int16), 512)
        face_area_raw_512 = cad.pad_face_area(face_areas.astype(np.float32), 512)
        face_type_512 = cad.pad_1d_int(np.asarray(face_types, dtype=np.int16), 512)
        face_pc_raw_512 = cad.pad_face_pc(np.asarray(points_array_origin, dtype=np.float32), 512)

        points_array, norm_vecs_array, lines_array = [], [], []
        for face in origin_faces:
            if face.SolidFaceType.value == 10:
                pts, norms, lines = sample_convergent_face_points(face)
            else:
                pts, norms, lines = sample_face_points(face)
            points_array.append(pts)
            norm_vecs_array.append(norms)
            lines_array.append(lines)
        cad._log_memory("single_debug:after_static_prep", raw_faces=raw_face_count)

        flat_tools: List[str] = []
        ball_tools: List[str] = []
        for diameter in [20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 4.0]:
            cam_utils.create_cam_tool(session, work_part, tool_diameter=diameter, tool_type="MILL", tool_list=flat_tools)
        for diameter in [8.0, 6.0, 4.0]:
            cam_utils.create_cam_tool(session, work_part, tool_diameter=diameter, tool_type="BALL_MILL", tool_list=ball_tools)

        out_dir = _create_run_output_dir(out_root, part_name, seed)
        target_body_mesh_path = cad.export_body_to_obj(
            session,
            work_part,
            origin_body,
            os.path.join(out_dir, "target_body.obj"),
        )

        node_mask_512 = cad.build_node_mask_512(raw_face_count)
        point_mask_512x100 = cad.build_point_mask_512x100(face_pc_raw_512, raw_face_count)
        measurement_point_coords = cad.extract_point_coordinates(points_array)
        normalization_center_xyz, normalization_scale, bbox_extent_xyz = cad.compute_reference_frame(
            face_pc_raw_512,
            node_mask_512,
        )
        cad.orient_sample_directions_and_lines_outward(
            work_part,
            points_array,
            norm_vecs_array,
            lines_array,
            normalization_center_xyz,
        )

        face_normals_list = []
        for idx, current_norms_raw in enumerate(norm_vecs_array):
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
                mean_normal = np.mean(clean_vectors, axis=0)
                norm = float(np.linalg.norm(mean_normal))
                if norm > 1e-9:
                    mean_normal = mean_normal / norm
                else:
                    mean_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            face_points = measurement_point_coords[idx] if idx < len(measurement_point_coords) else np.empty((0, 3), dtype=np.float32)
            mean_normal = cad.orient_normal_outward_from_center(
                mean_normal,
                face_points,
                normalization_center_xyz,
            )
            face_normals_list.append(mean_normal)
        face_normal_512 = cad.build_group_face_normals_512(groups_face, face_normals_list, max_nodes=512)

        face_pc = cad.normalize_face_points(face_pc_raw_512, normalization_center_xyz, normalization_scale)
        face_area = cad.normalize_face_area(face_area_raw_512, normalization_scale)
        workpiece_name_chain: List[str] = []
        base_object_blank, _, _ = geometry.create_geometry(
            session,
            work_part,
            prt_file_path,
            workpiece_name_chain,
            origin_body,
            True,
            False,
        )

        meta = {
            "prt_file_path": os.path.abspath(prt_file_path),
            "target_body_mesh_path": target_body_mesh_path,
            "part_name": part_name,
            "seed": int(seed),
            "num_faces": int(raw_face_count),
            "note": {
                "mode": "single_scenario_debug_dataset",
                "candidate_strategy": "single_scenario_best_only_debug",
                "beam_width": 1,
                "sampled_branches": 0,
                "settings": settings,
            },
        }
        cad._json_dump(os.path.join(out_dir, "meta.json"), meta)
        cad._np_save(os.path.join(out_dir, "embed_centrality.npy"), centrality)
        cad._np_save(os.path.join(out_dir, "embed_spatial_pos.npy"), spatial_pos)
        cad._np_save(os.path.join(out_dir, "embed_face_area.npy"), face_area)
        cad._np_save(os.path.join(out_dir, "embed_face_type.npy"), face_type_512)
        cad._np_save(os.path.join(out_dir, "embed_face_pc.npy"), face_pc)
        cad._np_save(os.path.join(out_dir, "embed_face_normal.npy"), face_normal_512)
        cad._np_save(os.path.join(out_dir, "embed_node_mask.npy"), node_mask_512)
        cad._np_save(os.path.join(out_dir, "embed_point_mask.npy"), point_mask_512x100)
        cad._json_dump(os.path.join(out_dir, "graph_nx.json"), graph_json)

        shared_row_payload = _build_shared_row_payload(
            part_name=part_name,
            prt_file_path=prt_file_path,
            target_body_mesh_path=target_body_mesh_path,
            graph_json=graph_json,
            seed=seed,
            node_mask_512=node_mask_512,
            point_mask_512x100=point_mask_512x100,
            centrality=centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            face_type_512=face_type_512,
            normalization_center_xyz=normalization_center_xyz,
            normalization_scale=normalization_scale,
            bbox_extent_xyz=bbox_extent_xyz,
            settings=settings,
        )

        parquet_path = os.path.join(out_dir, f"{part_name}_seed{int(seed)}_single_scenario_dataset.parquet")
        parquet_stream = cad._make_parquet_stream_state(parquet_path)
        candidate_csv_path = os.path.join(out_dir, "candidate_debug.csv")
        finish_csv_path = os.path.join(out_dir, "finish_face_debug.csv")

        episode_record = {
            "part_name": part_name,
            "seed": int(seed),
            "num_decision_steps": int(settings["fixed_decision_steps"]),
            "rollout_mode": "fixed_steps",
            "fixed_decision_steps": int(settings["fixed_decision_steps"]),
            "early_rough_only_steps": int(settings["early_rough_only_steps"]),
            "surface_finish_tol": float(settings["surface_finish_tol"]),
            "min_effective_finish_local_drop_ratio": float(settings["min_effective_finish_local_drop_ratio"]),
            "row_unit": "one_executed_nx_operation",
            "beam_width": 1,
            "sampled_branches": 0,
            "steps": [],
        }

        rng = np.random.default_rng(int(seed))
        rough_done_cumulative_512 = np.zeros((512,), dtype=np.int16)
        prev_macro_class_id = -1
        initial_stock_volume = None
        scenario_id = "s0"
        history: List[Dict[str, Any]] = []
        termination_reason = "fixed_steps_complete"

        for t in range(int(settings["fixed_decision_steps"])):
            before_state = cad.measure_current_state_for_branch(
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
                workpiece_name_chain=workpiece_name_chain,
                state_depth=len(history),
                base_object_blank=base_object_blank,
            )

            state_volume = float(before_state["volume"])
            if initial_stock_volume is None:
                initial_stock_volume = float(max(state_volume, 1e-9))
            state_dev_red_512 = np.asarray(before_state["dev_red_512"], dtype=np.float32).reshape(512)
            state_done_mask_512, state_done_ratio = cad.compute_done_mask_from_dev_red(
                state_dev_red_512,
                float(settings["surface_finish_tol"]),
                node_mask_512=node_mask_512,
            )
            _, all_done = cad.compute_done_mask_from_dev_red_with_K(
                state_dev_red_512,
                float(settings["surface_finish_tol"]),
                raw_face_count,
            )
            state_removed_ratio = cad._compute_removed_ratio(state_volume, initial_stock_volume)
            finish_ready_current_512 = cad.build_finish_ready_mask_512(
                node_sdf_512=state_dev_red_512,
                rough_done_mask_512=rough_done_cumulative_512,
                node_mask_512=node_mask_512,
                done_tol=float(settings["surface_finish_tol"]),
                ready_tol=float(settings["finish_ready_tol"]),
            )

            _append_finish_reason_rows(
                rows=finish_reason_rows,
                decision_step=t,
                scenario_id=scenario_id,
                state_volume=state_volume,
                state_removed_ratio=state_removed_ratio,
                state_done_ratio=state_done_ratio,
                state_node_sdf_raw=state_dev_red_512,
                state_done_mask_512=state_done_mask_512,
                rough_done_mask_512=rough_done_cumulative_512,
                finish_ready_mask_512=finish_ready_current_512,
                node_mask_512=node_mask_512,
                face_normal_512=face_normal_512,
                surface_finish_tol=float(settings["surface_finish_tol"]),
                finish_ready_tol=float(settings["finish_ready_tol"]),
                raw_face_count=raw_face_count,
            )

            if all_done:
                episode_record["steps"].append(
                    {
                        "t": int(t),
                        "scenario_id": str(scenario_id),
                        "stopped": True,
                        "reason": "all_faces_done",
                        "state_volume_before": float(state_volume),
                        "state_removed_ratio_before": float(state_removed_ratio),
                        "state_done_ratio_before": float(state_done_ratio),
                    }
                )
                termination_reason = "all_faces_done"
                break

            candidates = cad.generate_action_candidates(
                state_node_sdf_512=state_dev_red_512,
                rough_done_mask_512=rough_done_cumulative_512,
                finish_ready_mask_512=finish_ready_current_512,
                node_mask_512=node_mask_512,
                face_normal_512x3=face_normal_512,
                max_rough_targets=int(settings["max_rough_targets"]),
                max_finish_targets=int(settings["max_finish_targets"]),
                max_tools_per_class=int(settings["max_tools_per_class"]),
                allow_finish=bool(t >= int(settings["early_rough_only_steps"])),
                rng=rng,
            )
            candidate_tools = sorted({f"{c['tool_kind']}_{float(c['tool_diameter']):.1f}" for c in candidates})
            candidate_macros = sorted({str(c["macro_class_name"]) for c in candidates})

            step_record: Dict[str, Any] = {
                "t": int(t),
                "num_input_branches": 1,
                "branch_records": [
                    {
                        "scenario_id": str(scenario_id),
                        "state_volume_before": float(state_volume),
                        "state_removed_ratio_before": float(state_removed_ratio),
                        "state_done_ratio_before": float(state_done_ratio),
                        "num_candidates": int(len(candidates)),
                        "candidate_tools": candidate_tools,
                        "candidate_macros": candidate_macros,
                        "reject_stats": {"cam_error": 0, "no_effect": 0, "below_min_effect": 0},
                        "reject_examples": [],
                    }
                ],
                "num_effective_transitions": 0,
                "num_next_branches": 0,
                "beam_width": 1,
                "sampled_branches": 0,
                "selected_branches": [],
            }
            branch_record = step_record["branch_records"][0]

            if not candidates:
                branch_record["stopped"] = True
                branch_record["reason"] = "no_candidates"
                episode_record["steps"].append(step_record)
                termination_reason = "no_candidates"
                break

            valid_pts = face_pc_raw_512[node_mask_512 == 0].reshape(-1, 3)
            oct_raw_min = valid_pts.min(axis=0)
            oct_raw_max = valid_pts.max(axis=0)
            oct_pad = (oct_raw_max - oct_raw_min) * float(settings["octree_bbox_padding"])
            oct_raw_min = oct_raw_min - oct_pad
            oct_raw_max = oct_raw_max + oct_pad

            ranked_candidates = sorted(
                [
                    {
                        "candidate_index": int(ci),
                        "action": dict(action),
                    }
                    for ci, action in enumerate(candidates)
                ],
                key=lambda item: _candidate_priority_key(item["action"], state_dev_red_512),
            )
            branch_record["selection_mode"] = "single_candidate_priority"
            for rank, item in enumerate(ranked_candidates):
                action = item["action"]
                candidate_debug_rows.append(
                    {
                        "decision_step": int(t),
                        "scenario_id": str(scenario_id),
                        "candidate_index": int(item["candidate_index"]),
                        "selection_rank": int(rank),
                        "selected": int(rank == 0),
                        "simulated": int(rank == 0),
                        "macro_class_name": str(action["macro_class_name"]),
                        "tool_kind": str(action["tool_kind"]),
                        "tool_diameter": float(action["tool_diameter"]),
                        "action_face_id": int(action["action_face_id"]),
                        "operation_name": str(action["optype"]),
                        "state_volume_before": float(state_volume),
                        "state_removed_ratio_before": float(state_removed_ratio),
                    }
                )

            chosen_item = ranked_candidates[0]
            chosen_action = dict(chosen_item["action"])
            chosen_ci = int(chosen_item["candidate_index"])
            child_id = f"{scenario_id}.{t}.{chosen_ci}"
            chosen_result = None
            chosen_error_text = ""
            try:
                chosen_result = cad.simulate_single_action(
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
                    action=chosen_action,
                    surface_finish_tol=float(settings["surface_finish_tol"]),
                    workpiece_name_chain=workpiece_name_chain,
                    commit_to_chain=True,
                    operation_depth=len(history),
                    base_object_blank=base_object_blank,
                    precomputed_before_state=before_state,
                    octree_bbox_min_raw=oct_raw_min,
                    octree_bbox_max_raw=oct_raw_max,
                    octree_coarse_depth=int(settings["octree_coarse_depth"]),
                    octree_fine_depth=int(settings["octree_fine_depth"]),
                    octree_max_nodes=int(settings["octree_max_nodes"]),
                    octree_bbox_padding=float(settings["octree_bbox_padding"]),
                    octree_enabled=bool(settings["octree_enabled"]),
                )
            except Exception as exc:
                chosen_error_text = repr(exc)

            if chosen_result is None:
                branch_record["reject_stats"]["cam_error"] += 1
                branch_record["num_effective"] = 0
                branch_record["stopped"] = True
                branch_record["reason"] = "cam_error"
                if len(branch_record["reject_examples"]) < 5:
                    branch_record["reject_examples"].append(
                        f"c{chosen_ci}:{chosen_action['macro_class_name']} "
                        f"{chosen_action['tool_kind']}_{float(chosen_action['tool_diameter']):.1f} "
                        f"face={chosen_action['action_face_id']} cam_error err={chosen_error_text}"
                    )
                for row in reversed(candidate_debug_rows):
                    if int(row.get("decision_step", -1)) == int(t) and int(row.get("candidate_index", -1)) == int(chosen_ci):
                        row["accepted"] = 0
                        row["reject_reason"] = "cam_error"
                        row["error"] = chosen_error_text
                        break
                step_record["num_effective_transitions"] = 0
                episode_record["steps"].append(step_record)
                termination_reason = "cam_error"
                break

            removed = float(chosen_result.get("removed_volume", 0.0) or 0.0)
            next_sdf = np.asarray(chosen_result["dev_after_red_512"], dtype=np.float32)
            delta = np.maximum(state_dev_red_512 - next_sdf, 0.0)
            max_delta = float(np.nanmax(delta)) if delta.size else 0.0
            max_abs_delta = float(np.nanmax(np.abs(next_sdf - state_dev_red_512))) if next_sdf.size else 0.0
            vol_before = float(chosen_result.get("volume_before", 1.0) or 1.0)
            selected_volume_after = float(chosen_result.get("volume_after", vol_before) or vol_before)
            removed_ratio = removed / max(vol_before, 1e-9)
            selected_removed_ratio_after = cad._compute_removed_ratio(selected_volume_after, initial_stock_volume)
            removed_ratio_gain = max(float(selected_removed_ratio_after - state_removed_ratio), 0.0)
            _, selected_done_ratio = cad.compute_done_mask_from_dev_red(
                next_sdf,
                float(settings["surface_finish_tol"]),
                node_mask_512=node_mask_512,
            )
            effect_eval = _compute_effective_transition(
                action=chosen_action,
                state_sdf_512=state_dev_red_512,
                next_sdf_512=next_sdf,
                removed=removed,
                removed_ratio=removed_ratio,
                max_delta=max_delta,
                removed_ratio_gain=removed_ratio_gain,
                min_removed_volume=float(settings["min_effective_removed_volume"]),
                min_removed_ratio=float(settings["min_effective_removed_ratio"]),
                min_max_node_delta=float(settings["min_effective_max_node_delta"]),
                min_removed_ratio_gain=float(settings["min_effective_removed_ratio_gain"]),
                min_finish_local_drop_ratio=float(settings["min_effective_finish_local_drop_ratio"]),
            )
            is_zero_effect = bool(effect_eval["is_zero_effect"])
            is_effective = bool(effect_eval["is_effective"])
            action_score = float(6.0 * removed_ratio_gain - 0.002 * float(chosen_result.get("cycle_time", 0.0) or 0.0))
            chosen_reject_reason = ""
            if is_zero_effect:
                chosen_reject_reason = "no_effect"
                branch_record["reject_stats"]["no_effect"] += 1
            elif not is_effective:
                chosen_reject_reason = "below_min_effect"
                branch_record["reject_stats"]["below_min_effect"] += 1

            for row in reversed(candidate_debug_rows):
                if int(row.get("decision_step", -1)) == int(t) and int(row.get("candidate_index", -1)) == int(chosen_ci):
                    row["next_state_volume"] = float(selected_volume_after)
                    row["removed_volume"] = float(removed)
                    row["removed_ratio"] = float(removed_ratio)
                    row["removed_ratio_after"] = float(selected_removed_ratio_after)
                    row["removed_ratio_gain"] = float(removed_ratio_gain)
                    row["max_delta"] = float(max_delta)
                    row["max_abs_delta"] = float(max_abs_delta)
                    row["state_done_ratio_before"] = float(state_done_ratio)
                    row["next_done_ratio"] = float(selected_done_ratio)
                    row["cycle_time"] = float(chosen_result.get("cycle_time", 0.0) or 0.0)
                    row["local_face_before"] = float(effect_eval["local_face_before"])
                    row["local_face_after"] = float(effect_eval["local_face_after"])
                    row["local_face_drop"] = float(effect_eval["local_face_drop"])
                    row["local_face_drop_ratio"] = float(effect_eval["local_face_drop_ratio"])
                    row["effect_criterion"] = str(effect_eval["criterion"])
                    row["accepted"] = int(chosen_reject_reason == "")
                    row["reject_reason"] = chosen_reject_reason
                    row["score"] = float(action_score)
                    break

            if chosen_reject_reason:
                branch_record["num_effective"] = 0
                step_record["num_effective_transitions"] = 0
                branch_record["stopped"] = True
                branch_record["reason"] = "no_effective_candidates"
                if len(branch_record["reject_examples"]) < 5:
                    branch_record["reject_examples"].append(
                        f"c{chosen_ci}:{chosen_action['macro_class_name']} "
                        f"{chosen_action['tool_kind']}_{float(chosen_action['tool_diameter']):.1f} "
                        f"face={chosen_action['action_face_id']} {chosen_reject_reason} "
                        f"vol={vol_before:.6f}->{selected_volume_after:.6f} "
                        f"removed={removed:.6g} ratio={removed_ratio:.6g} "
                        f"max_delta={max_delta:.6g} removed_gain={removed_ratio_gain:.6g} "
                        f"local_sdf={float(effect_eval['local_face_before']):.6g}"
                        f"->{float(effect_eval['local_face_after']):.6g} "
                        f"local_drop={float(effect_eval['local_face_drop']):.6g} "
                        f"local_drop_ratio={float(effect_eval['local_face_drop_ratio']):.6g} "
                        f"criterion={effect_eval['criterion']} "
                        f"max_abs_delta={max_abs_delta:.6g} "
                        f"ct={float(chosen_result.get('cycle_time', 0.0) or 0.0):.6g}"
                    )
                episode_record["steps"].append(step_record)
                termination_reason = "no_effective_candidates"
                break

            branch_record["num_effective"] = 1
            step_record["num_effective_transitions"] = 1

            serialized_row = _build_transition_row(
                shared_row_payload=shared_row_payload,
                action=chosen_action,
                result=chosen_result,
                state_node_sdf_raw=state_dev_red_512,
                rough_done_mask_for_row=rough_done_cumulative_512,
                prev_macro_class_id_for_row=prev_macro_class_id,
                target_node_id=int(chosen_action["action_face_id"]),
                decision_step=t,
                candidate_index=chosen_ci,
                scenario_id=child_id,
                parent_scenario_id=scenario_id,
                node_mask_512=node_mask_512,
                normalization_center_xyz=normalization_center_xyz,
                normalization_scale=normalization_scale,
                bbox_extent_xyz=bbox_extent_xyz,
                surface_finish_tol=float(settings["surface_finish_tol"]),
                finish_ready_tol=float(settings["finish_ready_tol"]),
                initial_stock_volume=float(initial_stock_volume),
                octree_enabled=bool(settings["octree_enabled"]),
            )
            cad._append_serialized_rows_to_parquet_stream(parquet_stream, [serialized_row])

            next_sdf = np.asarray(chosen_result["dev_after_red_512"], dtype=np.float32)
            delta = np.maximum(state_dev_red_512 - next_sdf, 0.0)
            macro_class_name = str(chosen_action["macro_class_name"])
            if macro_class_name == "indexed_rough":
                rough_impacted = (
                    (delta > float(cad.ROUGH_DONE_DELTA_EPS))
                    & (node_mask_512 == 0)
                ).astype(np.int16)
                rough_done_cumulative_512 = np.maximum(rough_done_cumulative_512, rough_impacted)
            prev_macro_class_id = int(cad.MACRO_CLASS_TO_ID[macro_class_name])

            selected_volume_after = float(chosen_result.get("volume_after", state_volume) or state_volume)
            selected_removed_ratio_after = cad._compute_removed_ratio(selected_volume_after, initial_stock_volume)
            _, selected_done_ratio = cad.compute_done_mask_from_dev_red(
                next_sdf,
                float(settings["surface_finish_tol"]),
                node_mask_512=node_mask_512,
            )

            branch_record["best_candidate_index"] = int(chosen_ci)
            branch_record["best_macro"] = str(chosen_action["macro_class_name"])
            branch_record["best_tool"] = f"{chosen_action['tool_kind']}_{float(chosen_action['tool_diameter']):.1f}"
            branch_record["best_action_face"] = int(chosen_action["action_face_id"])
            branch_record["best_score"] = float(action_score)
            step_record["num_next_branches"] = 1
            step_record["selected_branches"] = [
                {
                    "scenario_id": str(child_id),
                    "parent_scenario_id": str(scenario_id),
                    "candidate_index": int(chosen_ci),
                    "macro": str(chosen_action["macro_class_name"]),
                    "tool": f"{chosen_action['tool_kind']}_{float(chosen_action['tool_diameter']):.1f}",
                    "action_face": int(chosen_action["action_face_id"]),
                    "score": float(action_score),
                    "state_volume_after": float(selected_volume_after),
                    "state_removed_ratio_after": float(selected_removed_ratio_after),
                    "state_done_ratio_after": float(selected_done_ratio),
                }
            ]
            step_record["selected_removed_mean"] = float(chosen_result.get("removed_volume", 0.0) or 0.0)
            step_record["selected_removed_ratio_mean"] = float(selected_removed_ratio_after)
            step_record["selected_removed_ratio_gain"] = float(max(selected_removed_ratio_after - state_removed_ratio, 0.0))
            step_record["selected_done_ratio_mean"] = float(selected_done_ratio)
            episode_record["steps"].append(step_record)

            scenario_id = child_id
            history.append(dict(chosen_action))

        episode_record["termination"] = {
            "reason": str(termination_reason),
            "executed_steps": int(len(episode_record["steps"])),
            "planned_max_steps": int(settings["fixed_decision_steps"]),
        }
        cad._json_dump(os.path.join(out_dir, "episode_record.json"), episode_record)
        _write_csv(candidate_csv_path, candidate_debug_rows)
        _write_csv(finish_csv_path, finish_reason_rows)

    finally:
        if parquet_stream is not None:
            num_rows, _ = cad._close_parquet_stream(parquet_stream)
        if out_dir:
            if candidate_csv_path:
                _write_csv(candidate_csv_path, candidate_debug_rows)
            if finish_csv_path:
                _write_csv(finish_csv_path, finish_reason_rows)
            if episode_record:
                cad._json_dump(os.path.join(out_dir, "episode_record.json"), episode_record)
        if session is not None and prt_file_path:
            cad.force_close_part_by_path(prt_file_path)

    return {
        "out_dir": out_dir,
        "parquet_path": parquet_path,
        "candidate_csv_path": candidate_csv_path,
        "finish_csv_path": finish_csv_path,
        "num_rows": int(num_rows),
        "num_decision_steps": int(len(episode_record.get("steps", []))),
        "termination_reason": str(episode_record.get("termination", {}).get("reason", "")),
    }

def main() -> int:
    # Edit these values before running.
    prt_path = r"Y:\04_개별폴더\22. 통합과정 오인욱\prt_dataset\3dDataset0000.prt"
    out_root = r"C:\Users\inwoo\Desktop"
    seed = 0

    prt_path = os.path.abspath(prt_path)
    out_root = os.path.abspath(out_root)
    if not prt_path or prt_path == os.path.abspath(""):
        print("[Error] Set `prt_path` in main() before running.", file=sys.stderr)
        return 1
    if not out_root or out_root == os.path.abspath(""):
        print("[Error] Set `out_root` in main() before running.", file=sys.stderr)
        return 1
    if not os.path.isfile(prt_path):
        print(f"[Error] PRT file not found: {prt_path}", file=sys.stderr)
        return 1
    cad._ensure_dir(out_root)

    settings = {
        "fixed_decision_steps": 6,
        "early_rough_only_steps": 3,
        "max_rough_targets": 3,
        "max_finish_targets": 2,
        "max_tools_per_class": 2,
        "surface_finish_tol": 0.01,
        "finish_ready_tol": float(cad.FINISH_READY_TOL),
        "min_effective_removed_volume": 0.1,
        "min_effective_removed_ratio": 0.0,
        "min_effective_max_node_delta": 0.0,
        "min_effective_removed_ratio_gain": 0.0,
        "min_effective_finish_local_drop_ratio": float(os.getenv("MIN_EFFECTIVE_FINISH_LOCAL_DROP_RATIO", "0.05")),
        "octree_enabled": bool(cad.OCTREE_ENABLED),
        "octree_coarse_depth": int(cad.OCTREE_COARSE_DEPTH),
        "octree_fine_depth": int(cad.OCTREE_FINE_DEPTH),
        "octree_max_nodes": int(cad.OCTREE_MAX_NODES),
        "octree_bbox_padding": float(cad.OCTREE_BBOX_PADDING),
    }
    try:
        result = collect_single_scenario_debug(
            prt_file_path=prt_path,
            out_root=out_root,
            seed=int(seed),
            settings=settings,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[Critical Error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
