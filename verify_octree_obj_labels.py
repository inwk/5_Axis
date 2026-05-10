"""Verify parquet octree labels against exported s_t / s_t+1 OBJ meshes.

Edit the constants below and run directly from VS Code. This script does not
require NX; it recomputes point-in-mesh occupancy with trimesh-compatible OBJ
files already exported by collect_axis_dataset.py.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
PARQUET_PATH = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\_ALL_PARQUET_FILES\3dDataset2451_seed0_20260508_222741.parquet"
PARQUET_DIR = r""
PARQUET_GLOB = "*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []

OUTPUT_DIR = r"octree_obj_label_audits"
MAX_FILES = 1           # 0 = all
MAX_ROWS_PER_FILE = 0   # 0 = all rows in each file
MAX_ROWS_TOTAL = 100    # 0 = all matched rows; keep small because OBJ checks are expensive
ROW_INDEX_FILTER: list[int] = []

# Use this to skip tiny/no-op transitions when inspecting manually.
MIN_REMOVED_VOLUME = 0.0

# Same bounded-memory ray-parity mode used by CAM/utils.py by default.
RAY_POINT_CHUNK = int(os.getenv("OCTREE_RAY_POINT_CHUNK", "32"))
RAY_TRI_CHUNK = int(os.getenv("OCTREE_RAY_TRI_CHUNK", "1024"))


WANTED_COLUMNS = (
    "part_name",
    "decision_step",
    "candidate_index",
    "scenario_id",
    "macro_class_name",
    "tool_choice_name",
    "action_face_id",
    "out_removed_volume",
    "out_removed_ratio",
    "state_volume",
    "next_state_volume",
    "normalization_center_xyz",
    "normalization_scale",
    "octree_centers",
    "octree_occ_labels_before",
    "octree_occ_labels",
    "state_obj_path",
    "next_state_obj_path",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _array(value: Any, dtype) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype).reshape(-1)
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
        if not chunks:
            return np.asarray([], dtype=dtype)
        return np.concatenate(chunks).astype(dtype, copy=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _available_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return set(pd.read_parquet(path, columns=[]).columns)


def _resolve_parquet_files() -> list[Path]:
    if EXPLICIT_PARQUET_PATHS:
        files = [Path(p).expanduser().resolve() for p in EXPLICIT_PARQUET_PATHS if str(p).strip()]
    elif PARQUET_PATH:
        files = [Path(PARQUET_PATH).expanduser().resolve()]
    elif PARQUET_DIR:
        files = sorted(Path(PARQUET_DIR).expanduser().resolve().glob(PARQUET_GLOB))
    else:
        raise ValueError("Set PARQUET_PATH, PARQUET_DIR, or EXPLICIT_PARQUET_PATHS.")

    files = [path for path in files if path.is_file()]
    if MAX_FILES > 0:
        files = files[: int(MAX_FILES)]
    if not files:
        raise ValueError("No parquet files found.")
    return files


def _read_frame(path: Path) -> pd.DataFrame:
    available = _available_columns(path)
    required = {
        "normalization_center_xyz",
        "normalization_scale",
        "octree_centers",
        "octree_occ_labels_before",
        "octree_occ_labels",
        "state_obj_path",
        "next_state_obj_path",
    }
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    columns = [name for name in WANTED_COLUMNS if name in available]
    df = pd.read_parquet(path, columns=columns)
    if MAX_ROWS_PER_FILE > 0:
        df = df.iloc[: int(MAX_ROWS_PER_FILE)]
    return df


def _load_obj_as_trimesh(obj_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"OBJ did not load as trimesh.Trimesh: {obj_path}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"OBJ mesh is empty: {obj_path}")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    if mesh.faces.size:
        tri = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
        area2 = np.linalg.norm(
            np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :]),
            axis=1,
        )
        valid = np.isfinite(area2) & (area2 > 1e-12)
        if not np.all(valid):
            mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float64),
                faces=np.asarray(mesh.faces, dtype=np.int64)[valid],
                process=False,
            )
    mesh.remove_unreferenced_vertices()
    return mesh


@lru_cache(maxsize=4)
def _cached_load_mesh(path: str) -> trimesh.Trimesh:
    return _load_obj_as_trimesh(path)


def _contains_points_ray_parity(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.zeros((points.shape[0],), dtype=bool)
    if points.size == 0:
        return labels

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if triangles.size == 0:
        return labels

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    eps = 1e-9
    inside_bbox = np.all(points >= (bounds[0] - eps), axis=1) & np.all(points <= (bounds[1] + eps), axis=1)
    active_indices = np.flatnonzero(inside_bbox)
    if active_indices.size == 0:
        return labels

    active_points = points[active_indices]
    ray_dir = np.asarray([0.5773502691896258, 0.3713906763541037, 0.7276068751089989], dtype=np.float64)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    point_chunk = max(1, int(RAY_POINT_CHUNK))
    tri_chunk = max(1, int(RAY_TRI_CHUNK))
    counts = np.zeros((active_points.shape[0],), dtype=np.int32)
    for point_start in range(0, active_points.shape[0], point_chunk):
        pts = active_points[point_start : point_start + point_chunk]
        chunk_counts = np.zeros((pts.shape[0],), dtype=np.int32)
        for tri_start in range(0, triangles.shape[0], tri_chunk):
            tri = triangles[tri_start : tri_start + tri_chunk]
            v0 = tri[:, 0, :]
            e1 = tri[:, 1, :] - v0
            e2 = tri[:, 2, :] - v0
            h = np.cross(np.broadcast_to(ray_dir, e2.shape), e2)
            a = np.einsum("ij,ij->i", e1, h)
            valid_tri = np.abs(a) > eps
            if not np.any(valid_tri):
                continue
            v0 = v0[valid_tri]
            e1 = e1[valid_tri]
            e2 = e2[valid_tri]
            h = h[valid_tri]
            inv_a = 1.0 / a[valid_tri]

            s = pts[:, None, :] - v0[None, :, :]
            u = inv_a[None, :] * np.einsum("ptj,tj->pt", s, h)
            mask = (u >= -eps) & (u <= 1.0 + eps)
            if not np.any(mask):
                continue

            q = np.cross(s, e1[None, :, :])
            v = inv_a[None, :] * np.einsum("ptj,j->pt", q, ray_dir)
            mask &= (v >= -eps) & ((u + v) <= 1.0 + eps)
            if not np.any(mask):
                continue

            t = inv_a[None, :] * np.einsum("tj,ptj->pt", e2, q)
            mask &= t > eps
            chunk_counts += np.count_nonzero(mask, axis=1).astype(np.int32)
        counts[point_start : point_start + pts.shape[0]] = chunk_counts

    labels[active_indices] = (counts % 2) == 1
    return labels


def _path_from_cell(value: Any) -> Path | None:
    if _is_missing(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path if path.is_file() else None


def _row_to_centers_raw(row: pd.Series) -> np.ndarray:
    centers_norm = _array(row["octree_centers"], np.float32)
    if centers_norm.size % 3 != 0:
        centers_norm = centers_norm[: centers_norm.size - (centers_norm.size % 3)]
    centers_norm = centers_norm.reshape(-1, 3)
    center_xyz = _array(row["normalization_center_xyz"], np.float32)
    if center_xyz.size < 3:
        raise ValueError("normalization_center_xyz has fewer than 3 values")
    scale = _safe_float(row["normalization_scale"], default=1.0)
    return (centers_norm * max(scale, 1e-6)) + center_xyz[:3].reshape(1, 3)


def _audit_row(path: Path, row_index: int, row: pd.Series) -> dict[str, Any] | None:
    removed_volume = _safe_float(row.get("out_removed_volume", 0.0))
    if removed_volume < float(MIN_REMOVED_VOLUME):
        return None

    before_obj_path = _path_from_cell(row.get("state_obj_path", ""))
    after_obj_path = _path_from_cell(row.get("next_state_obj_path", ""))
    if before_obj_path is None or after_obj_path is None:
        return {
            "file": str(path),
            "row_index": int(row_index),
            "status": "missing_obj",
            "state_obj_path": str(row.get("state_obj_path", "")),
            "next_state_obj_path": str(row.get("next_state_obj_path", "")),
        }

    centers_raw = _row_to_centers_raw(row)
    parquet_before = _array(row["octree_occ_labels_before"], np.float32) >= 0.5
    parquet_after = _array(row["octree_occ_labels"], np.float32) >= 0.5
    count = min(centers_raw.shape[0], parquet_before.size, parquet_after.size)
    if count <= 0:
        return {
            "file": str(path),
            "row_index": int(row_index),
            "status": "empty_octree",
            "state_obj_path": str(before_obj_path),
            "next_state_obj_path": str(after_obj_path),
        }

    centers_raw = centers_raw[:count]
    parquet_before = parquet_before[:count]
    parquet_after = parquet_after[:count]

    before_mesh = _cached_load_mesh(str(before_obj_path))
    after_mesh = _cached_load_mesh(str(after_obj_path))
    obj_before = _contains_points_ray_parity(before_mesh, centers_raw)
    obj_after = _contains_points_ray_parity(after_mesh, centers_raw)

    parquet_changed = parquet_before != parquet_after
    obj_changed = obj_before != obj_after
    parquet_removed = parquet_before & (~parquet_after)
    obj_removed = obj_before & (~obj_after)

    before_match = obj_before == parquet_before
    after_match = obj_after == parquet_after
    changed_match = obj_changed == parquet_changed

    return {
        "file": str(path),
        "row_index": int(row_index),
        "status": "ok",
        "part_name": str(row.get("part_name", "")),
        "decision_step": int(_safe_float(row.get("decision_step", 0))),
        "candidate_index": int(_safe_float(row.get("candidate_index", 0))),
        "scenario_id": str(row.get("scenario_id", "")),
        "macro_class_name": str(row.get("macro_class_name", "")),
        "tool_choice_name": str(row.get("tool_choice_name", "")),
        "action_face_id": int(_safe_float(row.get("action_face_id", -1), -1)),
        "out_removed_volume": removed_volume,
        "out_removed_ratio": _safe_float(row.get("out_removed_ratio", 0.0)),
        "state_volume": _safe_float(row.get("state_volume", 0.0)),
        "next_state_volume": _safe_float(row.get("next_state_volume", 0.0)),
        "cells": int(count),
        "parquet_before_pos": int(parquet_before.sum()),
        "parquet_after_pos": int(parquet_after.sum()),
        "obj_before_pos": int(obj_before.sum()),
        "obj_after_pos": int(obj_after.sum()),
        "parquet_changed_cells": int(parquet_changed.sum()),
        "obj_changed_cells": int(obj_changed.sum()),
        "parquet_removed_cells": int(parquet_removed.sum()),
        "obj_removed_cells": int(obj_removed.sum()),
        "before_mismatch_cells": int((~before_match).sum()),
        "after_mismatch_cells": int((~after_match).sum()),
        "changed_mask_mismatch_cells": int((~changed_match).sum()),
        "before_agreement": float(before_match.mean()),
        "after_agreement": float(after_match.mean()),
        "changed_mask_agreement": float(changed_match.mean()),
        "parquet_changed_ratio": float(parquet_changed.mean()),
        "obj_changed_ratio": float(obj_changed.mean()),
        "parquet_removed_ratio": float(parquet_removed.mean()),
        "obj_removed_ratio": float(obj_removed.mean()),
        "state_obj_path": str(before_obj_path),
        "next_state_obj_path": str(after_obj_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get("status") == "ok" and key in row]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    files = _resolve_parquet_files()
    row_filter = set(int(x) for x in ROW_INDEX_FILTER)
    out_dir = Path(OUTPUT_DIR).expanduser().resolve() / f"octree_obj_label_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    audited_count = 0
    rows_seen = 0
    for file_idx, parquet_path in enumerate(files, start=1):
        df = _read_frame(parquet_path)
        for row_index, row in df.iterrows():
            rows_seen += 1
            if row_filter and int(row_index) not in row_filter:
                continue
            if MAX_ROWS_TOTAL > 0 and audited_count >= int(MAX_ROWS_TOTAL):
                break
            record = _audit_row(parquet_path, int(row_index), row)
            if record is None:
                continue
            records.append(record)
            if record.get("status") == "ok":
                audited_count += 1
            if len(records) % 10 == 0:
                print(f"[Progress] records={len(records)} audited_ok={audited_count}")
        if MAX_ROWS_TOTAL > 0 and audited_count >= int(MAX_ROWS_TOTAL):
            break
        print(f"[File] {file_idx}/{len(files)} {parquet_path}")

    ok_records = [row for row in records if row.get("status") == "ok"]
    mismatch_records = [
        row for row in ok_records
        if int(row.get("before_mismatch_cells", 0)) > 0
        or int(row.get("after_mismatch_cells", 0)) > 0
        or int(row.get("changed_mask_mismatch_cells", 0)) > 0
    ]
    missing_obj_records = [row for row in records if row.get("status") == "missing_obj"]

    total_cells = int(sum(int(row.get("cells", 0)) for row in ok_records))
    total_before_mismatch = int(sum(int(row.get("before_mismatch_cells", 0)) for row in ok_records))
    total_after_mismatch = int(sum(int(row.get("after_mismatch_cells", 0)) for row in ok_records))
    total_changed_mismatch = int(sum(int(row.get("changed_mask_mismatch_cells", 0)) for row in ok_records))

    summary = {
        "num_files": int(len(files)),
        "rows_seen": int(rows_seen),
        "records": int(len(records)),
        "audited_rows": int(len(ok_records)),
        "missing_obj_rows": int(len(missing_obj_records)),
        "mismatch_rows": int(len(mismatch_records)),
        "total_cells": int(total_cells),
        "weighted_before_agreement": float(1.0 - total_before_mismatch / max(total_cells, 1)),
        "weighted_after_agreement": float(1.0 - total_after_mismatch / max(total_cells, 1)),
        "weighted_changed_mask_agreement": float(1.0 - total_changed_mismatch / max(total_cells, 1)),
        "mean_before_agreement": _mean(ok_records, "before_agreement"),
        "mean_after_agreement": _mean(ok_records, "after_agreement"),
        "mean_changed_mask_agreement": _mean(ok_records, "changed_mask_agreement"),
        "rows_parquet_changed": int(sum(int(row.get("parquet_changed_cells", 0)) > 0 for row in ok_records)),
        "rows_obj_changed": int(sum(int(row.get("obj_changed_cells", 0)) > 0 for row in ok_records)),
        "mean_parquet_changed_ratio": _mean(ok_records, "parquet_changed_ratio"),
        "mean_obj_changed_ratio": _mean(ok_records, "obj_changed_ratio"),
        "max_parquet_changed_ratio": float(max((float(row.get("parquet_changed_ratio", 0.0)) for row in ok_records), default=0.0)),
        "max_obj_changed_ratio": float(max((float(row.get("obj_changed_ratio", 0.0)) for row in ok_records), default=0.0)),
    }

    _write_csv(out_dir / "rows.csv", records)
    _write_csv(out_dir / "mismatch_rows.csv", mismatch_records)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] output_dir={out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
