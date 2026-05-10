"""Audit whether octree before/after labels actually change.

Edit the constants below and run directly from VS Code.
The script reads parquet files one at a time, so it does not concatenate the
dataset into RAM.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
PARQUET_DIR = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\_ALL_PARQUET_FILES"
PARQUET_GLOB = "*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []
OUTPUT_DIR = r"octree_transition_audits"
MAX_FILES = 0          # 0 = all
MAX_ROWS_PER_FILE = 0  # 0 = all


WANTED_COLUMNS = (
    "part_name",
    "macro_class_name",
    "out_removed_volume",
    "out_removed_ratio",
    "state_volume",
    "next_state_volume",
    "octree_occ_labels_before",
    "octree_occ_labels",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
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


def _available_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return set(pd.read_parquet(path, columns=[]).columns)


def _resolve_files() -> list[Path]:
    if EXPLICIT_PARQUET_PATHS:
        files = [Path(p).expanduser().resolve() for p in EXPLICIT_PARQUET_PATHS if str(p).strip()]
    elif PARQUET_DIR:
        files = sorted(Path(PARQUET_DIR).expanduser().resolve().glob(PARQUET_GLOB))
    else:
        raise ValueError("Set PARQUET_DIR or EXPLICIT_PARQUET_PATHS at the top of this script.")

    files = [path for path in files if path.is_file()]
    if MAX_FILES > 0:
        files = files[: int(MAX_FILES)]
    if not files:
        raise ValueError("No parquet files found.")
    return files


def _part_name_from_path(path: Path) -> str:
    stem = path.stem
    marker = "_seed"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def _safe_float(value: Any) -> float:
    if _is_missing(value):
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _read_frame(path: Path) -> pd.DataFrame:
    available = _available_columns(path)
    columns = [name for name in WANTED_COLUMNS if name in available]
    missing_required = [name for name in ("octree_occ_labels_before", "octree_occ_labels") if name not in available]
    if missing_required:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def _audit_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = _read_frame(path)
    row_records: list[dict[str, Any]] = []
    if df.empty:
        return {
            "path": str(path),
            "part_name": _part_name_from_path(path),
            "rows": 0,
            "valid_rows": 0,
            "rows_with_changes": 0,
            "rows_with_removed_cells": 0,
            "mean_changed_ratio": 0.0,
            "max_changed_ratio": 0.0,
            "mean_removed_cell_ratio": 0.0,
            "max_removed_cell_ratio": 0.0,
            "mean_removed_volume": 0.0,
            "rows_removed_volume_positive": 0,
            "status": "missing_required_or_empty",
        }, row_records

    if MAX_ROWS_PER_FILE > 0:
        df = df.iloc[: int(MAX_ROWS_PER_FILE)]

    valid_rows = 0
    rows_with_changes = 0
    rows_with_removed_cells = 0
    changed_ratios: list[float] = []
    removed_cell_ratios: list[float] = []
    removed_volumes: list[float] = []
    rows_removed_volume_positive = 0
    part_name = _part_name_from_path(path)

    for row_idx, row in df.iterrows():
        before_raw = row.get("octree_occ_labels_before")
        after_raw = row.get("octree_occ_labels")
        if _is_missing(before_raw) or _is_missing(after_raw):
            continue
        before = (_array(before_raw, np.float32) >= 0.5)
        after = (_array(after_raw, np.float32) >= 0.5)
        count = min(before.size, after.size)
        if count <= 0:
            continue
        before = before[:count]
        after = after[:count]

        changed = before != after
        removed = before & (~after)
        added = (~before) & after
        changed_count = int(changed.sum())
        removed_count = int(removed.sum())
        added_count = int(added.sum())
        changed_ratio = float(changed_count / count)
        removed_cell_ratio = float(removed_count / count)
        added_cell_ratio = float(added_count / count)
        removed_volume = _safe_float(row.get("out_removed_volume", 0.0))
        removed_ratio = _safe_float(row.get("out_removed_ratio", 0.0))

        valid_rows += 1
        rows_with_changes += int(changed_count > 0)
        rows_with_removed_cells += int(removed_count > 0)
        rows_removed_volume_positive += int(removed_volume > 0.0)
        changed_ratios.append(changed_ratio)
        removed_cell_ratios.append(removed_cell_ratio)
        removed_volumes.append(removed_volume)

        if "part_name" in row.index and not _is_missing(row.get("part_name")):
            part_name = str(row.get("part_name"))

        row_records.append({
            "path": str(path),
            "row_index": int(row_idx),
            "part_name": part_name,
            "macro_class_name": str(row.get("macro_class_name", "")),
            "cells": int(count),
            "changed_cells": changed_count,
            "removed_cells": removed_count,
            "added_cells": added_count,
            "changed_ratio": changed_ratio,
            "removed_cell_ratio": removed_cell_ratio,
            "added_cell_ratio": added_cell_ratio,
            "out_removed_volume": removed_volume,
            "out_removed_ratio": removed_ratio,
            "state_volume": _safe_float(row.get("state_volume", 0.0)),
            "next_state_volume": _safe_float(row.get("next_state_volume", 0.0)),
        })

    file_record = {
        "path": str(path),
        "part_name": part_name,
        "rows": int(len(df)),
        "valid_rows": int(valid_rows),
        "rows_with_changes": int(rows_with_changes),
        "rows_with_removed_cells": int(rows_with_removed_cells),
        "mean_changed_ratio": float(np.mean(changed_ratios)) if changed_ratios else 0.0,
        "max_changed_ratio": float(np.max(changed_ratios)) if changed_ratios else 0.0,
        "mean_removed_cell_ratio": float(np.mean(removed_cell_ratios)) if removed_cell_ratios else 0.0,
        "max_removed_cell_ratio": float(np.max(removed_cell_ratios)) if removed_cell_ratios else 0.0,
        "mean_removed_volume": float(np.mean(removed_volumes)) if removed_volumes else 0.0,
        "rows_removed_volume_positive": int(rows_removed_volume_positive),
        "status": "ok",
    }
    return file_record, row_records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = _resolve_files()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve() / f"octree_change_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    for file_idx, path in enumerate(files, start=1):
        file_record, rows = _audit_file(path)
        file_records.append(file_record)
        row_records.extend(rows)
        if file_idx % 25 == 0 or file_idx == len(files):
            print(f"[Progress] {file_idx}/{len(files)} files")

    valid_file_records = [r for r in file_records if int(r["valid_rows"]) > 0]
    valid_row_records = row_records
    changed_rows = [r for r in valid_row_records if int(r["changed_cells"]) > 0]
    removed_rows = [r for r in valid_row_records if int(r["removed_cells"]) > 0]
    removed_volume_positive_rows = [r for r in valid_row_records if float(r["out_removed_volume"]) > 0.0]
    volume_pos_no_change = [
        r for r in removed_volume_positive_rows
        if int(r["changed_cells"]) == 0
    ]

    summary = {
        "num_files": len(files),
        "valid_files": len(valid_file_records),
        "num_rows": len(valid_row_records),
        "rows_with_changes": len(changed_rows),
        "rows_with_removed_cells": len(removed_rows),
        "rows_removed_volume_positive": len(removed_volume_positive_rows),
        "rows_removed_volume_positive_but_no_occ_change": len(volume_pos_no_change),
        "row_changed_ratio": len(changed_rows) / max(len(valid_row_records), 1),
        "row_removed_cell_ratio": len(removed_rows) / max(len(valid_row_records), 1),
        "mean_changed_ratio": float(np.mean([r["changed_ratio"] for r in valid_row_records])) if valid_row_records else 0.0,
        "max_changed_ratio": float(np.max([r["changed_ratio"] for r in valid_row_records])) if valid_row_records else 0.0,
        "mean_removed_cell_ratio": float(np.mean([r["removed_cell_ratio"] for r in valid_row_records])) if valid_row_records else 0.0,
        "max_removed_cell_ratio": float(np.max([r["removed_cell_ratio"] for r in valid_row_records])) if valid_row_records else 0.0,
        "mean_removed_volume": float(np.mean([r["out_removed_volume"] for r in valid_row_records])) if valid_row_records else 0.0,
        "max_removed_volume": float(np.max([r["out_removed_volume"] for r in valid_row_records])) if valid_row_records else 0.0,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "file_audit.csv", file_records)
    _write_csv(out_dir / "row_audit.csv", row_records)
    _write_csv(
        out_dir / "removed_volume_positive_but_no_occ_change.csv",
        volume_pos_no_change,
    )

    print(f"[Done] output_dir={out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
