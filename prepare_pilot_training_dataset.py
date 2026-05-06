"""Prepare a pilot parquet dataset for SSL and transition training.

This script scans collected process-skeleton parquet files, filters unusable
files, computes lightweight dataset statistics, and writes a part-level
train/val/test split manifest.

Edit the "User config" constants below and run this file directly from VS Code.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
PARQUET_DIR = r""
OUTPUT_DIR = r"pilot_training_splits"
PARQUET_GLOB = "*.parquet"
SEED = 0
VAL_RATIO = 0.10
TEST_RATIO = 0.10
MAX_FILES = 0
STATS_SAMPLE_ROWS_PER_FILE = 32

REQUIRED_OCTREE_COLUMNS = ("octree_centers", "octree_depths", "octree_occ_labels")
OPTIONAL_STATS_COLUMNS = (
    "part_name",
    "static_feature_dir",
    "macro_class_name",
    "is_chosen",
    "octree_occ_labels_before",
)
REQUIRED_STATIC_FILES = (
    "embed_face_pc.npy",
    "embed_face_normal.npy",
    "embed_node_mask.npy",
    "embed_point_mask.npy",
    "embed_centrality.npy",
    "embed_spatial_pos.npy",
    "embed_face_area.npy",
    "embed_face_type.npy",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _array(value: Any, dtype) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype)
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


def _read_stats_frame(path: Path) -> pd.DataFrame:
    available = _available_columns(path)
    wanted = list(REQUIRED_OCTREE_COLUMNS + OPTIONAL_STATS_COLUMNS)
    columns = [name for name in wanted if name in available]
    if not columns:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=columns)


def _part_name_from_path(path: Path) -> str:
    stem = path.stem
    marker = "_seed"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def _part_name_from_frame(path: Path, df: pd.DataFrame) -> str:
    if "part_name" in df.columns and len(df) > 0:
        for value in df["part_name"].tolist():
            if not _is_missing(value) and str(value).strip():
                return str(value).strip()
    return _part_name_from_path(path)


def _first_static_feature_dir(df: pd.DataFrame) -> str:
    if "static_feature_dir" not in df.columns:
        return ""
    for value in df["static_feature_dir"].tolist():
        if not _is_missing(value) and str(value).strip():
            return str(Path(str(value)).expanduser().resolve())
    return ""


def _missing_static_files(static_feature_dir: str) -> list[str]:
    if not static_feature_dir:
        return list(REQUIRED_STATIC_FILES)
    root = Path(static_feature_dir)
    return [name for name in REQUIRED_STATIC_FILES if not (root / name).is_file()]


def _has_octree_row(row: pd.Series) -> bool:
    return all(col in row.index and not _is_missing(row[col]) for col in REQUIRED_OCTREE_COLUMNS)


def _first_valid_row_index(df: pd.DataFrame, prefer_chosen: bool) -> int:
    fallback = -1
    for idx, row in df.iterrows():
        if not _has_octree_row(row):
            continue
        if fallback < 0:
            fallback = int(idx)
        if prefer_chosen and "is_chosen" in row.index and int(row.get("is_chosen", 0) or 0) == 1:
            return int(idx)
    return fallback


def _sample_occ_ratio(df: pd.DataFrame, sample_rows: int) -> tuple[float, int]:
    if "octree_occ_labels" not in df.columns or len(df) == 0:
        return 0.0, 0
    valid_indices = [int(idx) for idx, row in df.iterrows() if not _is_missing(row["octree_occ_labels"])]
    if not valid_indices:
        return 0.0, 0
    if sample_rows > 0 and len(valid_indices) > sample_rows:
        valid_indices = valid_indices[:sample_rows]
    occ_sum = 0.0
    occ_count = 0
    for idx in valid_indices:
        labels = _array(df.loc[idx, "octree_occ_labels"], np.float32).reshape(-1)
        if labels.size == 0:
            continue
        occ_sum += float((labels >= 0.5).sum())
        occ_count += int(labels.size)
    return (occ_sum / float(occ_count)) if occ_count > 0 else 0.0, occ_count


def _scan_one(path: Path, sample_rows: int) -> dict[str, Any]:
    available_columns: set[str] = set()
    try:
        available_columns = _available_columns(path)
        df = _read_stats_frame(path)
    except Exception as exc:
        return {
            "path": str(path),
            "part_name": _part_name_from_path(path),
            "valid": False,
            "invalid_reason": f"read_failed:{exc}",
            "rows": 0,
            "valid_octree_rows": 0,
            "chosen_rows": 0,
            "before_occ_rows": 0,
            "occ_ratio_sample": 0.0,
            "occ_ratio_sample_cells": 0,
            "first_valid_row_index": -1,
            "first_chosen_valid_row_index": -1,
            "static_feature_dir": "",
            "missing_static_files": [],
            "macro_counts": {},
        }

    rows = int(len(df))
    missing_required = [col for col in REQUIRED_OCTREE_COLUMNS if col not in df.columns]
    valid_octree_rows = 0
    before_occ_rows = 0
    if not missing_required:
        for _, row in df.iterrows():
            if _has_octree_row(row):
                valid_octree_rows += 1
            if "octree_occ_labels_before" in row.index and not _is_missing(row["octree_occ_labels_before"]):
                before_occ_rows += 1

    chosen_rows = int(df["is_chosen"].fillna(0).astype(int).sum()) if "is_chosen" in df.columns else 0
    macro_counts = (
        {str(k): int(v) for k, v in df["macro_class_name"].value_counts().to_dict().items()}
        if "macro_class_name" in df.columns
        else {}
    )
    occ_ratio, occ_cells = _sample_occ_ratio(df, sample_rows=sample_rows)
    static_feature_dir = _first_static_feature_dir(df)
    missing_static_files = []
    if "state_points" not in available_columns:
        missing_static_files = _missing_static_files(static_feature_dir)

    invalid_reason = ""
    if rows <= 0:
        invalid_reason = "empty_parquet"
    elif missing_required:
        invalid_reason = "missing_columns:" + ",".join(missing_required)
    elif valid_octree_rows <= 0:
        invalid_reason = "no_complete_octree_rows"
    elif "state_points" not in available_columns and not static_feature_dir:
        invalid_reason = "missing_state_points_or_static_feature_dir"
    elif missing_static_files:
        invalid_reason = "missing_static_files:" + ",".join(missing_static_files)

    return {
        "path": str(path),
        "part_name": _part_name_from_frame(path, df),
        "valid": not bool(invalid_reason),
        "invalid_reason": invalid_reason,
        "rows": rows,
        "valid_octree_rows": int(valid_octree_rows),
        "chosen_rows": int(chosen_rows),
        "before_occ_rows": int(before_occ_rows),
        "occ_ratio_sample": float(occ_ratio),
        "occ_ratio_sample_cells": int(occ_cells),
        "first_valid_row_index": int(_first_valid_row_index(df, prefer_chosen=False)),
        "first_chosen_valid_row_index": int(_first_valid_row_index(df, prefer_chosen=True)),
        "static_feature_dir": static_feature_dir,
        "missing_static_files": missing_static_files,
        "macro_counts": macro_counts,
    }


def _split_parts(parts: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, set[str]]:
    shuffled = list(parts)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": set(), "val": set(), "test": set()}
    test_count = int(round(n * float(test_ratio))) if test_ratio > 0 else 0
    val_count = int(round(n * float(val_ratio))) if val_ratio > 0 else 0
    if n >= 3 and test_ratio > 0:
        test_count = max(1, test_count)
    if n >= 3 and val_ratio > 0:
        val_count = max(1, val_count)
    if test_count + val_count >= n:
        overflow = test_count + val_count - (n - 1)
        val_count = max(0, val_count - overflow)
    test_parts = set(shuffled[:test_count])
    val_parts = set(shuffled[test_count:test_count + val_count])
    train_parts = set(shuffled[test_count + val_count:])
    return {"train": train_parts, "val": val_parts, "test": test_parts}


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_audit_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "valid",
        "invalid_reason",
        "part_name",
        "rows",
        "valid_octree_rows",
        "chosen_rows",
        "before_occ_rows",
        "occ_ratio_sample",
        "occ_ratio_sample_cells",
        "first_valid_row_index",
        "first_chosen_valid_row_index",
        "static_feature_dir",
        "missing_static_files",
        "path",
        "macro_counts_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fieldnames}
            row["missing_static_files"] = ",".join(record.get("missing_static_files", []))
            row["macro_counts_json"] = json.dumps(record.get("macro_counts", {}), ensure_ascii=True)
            writer.writerow(row)


def _sum_macro_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for macro, count in dict(record.get("macro_counts", {})).items():
            counter[str(macro)] += int(count)
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def build_manifest(
    parquet_dir: str,
    parquet_glob: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    max_files: int,
    stats_sample_rows_per_file: int,
) -> dict[str, Any]:
    parquet_dir_path = Path(parquet_dir).expanduser().resolve()
    if not parquet_dir:
        raise ValueError("Set PARQUET_DIR at the top of prepare_pilot_training_dataset.py")
    if not parquet_dir_path.is_dir():
        raise NotADirectoryError(f"Parquet directory not found: {parquet_dir_path}")

    files = sorted(parquet_dir_path.glob(parquet_glob))
    if max_files > 0:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No parquet files matched {parquet_glob!r} under {parquet_dir_path}")

    records = []
    for i, path in enumerate(files, start=1):
        record = _scan_one(path, sample_rows=int(stats_sample_rows_per_file))
        records.append(record)
        if i == 1 or i % 50 == 0 or i == len(files):
            print(f"[Scan] {i}/{len(files)} files")

    valid_records = [record for record in records if bool(record["valid"])]
    parts = sorted({str(record["part_name"]) for record in valid_records})
    split_parts = _split_parts(parts, val_ratio, test_ratio, seed)

    split_files: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    invalid_records = []
    for record in records:
        if not bool(record["valid"]):
            record["split"] = "invalid"
            invalid_records.append(record)
            continue
        split = "train"
        for name, part_set in split_parts.items():
            if str(record["part_name"]) in part_set:
                split = name
                break
        record["split"] = split
        split_files[split].append(str(record["path"]))
        split_records[split].append(record)

    overfit_record = None
    for record in split_records["train"]:
        if int(record.get("first_chosen_valid_row_index", -1)) >= 0:
            overfit_record = record
            break
    if overfit_record is None and split_records["train"]:
        overfit_record = split_records["train"][0]

    overfit_sample = None
    if overfit_record is not None:
        row_index = int(overfit_record.get("first_chosen_valid_row_index", -1))
        if row_index < 0:
            row_index = int(overfit_record.get("first_valid_row_index", 0))
        overfit_sample = {
            "parquet_path": str(overfit_record["path"]),
            "row_index": int(row_index),
        }

    def split_summary(name: str) -> dict[str, Any]:
        rows = split_records[name]
        return {
            "files": len(split_files[name]),
            "parts": len({str(record["part_name"]) for record in rows}),
            "rows": int(sum(int(record["rows"]) for record in rows)),
            "valid_octree_rows": int(sum(int(record["valid_octree_rows"]) for record in rows)),
            "chosen_rows": int(sum(int(record["chosen_rows"]) for record in rows)),
            "macro_counts": _sum_macro_counts(rows),
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_parquet_dir": str(parquet_dir_path),
        "parquet_glob": parquet_glob,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "splits": split_files,
        "split_parts": {key: sorted(value) for key, value in split_parts.items()},
        "summary": {
            "total_files_scanned": len(records),
            "valid_files": len(valid_records),
            "invalid_files": len(invalid_records),
            "total_parts": len(parts),
            "train": split_summary("train"),
            "val": split_summary("val"),
            "test": split_summary("test"),
        },
        "overfit_sample": overfit_sample,
        "audit_records": records,
    }


def main() -> None:
    manifest = build_manifest(
        parquet_dir=PARQUET_DIR,
        parquet_glob=PARQUET_GLOB,
        seed=SEED,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        max_files=MAX_FILES,
        stats_sample_rows_per_file=STATS_SAMPLE_ROWS_PER_FILE,
    )
    out_root = Path(OUTPUT_DIR).expanduser().resolve()
    run_dir = out_root / f"pilot_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    audit_records = manifest.pop("audit_records")
    manifest_path = run_dir / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_audit_csv(run_dir / "parquet_audit.csv", audit_records)
    for split_name, files in manifest["splits"].items():
        _write_lines(run_dir / f"{split_name}_files.txt", files)

    print(f"[Done] manifest={manifest_path}")
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    if manifest.get("overfit_sample"):
        print("[Overfit Sample]")
        print(json.dumps(manifest["overfit_sample"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
