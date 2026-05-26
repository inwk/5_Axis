"""Audit SDF-query transition parquet files.

Edit the constants below and run directly from VS Code. This does not require
NX; it only checks whether SDF query supervision was written consistently.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
PARQUET_PATH = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\_ALL_PARQUET_FILES\3dDataset2771_seed0_20260522_084415.parquet"
PARQUET_DIR = r""
PARQUET_GLOB = "*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []

OUTPUT_DIR = r"sdf_query_audits"
MAX_FILES = 1          # 0 = all
MAX_ROWS_PER_FILE = 0  # 0 = all
MAX_ROWS_TOTAL = 0     # 0 = all

EXPECTED_QUERY_COUNT = 0  # 0 = do not enforce
TSDF_CHANGE_EPS = 1e-3
MONOTONICITY_EPS = 1e-3   # removal should satisfy after_tsdf >= before_tsdf
MAX_CORRUPT_FILES = 0
MAX_BAD_QUERY_ROWS = 0
MAX_NONFINITE_ROWS = 0
MAX_OUT_OF_RANGE_ROWS = 0
MIN_POINT_CHANGED_RATIO = 0.15
MAX_POINT_MONOTONICITY_VIOLATION_RATIO = 0.10


REQUIRED_COLUMNS = (
    "sdf_query_points",
    "sdf_tsdf_before",
    "sdf_tsdf_after",
    "sdf_delta_tsdf",
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


def _resolve_files() -> list[Path]:
    files: list[Path] = []
    if EXPLICIT_PARQUET_PATHS:
        files.extend(Path(p).expanduser().resolve() for p in EXPLICIT_PARQUET_PATHS if str(p).strip())
    elif PARQUET_PATH.strip():
        files.append(Path(PARQUET_PATH).expanduser().resolve())
    elif PARQUET_DIR.strip():
        files.extend(sorted(Path(PARQUET_DIR).expanduser().resolve().glob(PARQUET_GLOB)))
    else:
        raise ValueError("Set PARQUET_PATH, PARQUET_DIR, or EXPLICIT_PARQUET_PATHS.")

    out: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            raise FileNotFoundError(path)
        out.append(path)
    if MAX_FILES > 0:
        out = out[:MAX_FILES]
    if not out:
        raise ValueError("No parquet files found.")
    return out


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    files = _resolve_files()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve() / f"sdf_query_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_files": len(files),
        "valid_files": 0,
        "corrupt_files": 0,
        "rows_seen": 0,
        "rows_with_required_columns": 0,
        "rows_with_target_tsdf": 0,
        "rows_bad_query_count": 0,
        "rows_nonfinite": 0,
        "rows_out_of_range": 0,
        "rows_with_tsdf_change": 0,
        "rows_with_monotonicity_violation": 0,
        "total_points": 0,
        "changed_points": 0,
        "monotonicity_violating_points": 0,
        "mean_abs_delta_sum": 0.0,
        "max_abs_delta": 0.0,
        "mean_before_tsdf_sum": 0.0,
        "mean_after_tsdf_sum": 0.0,
        "mean_target_tsdf_sum": 0.0,
    }
    records: list[dict[str, Any]] = []

    for file_index, path in enumerate(files, start=1):
        try:
            df = pd.read_parquet(path)
            summary["valid_files"] += 1
        except Exception as exc:
            summary["corrupt_files"] += 1
            records.append({
                "file": str(path),
                "row_index": -1,
                "status": "corrupt_parquet",
                "error": repr(exc),
                "size_bytes": int(path.stat().st_size) if path.exists() else -1,
            })
            print(f"[WARN] Skipping corrupt parquet: {path} err={exc!r}")
            continue
        if MAX_ROWS_PER_FILE > 0:
            df = df.head(MAX_ROWS_PER_FILE)
        for row_index, row in df.iterrows():
            if MAX_ROWS_TOTAL > 0 and summary["rows_seen"] >= MAX_ROWS_TOTAL:
                break
            summary["rows_seen"] += 1
            missing = [col for col in REQUIRED_COLUMNS if col not in row.index or _is_missing(row[col])]
            if missing:
                records.append({
                    "file": str(path),
                    "row_index": int(row_index),
                    "status": "missing_required",
                    "missing": missing,
                })
                continue

            points = _array(row["sdf_query_points"], np.float32).reshape(-1, 3)
            before = _array(row["sdf_tsdf_before"], np.float32)
            after = _array(row["sdf_tsdf_after"], np.float32)
            delta = _array(row["sdf_delta_tsdf"], np.float32)
            count = min(points.shape[0], before.shape[0], after.shape[0], delta.shape[0])
            points = points[:count]
            before = before[:count]
            after = after[:count]
            delta = delta[:count]
            target = None
            if "sdf_target_tsdf" in row.index and not _is_missing(row["sdf_target_tsdf"]):
                target = _array(row["sdf_target_tsdf"], np.float32)[:count]
                summary["rows_with_target_tsdf"] += 1

            summary["rows_with_required_columns"] += 1
            if EXPECTED_QUERY_COUNT > 0 and count != int(EXPECTED_QUERY_COUNT):
                summary["rows_bad_query_count"] += 1

            finite_ok = (
                np.isfinite(points).all()
                and np.isfinite(before).all()
                and np.isfinite(after).all()
                and np.isfinite(delta).all()
                and (target is None or np.isfinite(target).all())
            )
            if not finite_ok:
                summary["rows_nonfinite"] += 1

            range_ok = (
                np.all((before >= -1.0001) & (before <= 1.0001))
                and np.all((after >= -1.0001) & (after <= 1.0001))
                and (target is None or np.all((target >= -1.0001) & (target <= 1.0001)))
            )
            if not range_ok:
                summary["rows_out_of_range"] += 1

            abs_delta = np.abs(after - before)
            changed = abs_delta > float(TSDF_CHANGE_EPS)
            mono_violation = after < (before - float(MONOTONICITY_EPS))
            if bool(changed.any()):
                summary["rows_with_tsdf_change"] += 1
            if bool(mono_violation.any()):
                summary["rows_with_monotonicity_violation"] += 1

            summary["total_points"] += int(count)
            summary["changed_points"] += int(changed.sum())
            summary["monotonicity_violating_points"] += int(mono_violation.sum())
            summary["mean_abs_delta_sum"] += float(abs_delta.mean()) if count > 0 else 0.0
            summary["max_abs_delta"] = max(float(summary["max_abs_delta"]), float(abs_delta.max()) if count > 0 else 0.0)
            summary["mean_before_tsdf_sum"] += float(before.mean()) if count > 0 else 0.0
            summary["mean_after_tsdf_sum"] += float(after.mean()) if count > 0 else 0.0
            if target is not None and target.size > 0:
                summary["mean_target_tsdf_sum"] += float(target.mean())

            records.append({
                "file": str(path),
                "row_index": int(row_index),
                "part_name": str(row.get("part_name", "")),
                "decision_step": int(row.get("decision_step", -1)),
                "candidate_index": int(row.get("candidate_index", -1)),
                "macro_class_name": str(row.get("macro_class_name", "")),
                "tool_choice_name": str(row.get("tool_choice_name", "")),
                "query_count": int(count),
                "mean_before_tsdf": float(before.mean()) if count > 0 else 0.0,
                "mean_after_tsdf": float(after.mean()) if count > 0 else 0.0,
                "mean_abs_delta": float(abs_delta.mean()) if count > 0 else 0.0,
                "max_abs_delta": float(abs_delta.max()) if count > 0 else 0.0,
                "changed_ratio": _safe_ratio(float(changed.sum()), float(count)),
                "monotonicity_violation_ratio": _safe_ratio(float(mono_violation.sum()), float(count)),
                "removed_volume": float(row.get("out_removed_volume", 0.0) or 0.0),
            })

        if MAX_ROWS_TOTAL > 0 and summary["rows_seen"] >= MAX_ROWS_TOTAL:
            break
        print(f"[File] {file_index}/{len(files)} {path}")

    valid_rows = int(summary["rows_with_required_columns"])
    summary["point_changed_ratio"] = _safe_ratio(float(summary["changed_points"]), float(summary["total_points"]))
    summary["point_monotonicity_violation_ratio"] = _safe_ratio(
        float(summary["monotonicity_violating_points"]),
        float(summary["total_points"]),
    )
    summary["mean_abs_delta"] = _safe_ratio(float(summary["mean_abs_delta_sum"]), float(valid_rows))
    summary["mean_before_tsdf"] = _safe_ratio(float(summary["mean_before_tsdf_sum"]), float(valid_rows))
    summary["mean_after_tsdf"] = _safe_ratio(float(summary["mean_after_tsdf_sum"]), float(valid_rows))
    summary["mean_target_tsdf"] = _safe_ratio(
        float(summary["mean_target_tsdf_sum"]),
        float(summary["rows_with_target_tsdf"]),
    )
    thresholds = {
        "max_corrupt_files": int(MAX_CORRUPT_FILES),
        "max_bad_query_rows": int(MAX_BAD_QUERY_ROWS),
        "max_nonfinite_rows": int(MAX_NONFINITE_ROWS),
        "max_out_of_range_rows": int(MAX_OUT_OF_RANGE_ROWS),
        "min_point_changed_ratio": float(MIN_POINT_CHANGED_RATIO),
        "max_point_monotonicity_violation_ratio": float(MAX_POINT_MONOTONICITY_VIOLATION_RATIO),
    }
    failures: list[str] = []
    if int(summary["corrupt_files"]) > int(MAX_CORRUPT_FILES):
        failures.append(f"corrupt_files={summary['corrupt_files']} > {MAX_CORRUPT_FILES}")
    if int(summary["rows_bad_query_count"]) > int(MAX_BAD_QUERY_ROWS):
        failures.append(f"rows_bad_query_count={summary['rows_bad_query_count']} > {MAX_BAD_QUERY_ROWS}")
    if int(summary["rows_nonfinite"]) > int(MAX_NONFINITE_ROWS):
        failures.append(f"rows_nonfinite={summary['rows_nonfinite']} > {MAX_NONFINITE_ROWS}")
    if int(summary["rows_out_of_range"]) > int(MAX_OUT_OF_RANGE_ROWS):
        failures.append(f"rows_out_of_range={summary['rows_out_of_range']} > {MAX_OUT_OF_RANGE_ROWS}")
    if float(summary["point_changed_ratio"]) < float(MIN_POINT_CHANGED_RATIO):
        failures.append(f"point_changed_ratio={summary['point_changed_ratio']:.6f} < {MIN_POINT_CHANGED_RATIO}")
    if float(summary["point_monotonicity_violation_ratio"]) > float(MAX_POINT_MONOTONICITY_VIOLATION_RATIO):
        failures.append(
            "point_monotonicity_violation_ratio="
            f"{summary['point_monotonicity_violation_ratio']:.6f} > {MAX_POINT_MONOTONICITY_VIOLATION_RATIO}"
        )
    summary["audit_thresholds"] = thresholds
    summary["audit_failures"] = failures
    summary["audit_pass"] = len(failures) == 0

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(records).to_csv(out_dir / "rows.csv", index=False, encoding="utf-8-sig")
    print("[AUDIT] PASS" if summary["audit_pass"] else f"[AUDIT] FAIL: {failures}")
    print(f"[Done] output_dir={out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
