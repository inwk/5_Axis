"""Analyze face-type class imbalance in StateEncoder SSL parquet files.

Edit the constants below and run directly from VSCode / terminal.
Only `face_type_512`, `node_mask`, and optional `part_name` are read.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from graph_sdf import GraphSdfModelConfig
from ssl_pretraining.state_encoder_dataset import resolve_parquet_files


# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode.
# ---------------------------------------------------------------------------
PARQUET_DIR = r""
PARQUET_GLOB = "**/*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []

OUTPUT_DIR = r""  # Empty = write analysis next to PARQUET_DIR, or cwd if explicit paths are used.

# Padding nodes have node_mask == 1 and should not count as real face classes.
EXCLUDE_PADDED_NODES = True


def _array(value: Any, dtype: np.dtype) -> np.ndarray:
    """Converts parquet list/object values into a flat numpy array."""
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


def _read_needed_columns(path: str) -> pd.DataFrame:
    wanted = ["part_name", "face_type_512", "node_mask"]
    try:
        import pyarrow.parquet as pq  # type: ignore

        available = set(pq.ParquetFile(path).schema_arrow.names)
        if "face_type_512" not in available:
            raise ValueError(f"{path} is missing required column: face_type_512")
        columns = [name for name in wanted if name in available]
        return pd.read_parquet(path, columns=columns)
    except ImportError:
        frame = pd.read_parquet(path)
        if "face_type_512" not in frame.columns:
            raise ValueError(f"{path} is missing required column: face_type_512")
        columns = [name for name in wanted if name in frame.columns]
        return frame[columns]


def _output_dir() -> Path:
    if OUTPUT_DIR.strip():
        root = Path(OUTPUT_DIR).expanduser().resolve()
    elif PARQUET_DIR.strip():
        root = Path(PARQUET_DIR).expanduser().resolve()
    else:
        root = Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den > 0 else 0.0


def _effective_number_weight(count: int, beta: float = 0.999) -> float:
    """Class-balanced weight from Cui et al.; normalized later for readability."""
    if count <= 0:
        return 0.0
    return float((1.0 - beta) / (1.0 - math.pow(beta, int(count))))


def main() -> None:
    parquet_files = resolve_parquet_files(
        parquet_dir=PARQUET_DIR,
        parquet_glob=PARQUET_GLOB,
        explicit_parquet_paths=EXPLICIT_PARQUET_PATHS,
        caller_name="ssl_pretraining/analyze_face_type_distribution.py",
    )
    counts: dict[int, int] = {}
    padded_counts: dict[int, int] = {}
    parts_with_type: dict[int, set[str]] = {}
    row_count = 0
    valid_node_count = 0
    padded_node_count = 0
    errors: list[dict[str, str]] = []

    print(f"[INFO] parquet files={len(parquet_files)}")
    for file_idx, path in enumerate(parquet_files, start=1):
        try:
            frame = _read_needed_columns(path)
            for row_idx, row in frame.iterrows():
                row_count += 1
                part_name = str(row.get("part_name", Path(path).stem))
                face_type = _array(row["face_type_512"], np.int16).reshape(-1)
                if "node_mask" in row.index:
                    node_mask = _array(row["node_mask"], np.int16).reshape(-1)
                else:
                    node_mask = np.zeros_like(face_type, dtype=np.int16)
                count = min(len(face_type), len(node_mask))
                face_type = face_type[:count]
                node_mask = node_mask[:count]

                real_mask = node_mask == 0
                padded_mask = ~real_mask
                valid_node_count += int(real_mask.sum())
                padded_node_count += int(padded_mask.sum())

                for type_id in face_type[padded_mask]:
                    key = int(type_id)
                    padded_counts[key] = padded_counts.get(key, 0) + 1

                values = face_type[real_mask] if EXCLUDE_PADDED_NODES else face_type
                for type_id in values:
                    key = int(type_id)
                    counts[key] = counts.get(key, 0) + 1
                    parts_with_type.setdefault(key, set()).add(part_name)
        except Exception as exc:
            errors.append({"parquet_path": path, "error": repr(exc)})
            print(f"[WARN] failed {path}: {exc!r}")

        if file_idx == 1 or file_idx % 100 == 0 or file_idx == len(parquet_files):
            print(f"[INFO] processed files={file_idx}/{len(parquet_files)} rows={row_count}")

    if not counts:
        raise RuntimeError(f"No face type values found. errors={errors[:5]}")

    total = int(sum(counts.values()))
    rows = []
    for type_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "face_type_id": int(type_id),
                "count": int(count),
                "ratio": _safe_ratio(int(count), total),
                "num_parts_with_type": len(parts_with_type.get(int(type_id), set())),
                "padded_count_with_same_id": int(padded_counts.get(int(type_id), 0)),
                "inverse_freq_weight_raw": _safe_ratio(total, int(count)),
                "effective_num_weight_raw": _effective_number_weight(int(count)),
            }
        )

    eff_weights = np.asarray([row["effective_num_weight_raw"] for row in rows], dtype=np.float64)
    positive = eff_weights > 0
    if positive.any():
        eff_weights = eff_weights / float(eff_weights[positive].mean())
    for row, weight in zip(rows, eff_weights):
        row["effective_num_weight_norm"] = float(weight)

    out_dir = _output_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"face_type_distribution_{stamp}.csv"
    json_path = out_dir / f"face_type_distribution_{stamp}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "parquet_files": parquet_files,
        "num_parquet_files": len(parquet_files),
        "num_rows": row_count,
        "total_counted_nodes": total,
        "valid_node_count": valid_node_count,
        "padded_node_count": padded_node_count,
        "exclude_padded_nodes": bool(EXCLUDE_PADDED_NODES),
        "face_type_vocab_size": int(GraphSdfModelConfig().face_type_vocab_size),
        "reserved_ssl_mask_face_type_id": int(GraphSdfModelConfig().face_type_vocab_size - 1),
        "distribution": rows,
        "errors": errors,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[Summary]")
    print(f"  rows                 : {row_count}")
    print(f"  valid nodes          : {valid_node_count}")
    print(f"  padded nodes         : {padded_node_count}")
    print(f"  counted nodes        : {total}")
    print(f"  unique face types    : {len(rows)}")
    print(f"  reserved mask id     : {GraphSdfModelConfig().face_type_vocab_size - 1}")
    print("[Top Classes]")
    for row in rows[:10]:
        print(
            f"  type={row['face_type_id']:>3} "
            f"count={row['count']:>8} "
            f"ratio={row['ratio']:.4f} "
            f"parts={row['num_parts_with_type']:>5} "
            f"eff_w={row['effective_num_weight_norm']:.3f}"
        )
    print(f"[OK] csv saved : {csv_path}")
    print(f"[OK] json saved: {json_path}")


if __name__ == "__main__":
    main()
