"""Parquet dataset helpers for StateEncoder SSL pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StateEncoderSslParquetDataset(Dataset):
    """Loads only the columns required for StateEncoder SSL pretraining."""

    REQUIRED_COLUMNS = ["state_points", "node_mask", "point_mask"]
    OPTIONAL_COLUMNS = [
        "node_process_state",
        "centrality_512",
        "spatial_pos_512x512",
        "face_area_512x1",
        "face_type_512",
    ]

    def __init__(self, parquet_files: list[str]) -> None:
        frames = [self._read_needed_columns(path) for path in parquet_files]
        self.df = pd.concat(frames, ignore_index=True)
        if self.df.empty:
            raise ValueError("StateEncoder SSL dataset has no rows.")

    @staticmethod
    def _read_needed_columns(path: str) -> pd.DataFrame:
        wanted = StateEncoderSslParquetDataset.REQUIRED_COLUMNS + StateEncoderSslParquetDataset.OPTIONAL_COLUMNS
        try:
            import pyarrow.parquet as pq  # type: ignore

            # ParquetSchema.names returns nested leaf names such as "element"
            # for list columns. schema_arrow.names preserves top-level columns.
            available = set(pq.ParquetFile(path).schema_arrow.names)
            columns = [name for name in wanted if name in available]
            missing_required = [name for name in StateEncoderSslParquetDataset.REQUIRED_COLUMNS if name not in available]
            if missing_required:
                raise ValueError(f"{path} is missing required columns: {missing_required}")
            return pd.read_parquet(path, columns=columns)
        except ImportError:
            frame = pd.read_parquet(path)
            missing_required = [name for name in StateEncoderSslParquetDataset.REQUIRED_COLUMNS if name not in frame.columns]
            if missing_required:
                raise ValueError(f"{path} is missing required columns: {missing_required}")
            columns = [name for name in wanted if name in frame.columns]
            return frame[columns]

    def __len__(self) -> int:
        return int(len(self.df))

    @staticmethod
    def _array(value, dtype) -> np.ndarray:
        """Converts parquet list/object values into numpy arrays."""
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

    @staticmethod
    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        return False

    @staticmethod
    def _optional_array(value, dtype, default_shape: tuple[int, ...]) -> np.ndarray:
        if StateEncoderSslParquetDataset._is_missing(value):
            return np.zeros(default_shape, dtype=dtype)
        arr = StateEncoderSslParquetDataset._array(value, dtype)
        if arr.size == 0:
            return np.zeros(default_shape, dtype=dtype)
        return arr

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[int(index)]
        state_points = self._array(row["state_points"], np.float32).reshape(512, 100, 7)
        num_nodes = int(state_points.shape[0])
        points_per_node = int(state_points.shape[1])
        node_mask = self._array(row["node_mask"], np.int16).reshape(num_nodes).astype(np.bool_)
        point_mask = self._array(row["point_mask"], np.int16).reshape(num_nodes, points_per_node).astype(np.bool_)
        batch = {
            "state_points": torch.from_numpy(state_points),
            "node_mask": torch.from_numpy(node_mask),
            "point_mask": torch.from_numpy(point_mask),
        }

        if "node_process_state" in row.index:
            batch["node_process_state"] = torch.from_numpy(
                self._optional_array(row["node_process_state"], np.float32, (num_nodes, 2)).reshape(num_nodes, 2)
            )
        if "centrality_512" in row.index:
            centrality = self._array(row["centrality_512"], np.int16).reshape(num_nodes)
            batch["centrality_512"] = torch.from_numpy(centrality)
            batch["node_centrality"] = batch["centrality_512"]
        if "spatial_pos_512x512" in row.index:
            spatial_pos = self._array(row["spatial_pos_512x512"], np.int16).reshape(num_nodes, num_nodes)
            batch["spatial_pos_512x512"] = torch.from_numpy(spatial_pos)
            batch["spatial_pos"] = batch["spatial_pos_512x512"]
        if "face_area_512x1" in row.index:
            face_area = self._array(row["face_area_512x1"], np.float32).reshape(num_nodes, 1)
            batch["face_area_512x1"] = torch.from_numpy(face_area)
            batch["face_area"] = batch["face_area_512x1"]
        if "face_type_512" in row.index:
            face_type = self._array(row["face_type_512"], np.int16).reshape(num_nodes)
        else:
            face_type = np.zeros((num_nodes,), dtype=np.int16)
        batch["face_type_512"] = torch.from_numpy(face_type)
        batch["node_face_type"] = batch["face_type_512"]
        return batch


def resolve_parquet_files(
    parquet_dir: str,
    parquet_glob: str,
    explicit_parquet_paths: list[str],
    caller_name: str,
) -> list[str]:
    """Resolves either explicit parquet paths or all matching files in a folder."""
    files: list[Path] = []
    if explicit_parquet_paths:
        files.extend(Path(p).expanduser().resolve() for p in explicit_parquet_paths if str(p).strip())
    elif parquet_dir:
        files.extend(sorted(Path(parquet_dir).expanduser().resolve().glob(parquet_glob)))
    else:
        raise ValueError(f"Set either EXPLICIT_PARQUET_PATHS or PARQUET_DIR at the top of {caller_name}")

    unique_files: list[str] = []
    seen: set[str] = set()
    for path in files:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        unique_files.append(str(path))

    if not unique_files:
        raise ValueError("No parquet files matched the configured paths.")
    return unique_files


def split_indices(count: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Splits row indices into train/validation subsets."""
    if count <= 0:
        raise ValueError("Dataset has no rows.")
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError("VAL_RATIO must be in [0, 1).")
    if count < 2:
        return [0], [0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.arange(count, dtype=np.int64))
    val_count = int(round(count * float(val_ratio)))
    val_count = min(max(val_count, 1), count - 1)
    return perm[val_count:].tolist(), perm[:val_count].tolist()
