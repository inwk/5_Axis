"""PyTorch dataset for process-skeleton parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import TOOL_CHOICE_TO_ID, tool_choice_key


class ProcessSkeletonParquetDataset(Dataset):
    """Loads process-skeleton rows from one or more parquet files."""

    def __init__(self, parquet_files: Iterable[str | Path]) -> None:
        """Reads parquet files into a single dataframe index."""
        files = [str(Path(path)) for path in parquet_files]
        if not files:
            raise ValueError("At least one parquet file is required")

        frames = [pd.read_parquet(path) for path in files]
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        """Returns the number of operation rows."""
        return int(len(self.df))

    @staticmethod
    def _array(value, dtype) -> np.ndarray:
        """Converts parquet list/object values into numpy arrays."""
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def _optional_array(value, dtype, default_shape: tuple[int, ...]) -> np.ndarray:
        """Converts an optional parquet value or returns a zero default tensor."""
        if value is None:
            return np.zeros(default_shape, dtype=dtype)
        if isinstance(value, float) and np.isnan(value):
            return np.zeros(default_shape, dtype=dtype)
        arr = np.asarray(value, dtype=dtype)
        if arr.size == 0:
            return np.zeros(default_shape, dtype=dtype)
        return arr

    @staticmethod
    def _fit_global_process_state(value, target_dim: int = 11) -> np.ndarray:
        """Pads or truncates process-history vectors for schema compatibility."""
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = np.zeros((target_dim,), dtype=np.float32)
        count = min(len(arr), target_dim)
        out[:count] = arr[:count]
        return out

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Converts one dataframe row into the training batch schema."""
        row = self.df.iloc[int(index)]
        state_points = self._array(row["state_points"], np.float32)
        num_nodes = int(state_points.shape[0])
        node_process_state = self._optional_array(
            row["node_process_state"] if "node_process_state" in row.index else None,
            np.float32,
            (num_nodes, 2),
        )
        next_point_sdf = None
        if "next_point_sdf" in row.index:
            next_point_sdf = self._optional_array(
                row["next_point_sdf"],
                np.float32,
                (num_nodes, state_points.shape[1]),
            )
        elif "next_state_points" in row.index:
            next_state_points = self._optional_array(
                row["next_state_points"],
                np.float32,
                state_points.shape,
            )
            next_point_sdf = next_state_points[..., 6]

        next_node_sdf = None
        if "next_node_sdf" in row.index:
            next_node_sdf = self._optional_array(row["next_node_sdf"], np.float32, (num_nodes,))
        elif next_point_sdf is not None:
            next_node_sdf = next_point_sdf.mean(axis=1, dtype=np.float32)
        else:
            next_node_sdf = np.zeros((num_nodes,), dtype=np.float32)

        tool_choice_id = row["tool_choice_id"] if "tool_choice_id" in row.index else None
        if tool_choice_id is None or (isinstance(tool_choice_id, float) and np.isnan(tool_choice_id)):
            tool_kind = row["tool_type_name"] if "tool_type_name" in row.index else None
            tool_diameter = row["tool_diameter"] if "tool_diameter" in row.index else None
            if tool_kind is not None and tool_diameter is not None:
                key = tool_choice_key(str(tool_kind), float(tool_diameter))
                tool_choice_id = TOOL_CHOICE_TO_ID.get(key, -1)
            else:
                tool_choice_id = -1
        tool_choice_valid = row["tool_choice_valid"] if "tool_choice_valid" in row.index else None
        if tool_choice_valid is None or (isinstance(tool_choice_valid, float) and np.isnan(tool_choice_valid)):
            tool_choice_valid = 1 if int(tool_choice_id) >= 0 else 0

        action_face_id = None
        if "action_face_id" in row.index:
            action_face_id = row["action_face_id"]
        elif "target_node_id" in row.index:
            action_face_id = row["target_node_id"]
        elif "target_face_id" in row.index:
            action_face_id = row["target_face_id"]
        elif "anchor_face_id" in row.index:
            action_face_id = row["anchor_face_id"]
        if action_face_id is None or (isinstance(action_face_id, float) and np.isnan(action_face_id)):
            action_face_id = -1

        action_face_valid = row["action_face_valid"] if "action_face_valid" in row.index else None
        if action_face_valid is None or (isinstance(action_face_valid, float) and np.isnan(action_face_valid)):
            legacy_valid = row["target_node_valid"] if "target_node_valid" in row.index else None
            if legacy_valid is None or (isinstance(legacy_valid, float) and np.isnan(legacy_valid)):
                action_face_valid = 1 if int(action_face_id) >= 0 else 0
            else:
                action_face_valid = legacy_valid

        batch = {
            "state_points": torch.from_numpy(state_points),
            "node_process_state": torch.from_numpy(node_process_state),
            "next_node_sdf": torch.from_numpy(next_node_sdf),
            "node_mask": torch.from_numpy(self._array(row["node_mask"], np.int16).astype(np.bool_)),
            "point_mask": torch.from_numpy(self._array(row["point_mask"], np.int16).astype(np.bool_)),
            "macro_class_id": torch.tensor(int(row["macro_class_id"]), dtype=torch.long),
            "tool_choice_id": torch.tensor(int(max(int(tool_choice_id), 0)), dtype=torch.long),
            "action_face_id": torch.tensor(int(action_face_id), dtype=torch.long),
            "action_face_valid": torch.tensor(int(action_face_valid), dtype=torch.float32),
            "tool_choice_valid": torch.tensor(int(tool_choice_valid), dtype=torch.float32),
        }
        if next_point_sdf is not None:
            batch["next_point_sdf"] = torch.from_numpy(next_point_sdf)

        if "is_chosen" in row.index:
            batch["is_chosen"] = torch.tensor(int(row["is_chosen"]), dtype=torch.bool)

        if "global_process_state" in row.index:
            batch["global_process_state"] = torch.from_numpy(
                self._fit_global_process_state(row["global_process_state"], target_dim=11)
            )
        if "action_face_mask" in row.index:
            batch["action_face_mask"] = torch.from_numpy(
                self._array(row["action_face_mask"], np.int16).astype(np.bool_)
            )
        elif "target_node_mask" in row.index:
            batch["action_face_mask"] = torch.from_numpy(
                self._array(row["target_node_mask"], np.int16).astype(np.bool_)
            )
        if "macro_class_mask" in row.index:
            batch["macro_class_mask"] = torch.from_numpy(
                self._array(row["macro_class_mask"], np.int16).astype(np.bool_)
            )
        if "tool_choice_mask" in row.index:
            batch["tool_choice_mask"] = torch.from_numpy(
                self._array(row["tool_choice_mask"], np.int16).astype(np.bool_)
            )
        if "centrality_512" in row.index:
            node_centrality = torch.from_numpy(self._array(row["centrality_512"], np.int16))
            batch["centrality_512"] = node_centrality
            batch["node_centrality"] = node_centrality
        if "spatial_pos_512x512" in row.index:
            spatial_pos = torch.from_numpy(self._array(row["spatial_pos_512x512"], np.int16))
            batch["spatial_pos_512x512"] = spatial_pos
            batch["spatial_pos"] = spatial_pos
        if "face_area_512x1" in row.index:
            face_area = torch.from_numpy(self._array(row["face_area_512x1"], np.float32))
            batch["face_area_512x1"] = face_area
            batch["face_area"] = face_area
        if "axis_visible_512" in row.index:
            axis_visible = torch.from_numpy(self._array(row["axis_visible_512"], np.int16))
            batch["axis_visible_512"] = axis_visible
            batch["axis_visible"] = axis_visible

        return batch
