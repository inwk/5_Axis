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

    def __init__(self, parquet_files: Iterable[str | Path], octree_query_nodes: int | None = 2048) -> None:
        """Reads parquet files into a single dataframe index."""
        files = [str(Path(path)) for path in parquet_files]
        if not files:
            raise ValueError("At least one parquet file is required")

        frames = [pd.read_parquet(path) for path in files]
        self.df = pd.concat(frames, ignore_index=True)
        self.octree_query_nodes = octree_query_nodes

    def __len__(self) -> int:
        """Returns the number of operation rows."""
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
    def _optional_array(value, dtype, default_shape: tuple[int, ...]) -> np.ndarray:
        """Converts an optional parquet value or returns a zero default tensor."""
        if ProcessSkeletonParquetDataset._is_missing(value):
            return np.zeros(default_shape, dtype=dtype)
        arr = ProcessSkeletonParquetDataset._array(value, dtype)
        if arr.size == 0:
            return np.zeros(default_shape, dtype=dtype)
        return arr

    @staticmethod
    def _is_missing(value) -> bool:
        """Returns True when a parquet cell represents a missing optional value."""
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        return False

    @staticmethod
    def _fit_global_process_state(value, target_dim: int = 11) -> np.ndarray:
        """Pads or truncates process-history vectors for schema compatibility."""
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = np.zeros((target_dim,), dtype=np.float32)
        count = min(len(arr), target_dim)
        out[:count] = arr[:count]
        return out

    def _fit_octree_sample(
        self,
        centers: np.ndarray,
        depths: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples/repeats octree leaves to a fixed K so DataLoader can batch.

        Args:
            rng: Optional random generator.  Pass the same instance to two calls
                 so that before- and after-labels are sampled at the same indices.
        """
        target = self.octree_query_nodes
        if target is None:
            return centers, depths, labels
        target = int(target)
        if target <= 0:
            return centers, depths, labels

        count = int(min(centers.shape[0], depths.shape[0], labels.shape[0]))
        centers = centers[:count]
        depths = depths[:count]
        labels = labels[:count]
        if count == target:
            return centers, depths, labels
        if count <= 0:
            return (
                np.zeros((target, 3), dtype=np.float32),
                np.zeros((target,), dtype=np.int64),
                np.zeros((target,), dtype=np.float32),
            )

        if rng is None:
            rng = np.random.default_rng()
        if count > target:
            indices = rng.choice(count, size=target, replace=False)
        else:
            extra = rng.choice(count, size=target - count, replace=True)
            indices = np.concatenate([np.arange(count), extra])
            rng.shuffle(indices)
        return centers[indices], depths[indices], labels[indices]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Converts one dataframe row into the training batch schema."""
        row = self.df.iloc[int(index)]
        state_points = self._array(row["state_points"], np.float32).reshape(512, 100, 7)
        num_nodes = int(state_points.shape[0])
        points_per_node = int(state_points.shape[1])
        node_process_state = self._optional_array(
            row["node_process_state"] if "node_process_state" in row.index else None,
            np.float32,
            (num_nodes, 2),
        ).reshape(num_nodes, 2)
        next_point_sdf = None
        if "next_point_sdf" in row.index:
            next_point_sdf = self._optional_array(
                row["next_point_sdf"],
                np.float32,
                (num_nodes, points_per_node),
            ).reshape(num_nodes, points_per_node)
        elif "next_state_points" in row.index:
            next_state_points = self._optional_array(
                row["next_state_points"],
                np.float32,
                state_points.shape,
            ).reshape(state_points.shape)
            next_point_sdf = next_state_points[..., 6]

        next_node_sdf = None
        if "next_node_sdf" in row.index:
            next_node_sdf = self._optional_array(row["next_node_sdf"], np.float32, (num_nodes,)).reshape(num_nodes)
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
            "node_mask": torch.from_numpy(self._array(row["node_mask"], np.int16).reshape(num_nodes).astype(np.bool_)),
            "point_mask": torch.from_numpy(
                self._array(row["point_mask"], np.int16).reshape(num_nodes, points_per_node).astype(np.bool_)
            ),
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
                self._array(row["action_face_mask"], np.int16).reshape(num_nodes).astype(np.bool_)
            )
        elif "target_node_mask" in row.index:
            batch["action_face_mask"] = torch.from_numpy(
                self._array(row["target_node_mask"], np.int16).reshape(num_nodes).astype(np.bool_)
            )
        if "macro_class_mask" in row.index:
            batch["macro_class_mask"] = torch.from_numpy(
                self._array(row["macro_class_mask"], np.int16).reshape(-1).astype(np.bool_)
            )
        if "tool_choice_mask" in row.index:
            batch["tool_choice_mask"] = torch.from_numpy(
                self._array(row["tool_choice_mask"], np.int16).reshape(-1).astype(np.bool_)
            )
        if "centrality_512" in row.index:
            node_centrality = torch.from_numpy(self._array(row["centrality_512"], np.int16).reshape(num_nodes))
            batch["centrality_512"] = node_centrality
            batch["node_centrality"] = node_centrality
        if "spatial_pos_512x512" in row.index:
            spatial_pos = torch.from_numpy(self._array(row["spatial_pos_512x512"], np.int16).reshape(num_nodes, num_nodes))
            batch["spatial_pos_512x512"] = spatial_pos
            batch["spatial_pos"] = spatial_pos
        if "face_area_512x1" in row.index:
            face_area = torch.from_numpy(self._array(row["face_area_512x1"], np.float32).reshape(num_nodes, 1))
            batch["face_area_512x1"] = face_area
            batch["face_area"] = face_area
        if "axis_visible_512" in row.index:
            axis_visible = torch.from_numpy(self._array(row["axis_visible_512"], np.int16).reshape(num_nodes))
            batch["axis_visible_512"] = axis_visible
            batch["axis_visible"] = axis_visible

        # Adaptive octree occupancy data.
        # New schema:
        #   octree_centers:    [K, 3] normalized leaf centers (stored flat)
        #   octree_depths:     [K]    integer octree depth per leaf
        #   octree_occ_labels: [K]    1.0 = inside material after op
        # Legacy fallback maps occ_query_xyz/next_occ_labels to fine-depth leaves.
        centers_raw = row["octree_centers"] if "octree_centers" in row.index else None
        labels_raw = row["octree_occ_labels"] if "octree_occ_labels" in row.index else None
        depths_raw = row["octree_depths"] if "octree_depths" in row.index else None

        if self._is_missing(centers_raw) and "occ_query_xyz" in row.index:
            centers_raw = row["occ_query_xyz"]
        if self._is_missing(labels_raw) and "next_occ_labels" in row.index:
            labels_raw = row["next_occ_labels"]

        if not self._is_missing(centers_raw) and not self._is_missing(labels_raw):
            centers = self._array(centers_raw, np.float32).reshape(-1, 3)
            labels = self._array(labels_raw, np.float32).reshape(-1)
            if not self._is_missing(depths_raw):
                depths = self._array(depths_raw, np.int16).reshape(-1)
            else:
                depths = np.full((centers.shape[0],), 5, dtype=np.int16)

            count = min(centers.shape[0], labels.shape[0], depths.shape[0])
            if count > 0:
                # Use a single seeded RNG so before- and after-labels are sampled
                # at the same indices (required for the monotonicity constraint).
                sample_rng = np.random.default_rng(seed=abs(int(index)) % (2 ** 31))

                centers, depths, labels = self._fit_octree_sample(
                    centers[:count],
                    depths[:count].astype(np.int64),
                    labels[:count],
                    rng=sample_rng,
                )
                batch["octree_centers"] = torch.from_numpy(centers)
                batch["octree_depths"] = torch.from_numpy(depths.astype(np.int64))
                batch["octree_occ_labels"] = torch.from_numpy(labels)

                # ── Before-state occupancy labels (for monotonicity loss) ──
                # octree_occ_labels_before[i] is the occupancy of the same octree
                # cell as octree_occ_labels[i] *before* the current operation.
                # Requires data collection to store this field (see collect_axis_dataset.py).
                labels_before_raw = (
                    row["octree_occ_labels_before"] if "octree_occ_labels_before" in row.index else None
                )
                if not self._is_missing(labels_before_raw):
                    labels_before_full = self._array(labels_before_raw, np.float32).reshape(-1)
                    bcount = min(labels_before_full.shape[0], count)
                    if bcount > 0:
                        # Reset the same seeded RNG → same indices as after-labels.
                        sample_rng_b = np.random.default_rng(seed=abs(int(index)) % (2 ** 31))
                        centers_orig = self._array(centers_raw, np.float32).reshape(-1, 3)
                        depths_orig = self._array(depths_raw, np.int16).reshape(-1) if not self._is_missing(depths_raw) else np.full((centers_orig.shape[0],), 5, dtype=np.int16)
                        _, _, labels_before_sampled = self._fit_octree_sample(
                            centers_orig[:count],
                            depths_orig[:count].astype(np.int64),
                            labels_before_full[:count],
                            rng=sample_rng_b,
                        )
                        batch["octree_occ_labels_before"] = torch.from_numpy(labels_before_sampled)

        bbox_min_raw = row["octree_bbox_min"] if "octree_bbox_min" in row.index else None
        bbox_max_raw = row["octree_bbox_max"] if "octree_bbox_max" in row.index else None
        if self._is_missing(bbox_min_raw) and "occ_bbox_min" in row.index:
            bbox_min_raw = row["occ_bbox_min"]
        if self._is_missing(bbox_max_raw) and "occ_bbox_max" in row.index:
            bbox_max_raw = row["occ_bbox_max"]

        if not self._is_missing(bbox_min_raw):
            batch["octree_bbox_min"] = torch.from_numpy(self._array(bbox_min_raw, np.float32).reshape(3))
        if not self._is_missing(bbox_max_raw):
            batch["octree_bbox_max"] = torch.from_numpy(self._array(bbox_max_raw, np.float32).reshape(3))
        return batch
