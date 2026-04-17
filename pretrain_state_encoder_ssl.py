"""Self-supervised pretraining for the StateEncoder.

Edit the constants below and run directly from VSCode/debug mode.
No CLI arguments are required.

Pretext tasks:
    1. Masked face type prediction.
    2. Masked face normal reconstruction.
    3. Masked node SDF/residual reconstruction.
    4. Masked face area reconstruction.
    5. Optional graph adjacency prediction from sampled node pairs.

This trains only the StateEncoder plus lightweight SSL heads.  Planner,
ActionEmbedding, and OctreeDecoder are not used.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from graph_sdf import GraphSdfModelConfig, StateEncoder


# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode.
# ---------------------------------------------------------------------------
PARQUET_DIR = r""
# Use "**/*.parquet" to include run subdirectories under PARQUET_DIR.
PARQUET_GLOB = "**/*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []

VAL_RATIO = 0.2
SEED = 0

BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 1

MASK_NODE_RATIO = 0.25
MASK_FACE_TYPE_ID = 0
MASK_NORMAL_AND_SDF = True
MASK_FACE_AREA = True

TYPE_LOSS_WEIGHT = 1.0
NORMAL_LOSS_WEIGHT = 1.0
SDF_LOSS_WEIGHT = 1.0
AREA_LOSS_WEIGHT = 0.3
EDGE_LOSS_WEIGHT = 0.5

USE_EDGE_LOSS = True
EDGE_PAIRS_PER_SAMPLE = 512

SAVE_CHECKPOINTS = True
CHECKPOINT_ROOT = r"C:\Users\inwoo\Desktop\5_Axis\checkpoints_state_encoder_ssl"
RUN_NAME = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_parquet_files() -> list[str]:
    files: list[Path] = []
    if EXPLICIT_PARQUET_PATHS:
        files.extend(Path(p).expanduser().resolve() for p in EXPLICIT_PARQUET_PATHS if str(p).strip())
    elif PARQUET_DIR:
        files.extend(sorted(Path(PARQUET_DIR).expanduser().resolve().glob(PARQUET_GLOB)))
    else:
        raise ValueError("Set either EXPLICIT_PARQUET_PATHS or PARQUET_DIR at the top of pretrain_state_encoder_ssl.py")

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


def _split_indices(count: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
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

            available = set(pq.ParquetFile(path).schema.names)
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


def _make_run_dir() -> Path | None:
    if not SAVE_CHECKPOINTS:
        return None
    run_name = RUN_NAME.strip() if RUN_NAME else f"state_encoder_ssl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(CHECKPOINT_ROOT).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class StateEncoderSslModel(nn.Module):
    """StateEncoder with small self-supervised prediction heads."""

    def __init__(self, config: GraphSdfModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        self.encoder = StateEncoder(config)
        self.face_type_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.face_type_vocab_size),
        )
        self.normal_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.sdf_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.area_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.edge_head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        state_points: torch.Tensor,
        node_process_state: torch.Tensor | None,
        node_centrality: torch.Tensor | None,
        spatial_pos: torch.Tensor | None,
        face_area: torch.Tensor | None,
        node_face_type: torch.Tensor | None,
        node_mask: torch.Tensor | None,
        point_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        node_embeddings = self.encoder(
            state_points,
            node_process_state=node_process_state,
            node_centrality=node_centrality,
            spatial_pos=spatial_pos,
            face_area=face_area,
            node_face_type=node_face_type,
            node_mask=node_mask,
            point_mask=point_mask,
        )
        return {
            "node_embeddings": node_embeddings,
            "face_type_logits": self.face_type_head(node_embeddings),
            "pred_normal": F.normalize(self.normal_head(node_embeddings), dim=-1, eps=1e-8),
            "pred_sdf": self.sdf_head(node_embeddings).squeeze(-1),
            "pred_log_area": self.area_head(node_embeddings).squeeze(-1),
        }

    def edge_logits(self, node_embeddings: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        """Predicts adjacency logits for batched node pairs.

        Args:
            node_embeddings: [B, N, H].
            src_idx: [B, M] source node indices.
            dst_idx: [B, M] destination node indices.
        """
        hidden = node_embeddings.shape[-1]
        src = torch.gather(node_embeddings, 1, src_idx.unsqueeze(-1).expand(-1, -1, hidden))
        dst = torch.gather(node_embeddings, 1, dst_idx.unsqueeze(-1).expand(-1, -1, hidden))
        pair_features = torch.cat([src, dst, torch.abs(src - dst)], dim=-1)
        return self.edge_head(pair_features).squeeze(-1)


def _optional_to_device(batch: dict, key: str, device: torch.device):
    value = batch.get(key)
    if value is None:
        return None
    return value.to(device)


def _masked_point_mean(values: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
    """Returns per-node mean over points, respecting point padding mask."""
    if point_mask is None:
        return values.mean(dim=2)
    valid = (~point_mask.bool()).unsqueeze(-1)
    if values.ndim == 3:
        valid = valid.squeeze(-1)
    count = valid.float().sum(dim=2 if values.ndim == 4 else 2).clamp_min(1.0)
    summed = (values * valid).sum(dim=2)
    return summed / count


def _node_targets(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    state_points = batch["state_points"].to(device)
    point_mask = _optional_to_device(batch, "point_mask", device)
    normals = state_points[..., 3:6]
    sdf = state_points[..., 6]
    target_normal = F.normalize(_masked_point_mean(normals, point_mask), dim=-1, eps=1e-8)
    target_sdf = _masked_point_mean(sdf, point_mask)

    node_count = state_points.shape[1]
    if batch.get("node_face_type") is not None:
        target_type = batch["node_face_type"].to(device).long()
    else:
        target_type = torch.zeros(state_points.shape[:2], dtype=torch.long, device=device)
    if batch.get("face_area") is not None:
        face_area = batch["face_area"].to(device).float().reshape(state_points.shape[0], node_count)
    else:
        face_area = torch.zeros(state_points.shape[:2], dtype=state_points.dtype, device=device)
    target_log_area = torch.log1p(face_area.clamp_min(0.0))
    return {
        "normal": target_normal,
        "sdf": target_sdf,
        "face_type": target_type,
        "log_area": target_log_area,
    }


def _build_node_ssl_mask(node_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Builds a random valid-node mask and guarantees at least one masked node per sample."""
    valid = ~node_mask.bool()
    ssl_mask = (torch.rand_like(valid.float()) < float(mask_ratio)) & valid
    for batch_idx in range(valid.shape[0]):
        if bool(valid[batch_idx].any()) and not bool(ssl_mask[batch_idx].any()):
            choices = valid[batch_idx].nonzero(as_tuple=True)[0]
            pick = choices[torch.randint(len(choices), (1,), device=choices.device)]
            ssl_mask[batch_idx, pick] = True
    return ssl_mask


def _corrupt_inputs(
    batch: dict,
    device: torch.device,
    ssl_mask: torch.Tensor,
    config: GraphSdfModelConfig,
) -> dict[str, torch.Tensor | None]:
    state_points = batch["state_points"].to(device).clone()
    node_process_state = _optional_to_device(batch, "node_process_state", device)
    node_centrality = _optional_to_device(batch, "node_centrality", device)
    spatial_pos = _optional_to_device(batch, "spatial_pos", device)
    node_mask = batch["node_mask"].to(device).bool()
    point_mask = _optional_to_device(batch, "point_mask", device)

    face_area = _optional_to_device(batch, "face_area", device)
    if face_area is None:
        face_area = state_points.new_zeros(state_points.shape[0], state_points.shape[1], config.face_area_feature_dim)
    else:
        face_area = face_area.clone()

    node_face_type = _optional_to_device(batch, "node_face_type", device)
    if node_face_type is None:
        node_face_type = torch.zeros(state_points.shape[:2], dtype=torch.long, device=device)
    else:
        node_face_type = node_face_type.clone()

    if MASK_NORMAL_AND_SDF:
        state_points[ssl_mask, :, 3:6] = 0.0
        state_points[ssl_mask, :, 6] = 0.0
    if MASK_FACE_AREA:
        face_area[ssl_mask] = 0.0
    node_face_type[ssl_mask] = int(MASK_FACE_TYPE_ID)

    return {
        "state_points": state_points,
        "node_process_state": node_process_state,
        "node_centrality": node_centrality,
        "spatial_pos": spatial_pos,
        "face_area": face_area,
        "node_face_type": node_face_type,
        "node_mask": node_mask,
        "point_mask": point_mask,
    }


def _sample_edge_pairs(
    spatial_pos: torch.Tensor,
    node_mask: torch.Tensor,
    pairs_per_sample: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Samples balanced positive/negative adjacency pairs from spatial_pos."""
    batch_size, node_count, _ = spatial_pos.shape
    src_all = []
    dst_all = []
    label_all = []
    for batch_idx in range(batch_size):
        valid = ~node_mask[batch_idx].bool()
        valid_pair = valid[:, None] & valid[None, :]
        not_self = ~torch.eye(node_count, dtype=torch.bool, device=spatial_pos.device)
        pos = ((spatial_pos[batch_idx] == 1) & valid_pair & not_self).nonzero(as_tuple=False)
        neg = ((spatial_pos[batch_idx] > 1) & valid_pair & not_self).nonzero(as_tuple=False)
        if pos.numel() == 0 or neg.numel() == 0:
            return None
        half = max(1, int(pairs_per_sample) // 2)
        pos_count = min(half, pos.shape[0])
        neg_count = min(int(pairs_per_sample) - pos_count, neg.shape[0])
        pos_idx = torch.randint(pos.shape[0], (pos_count,), device=spatial_pos.device)
        neg_idx = torch.randint(neg.shape[0], (neg_count,), device=spatial_pos.device)
        pairs = torch.cat([pos[pos_idx], neg[neg_idx]], dim=0)
        labels = torch.cat([
            torch.ones(pos_count, device=spatial_pos.device),
            torch.zeros(neg_count, device=spatial_pos.device),
        ])
        if pairs.shape[0] < int(pairs_per_sample):
            pad_count = int(pairs_per_sample) - pairs.shape[0]
            pad_idx = torch.randint(pairs.shape[0], (pad_count,), device=spatial_pos.device)
            pairs = torch.cat([pairs, pairs[pad_idx]], dim=0)
            labels = torch.cat([labels, labels[pad_idx]], dim=0)
        order = torch.randperm(pairs.shape[0], device=spatial_pos.device)
        pairs = pairs[order]
        labels = labels[order]
        src_all.append(pairs[:, 0])
        dst_all.append(pairs[:, 1])
        label_all.append(labels)

    return torch.stack(src_all, dim=0), torch.stack(dst_all, dim=0), torch.stack(label_all, dim=0)


def _compute_ssl_loss(
    model: StateEncoderSslModel,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = _node_targets(batch, device)
    node_mask = batch["node_mask"].to(device).bool()
    ssl_mask = _build_node_ssl_mask(node_mask, MASK_NODE_RATIO)
    corrupted = _corrupt_inputs(batch, device, ssl_mask, model.config)
    outputs = model(**corrupted)

    target_type = targets["face_type"].clamp(min=0, max=model.config.face_type_vocab_size - 1)
    type_loss = F.cross_entropy(outputs["face_type_logits"][ssl_mask], target_type[ssl_mask])

    normal_cos = (outputs["pred_normal"][ssl_mask] * targets["normal"][ssl_mask]).sum(dim=-1).clamp(-1.0, 1.0)
    normal_loss = (1.0 - normal_cos).mean()

    sdf_loss = F.smooth_l1_loss(outputs["pred_sdf"][ssl_mask], targets["sdf"][ssl_mask])
    area_loss = F.smooth_l1_loss(outputs["pred_log_area"][ssl_mask], targets["log_area"][ssl_mask])

    edge_loss = outputs["pred_sdf"].new_zeros(())
    edge_acc = float("nan")
    if USE_EDGE_LOSS and corrupted["spatial_pos"] is not None:
        edge_sample = _sample_edge_pairs(
            corrupted["spatial_pos"].long(),
            corrupted["node_mask"].bool(),
            EDGE_PAIRS_PER_SAMPLE,
        )
        if edge_sample is not None:
            src_idx, dst_idx, edge_labels = edge_sample
            edge_logits = model.edge_logits(outputs["node_embeddings"], src_idx, dst_idx)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            edge_pred = (torch.sigmoid(edge_logits) >= 0.5).float()
            edge_acc = float((edge_pred == edge_labels).float().mean().item())

    loss = (
        TYPE_LOSS_WEIGHT * type_loss
        + NORMAL_LOSS_WEIGHT * normal_loss
        + SDF_LOSS_WEIGHT * sdf_loss
        + AREA_LOSS_WEIGHT * area_loss
        + EDGE_LOSS_WEIGHT * edge_loss
    )

    with torch.no_grad():
        type_acc = float((outputs["face_type_logits"][ssl_mask].argmax(dim=-1) == target_type[ssl_mask]).float().mean().item())
        sdf_mae = float(torch.abs(outputs["pred_sdf"][ssl_mask] - targets["sdf"][ssl_mask]).mean().item())
        area_mae = float(torch.abs(outputs["pred_log_area"][ssl_mask] - targets["log_area"][ssl_mask]).mean().item())
        normal_cos_mean = float(normal_cos.mean().item())

    metrics = {
        "loss": float(loss.detach().item()),
        "type_loss": float(type_loss.detach().item()),
        "normal_loss": float(normal_loss.detach().item()),
        "sdf_loss": float(sdf_loss.detach().item()),
        "area_loss": float(area_loss.detach().item()),
        "edge_loss": float(edge_loss.detach().item()),
        "type_acc": type_acc,
        "normal_cos": normal_cos_mean,
        "sdf_mae": sdf_mae,
        "area_mae": area_mae,
        "edge_acc": edge_acc,
        "masked_ratio": float(ssl_mask.float().mean().item()),
    }
    return loss, metrics


def _merge_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in items if not np.isnan(m[key])]
        out[key] = float(sum(values) / max(len(values), 1)) if values else float("nan")
    return out


def _save_checkpoint(
    path: Path,
    model: StateEncoderSslModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    config: GraphSdfModelConfig,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "model_config": asdict(config),
        },
        path,
    )


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parquet_files = _resolve_parquet_files()

    dataset = StateEncoderSslParquetDataset(parquet_files)
    train_indices, val_indices = _split_indices(len(dataset), VAL_RATIO, SEED)
    train_dataset: Dataset = Subset(dataset, train_indices)
    val_dataset: Dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model_cfg = GraphSdfModelConfig()
    model = StateEncoderSslModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    run_dir = _make_run_dir()
    if run_dir is not None:
        _save_json(
            run_dir / "run_config.json",
            {
                "parquet_files": parquet_files,
                "seed": SEED,
                "val_ratio": VAL_RATIO,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "mask_node_ratio": MASK_NODE_RATIO,
                "loss_weights": {
                    "type": TYPE_LOSS_WEIGHT,
                    "normal": NORMAL_LOSS_WEIGHT,
                    "sdf": SDF_LOSS_WEIGHT,
                    "area": AREA_LOSS_WEIGHT,
                    "edge": EDGE_LOSS_WEIGHT,
                },
                "use_edge_loss": USE_EDGE_LOSS,
                "edge_pairs_per_sample": EDGE_PAIRS_PER_SAMPLE,
                "model_config": asdict(model_cfg),
                "train_rows": len(train_indices),
                "val_rows": len(val_indices),
            },
        )

    print(f"[Device] {device}")
    print(f"[Files] {len(parquet_files)} parquet files")
    print(f"[Rows] total={len(dataset)} train={len(train_indices)} val={len(val_indices)}")
    print(f"[Train] epochs={NUM_EPOCHS} lr={LEARNING_RATE} batch={BATCH_SIZE} mask_ratio={MASK_NODE_RATIO}")
    print(
        f"[Loss] type={TYPE_LOSS_WEIGHT} normal={NORMAL_LOSS_WEIGHT} "
        f"sdf={SDF_LOSS_WEIGHT} area={AREA_LOSS_WEIGHT} edge={EDGE_LOSS_WEIGHT}"
    )
    if run_dir is not None:
        print(f"[Checkpoint Dir] {run_dir}")

    best_val_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_metrics = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _compute_ssl_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            train_metrics.append(metrics)

        model.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                _, metrics = _compute_ssl_loss(model, batch, device)
                val_metrics.append(metrics)

        train_avg = _merge_metrics(train_metrics)
        val_avg = _merge_metrics(val_metrics)

        if run_dir is not None and (epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS):
            _save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, train_avg, val_avg, model_cfg)
        if run_dir is not None and val_avg.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_avg["loss"]
            _save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, train_avg, val_avg, model_cfg)

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS:
            print(
                f"[Epoch {epoch:04d}] "
                f"train={train_avg.get('loss', float('nan')):.6f} "
                f"val={val_avg.get('loss', float('nan')):.6f} | "
                f"type_acc={val_avg.get('type_acc', float('nan')):.4f} "
                f"normal_cos={val_avg.get('normal_cos', float('nan')):.4f} "
                f"sdf_mae={val_avg.get('sdf_mae', float('nan')):.6f} "
                f"edge_acc={val_avg.get('edge_acc', float('nan')):.4f}"
            )

    print("[Done] StateEncoder SSL pretraining finished.")


if __name__ == "__main__":
    main()
