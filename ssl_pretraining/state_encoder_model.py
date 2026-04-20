"""StateEncoder SSL model and loss utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_sdf import GraphSdfModelConfig, StateEncoder


@dataclass(frozen=True)
class StateEncoderSslLossConfig:
    """Loss and corruption settings for StateEncoder SSL."""

    mask_node_ratio: float = 0.25
    # -1 means "use the last face-type vocab id" as a dedicated SSL mask token.
    # This keeps the mask token separate from real class 0 / padding.
    mask_face_type_id: int = -1
    mask_normal_and_sdf: bool = True
    mask_face_area: bool = True
    type_loss_weight: float = 1.0
    normal_loss_weight: float = 1.0
    sdf_loss_weight: float = 1.0
    area_loss_weight: float = 0.3
    edge_loss_weight: float = 0.5
    use_edge_loss: bool = True
    edge_pairs_per_sample: int = 512
    # Optional CE weights for face-type reconstruction. Length must match
    # GraphSdfModelConfig.face_type_vocab_size when provided.
    face_type_class_weights: tuple[float, ...] | None = None


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
        """Predicts adjacency logits for batched node pairs."""
        hidden = node_embeddings.shape[-1]
        src = torch.gather(node_embeddings, 1, src_idx.unsqueeze(-1).expand(-1, -1, hidden))
        dst = torch.gather(node_embeddings, 1, dst_idx.unsqueeze(-1).expand(-1, -1, hidden))
        pair_features = torch.cat([src, dst, torch.abs(src - dst)], dim=-1)
        return self.edge_head(pair_features).squeeze(-1)


def optional_to_device(batch: dict, key: str, device: torch.device):
    value = batch.get(key)
    if value is None:
        return None
    return value.to(device)


def masked_point_mean(values: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
    """Returns per-node mean over points, respecting point padding mask."""
    if point_mask is None:
        return values.mean(dim=2)
    valid = (~point_mask.bool()).unsqueeze(-1)
    if values.ndim == 3:
        valid = valid.squeeze(-1)
    count = valid.float().sum(dim=2).clamp_min(1.0)
    summed = (values * valid).sum(dim=2)
    return summed / count


def node_targets(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Builds clean SSL targets from an uncorrupted batch."""
    state_points = batch["state_points"].to(device)
    point_mask = optional_to_device(batch, "point_mask", device)
    normals = state_points[..., 3:6]
    sdf = state_points[..., 6]
    target_normal = F.normalize(masked_point_mean(normals, point_mask), dim=-1, eps=1e-8)
    target_sdf = masked_point_mean(sdf, point_mask)

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


def build_node_ssl_mask(node_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Builds a random valid-node mask and guarantees at least one masked node per sample."""
    valid = ~node_mask.bool()
    ssl_mask = (torch.rand_like(valid.float()) < float(mask_ratio)) & valid
    for batch_idx in range(valid.shape[0]):
        if bool(valid[batch_idx].any()) and not bool(ssl_mask[batch_idx].any()):
            choices = valid[batch_idx].nonzero(as_tuple=True)[0]
            pick = choices[torch.randint(len(choices), (1,), device=choices.device)]
            ssl_mask[batch_idx, pick] = True
    return ssl_mask


def corrupt_inputs(
    batch: dict,
    device: torch.device,
    ssl_mask: torch.Tensor,
    model_config: GraphSdfModelConfig,
    loss_config: StateEncoderSslLossConfig,
) -> dict[str, torch.Tensor | None]:
    """Applies the masked-node corruption used by the SSL reconstruction tasks."""
    state_points = batch["state_points"].to(device).clone()
    node_process_state = optional_to_device(batch, "node_process_state", device)
    node_centrality = optional_to_device(batch, "node_centrality", device)
    spatial_pos = optional_to_device(batch, "spatial_pos", device)
    node_mask = batch["node_mask"].to(device).bool()
    point_mask = optional_to_device(batch, "point_mask", device)

    face_area = optional_to_device(batch, "face_area", device)
    if face_area is None:
        face_area = state_points.new_zeros(state_points.shape[0], state_points.shape[1], model_config.face_area_feature_dim)
    else:
        face_area = face_area.clone()

    node_face_type = optional_to_device(batch, "node_face_type", device)
    if node_face_type is None:
        node_face_type = torch.zeros(state_points.shape[:2], dtype=torch.long, device=device)
    else:
        node_face_type = node_face_type.clone()

    if loss_config.mask_normal_and_sdf:
        state_points[ssl_mask, :, 3:6] = 0.0
        state_points[ssl_mask, :, 6] = 0.0
    if loss_config.mask_face_area:
        face_area[ssl_mask] = 0.0
    mask_face_type_id = int(loss_config.mask_face_type_id)
    if mask_face_type_id < 0:
        mask_face_type_id = int(model_config.face_type_vocab_size) - 1
    mask_face_type_id = max(0, min(mask_face_type_id, int(model_config.face_type_vocab_size) - 1))
    node_face_type[ssl_mask] = mask_face_type_id

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


def sample_edge_pairs(
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


def compute_ssl_loss(
    model: StateEncoderSslModel,
    batch: dict,
    device: torch.device,
    loss_config: StateEncoderSslLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes StateEncoder SSL loss and scalar metrics."""
    targets = node_targets(batch, device)
    node_mask = batch["node_mask"].to(device).bool()
    ssl_mask = build_node_ssl_mask(node_mask, loss_config.mask_node_ratio)
    corrupted = corrupt_inputs(batch, device, ssl_mask, model.config, loss_config)
    outputs = model(**corrupted)

    target_type = targets["face_type"].clamp(min=0, max=model.config.face_type_vocab_size - 1)
    face_type_weight = None
    if loss_config.face_type_class_weights is not None:
        if len(loss_config.face_type_class_weights) != int(model.config.face_type_vocab_size):
            raise ValueError(
                "face_type_class_weights length must match face_type_vocab_size "
                f"({len(loss_config.face_type_class_weights)} != {model.config.face_type_vocab_size})"
            )
        face_type_weight = outputs["face_type_logits"].new_tensor(loss_config.face_type_class_weights)
    type_loss = F.cross_entropy(
        outputs["face_type_logits"][ssl_mask],
        target_type[ssl_mask],
        weight=face_type_weight,
    )

    normal_cos = (outputs["pred_normal"][ssl_mask] * targets["normal"][ssl_mask]).sum(dim=-1).clamp(-1.0, 1.0)
    normal_loss = (1.0 - normal_cos).mean()

    sdf_loss = F.smooth_l1_loss(outputs["pred_sdf"][ssl_mask], targets["sdf"][ssl_mask])
    area_loss = F.smooth_l1_loss(outputs["pred_log_area"][ssl_mask], targets["log_area"][ssl_mask])

    edge_loss = outputs["pred_sdf"].new_zeros(())
    edge_acc = math.nan
    if loss_config.use_edge_loss and corrupted["spatial_pos"] is not None:
        edge_sample = sample_edge_pairs(
            corrupted["spatial_pos"].long(),
            corrupted["node_mask"].bool(),
            loss_config.edge_pairs_per_sample,
        )
        if edge_sample is not None:
            src_idx, dst_idx, edge_labels = edge_sample
            edge_logits = model.edge_logits(outputs["node_embeddings"], src_idx, dst_idx)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            edge_pred = (torch.sigmoid(edge_logits) >= 0.5).float()
            edge_acc = float((edge_pred == edge_labels).float().mean().item())

    loss = (
        loss_config.type_loss_weight * type_loss
        + loss_config.normal_loss_weight * normal_loss
        + loss_config.sdf_loss_weight * sdf_loss
        + loss_config.area_loss_weight * area_loss
        + loss_config.edge_loss_weight * edge_loss
    )

    with torch.no_grad():
        target_type_masked = target_type[ssl_mask]
        pred_type_masked = outputs["face_type_logits"][ssl_mask].argmax(dim=-1)
        type_acc = float((pred_type_masked == target_type_masked).float().mean().item())
        sdf_mae = float(torch.abs(outputs["pred_sdf"][ssl_mask] - targets["sdf"][ssl_mask]).mean().item())
        area_mae = float(torch.abs(outputs["pred_log_area"][ssl_mask] - targets["log_area"][ssl_mask]).mean().item())
        normal_cos_mean = float(normal_cos.mean().item())
        type_counts: dict[str, float] = {}
        for class_id in range(int(model.config.face_type_vocab_size)):
            class_mask = target_type_masked == class_id
            count = int(class_mask.sum().item())
            correct = int(((pred_type_masked == target_type_masked) & class_mask).sum().item())
            type_counts[f"type_count_{class_id}"] = float(count)
            type_counts[f"type_correct_{class_id}"] = float(correct)

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
    metrics.update(type_counts)
    return loss, metrics


def merge_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    """Averages metric dictionaries while ignoring NaNs."""
    if not items:
        return {}
    keys = items[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in items if not np.isnan(m[key])]
        if key.startswith("type_count_") or key.startswith("type_correct_"):
            out[key] = float(sum(values)) if values else 0.0
        else:
            out[key] = float(sum(values) / max(len(values), 1)) if values else math.nan
    return out
