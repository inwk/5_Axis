"""Train the transition-only octree model on multiple parquet files.

Edit the constants below and run directly from VSCode/debug mode.
No CLI args are required.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from graph_sdf import GraphSdfModelConfig, GraphSdfPlanningModel, ProcessSkeletonParquetDataset
from graph_sdf.schema import macro_class_name_from_id
from graph_sdf.training import transition_train_step, transition_validation_step


# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode.
# ---------------------------------------------------------------------------
PARQUET_DIR = r""
# Use "**/*.parquet" to include run subdirectories under PARQUET_DIR.
PARQUET_GLOB = "**/*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = [r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\_ALL_PARQUET_FILES\3dDataset3063_seed0_20260523_232918.parquet"]
SPLIT_MANIFEST_PATH = os.getenv("PILOT_SPLIT_MANIFEST", r"")
TRAIN_SPLIT_NAME = os.getenv("PILOT_TRAIN_SPLIT", "train")
VAL_SPLIT_NAME = os.getenv("PILOT_VAL_SPLIT", "val")

VAL_RATIO = 0.2
SEED = 0

BATCH_SIZE = 1
NUM_WORKERS = 2
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
PRINT_EVERY = 1
TRAIN_LOG_EVERY_BATCHES = 50
MAX_TRAIN_BATCHES_PER_EPOCH = 0  # 0 = use all train batches
MAX_VAL_BATCHES = 0             # cap pilot validation cost/memory churn
MAX_TRAIN_FILES = 32             # 0 = use all train files from the split
MAX_VAL_FILES = 8                # 0 = use all val files from the split
LAZY_PARQUET_LOADING = True      # keep RAM bounded by loading parquet row groups on demand
PARQUET_ROW_GROUP_CACHE_SIZE = 16
LOCALITY_SORT_LAZY_INDICES = True
SPLIT_BY_PART = bool(int(os.getenv("SPLIT_BY_PART", "1")))
SDF_RESAMPLE_EACH_EPOCH = False
USE_TARGET_TSDF_INPUT = True
OVERFIT_DEBUG_MODE = "one_part"
OVERFIT_PART_NAME = os.getenv("OVERFIT_PART_NAME", "").strip()

# Optional StateEncoder initialization from SSL pretraining.
# Set this to ssl_pretraining/train_state_encoder_ssl.py's best.pt.
PRETRAINED_ENCODER_CHECKPOINT = os.getenv("PRETRAINED_ENCODER_CHECKPOINT", r"")
PRETRAINED_ENCODER_STRICT = True
FREEZE_STATE_ENCODER = False
STATE_ENCODER_LR_MULTIPLIER = 1.0

# For SDF-only transition learning, query points are randomly subsampled per row.
SDF_QUERY_NODES = 32768
OCTREE_QUERY_NODES = 4096
OCTREE_POS_WEIGHT_FACTOR = 2.0
OCTREE_DEPTH_WEIGHT_BASE = 2.0
MONOTONICITY_WEIGHT = 0.1
OCCUPANCY_LOSS_WEIGHT = 0.1
OCTREE_FILL_FRACTION_WEIGHT = 0.5
OCTREE_REMOVED_FRACTION_WEIGHT = 2.0
TSDF_LOSS_WEIGHT = 1.0
DELTA_TSDF_LOSS_WEIGHT = 0.5
CHANGED_TSDF_LOSS_WEIGHT = 2.0
CHANGED_DELTA_TSDF_LOSS_WEIGHT = 1.0
TSDF_CHANGE_EPS = 1e-3
TSDF_MONOTONICITY_WEIGHT = 0.1
TSDF_MONOTONICITY_EMPTY_MARGIN = 0.2
AFFECTED_FACE_LOSS_WEIGHT = 0.2
AFFECTED_FACE_DELTA_LOSS_WEIGHT = 0.1

SAVE_CHECKPOINTS = False
CHECKPOINT_ROOT = r"C:\Users\inwoo\Desktop\5_Axis\checkpoints_transition"
RUN_NAME = os.getenv("RUN_NAME", "")


def _cuda_memory_text() -> str:
    """Returns a compact CUDA memory summary for progress logs."""
    if not torch.cuda.is_available():
        return "cuda=n/a"
    allocated = torch.cuda.memory_allocated() / (1024.0 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024.0 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    return f"cuda_alloc={allocated:.2f}GB cuda_reserved={reserved:.2f}GB cuda_peak={peak:.2f}GB"


LOSS_COMPONENT_LOG_ORDER = [
    "tsdf",
    "delta_tsdf",
    "changed_tsdf",
    "changed_delta_tsdf",
    "tsdf_mono",
    "affected_face",
    "affected_delta",
]


def _accumulate_loss_components(sums: dict[str, float], components: dict[str, float]) -> None:
    """Accumulates weighted loss contributions for epoch-level logging."""
    for key, value in components.items():
        sums[key] = sums.get(key, 0.0) + float(value)


def _average_loss_components(sums: dict[str, float], count: int) -> dict[str, float]:
    denom = max(int(count), 1)
    return {key: value / denom for key, value in sums.items()}


def _format_loss_components(label: str, components: dict[str, float]) -> str:
    if not components:
        return ""
    ordered = [key for key in LOSS_COMPONENT_LOG_ORDER if key in components]
    ordered.extend(sorted(key for key in components if key not in set(ordered)))
    body = " ".join(f"{key}={components[key]:.4f}" for key in ordered)
    return f" {label}_losses({body})"


def _format_macro_metrics(metrics: dict[str, float]) -> str:
    """Formats per-macro SDF validation metrics compactly."""
    macro_names = sorted({
        key.split(".")[1]
        for key in metrics
        if key.startswith("macro.") and len(key.split(".")) >= 3
    })
    if not macro_names:
        return ""
    parts = []
    for name in macro_names:
        prefix = f"macro.{name}"
        fields = []
        tsdf_key = f"{prefix}.tsdf_mae"
        changed_key = f"{prefix}.changed_tsdf_mae"
        ratio_key = f"{prefix}.tsdf_changed_ratio"
        if tsdf_key in metrics:
            fields.append(f"tsdf={metrics[tsdf_key]:.4f}")
        if changed_key in metrics:
            fields.append(f"changed={metrics[changed_key]:.4f}")
        if ratio_key in metrics:
            fields.append(f"ratio={metrics[ratio_key]:.4f}")
        if fields:
            parts.append(f"{name}(" + " ".join(fields) + ")")
    return " val_macro_metrics=" + ";".join(parts) if parts else ""


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


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
        raise ValueError("Set either EXPLICIT_PARQUET_PATHS or PARQUET_DIR at the top of train_transition_dataset.py")

    unique_files = []
    seen = set()
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


def _load_manifest_files(manifest_path: str, split_name: str) -> list[str]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SPLIT_MANIFEST_PATH not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits", {})
    if split_name not in splits:
        raise KeyError(f"Split {split_name!r} not found in manifest: {path}")
    files = [str(Path(p).expanduser().resolve()) for p in splits[split_name]]
    if not files:
        raise ValueError(f"Manifest split {split_name!r} has no parquet files: {path}")
    for file_path in files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Parquet file from manifest not found: {file_path}")
    return files


def _cap_files(files: list[str], max_files: int) -> list[str]:
    """Returns a deterministic pilot subset to avoid eager-loading every parquet."""
    limit = int(max_files)
    if limit <= 0 or len(files) <= limit:
        return files
    return files[:limit]


def _build_valid_transition_indices(dataset: ProcessSkeletonParquetDataset) -> list[int]:
    if hasattr(dataset, "valid_sdf_indices"):
        indices = dataset.valid_sdf_indices(required=("sdf_query_points", "sdf_tsdf_after"))
        if not indices:
            raise ValueError("No rows with complete SDF query supervision were found.")
        return indices

    if getattr(dataset, "lazy_load", False):
        required = {"sdf_query_points", "sdf_tsdf_after"}
        indices: list[int] = []
        for idx in dataset.row_indices():
            row = dataset._row_from_lazy_index(int(idx))
            if all(col in row.index and not dataset._is_missing(row[col]) for col in required):
                indices.append(int(idx))
        if not indices:
            raise ValueError("No rows with complete SDF query supervision were found.")
        return indices

    indices: list[int] = []
    df = dataset.df
    required = {"sdf_query_points", "sdf_tsdf_after"}
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required SDF query columns: {missing_cols}")

    for idx, row in df.iterrows():
        if (
            not dataset._is_missing(row["sdf_query_points"])
            and not dataset._is_missing(row["sdf_tsdf_after"])
        ):
            indices.append(int(idx))
    if not indices:
        raise ValueError("No rows with complete SDF query supervision were found.")
    return indices


def _split_indices(indices: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError("VAL_RATIO must be in [0, 1).")
    if len(indices) < 2:
        return indices, indices

    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(indices, dtype=np.int64))
    val_count = int(round(len(indices) * float(val_ratio)))
    val_count = min(max(val_count, 1), len(indices) - 1)
    val_indices = perm[:val_count].tolist()
    train_indices = perm[val_count:].tolist()
    return train_indices, val_indices


def _metadata_group_key(values: dict[str, object], fallback_index: int) -> str:
    """Returns a stable part/run grouping key for leakage-resistant splits."""
    part_name = values.get("part_name")
    if part_name is not None and str(part_name).strip():
        return f"part:{str(part_name).strip()}"

    prt_path = values.get("prt_file_path")
    if prt_path is not None and str(prt_path).strip():
        return f"prt:{Path(str(prt_path)).stem}"

    static_dir = values.get("static_feature_dir")
    if static_dir is not None and str(static_dir).strip():
        return f"static:{Path(str(static_dir)).name}"

    return f"row:{int(fallback_index)}"


def _build_part_groups(
    dataset: ProcessSkeletonParquetDataset,
    indices: list[int],
) -> dict[str, list[int]]:
    """Groups row indices by part/run metadata without loading payload arrays."""
    metadata = dataset.row_metadata(
        indices,
        columns=("part_name", "prt_file_path", "static_feature_dir"),
    )
    groups: dict[str, list[int]] = {}
    for idx in indices:
        key = _metadata_group_key(metadata.get(int(idx), {}), int(idx))
        groups.setdefault(key, []).append(int(idx))
    return {key: sorted(values) for key, values in groups.items()}


def _split_indices_by_part(
    dataset: ProcessSkeletonParquetDataset,
    indices: list[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], int]:
    """Splits by part/run group so one part does not leak into both train and val."""
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError("VAL_RATIO must be in [0, 1).")
    if len(indices) < 2:
        return indices, indices, 1

    groups = _build_part_groups(dataset, indices)
    if len(groups) < 2:
        train_indices, val_indices = _split_indices(indices, val_ratio, seed)
        return train_indices, val_indices, len(groups)

    rng = np.random.default_rng(seed)
    group_keys = np.asarray(sorted(groups.keys()), dtype=object)
    permuted_keys = rng.permutation(group_keys).tolist()
    target_val_rows = int(round(len(indices) * float(val_ratio)))
    target_val_rows = min(max(target_val_rows, 1), len(indices) - 1)

    val_keys: set[str] = set()
    val_count = 0
    for key in permuted_keys:
        if len(val_keys) >= len(groups) - 1:
            break
        if val_keys and val_count >= target_val_rows:
            break
        val_keys.add(str(key))
        val_count += len(groups[str(key)])

    if not val_keys:
        val_keys.add(str(permuted_keys[0]))

    train_indices: list[int] = []
    val_indices: list[int] = []
    for key, row_indices in groups.items():
        if key in val_keys:
            val_indices.extend(row_indices)
        else:
            train_indices.extend(row_indices)

    if not train_indices or not val_indices:
        fallback_train, fallback_val = _split_indices(indices, val_ratio, seed)
        return fallback_train, fallback_val, len(groups)
    return train_indices, val_indices, len(groups)


def _apply_overfit_debug_split(
    dataset: ProcessSkeletonParquetDataset,
    indices: list[int],
    mode: str,
    part_name: str,
) -> tuple[list[int], list[int], str]:
    """Uses train=val on one row or one part to test whether the model can overfit."""
    normalized = str(mode).strip().lower()
    if not normalized:
        return [], [], ""
    if not indices:
        raise ValueError("OVERFIT_DEBUG_MODE requested but no valid rows are available.")

    if normalized in {"one_row", "row"}:
        idx = int(indices[0])
        return [idx], [idx], f"one_row:index={idx}"

    if normalized in {"one_part", "part"}:
        groups = _build_part_groups(dataset, indices)
        if not groups:
            raise ValueError("OVERFIT_DEBUG_MODE=one_part requested but no part groups were found.")
        selected_key = ""
        if part_name:
            for key in groups:
                if key == part_name or key.endswith(f":{part_name}"):
                    selected_key = key
                    break
            if not selected_key:
                raise ValueError(f"OVERFIT_PART_NAME={part_name!r} did not match any part group.")
        else:
            selected_key = max(sorted(groups.keys()), key=lambda key: len(groups[key]))
        selected_indices = groups[selected_key]
        return selected_indices, selected_indices, f"one_part:{selected_key}:rows={len(selected_indices)}"

    raise ValueError("OVERFIT_DEBUG_MODE must be '', 'one_row', or 'one_part'.")


def _macro_distribution(dataset: ProcessSkeletonParquetDataset, indices: list[int]) -> dict[str, int]:
    if hasattr(dataset, "macro_distribution"):
        return dataset.macro_distribution(indices)

    out: dict[str, int] = {}
    if "macro_class_name" not in dataset.df.columns:
        return out
    for idx in indices:
        name = str(dataset.df.iloc[int(idx)].get("macro_class_name", "unknown"))
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _locality_sort_indices(dataset: ProcessSkeletonParquetDataset, indices: list[int]) -> list[int]:
    """Keeps random split membership but orders rows for sequential parquet reads."""
    if not bool(LOCALITY_SORT_LAZY_INDICES) or not getattr(dataset, "lazy_load", False):
        return indices
    if hasattr(dataset, "sort_indices_by_storage_order"):
        return dataset.sort_indices_by_storage_order(indices)
    return sorted(int(idx) for idx in indices)


def _make_dataloader(dataset: Dataset, shuffle: bool, persistent_workers: bool | None = None) -> DataLoader:
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": bool(shuffle),
        "num_workers": int(NUM_WORKERS),
        "pin_memory": torch.cuda.is_available(),
    }
    use_persistent_workers = (
        bool(persistent_workers)
        if persistent_workers is not None
        else True
    )
    if int(NUM_WORKERS) > 0:
        kwargs["persistent_workers"] = use_persistent_workers
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _set_dataset_sdf_sample_epoch(dataset: Dataset, epoch: int) -> None:
    """Propagates the SDF query sample epoch through Subset wrappers."""
    if hasattr(dataset, "set_sdf_sample_epoch"):
        dataset.set_sdf_sample_epoch(epoch)
    if isinstance(dataset, Subset):
        _set_dataset_sdf_sample_epoch(dataset.dataset, epoch)


def _strip_prefix_state_dict(state_dict: dict, prefix: str) -> dict:
    """Strips a module prefix from every key that has the prefix."""
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if str(key).startswith(prefix)
    }


def _extract_encoder_state_dict(checkpoint: object) -> tuple[dict, str]:
    """Extracts a StateEncoder state dict from supported checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "encoder_state_dict" in checkpoint:
            return checkpoint["encoder_state_dict"], "encoder_state_dict"

        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            state_encoder_state = _strip_prefix_state_dict(model_state, "state_encoder.")
            if state_encoder_state:
                return state_encoder_state, "model_state_dict:state_encoder."
            ssl_encoder_state = _strip_prefix_state_dict(model_state, "encoder.")
            if ssl_encoder_state:
                return ssl_encoder_state, "model_state_dict:encoder."

        # Allow a raw state dict saved directly from StateEncoder.state_dict().
        if (
            checkpoint
            and all(isinstance(key, str) for key in checkpoint.keys())
            and all(torch.is_tensor(value) for value in checkpoint.values())
        ):
            return checkpoint, "raw_state_dict"

    raise ValueError(
        "Could not find an encoder state dict. Expected one of: "
        "'encoder_state_dict', 'model_state_dict' with 'encoder.' prefix, "
        "'model_state_dict' with 'state_encoder.' prefix, or a raw state dict."
    )


def _load_pretrained_encoder(
    model: GraphSdfPlanningModel,
    checkpoint_path: str,
    strict: bool,
) -> dict[str, object]:
    """Loads a pretrained SSL StateEncoder into the transition model."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PRETRAINED_ENCODER_CHECKPOINT not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    encoder_state, source_key = _extract_encoder_state_dict(checkpoint)
    result = model.state_encoder.load_state_dict(encoder_state, strict=bool(strict))
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return {
        "checkpoint_path": str(path),
        "source_key": source_key,
        "strict": bool(strict),
        "num_tensors": len(encoder_state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _set_state_encoder_trainability(model: GraphSdfPlanningModel, freeze: bool) -> None:
    """Freezes/unfreezes the StateEncoder parameters."""
    for param in model.state_encoder.parameters():
        param.requires_grad = not bool(freeze)


def _build_optimizer(
    model: GraphSdfPlanningModel,
    learning_rate: float,
    encoder_lr_multiplier: float,
    freeze_state_encoder: bool,
) -> torch.optim.Optimizer:
    """Builds AdamW with optional lower LR for pretrained encoder parameters."""
    _set_state_encoder_trainability(model, freeze=freeze_state_encoder)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after applying freeze settings.")

    if freeze_state_encoder or abs(float(encoder_lr_multiplier) - 1.0) < 1e-12:
        return torch.optim.AdamW(trainable_params, lr=learning_rate)

    encoder_param_ids = {id(param) for param in model.state_encoder.parameters()}
    encoder_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) in encoder_param_ids
    ]
    other_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in encoder_param_ids
    ]
    param_groups = []
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": float(learning_rate) * float(encoder_lr_multiplier),
        })
    if other_params:
        param_groups.append({"params": other_params, "lr": float(learning_rate)})
    return torch.optim.AdamW(param_groups)


def _octree_before_for_decoder(batch: dict, device: torch.device) -> torch.Tensor | None:
    """Builds [before_tsdf, target_tsdf, before_fill_or_occ, known] query features."""
    before_fill = None
    if batch.get("octree_fill_before") is not None:
        before_fill = batch["octree_fill_before"].to(device).float().clamp(0.0, 1.0)
    elif batch.get("octree_occ_labels_before") is not None:
        before_valid = batch.get("octree_occ_labels_before_valid")
        if before_valid is None or bool(before_valid.sum().item() >= before_valid.numel()):
            before_fill = batch["octree_occ_labels_before"].to(device).float().clamp(0.0, 1.0)

    before_tsdf = (
        batch["octree_tsdf_before"].to(device).float().clamp(-1.0, 1.0)
        if batch.get("octree_tsdf_before") is not None
        else None
    )
    target_tsdf = (
        batch["octree_target_tsdf"].to(device).float().clamp(-1.0, 1.0)
        if batch.get("octree_target_tsdf") is not None
        else None
    )
    if before_fill is None and before_tsdf is None and target_tsdf is None:
        return None
    ref = before_tsdf if before_tsdf is not None else target_tsdf if target_tsdf is not None else before_fill
    zeros = torch.zeros_like(ref)
    ones = torch.ones_like(ref)
    return torch.stack([
        before_tsdf if before_tsdf is not None else zeros,
        target_tsdf if target_tsdf is not None else zeros,
        before_fill if before_fill is not None else zeros,
        ones,
    ], dim=-1)


@torch.no_grad()
def _evaluate_octree_metrics(model: GraphSdfPlanningModel, batch: dict, device: torch.device) -> dict[str, float]:
    model.eval()
    state_embedding = model.encode_state(
        state_points=batch["state_points"].to(device),
        node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
        node_face_type=batch.get("node_face_type").to(device) if batch.get("node_face_type") is not None else None,
        node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
    )
    out = model.forward_octree(
        state_points=batch["state_points"].to(device),
        macro_class_id=batch["macro_class_id"].to(device),
        tool_choice_id=batch["tool_choice_id"].to(device),
        action_face_id=batch["action_face_id"].clamp_min(0).to(device),
        octree_centers=batch["octree_centers"].to(device),
        octree_depths=batch["octree_depths"].to(device),
        octree_occ_before=_octree_before_for_decoder(batch, device),
        axis_visible=batch.get("axis_visible").to(device) if batch.get("axis_visible") is not None else None,
        axis_dir=batch.get("axis_dir").to(device) if batch.get("axis_dir") is not None else None,
        tool_radius_norm=batch.get("tool_radius_norm").to(device) if batch.get("tool_radius_norm") is not None else None,
        node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
        node_face_type=batch.get("node_face_type").to(device) if batch.get("node_face_type") is not None else None,
        node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
        state_embedding=state_embedding,
    )
    prob = torch.sigmoid(out["occ_logits"])
    pred = (prob >= 0.5).float()
    gt = batch["octree_occ_labels"].to(device)
    metrics = {
        "acc": float((pred == gt).float().mean().item()),
        "pred_pos": float(pred.mean().item()),
        "gt_pos": float(gt.mean().item()),
        "prob_mean": float(prob.mean().item()),
        "num_cells": float(gt.numel()),
    }
    if batch.get("octree_tsdf_after") is not None and out.get("tsdf") is not None:
        pred_tsdf = out["tsdf"]
        target_tsdf = batch["octree_tsdf_after"].to(device).float()
        metrics["tsdf_mae"] = float((pred_tsdf - target_tsdf).abs().mean().item())
        metrics["pred_tsdf_mean"] = float(pred_tsdf.mean().item())
        metrics["gt_tsdf_mean"] = float(target_tsdf.mean().item())
        if batch.get("octree_tsdf_before") is not None:
            before_tsdf = batch["octree_tsdf_before"].to(device).float()
            target_delta = (
                batch["octree_delta_tsdf"].to(device).float()
                if batch.get("octree_delta_tsdf") is not None
                else target_tsdf - before_tsdf
            )
            pred_delta = pred_tsdf - before_tsdf
            changed_tsdf = target_delta.abs() > float(TSDF_CHANGE_EPS)
            metrics["delta_tsdf_mae"] = float((pred_delta - target_delta).abs().mean().item())
            metrics["tsdf_changed_ratio"] = float(changed_tsdf.float().mean().item())
            if bool(changed_tsdf.any().item()):
                changed_err = (pred_delta[changed_tsdf] - target_delta[changed_tsdf]).abs()
                metrics["changed_tsdf_mae"] = float(changed_err.mean().item())
                metrics["changed_delta_tsdf_mae"] = float(changed_err.mean().item())
    if batch.get("octree_fill_after") is not None:
        fill_after = batch["octree_fill_after"].to(device).float()
        metrics["fill_mae"] = float((prob - fill_after).abs().mean().item())
        metrics["gt_fill_mean"] = float(fill_after.mean().item())
    if batch.get("octree_removed_fraction") is not None:
        before_fill = (
            batch["octree_fill_before"].to(device).float()
            if batch.get("octree_fill_before") is not None
            else batch["octree_occ_labels_before"].to(device).float()
            if batch.get("octree_occ_labels_before") is not None
            else None
        )
        if before_fill is not None:
            removed_target = batch["octree_removed_fraction"].to(device).float()
            removed_pred = torch.relu(before_fill - prob)
            metrics["removed_fraction_mae"] = float((removed_pred - removed_target).abs().mean().item())
            metrics["gt_removed_fraction_mean"] = float(removed_target.mean().item())
    if batch.get("octree_occ_labels_before") is not None:
        before = batch["octree_occ_labels_before"].to(device)
        valid = (
            batch["octree_occ_labels_before_valid"].to(device)
            if batch.get("octree_occ_labels_before_valid") is not None
            else torch.ones_like(before)
        )
        valid_count = float(valid.sum().item())
        if valid_count > 0.0:
            before_bin = (before >= 0.5).float()
            gt_bin = (gt >= 0.5).float()
            valid_float = valid.float()
            changed_mask = ((before_bin != gt_bin).float() * valid_float)
            removed_mask = ((before_bin > 0.5) & (gt_bin < 0.5)).float() * valid_float
            added_mask = ((before_bin < 0.5) & (gt_bin > 0.5)).float() * valid_float
            changed_count = float(changed_mask.sum().item())
            removed_count = float(removed_mask.sum().item())
            added_count = float(added_mask.sum().item())

            metrics["mono_viol"] = float((((pred > before).float() * valid_float).sum() / valid_float.sum().clamp_min(1.0)).item())
            metrics["mono_valid_cells"] = valid_count
            metrics["before_valid_cells"] = valid_count
            metrics["changed_cell_ratio"] = changed_count / max(valid_count, 1.0)
            metrics["changed_cells"] = changed_count
            metrics["removed_cell_ratio"] = removed_count / max(valid_count, 1.0)
            metrics["removed_cells"] = removed_count
            metrics["gt_added_cell_ratio"] = added_count / max(valid_count, 1.0)
            metrics["added_cells"] = added_count
            if changed_count > 0.0:
                metrics["changed_cell_acc"] = float((((pred == gt_bin).float() * changed_mask).sum() / changed_mask.sum().clamp_min(1.0)).item())
            if removed_count > 0.0:
                metrics["removed_cell_recall"] = float((((pred < 0.5).float() * removed_mask).sum() / removed_mask.sum().clamp_min(1.0)).item())
    return metrics


@torch.no_grad()
def _evaluate_sdf_metrics(
    model: GraphSdfPlanningModel,
    batch: dict,
    device: torch.device,
    use_target_tsdf_input: bool = True,
) -> dict[str, float]:
    model.eval()
    state_embedding = model.encode_state(
        state_points=batch["state_points"].to(device),
        node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
        node_face_type=batch.get("node_face_type").to(device) if batch.get("node_face_type") is not None else None,
        node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
    )
    query_state = None
    before = batch["sdf_tsdf_before"].to(device).float() if batch.get("sdf_tsdf_before") is not None else None
    target = batch["sdf_target_tsdf"].to(device).float() if batch.get("sdf_target_tsdf") is not None else None
    if not bool(use_target_tsdf_input):
        target = None
    if before is not None or target is not None:
        ref = before if before is not None else target
        zeros = torch.zeros_like(ref)
        ones = torch.ones_like(ref)
        query_state = torch.stack([
            before if before is not None else zeros,
            target if target is not None else zeros,
            ones,
        ], dim=-1)
    out = model.forward_sdf_query(
        state_points=batch["state_points"].to(device),
        macro_class_id=batch["macro_class_id"].to(device),
        tool_choice_id=batch["tool_choice_id"].to(device),
        action_face_id=batch["action_face_id"].clamp_min(0).to(device),
        sdf_query_points=batch["sdf_query_points"].to(device),
        sdf_query_state=query_state,
        axis_visible=batch.get("axis_visible").to(device) if batch.get("axis_visible") is not None else None,
        axis_dir=batch.get("axis_dir").to(device) if batch.get("axis_dir") is not None else None,
        tool_radius_norm=batch.get("tool_radius_norm").to(device) if batch.get("tool_radius_norm") is not None else None,
        node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
        node_face_type=batch.get("node_face_type").to(device) if batch.get("node_face_type") is not None else None,
        node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
        state_embedding=state_embedding,
    )
    pred = out["sdf_tsdf"]
    gt = batch["sdf_tsdf_after"].to(device).float()
    metrics = {
        "tsdf_mae": float((pred - gt).abs().mean().item()),
        "pred_tsdf_mean": float(pred.mean().item()),
        "gt_tsdf_mean": float(gt.mean().item()),
        "num_points": float(gt.numel()),
    }
    if before is not None:
        target_delta = batch["sdf_delta_tsdf"].to(device).float() if batch.get("sdf_delta_tsdf") is not None else gt - before
        pred_delta = pred - before
        changed = target_delta.abs() > float(TSDF_CHANGE_EPS)
        metrics["delta_tsdf_mae"] = float((pred_delta - target_delta).abs().mean().item())
        metrics["tsdf_changed_ratio"] = float(changed.float().mean().item())
        if bool(changed.any().item()):
            changed_err = (pred_delta[changed] - target_delta[changed]).abs()
            metrics["changed_tsdf_mae"] = float(changed_err.mean().item())
            metrics["changed_delta_tsdf_mae"] = float(changed_err.mean().item())

        macro_ids = batch["macro_class_id"].to(device)
        for macro_id_tensor in torch.unique(macro_ids):
            macro_id = int(macro_id_tensor.item())
            name = macro_class_name_from_id(macro_id)
            sample_mask = macro_ids == macro_id
            point_mask = sample_mask[:, None].expand_as(gt)
            point_count = float(point_mask.sum().item())
            if point_count <= 0.0:
                continue
            macro_prefix = f"macro.{name}"
            metrics[f"{macro_prefix}.num_points"] = point_count
            metrics[f"{macro_prefix}.tsdf_mae"] = float((pred - gt).abs()[point_mask].mean().item())
            macro_changed = changed & point_mask
            changed_count = float(macro_changed.sum().item())
            metrics[f"{macro_prefix}.changed_points"] = changed_count
            metrics[f"{macro_prefix}.tsdf_changed_ratio"] = changed_count / max(point_count, 1.0)
            if changed_count > 0.0:
                macro_changed_err = (pred_delta[macro_changed] - target_delta[macro_changed]).abs()
                metrics[f"{macro_prefix}.changed_tsdf_mae"] = float(macro_changed_err.mean().item())
    if batch.get("affected_face_mask") is not None:
        affected_out = model.forward_affected_faces(
            state_points=batch["state_points"].to(device),
            macro_class_id=batch["macro_class_id"].to(device),
            tool_choice_id=batch["tool_choice_id"].to(device),
            action_face_id=batch["action_face_id"].clamp_min(0).to(device),
            axis_visible=batch.get("axis_visible").to(device) if batch.get("axis_visible") is not None else None,
            axis_dir=batch.get("axis_dir").to(device) if batch.get("axis_dir") is not None else None,
            tool_radius_norm=batch.get("tool_radius_norm").to(device) if batch.get("tool_radius_norm") is not None else None,
            node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
            node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
            spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
            face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
            node_face_type=batch.get("node_face_type").to(device) if batch.get("node_face_type") is not None else None,
            node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
            point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
            state_embedding=state_embedding,
        )
        affected_gt = batch["affected_face_mask"].to(device).float().clamp(0.0, 1.0)
        affected_prob = torch.sigmoid(affected_out["affected_logits"])
        affected_pred = (affected_prob >= 0.5).float()
        valid = ~batch["node_mask"].to(device) if batch.get("node_mask") is not None else torch.ones_like(affected_gt, dtype=torch.bool)
        valid_count = valid.float().sum().clamp_min(1.0)
        tp = ((affected_pred == 1.0) & (affected_gt == 1.0) & valid).float().sum()
        pred_pos = ((affected_pred == 1.0) & valid).float().sum()
        gt_pos = ((affected_gt == 1.0) & valid).float().sum()
        metrics["affected_face_acc"] = float(((affected_pred == affected_gt).float()[valid].sum() / valid_count).item())
        metrics["affected_pred_pos"] = float((pred_pos / valid_count).item())
        metrics["affected_gt_pos"] = float((gt_pos / valid_count).item())
        metrics["affected_precision"] = float((tp / pred_pos.clamp_min(1.0)).item())
        metrics["affected_recall"] = float((tp / gt_pos.clamp_min(1.0)).item())
        metrics["num_faces"] = float(valid_count.item())
        metrics["affected_faces"] = float(gt_pos.item())
    return metrics


def _make_run_dir() -> Path | None:
    if not SAVE_CHECKPOINTS:
        return None
    run_name = RUN_NAME.strip() if RUN_NAME else f"transition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(CHECKPOINT_ROOT).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_checkpoint(
    path: Path,
    model: GraphSdfPlanningModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: GraphSdfModelConfig,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "model_config": asdict(config),
        },
        path,
    )


def _limited_batches(loader, max_batches: int):
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        yield batch_idx, batch


def main() -> None:
    if SDF_QUERY_NODES is None or int(SDF_QUERY_NODES) <= 0:
        raise ValueError("For SDF-only training, set SDF_QUERY_NODES to a positive integer.")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")

    set_seed(SEED)
    if SPLIT_MANIFEST_PATH.strip():
        train_files_all = _load_manifest_files(SPLIT_MANIFEST_PATH, TRAIN_SPLIT_NAME)
        val_files_all = _load_manifest_files(SPLIT_MANIFEST_PATH, VAL_SPLIT_NAME)
        train_files = _cap_files(train_files_all, MAX_TRAIN_FILES)
        val_files = _cap_files(val_files_all, MAX_VAL_FILES)
        parquet_files = train_files + val_files
        train_base_dataset = ProcessSkeletonParquetDataset(
            train_files,
            octree_query_nodes=int(OCTREE_QUERY_NODES),
            sdf_query_nodes=int(SDF_QUERY_NODES),
            lazy_load=LAZY_PARQUET_LOADING,
            parquet_cache_size=PARQUET_ROW_GROUP_CACHE_SIZE,
        )
        val_base_dataset = ProcessSkeletonParquetDataset(
            val_files,
            octree_query_nodes=int(OCTREE_QUERY_NODES),
            sdf_query_nodes=int(SDF_QUERY_NODES),
            lazy_load=LAZY_PARQUET_LOADING,
            parquet_cache_size=PARQUET_ROW_GROUP_CACHE_SIZE,
        )
        train_indices = _build_valid_transition_indices(train_base_dataset)
        val_indices = _build_valid_transition_indices(val_base_dataset)
        split_description = f"manifest:{TRAIN_SPLIT_NAME}/{VAL_SPLIT_NAME}"
        if OVERFIT_DEBUG_MODE:
            train_indices, val_indices, split_description = _apply_overfit_debug_split(
                train_base_dataset,
                train_indices,
                OVERFIT_DEBUG_MODE,
                OVERFIT_PART_NAME,
            )
            val_base_dataset = train_base_dataset
        train_indices = _locality_sort_indices(train_base_dataset, train_indices)
        val_indices = _locality_sort_indices(val_base_dataset, val_indices)
        train_dataset: Dataset = Subset(train_base_dataset, train_indices)
        val_dataset: Dataset = Subset(val_base_dataset, val_indices)
        macro_train_dataset = train_base_dataset
        macro_val_dataset = val_base_dataset
    else:
        parquet_files_all = _resolve_parquet_files()
        total_file_cap = (MAX_TRAIN_FILES if MAX_TRAIN_FILES > 0 else len(parquet_files_all)) + (
            MAX_VAL_FILES if MAX_VAL_FILES > 0 else 0
        )
        parquet_files = _cap_files(parquet_files_all, total_file_cap)
        base_dataset = ProcessSkeletonParquetDataset(
            parquet_files,
            octree_query_nodes=int(OCTREE_QUERY_NODES),
            sdf_query_nodes=int(SDF_QUERY_NODES),
            lazy_load=LAZY_PARQUET_LOADING,
            parquet_cache_size=PARQUET_ROW_GROUP_CACHE_SIZE,
        )
        valid_indices = _build_valid_transition_indices(base_dataset)
        if OVERFIT_DEBUG_MODE:
            train_indices, val_indices, split_description = _apply_overfit_debug_split(
                base_dataset,
                valid_indices,
                OVERFIT_DEBUG_MODE,
                OVERFIT_PART_NAME,
            )
        elif SPLIT_BY_PART:
            train_indices, val_indices, split_group_count = _split_indices_by_part(
                base_dataset,
                valid_indices,
                VAL_RATIO,
                SEED,
            )
            split_description = f"part_level:groups={split_group_count}"
        else:
            train_indices, val_indices = _split_indices(valid_indices, VAL_RATIO, SEED)
            split_description = "row_level"
        train_indices = _locality_sort_indices(base_dataset, train_indices)
        val_indices = _locality_sort_indices(base_dataset, val_indices)
        train_dataset = Subset(base_dataset, train_indices)
        val_dataset = Subset(base_dataset, val_indices)
        macro_train_dataset = base_dataset
        macro_val_dataset = base_dataset

    train_loader = _make_dataloader(
        train_dataset,
        shuffle=not bool(LAZY_PARQUET_LOADING),
        persistent_workers=not bool(SDF_RESAMPLE_EACH_EPOCH),
    )
    val_loader = _make_dataloader(val_dataset, shuffle=False, persistent_workers=True)

    model_cfg = replace(
        GraphSdfModelConfig(),
        octree_query_nodes=int(OCTREE_QUERY_NODES),
        sdf_query_nodes=int(SDF_QUERY_NODES),
        use_octree_decoder=False,
        use_sdf_query_decoder=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSdfPlanningModel(model_cfg).to(device)
    pretrained_encoder_info: dict[str, object] | None = None
    if PRETRAINED_ENCODER_CHECKPOINT.strip():
        pretrained_encoder_info = _load_pretrained_encoder(
            model,
            PRETRAINED_ENCODER_CHECKPOINT,
            strict=PRETRAINED_ENCODER_STRICT,
        )
        print(
            "[Pretrained Encoder] "
            f"loaded={pretrained_encoder_info['checkpoint_path']} "
            f"source={pretrained_encoder_info['source_key']} "
            f"tensors={pretrained_encoder_info['num_tensors']} "
            f"strict={pretrained_encoder_info['strict']}"
        )
        if pretrained_encoder_info["missing_keys"] or pretrained_encoder_info["unexpected_keys"]:
            print(f"[Pretrained Encoder] missing={pretrained_encoder_info['missing_keys']}")
            print(f"[Pretrained Encoder] unexpected={pretrained_encoder_info['unexpected_keys']}")

    optimizer = _build_optimizer(
        model,
        learning_rate=LEARNING_RATE,
        encoder_lr_multiplier=STATE_ENCODER_LR_MULTIPLIER,
        freeze_state_encoder=FREEZE_STATE_ENCODER,
    )

    run_dir = _make_run_dir()
    if run_dir is not None:
        _save_json(
            run_dir / "run_config.json",
            {
                "parquet_files": parquet_files,
                "split_manifest_path": SPLIT_MANIFEST_PATH,
                "train_split_name": TRAIN_SPLIT_NAME,
                "val_split_name": VAL_SPLIT_NAME,
                "seed": SEED,
                "val_ratio": VAL_RATIO,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "num_epochs": NUM_EPOCHS,
                "max_train_batches_per_epoch": MAX_TRAIN_BATCHES_PER_EPOCH,
                "max_val_batches": MAX_VAL_BATCHES,
                "max_train_files": MAX_TRAIN_FILES,
                "max_val_files": MAX_VAL_FILES,
                "lazy_parquet_loading": LAZY_PARQUET_LOADING,
                "parquet_row_group_cache_size": PARQUET_ROW_GROUP_CACHE_SIZE,
                "locality_sort_lazy_indices": LOCALITY_SORT_LAZY_INDICES,
                "split_description": split_description,
                "split_by_part": SPLIT_BY_PART,
                "sdf_resample_each_epoch": SDF_RESAMPLE_EACH_EPOCH,
                "use_target_tsdf_input": USE_TARGET_TSDF_INPUT,
                "overfit_debug_mode": OVERFIT_DEBUG_MODE,
                "overfit_part_name": OVERFIT_PART_NAME,
                "learning_rate": LEARNING_RATE,
                "sdf_query_nodes": int(SDF_QUERY_NODES),
                "octree_query_nodes": int(OCTREE_QUERY_NODES),
                "octree_pos_weight_factor": OCTREE_POS_WEIGHT_FACTOR,
                "octree_depth_weight_base": OCTREE_DEPTH_WEIGHT_BASE,
                "monotonicity_weight": MONOTONICITY_WEIGHT,
                "occupancy_loss_weight": OCCUPANCY_LOSS_WEIGHT,
                "octree_fill_fraction_weight": OCTREE_FILL_FRACTION_WEIGHT,
                "octree_removed_fraction_weight": OCTREE_REMOVED_FRACTION_WEIGHT,
                "tsdf_loss_weight": TSDF_LOSS_WEIGHT,
                "delta_tsdf_loss_weight": DELTA_TSDF_LOSS_WEIGHT,
                "changed_tsdf_loss_weight": CHANGED_TSDF_LOSS_WEIGHT,
                "changed_delta_tsdf_loss_weight": CHANGED_DELTA_TSDF_LOSS_WEIGHT,
                "tsdf_change_eps": TSDF_CHANGE_EPS,
                "tsdf_monotonicity_weight": TSDF_MONOTONICITY_WEIGHT,
                "tsdf_monotonicity_empty_margin": TSDF_MONOTONICITY_EMPTY_MARGIN,
                "affected_face_loss_weight": AFFECTED_FACE_LOSS_WEIGHT,
                "affected_face_delta_loss_weight": AFFECTED_FACE_DELTA_LOSS_WEIGHT,
                "pretrained_encoder_checkpoint": PRETRAINED_ENCODER_CHECKPOINT,
                "pretrained_encoder_strict": PRETRAINED_ENCODER_STRICT,
                "pretrained_encoder_info": pretrained_encoder_info,
                "freeze_state_encoder": FREEZE_STATE_ENCODER,
                "state_encoder_lr_multiplier": STATE_ENCODER_LR_MULTIPLIER,
                "model_config": asdict(model_cfg),
                "train_rows": len(train_indices),
                "val_rows": len(val_indices),
            },
        )

    print(f"[Device] {device}")
    print(f"[Files] {len(parquet_files)} parquet files")
    print(f"[File Caps] max_train_files={MAX_TRAIN_FILES or 'all'} max_val_files={MAX_VAL_FILES or 'all'}")
    print(
        f"[Parquet Loading] lazy={LAZY_PARQUET_LOADING} "
        f"row_group_cache={PARQUET_ROW_GROUP_CACHE_SIZE} "
        f"locality_sort={LOCALITY_SORT_LAZY_INDICES}"
    )
    print(f"[Rows] train={len(train_indices)} val={len(val_indices)}")
    print(f"[Split] {split_description}")
    print(f"[Train] epochs={NUM_EPOCHS} lr={LEARNING_RATE} batch={BATCH_SIZE} sdf_query_nodes={SDF_QUERY_NODES}")
    print(
        f"[Pilot Limits] max_train_batches={MAX_TRAIN_BATCHES_PER_EPOCH or 'all'} "
        f"max_val_batches={MAX_VAL_BATCHES or 'all'}"
    )
    print(
        f"[Encoder] pretrained={bool(PRETRAINED_ENCODER_CHECKPOINT.strip())} "
        f"freeze={FREEZE_STATE_ENCODER} lr_multiplier={STATE_ENCODER_LR_MULTIPLIER}"
    )
    print(
        f"[Loss] pos_weight={OCTREE_POS_WEIGHT_FACTOR} "
        f"depth_weight_base={OCTREE_DEPTH_WEIGHT_BASE} "
        f"monotonicity_weight={MONOTONICITY_WEIGHT} "
        f"occ_weight={OCCUPANCY_LOSS_WEIGHT} "
        f"fill_weight={OCTREE_FILL_FRACTION_WEIGHT} "
        f"removed_weight={OCTREE_REMOVED_FRACTION_WEIGHT} "
        f"tsdf_weight={TSDF_LOSS_WEIGHT} "
        f"delta_tsdf_weight={DELTA_TSDF_LOSS_WEIGHT} "
        f"changed_tsdf_weight={CHANGED_TSDF_LOSS_WEIGHT} "
        f"changed_delta_tsdf_weight={CHANGED_DELTA_TSDF_LOSS_WEIGHT} "
        f"tsdf_change_eps={TSDF_CHANGE_EPS} "
        f"tsdf_mono_empty_margin={TSDF_MONOTONICITY_EMPTY_MARGIN} "
        f"affected_face_weight={AFFECTED_FACE_LOSS_WEIGHT} "
        f"affected_delta_weight={AFFECTED_FACE_DELTA_LOSS_WEIGHT} "
        f"use_target_tsdf_input={USE_TARGET_TSDF_INPUT}"
    )
    print(f"[SDF Sampling] resample_each_epoch={SDF_RESAMPLE_EACH_EPOCH}")
    print(f"[Train Macro Dist] { _macro_distribution(macro_train_dataset, train_indices) }")
    print(f"[Val Macro Dist] { _macro_distribution(macro_val_dataset, val_indices) }")
    if run_dir is not None:
        print(f"[Checkpoint Dir] {run_dir}")
    print(f"[Memory] {_cuda_memory_text()}")

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _set_dataset_sdf_sample_epoch(train_dataset, epoch if SDF_RESAMPLE_EACH_EPOCH else 0)
        _set_dataset_sdf_sample_epoch(val_dataset, 0)
        train_losses = []
        train_component_sums: dict[str, float] = {}
        train_component_count = 0
        for batch_idx, batch in _limited_batches(train_loader, MAX_TRAIN_BATCHES_PER_EPOCH):
            try:
                step_result = transition_train_step(
                    model,
                    batch,
                    optimizer,
                    device,
                    octree_pos_weight_factor=OCTREE_POS_WEIGHT_FACTOR,
                    octree_depth_weight_base=OCTREE_DEPTH_WEIGHT_BASE,
                    monotonicity_weight=MONOTONICITY_WEIGHT,
                    fill_fraction_weight=OCTREE_FILL_FRACTION_WEIGHT,
                    removed_fraction_weight=OCTREE_REMOVED_FRACTION_WEIGHT,
                    tsdf_loss_weight=TSDF_LOSS_WEIGHT,
                    delta_tsdf_loss_weight=DELTA_TSDF_LOSS_WEIGHT,
                    changed_tsdf_loss_weight=CHANGED_TSDF_LOSS_WEIGHT,
                    changed_delta_tsdf_loss_weight=CHANGED_DELTA_TSDF_LOSS_WEIGHT,
                    tsdf_change_eps=TSDF_CHANGE_EPS,
                    tsdf_monotonicity_weight=TSDF_MONOTONICITY_WEIGHT,
                    tsdf_monotonicity_empty_margin=TSDF_MONOTONICITY_EMPTY_MARGIN,
                    occupancy_loss_weight=OCCUPANCY_LOSS_WEIGHT,
                    affected_face_loss_weight=AFFECTED_FACE_LOSS_WEIGHT,
                    affected_delta_loss_weight=AFFECTED_FACE_DELTA_LOSS_WEIGHT,
                    use_target_tsdf_input=USE_TARGET_TSDF_INPUT,
                    return_components=True,
                )
                if isinstance(step_result, tuple):
                    loss, loss_components = step_result
                else:
                    loss, loss_components = step_result, {}
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        "[OOM] CUDA ran out of memory during training. "
                        "Reduce BATCH_SIZE first, then SDF_QUERY_NODES. "
                        f"Current batch={BATCH_SIZE} sdf_query_nodes={SDF_QUERY_NODES} "
                        f"{_cuda_memory_text()}",
                        flush=True,
                    )
                raise
            train_losses.append(loss)
            _accumulate_loss_components(train_component_sums, loss_components)
            train_component_count += 1
            if TRAIN_LOG_EVERY_BATCHES > 0 and batch_idx % TRAIN_LOG_EVERY_BATCHES == 0:
                print(
                    f"[Epoch {epoch:04d} Batch {batch_idx:05d}] "
                    f"loss={float(loss):.6f} {_cuda_memory_text()}",
                    flush=True,
                )

        val_losses = []
        val_component_sums: dict[str, float] = {}
        val_component_count = 0
        val_metric_sums: dict[str, float] = {}
        val_metric_weights: dict[str, float] = {}
        for _, batch in _limited_batches(val_loader, MAX_VAL_BATCHES):
            try:
                val_result = transition_validation_step(
                    model,
                    batch,
                    device,
                    octree_pos_weight_factor=OCTREE_POS_WEIGHT_FACTOR,
                    octree_depth_weight_base=OCTREE_DEPTH_WEIGHT_BASE,
                    monotonicity_weight=MONOTONICITY_WEIGHT,
                    fill_fraction_weight=OCTREE_FILL_FRACTION_WEIGHT,
                    removed_fraction_weight=OCTREE_REMOVED_FRACTION_WEIGHT,
                    tsdf_loss_weight=TSDF_LOSS_WEIGHT,
                    delta_tsdf_loss_weight=DELTA_TSDF_LOSS_WEIGHT,
                    changed_tsdf_loss_weight=CHANGED_TSDF_LOSS_WEIGHT,
                    changed_delta_tsdf_loss_weight=CHANGED_DELTA_TSDF_LOSS_WEIGHT,
                    tsdf_change_eps=TSDF_CHANGE_EPS,
                    tsdf_monotonicity_weight=TSDF_MONOTONICITY_WEIGHT,
                    tsdf_monotonicity_empty_margin=TSDF_MONOTONICITY_EMPTY_MARGIN,
                    occupancy_loss_weight=OCCUPANCY_LOSS_WEIGHT,
                    affected_face_loss_weight=AFFECTED_FACE_LOSS_WEIGHT,
                    affected_delta_loss_weight=AFFECTED_FACE_DELTA_LOSS_WEIGHT,
                    use_target_tsdf_input=USE_TARGET_TSDF_INPUT,
                    return_components=True,
                )
                if isinstance(val_result, tuple):
                    val_loss_item, val_loss_components = val_result
                else:
                    val_loss_item, val_loss_components = val_result, {}
                val_losses.append(val_loss_item)
                _accumulate_loss_components(val_component_sums, val_loss_components)
                val_component_count += 1
                if batch.get("sdf_query_points") is not None:
                    metrics = _evaluate_sdf_metrics(
                        model,
                        batch,
                        device,
                        use_target_tsdf_input=USE_TARGET_TSDF_INPUT,
                    )
                else:
                    metrics = _evaluate_octree_metrics(model, batch, device)
                for key, value in metrics.items():
                    if key in {"num_cells", "num_points", "num_faces", "affected_faces", "mono_valid_cells", "before_valid_cells", "changed_cells", "removed_cells", "added_cells"}:
                        continue
                    if key.startswith("macro.") and (key.endswith(".num_points") or key.endswith(".changed_points")):
                        continue
                    if key.startswith("macro."):
                        parts = key.split(".")
                        macro_prefix = ".".join(parts[:2]) if len(parts) >= 3 else key
                        metric_name = parts[2] if len(parts) >= 3 else ""
                        if metric_name == "changed_tsdf_mae":
                            metric_weight = max(float(metrics.get(f"{macro_prefix}.changed_points", 0.0)), 1.0)
                        else:
                            metric_weight = max(float(metrics.get(f"{macro_prefix}.num_points", 0.0)), 1.0)
                    elif key == "mono_viol":
                        metric_weight = max(float(metrics.get("mono_valid_cells", 0.0)), 1.0)
                    elif key in {"changed_cell_acc"}:
                        metric_weight = max(float(metrics.get("changed_cells", 0.0)), 1.0)
                    elif key in {"removed_cell_recall"}:
                        metric_weight = max(float(metrics.get("removed_cells", 0.0)), 1.0)
                    elif key in {"changed_cell_ratio", "removed_cell_ratio", "gt_added_cell_ratio"}:
                        metric_weight = max(float(metrics.get("before_valid_cells", 0.0)), 1.0)
                    elif key.startswith("affected_"):
                        metric_weight = max(float(metrics.get("num_faces", 1.0)), 1.0)
                    else:
                        metric_weight = max(float(metrics.get("num_points", metrics.get("num_cells", 1.0))), 1.0)
                    val_metric_sums[key] = val_metric_sums.get(key, 0.0) + float(value) * metric_weight
                    val_metric_weights[key] = val_metric_weights.get(key, 0.0) + metric_weight
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        "[OOM] CUDA ran out of memory during validation. "
                        "Reduce MAX_VAL_BATCHES, BATCH_SIZE, or SDF_QUERY_NODES. "
                        f"Current max_val_batches={MAX_VAL_BATCHES} batch={BATCH_SIZE} "
                        f"sdf_query_nodes={SDF_QUERY_NODES} {_cuda_memory_text()}",
                        flush=True,
                    )
                raise

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss = float(sum(val_losses) / max(len(val_losses), 1))
        val_metrics = {
            key: val_metric_sums[key] / max(val_metric_weights.get(key, 0.0), 1.0)
            for key in val_metric_sums
        }
        train_loss_components = _average_loss_components(train_component_sums, train_component_count)
        val_loss_components = _average_loss_components(val_component_sums, val_component_count)

        if run_dir is not None and (epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS):
            _save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                model_cfg,
            )
        if run_dir is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                model_cfg,
            )

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS:
            log = f"[Epoch {epoch:04d}] train={train_loss:.6f} val={val_loss:.6f}"
            log += _format_loss_components("train", train_loss_components)
            log += _format_loss_components("val", val_loss_components)
            if val_metrics:
                if "acc" in val_metrics:
                    log += (
                        f" val_octree_acc={val_metrics['acc']:.4f}"
                        f" pred_pos={val_metrics['pred_pos']:.4f}"
                        f" gt_pos={val_metrics['gt_pos']:.4f}"
                        f" prob_mean={val_metrics['prob_mean']:.4f}"
                    )
                if "mono_viol" in val_metrics:
                    log += f" mono_viol={val_metrics['mono_viol']:.4f}"
                if "fill_mae" in val_metrics:
                    log += f" fill_mae={val_metrics['fill_mae']:.4f}"
                if "removed_fraction_mae" in val_metrics:
                    log += f" removed_frac_mae={val_metrics['removed_fraction_mae']:.4f}"
                if "tsdf_mae" in val_metrics:
                    log += f" tsdf_mae={val_metrics['tsdf_mae']:.4f}"
                if "delta_tsdf_mae" in val_metrics:
                    log += f" delta_tsdf_mae={val_metrics['delta_tsdf_mae']:.4f}"
                if "changed_tsdf_mae" in val_metrics:
                    log += f" changed_tsdf_mae={val_metrics['changed_tsdf_mae']:.4f}"
                if "changed_delta_tsdf_mae" in val_metrics:
                    log += f" changed_delta_tsdf_mae={val_metrics['changed_delta_tsdf_mae']:.4f}"
                if "tsdf_changed_ratio" in val_metrics:
                    log += f" tsdf_changed_ratio={val_metrics['tsdf_changed_ratio']:.4f}"
                if "affected_face_acc" in val_metrics:
                    log += f" affected_acc={val_metrics['affected_face_acc']:.4f}"
                if "affected_recall" in val_metrics:
                    log += f" affected_recall={val_metrics['affected_recall']:.4f}"
                if "affected_precision" in val_metrics:
                    log += f" affected_precision={val_metrics['affected_precision']:.4f}"
                if "affected_gt_pos" in val_metrics:
                    log += f" affected_gt_pos={val_metrics['affected_gt_pos']:.4f}"
                if "changed_cell_ratio" in val_metrics:
                    log += f" changed_ratio={val_metrics['changed_cell_ratio']:.4f}"
                if "changed_cell_acc" in val_metrics:
                    log += f" changed_acc={val_metrics['changed_cell_acc']:.4f}"
                if "removed_cell_recall" in val_metrics:
                    log += f" removed_recall={val_metrics['removed_cell_recall']:.4f}"
                log += _format_macro_metrics(val_metrics)
            log += f" {_cuda_memory_text()}"
            print(log)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[Done] transition-only dataset training finished.")


if __name__ == "__main__":
    main()
