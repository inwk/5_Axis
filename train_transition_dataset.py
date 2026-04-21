"""Train the transition-only octree model on multiple parquet files.

Edit the constants below and run directly from VSCode/debug mode.
No CLI args are required.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from graph_sdf import GraphSdfModelConfig, GraphSdfPlanningModel, ProcessSkeletonParquetDataset
from graph_sdf.training import transition_train_step, transition_validation_step


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
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PRINT_EVERY = 1

# Optional StateEncoder initialization from SSL pretraining.
# Set this to ssl_pretraining/train_state_encoder_ssl.py's best.pt.
PRETRAINED_ENCODER_CHECKPOINT = r""
PRETRAINED_ENCODER_STRICT = True
FREEZE_STATE_ENCODER = False
STATE_ENCODER_LR_MULTIPLIER = 1.0

# For batched training this should stay fixed and positive.
OCTREE_QUERY_NODES = 2048
OCTREE_POS_WEIGHT_FACTOR = 2.0
OCTREE_DEPTH_WEIGHT_BASE = 2.0
MONOTONICITY_WEIGHT = 0.1

SAVE_CHECKPOINTS = True
CHECKPOINT_ROOT = r"C:\Users\inwoo\Desktop\5_Axis\checkpoints_transition"
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


def _build_valid_transition_indices(dataset: ProcessSkeletonParquetDataset) -> list[int]:
    indices: list[int] = []
    df = dataset.df
    required = {"octree_centers", "octree_depths", "octree_occ_labels"}
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required octree columns: {missing_cols}")

    for idx, row in df.iterrows():
        if (
            not dataset._is_missing(row["octree_centers"])
            and not dataset._is_missing(row["octree_depths"])
            and not dataset._is_missing(row["octree_occ_labels"])
        ):
            indices.append(int(idx))
    if not indices:
        raise ValueError("No rows with complete octree supervision were found.")
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


def _macro_distribution(dataset: ProcessSkeletonParquetDataset, indices: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    if "macro_class_name" not in dataset.df.columns:
        return out
    for idx in indices:
        name = str(dataset.df.iloc[int(idx)].get("macro_class_name", "unknown"))
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


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


@torch.no_grad()
def _evaluate_octree_accuracy(model: GraphSdfPlanningModel, batch: dict, device: torch.device) -> float:
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
        octree_occ_before=batch.get("octree_occ_labels_before").to(device) if batch.get("octree_occ_labels_before") is not None else None,
        axis_visible=batch.get("axis_visible").to(device) if batch.get("axis_visible") is not None else None,
        node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
        node_face_type=batch.get("node_face_type").to(device) if batch.get("node_face_type") is not None else None,
        node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
        state_embedding=state_embedding,
    )
    pred = (torch.sigmoid(out["occ_logits"]) >= 0.5).float()
    gt = batch["octree_occ_labels"].to(device)
    return float((pred == gt).float().mean().item())


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


def main() -> None:
    if OCTREE_QUERY_NODES is None or int(OCTREE_QUERY_NODES) <= 0:
        raise ValueError("For full dataset batched training, set OCTREE_QUERY_NODES to a positive integer.")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")

    set_seed(SEED)
    parquet_files = _resolve_parquet_files()
    base_dataset = ProcessSkeletonParquetDataset(parquet_files, octree_query_nodes=int(OCTREE_QUERY_NODES))
    valid_indices = _build_valid_transition_indices(base_dataset)
    train_indices, val_indices = _split_indices(valid_indices, VAL_RATIO, SEED)

    train_dataset: Dataset = Subset(base_dataset, train_indices)
    val_dataset: Dataset = Subset(base_dataset, val_indices)

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

    model_cfg = replace(GraphSdfModelConfig(), octree_query_nodes=int(OCTREE_QUERY_NODES))
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
                "seed": SEED,
                "val_ratio": VAL_RATIO,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "octree_query_nodes": int(OCTREE_QUERY_NODES),
                "octree_pos_weight_factor": OCTREE_POS_WEIGHT_FACTOR,
                "octree_depth_weight_base": OCTREE_DEPTH_WEIGHT_BASE,
                "monotonicity_weight": MONOTONICITY_WEIGHT,
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
    print(f"[Rows] total_valid={len(valid_indices)} train={len(train_indices)} val={len(val_indices)}")
    print(f"[Train] epochs={NUM_EPOCHS} lr={LEARNING_RATE} batch={BATCH_SIZE} octree_query_nodes={OCTREE_QUERY_NODES}")
    print(
        f"[Encoder] pretrained={bool(PRETRAINED_ENCODER_CHECKPOINT.strip())} "
        f"freeze={FREEZE_STATE_ENCODER} lr_multiplier={STATE_ENCODER_LR_MULTIPLIER}"
    )
    print(
        f"[Loss] pos_weight={OCTREE_POS_WEIGHT_FACTOR} "
        f"depth_weight_base={OCTREE_DEPTH_WEIGHT_BASE} "
        f"monotonicity_weight={MONOTONICITY_WEIGHT}"
    )
    print(f"[Train Macro Dist] { _macro_distribution(base_dataset, train_indices) }")
    print(f"[Val Macro Dist] { _macro_distribution(base_dataset, val_indices) }")
    if run_dir is not None:
        print(f"[Checkpoint Dir] {run_dir}")

    best_val_loss = float("inf")
    first_val_batch = next(iter(val_loader)) if len(val_dataset) > 0 else None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_losses = []
        for batch in train_loader:
            loss = transition_train_step(
                model,
                batch,
                optimizer,
                device,
                octree_pos_weight_factor=OCTREE_POS_WEIGHT_FACTOR,
                octree_depth_weight_base=OCTREE_DEPTH_WEIGHT_BASE,
                monotonicity_weight=MONOTONICITY_WEIGHT,
            )
            train_losses.append(loss)

        val_losses = [
            transition_validation_step(
                model,
                batch,
                device,
                octree_pos_weight_factor=OCTREE_POS_WEIGHT_FACTOR,
                octree_depth_weight_base=OCTREE_DEPTH_WEIGHT_BASE,
                monotonicity_weight=MONOTONICITY_WEIGHT,
            )
            for batch in val_loader
        ]

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss = float(sum(val_losses) / max(len(val_losses), 1))

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
            if first_val_batch is not None:
                oct_acc = _evaluate_octree_accuracy(model, first_val_batch, device)
                log += f" val_octree_acc(sample_batch)={oct_acc:.4f}"
            print(log)

    print("[Done] transition-only dataset training finished.")


if __name__ == "__main__":
    main()
