"""Train StateEncoder with self-supervised face-graph reconstruction tasks.

Edit the constants below and run directly from VSCode/debug mode.
No CLI arguments are required.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from graph_sdf import GraphSdfModelConfig
from ssl_pretraining.state_encoder_dataset import (
    StateEncoderSslParquetDataset,
    resolve_parquet_files,
    split_indices,
)
from ssl_pretraining.state_encoder_model import (
    StateEncoderSslLossConfig,
    StateEncoderSslModel,
    compute_ssl_loss,
    merge_metrics,
)


# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode.
# ---------------------------------------------------------------------------
PARQUET_DIR = r""
# Use "**/*.parquet" to include run subdirectories under PARQUET_DIR.
PARQUET_GLOB = "**/*.parquet"
EXPLICIT_PARQUET_PATHS: list[str] = []
SPLIT_MANIFEST_PATH = os.getenv("PILOT_SPLIT_MANIFEST", r"")
SPLIT_NAME = os.getenv("PILOT_SSL_SPLIT", "train")

VAL_RATIO = 0.2
SEED = 0

BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 1

MASK_NODE_RATIO = 0.25
# -1 uses GraphSdfModelConfig.face_type_vocab_size - 1 as a dedicated SSL mask token.
# Keep this separate from real face type 0 / padding.
MASK_FACE_TYPE_ID = -1
MASK_NORMAL_AND_SDF = True
MASK_FACE_AREA = True

# For prt-only SSL datasets, state_points[..., 6] is zero-filled, so keep
# SDF_LOSS_WEIGHT=0.0 unless you are training on CAM/parquet state data.
TYPE_LOSS_WEIGHT = 1.0
NORMAL_LOSS_WEIGHT = 1.0
SDF_LOSS_WEIGHT = 0.3
AREA_LOSS_WEIGHT = 0.3
EDGE_LOSS_WEIGHT = 0.5

USE_EDGE_LOSS = True
EDGE_PAIRS_PER_SAMPLE = 512

# Face-type imbalance handling.
# Use "sqrt_inv_clipped" for the observed imbalanced face-type distribution.
# Set to "none" only when you need an unweighted ablation.
FACE_TYPE_CLASS_WEIGHT_MODE = "sqrt_inv_clipped"  # "none" or "sqrt_inv_clipped"
FACE_TYPE_CLASS_WEIGHT_MIN = 0.5
FACE_TYPE_CLASS_WEIGHT_MAX = 3.0
FACE_TYPE_REPORT_CLASSES = [1, 2, 3, 4, 5, 6]
FACE_TYPE_LOW_ACC_WARN_CLASSES = [3, 6]
FACE_TYPE_LOW_ACC_THRESHOLD = 0.60

SAVE_CHECKPOINTS = True
CHECKPOINT_ROOT = r"C:\Users\inwoo\Desktop\5_Axis\checkpoints_state_encoder_ssl"
RUN_NAME = os.getenv("RUN_NAME", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_run_dir() -> Path | None:
    if not SAVE_CHECKPOINTS:
        return None
    run_name = RUN_NAME.strip() if RUN_NAME else f"state_encoder_ssl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(CHECKPOINT_ROOT).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _face_type_counts(
    dataset: StateEncoderSslParquetDataset,
    indices: list[int],
    vocab_size: int,
) -> np.ndarray:
    """Counts valid-node face types for the selected dataset rows."""
    counts = np.zeros((int(vocab_size),), dtype=np.int64)
    for idx in indices:
        row = dataset.df.iloc[int(idx)]
        face_type = dataset._array(row["face_type_512"], np.int16).reshape(-1)
        if "node_mask" in row.index:
            node_mask = dataset._array(row["node_mask"], np.int16).reshape(-1)
        else:
            node_mask = np.zeros_like(face_type, dtype=np.int16)
        count = min(len(face_type), len(node_mask))
        valid = node_mask[:count] == 0
        valid_face_type = np.clip(face_type[:count][valid], 0, int(vocab_size) - 1)
        if valid_face_type.size:
            bincount = np.bincount(valid_face_type.astype(np.int64), minlength=int(vocab_size))
            counts += bincount[: int(vocab_size)]
    return counts


def _build_face_type_class_weights(
    counts: np.ndarray,
    mode: str,
    min_weight: float,
    max_weight: float,
) -> tuple[float, ...] | None:
    """Builds mild face-type CE weights from training-set counts."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"", "none", "off", "false", "0"}:
        return None
    if normalized_mode != "sqrt_inv_clipped":
        raise ValueError(f"Unsupported FACE_TYPE_CLASS_WEIGHT_MODE: {mode}")

    counts = np.asarray(counts, dtype=np.float64)
    weights = np.ones_like(counts, dtype=np.float64)
    present = counts > 0
    if not present.any():
        return tuple(float(x) for x in weights)

    freq = counts[present] / float(counts[present].sum())
    inv_sqrt = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    inv_sqrt = inv_sqrt / float(inv_sqrt.mean())
    inv_sqrt = np.clip(inv_sqrt, float(min_weight), float(max_weight))
    weights[present] = inv_sqrt
    return tuple(float(x) for x in weights)


def _face_type_acc(metrics: dict[str, float], class_id: int) -> float:
    count = float(metrics.get(f"type_count_{int(class_id)}", 0.0))
    correct = float(metrics.get(f"type_correct_{int(class_id)}", 0.0))
    return float(correct / count) if count > 0 else float("nan")


def _format_face_type_acc(metrics: dict[str, float], class_ids: list[int]) -> str:
    chunks = []
    for class_id in class_ids:
        count = int(metrics.get(f"type_count_{int(class_id)}", 0.0))
        acc = _face_type_acc(metrics, int(class_id))
        if np.isnan(acc):
            chunks.append(f"t{int(class_id)}=nan({count})")
        else:
            chunks.append(f"t{int(class_id)}={acc:.3f}({count})")
    return " ".join(chunks)


def _low_accuracy_warnings(metrics: dict[str, float], class_ids: list[int], threshold: float) -> list[str]:
    warnings = []
    for class_id in class_ids:
        count = int(metrics.get(f"type_count_{int(class_id)}", 0.0))
        acc = _face_type_acc(metrics, int(class_id))
        if count > 0 and not np.isnan(acc) and acc < float(threshold):
            warnings.append(f"type {int(class_id)} acc={acc:.3f} count={count}")
    return warnings


def _save_checkpoint(
    path: Path,
    model: StateEncoderSslModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    model_config: GraphSdfModelConfig,
    loss_config: StateEncoderSslLossConfig,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "model_config": asdict(model_config),
            "loss_config": asdict(loss_config),
        },
        path,
    )


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
    return files


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if SPLIT_MANIFEST_PATH.strip():
        parquet_files = _load_manifest_files(SPLIT_MANIFEST_PATH, SPLIT_NAME)
    else:
        parquet_files = resolve_parquet_files(
            parquet_dir=PARQUET_DIR,
            parquet_glob=PARQUET_GLOB,
            explicit_parquet_paths=EXPLICIT_PARQUET_PATHS,
            caller_name="ssl_pretraining/train_state_encoder_ssl.py",
        )

    dataset = StateEncoderSslParquetDataset(parquet_files)
    train_indices, val_indices = split_indices(len(dataset), VAL_RATIO, SEED)
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

    model_config = GraphSdfModelConfig()
    train_face_type_counts = _face_type_counts(dataset, train_indices, model_config.face_type_vocab_size)
    face_type_class_weights = _build_face_type_class_weights(
        train_face_type_counts,
        FACE_TYPE_CLASS_WEIGHT_MODE,
        FACE_TYPE_CLASS_WEIGHT_MIN,
        FACE_TYPE_CLASS_WEIGHT_MAX,
    )
    loss_config = StateEncoderSslLossConfig(
        mask_node_ratio=MASK_NODE_RATIO,
        mask_face_type_id=MASK_FACE_TYPE_ID,
        mask_normal_and_sdf=MASK_NORMAL_AND_SDF,
        mask_face_area=MASK_FACE_AREA,
        type_loss_weight=TYPE_LOSS_WEIGHT,
        normal_loss_weight=NORMAL_LOSS_WEIGHT,
        sdf_loss_weight=SDF_LOSS_WEIGHT,
        area_loss_weight=AREA_LOSS_WEIGHT,
        edge_loss_weight=EDGE_LOSS_WEIGHT,
        use_edge_loss=USE_EDGE_LOSS,
        edge_pairs_per_sample=EDGE_PAIRS_PER_SAMPLE,
        face_type_class_weights=face_type_class_weights,
    )
    model = StateEncoderSslModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    run_dir = _make_run_dir()
    if run_dir is not None:
        _save_json(
            run_dir / "run_config.json",
            {
                "parquet_files": parquet_files,
                "split_manifest_path": SPLIT_MANIFEST_PATH,
                "split_name": SPLIT_NAME,
                "seed": SEED,
                "val_ratio": VAL_RATIO,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "model_config": asdict(model_config),
                "loss_config": asdict(loss_config),
                "face_type_class_weight_mode": FACE_TYPE_CLASS_WEIGHT_MODE,
                "face_type_train_counts": train_face_type_counts.tolist(),
                "face_type_class_weights": list(face_type_class_weights) if face_type_class_weights is not None else None,
                "train_rows": len(train_indices),
                "val_rows": len(val_indices),
            },
        )

    print(f"[Device] {device}")
    print(f"[Files] {len(parquet_files)} parquet files")
    print(f"[Rows] total={len(dataset)} train={len(train_indices)} val={len(val_indices)}")
    print(f"[Train] epochs={NUM_EPOCHS} lr={LEARNING_RATE} batch={BATCH_SIZE} mask_ratio={MASK_NODE_RATIO}")
    resolved_mask_id = MASK_FACE_TYPE_ID if MASK_FACE_TYPE_ID >= 0 else model_config.face_type_vocab_size - 1
    print(f"[FaceType] vocab={model_config.face_type_vocab_size} mask_id={resolved_mask_id}")
    print(f"[FaceType Counts] { {i: int(c) for i, c in enumerate(train_face_type_counts.tolist()) if int(c) > 0} }")
    print(f"[FaceType Weight Mode] {FACE_TYPE_CLASS_WEIGHT_MODE}")
    if face_type_class_weights is not None:
        print(f"[FaceType Weights] { {i: round(float(w), 4) for i, w in enumerate(face_type_class_weights) if int(train_face_type_counts[i]) > 0} }")
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
            loss, metrics = compute_ssl_loss(model, batch, device, loss_config)
            loss.backward()
            optimizer.step()
            train_metrics.append(metrics)

        model.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                _, metrics = compute_ssl_loss(model, batch, device, loss_config)
                val_metrics.append(metrics)

        train_avg = merge_metrics(train_metrics)
        val_avg = merge_metrics(val_metrics)

        if run_dir is not None and (epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS):
            _save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, train_avg, val_avg, model_config, loss_config)
        if run_dir is not None and val_avg.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_avg["loss"]
            _save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, train_avg, val_avg, model_config, loss_config)

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
            print(
                "            "
                f"type_acc_by_class={_format_face_type_acc(val_avg, FACE_TYPE_REPORT_CLASSES)}"
            )
            low_acc = _low_accuracy_warnings(
                val_avg,
                FACE_TYPE_LOW_ACC_WARN_CLASSES,
                FACE_TYPE_LOW_ACC_THRESHOLD,
            )
            if low_acc and FACE_TYPE_CLASS_WEIGHT_MODE.strip().lower() in {"", "none", "off", "false", "0"}:
                print(
                    "            "
                    "[WARN] low minority face-type accuracy; consider "
                    "FACE_TYPE_CLASS_WEIGHT_MODE='sqrt_inv_clipped': "
                    + "; ".join(low_acc)
                )

    print("[Done] StateEncoder SSL pretraining finished.")


if __name__ == "__main__":
    main()
