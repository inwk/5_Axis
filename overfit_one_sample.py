"""Overfit sanity check on a single parquet row — or full training from a folder.

Two modes are controlled by the constants at the top:

OVERFIT MODE  (default)
    Set PARQUET_PATH to one .parquet file.  The script picks a single row,
    repeats it REPEAT_STEPS_PER_EPOCH times per epoch, and checks whether
    the model can drive the transition loss to near-zero.

FULL TRAIN MODE
    Set PARQUET_DIR to a folder that contains .parquet files (searched
    recursively).  ALL rows across all files are used.  REPEAT_STEPS_PER_EPOCH
    is ignored; each epoch iterates the entire dataset once.
    A TRAIN_VAL_SPLIT fraction of files go to training; the rest to validation.

Edit the constants below and run from VSCode / the terminal.
No CLI arguments are required.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from graph_sdf import GraphSdfModelConfig, GraphSdfPlanningModel, ProcessSkeletonParquetDataset
from graph_sdf.schema import ID_TO_MACRO_CLASS, ID_TO_TOOL_CHOICE
from graph_sdf.training import (
    EMALossBalancer,
    transition_train_step,
    transition_validation_step,
)


# ---------------------------------------------------------------------------
# ── Mode selection ───────────────────────────────────────────────────────────
# Set exactly one of PARQUET_PATH (overfit) or PARQUET_DIR (full training).
# ---------------------------------------------------------------------------
PARQUET_PATH = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\3dDataset0055_seed0_20260415_204610\3dDataset0055_seed0_process_skeleton_dataset.parquet"
PARQUET_DIR  = r""          # e.g. r"Y:\...\sdf_dataset_out"  ← set this for full training

# ---------------------------------------------------------------------------
# ── Overfit-mode config ──────────────────────────────────────────────────────
# Ignored when PARQUET_DIR is set.
# ---------------------------------------------------------------------------
USE_FIRST_CHOSEN_ROW    = False
ROW_INDEX               = 0        # used only when USE_FIRST_CHOSEN_ROW=False
OCTREE_QUERY_NODES      = None     # None = use full stored octree (deterministic overfit)
REPEAT_STEPS_PER_EPOCH  = 64      # repeated optimizer steps on the same sample per epoch

# ---------------------------------------------------------------------------
# ── Full-training config ─────────────────────────────────────────────────────
# Used when PARQUET_DIR is set.
# ---------------------------------------------------------------------------
TRAIN_VAL_SPLIT         = 0.9     # fraction of parquet files for training
BATCH_SIZE              = 4
NUM_WORKERS             = 0        # set > 0 only if not running inside NX

# ---------------------------------------------------------------------------
# ── Common hyperparameters ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------
NUM_EPOCHS              = 200
LEARNING_RATE           = 1e-4
PRINT_EVERY             = 10
SEED                    = 0

# ── Loss weights ──────────────────────────────────────────────────────────────
OCTREE_POS_WEIGHT_FACTOR  = 2.0   # BCE pos-class weight (compensate class imbalance)
OCTREE_DEPTH_WEIGHT_BASE  = 2.0   # fine cells get 2^(fine-coarse) × more gradient
MONOTONICITY_WEIGHT       = 0.1   # soft constraint: already-empty cells must stay empty
                                   # set 0.0 if octree_occ_labels_before is not in data

# ── EMA gradient balancer (only relevant for planner+octree joint training) ──
USE_LOSS_BALANCER         = False  # transition-only has just one loss term → no need
BALANCER_MOMENTUM         = 0.99


# ---------------------------------------------------------------------------
# ── Helpers ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RepeatSingleSampleDataset(Dataset):
    """Repeats one dataset row many times so each epoch has multiple optimizer steps."""

    def __init__(self, base_dataset: Dataset, base_index: int, repeat_count: int) -> None:
        self.base_dataset = base_dataset
        self.base_index = int(base_index)
        self.repeat_count = max(1, int(repeat_count))

    def __len__(self) -> int:
        return self.repeat_count

    def __getitem__(self, index: int):
        del index
        return self.base_dataset[self.base_index]


def _resolve_row_index(parquet_path: str, use_first_chosen_row: bool, row_index: int) -> int:
    df = pd.read_parquet(parquet_path)
    if use_first_chosen_row:
        if "is_chosen" not in df.columns:
            raise ValueError("USE_FIRST_CHOSEN_ROW=True but parquet has no 'is_chosen' column.")
        chosen_indices = df.index[df["is_chosen"].astype(int) == 1].tolist()
        if not chosen_indices:
            raise ValueError("No rows with is_chosen == 1 in the selected parquet file.")
        return int(chosen_indices[0])

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"ROW_INDEX={row_index} is out of range for parquet with {len(df)} rows.")
    return int(row_index)


def _scan_parquet_files(parquet_dir: str) -> list[Path]:
    """Returns all .parquet files found recursively under parquet_dir."""
    root = Path(parquet_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"PARQUET_DIR is not a directory: {root}")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under: {root}")
    return files


def _split_files(files: list[Path], train_fraction: float, seed: int):
    """Splits a file list into (train_files, val_files)."""
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * train_fraction))
    return shuffled[:split], shuffled[split:] or shuffled[:1]


def _describe_row(dataset: ProcessSkeletonParquetDataset, row_index: int) -> None:
    row = dataset.df.iloc[int(row_index)]
    macro_id = int(row["macro_class_id"])
    tool_id = (
        int(row["tool_choice_id"])
        if "tool_choice_id" in row.index and not pd.isna(row["tool_choice_id"])
        else -1
    )
    tool_name = ID_TO_TOOL_CHOICE.get(tool_id, ("unknown", -1.0))
    octree_count = 0
    if (
        "octree_centers" in row.index
        and row["octree_centers"] is not None
        and not (isinstance(row["octree_centers"], float) and np.isnan(row["octree_centers"]))
    ):
        octree_count = int(np.asarray(row["octree_centers"], dtype=object).size // 3)
    has_before = (
        "octree_occ_labels_before" in row.index
        and row["octree_occ_labels_before"] is not None
        and not (isinstance(row["octree_occ_labels_before"], float) and np.isnan(row["octree_occ_labels_before"]))
    )

    print("[Sample]")
    print(f"  source_row_index        : {row_index}")
    print(f"  part_name               : {row.get('part_name', '?')}")
    print(f"  decision_step           : {row.get('decision_step', '?')}")
    print(f"  candidate_index         : {row.get('candidate_index', '?')}")
    print(f"  is_chosen               : {row.get('is_chosen', '?')}")
    print(f"  macro_class             : {ID_TO_MACRO_CLASS.get(macro_id, 'unknown')} ({macro_id})")
    print(f"  tool_choice             : {tool_name[0]}_{tool_name[1]} ({tool_id})")
    print(f"  action_face_id          : {row.get('action_face_id', row.get('target_node_id', '?'))}")
    print(f"  octree_leaf_count       : {octree_count}")
    print(f"  has_before_occ_labels   : {has_before}  ← monotonicity loss {'ON' if has_before else 'OFF (set MONOTONICITY_WEIGHT=0.0)'}")


@torch.no_grad()
def _evaluate_transition(model: GraphSdfPlanningModel, batch: dict, device: torch.device) -> dict:
    model.eval()
    state_embedding = model.encode_state(
        state_points=batch["state_points"].to(device),
        node_process_state=batch["node_process_state"].to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch["node_centrality"].to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch["spatial_pos"].to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch["face_area"].to(device) if batch.get("face_area") is not None else None,
        node_mask=batch["node_mask"].to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch["point_mask"].to(device) if batch.get("point_mask") is not None else None,
    )
    oct_out = model.forward_octree(
        state_points=batch["state_points"].to(device),
        macro_class_id=batch["macro_class_id"].to(device),
        tool_choice_id=batch["tool_choice_id"].to(device),
        action_face_id=batch["action_face_id"].clamp_min(0).to(device),
        octree_centers=batch["octree_centers"].to(device),
        octree_depths=batch["octree_depths"].to(device),
        axis_visible=batch["axis_visible"].to(device) if batch.get("axis_visible") is not None else None,
        node_process_state=batch["node_process_state"].to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch["node_centrality"].to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch["spatial_pos"].to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch["face_area"].to(device) if batch.get("face_area") is not None else None,
        node_mask=batch["node_mask"].to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch["point_mask"].to(device) if batch.get("point_mask") is not None else None,
        state_embedding=state_embedding,
    )
    occ_prob = torch.sigmoid(oct_out["occ_logits"])
    occ_pred = (occ_prob >= 0.5).float()
    occ_gt   = batch["octree_occ_labels"].to(device)

    result = {
        "octree_acc":           float((occ_pred == occ_gt).float().mean().item()),
        "octree_pos_ratio_pred": float(occ_pred.mean().item()),
        "octree_pos_ratio_gt":   float(occ_gt.mean().item()),
        "octree_prob_mean":      float(occ_prob.mean().item()),
    }

    # Monotonicity violation rate: fraction of cells where pred > before (should be 0)
    if "octree_occ_labels_before" in batch and batch["octree_occ_labels_before"] is not None:
        before = batch["octree_occ_labels_before"].to(device)
        violation = (occ_pred > before).float()
        result["mono_violation_rate"] = float(violation.mean().item())

    return result


def _print_eval(epoch: int, train_loss: float, val_loss: float, pred: dict) -> None:
    mono_str = ""
    if "mono_violation_rate" in pred:
        mono_str = f" mono_viol={pred['mono_violation_rate']:.4f}"
    print(
        f"[Epoch {epoch:04d}] train={train_loss:.6f} val={val_loss:.6f}\n"
        f"            octree_acc={pred['octree_acc']:.4f} "
        f"pred_pos={pred['octree_pos_ratio_pred']:.4f} "
        f"gt_pos={pred['octree_pos_ratio_gt']:.4f} "
        f"prob_mean={pred['octree_prob_mean']:.4f}"
        f"{mono_str}"
    )


# ---------------------------------------------------------------------------
# ── Main ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Build DataLoaders ─────────────────────────────────────────────────
    full_train_mode = bool(PARQUET_DIR and PARQUET_DIR.strip())

    if full_train_mode:
        # ── FULL TRAINING: scan folder, split by file, iterate all rows ──
        all_files = _scan_parquet_files(PARQUET_DIR)
        train_files, val_files = _split_files(all_files, TRAIN_VAL_SPLIT, SEED)
        print(
            f"[FullTrain] {len(all_files)} parquet files found.\n"
            f"  train files : {len(train_files)}\n"
            f"  val   files : {len(val_files)}"
        )
        train_dataset = ProcessSkeletonParquetDataset(
            [str(f) for f in train_files],
            octree_query_nodes=OCTREE_QUERY_NODES,
        )
        val_dataset = ProcessSkeletonParquetDataset(
            [str(f) for f in val_files],
            octree_query_nodes=OCTREE_QUERY_NODES,
        )
        print(f"  train rows  : {len(train_dataset)}")
        print(f"  val   rows  : {len(val_dataset)}")
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS,
        )
        # Use first val batch for per-epoch evaluation output
        first_batch = next(iter(val_loader))

    else:
        # ── OVERFIT: single file, single row, repeated ────────────────────
        if not PARQUET_PATH:
            raise ValueError("Set PARQUET_PATH (overfit) or PARQUET_DIR (full training) at the top.")
        parquet_path = str(Path(PARQUET_PATH).expanduser().resolve())
        source_row_index = _resolve_row_index(parquet_path, USE_FIRST_CHOSEN_ROW, ROW_INDEX)
        base_dataset = ProcessSkeletonParquetDataset(
            [parquet_path], octree_query_nodes=OCTREE_QUERY_NODES
        )
        _describe_row(base_dataset, source_row_index)
        train_dataset = RepeatSingleSampleDataset(base_dataset, source_row_index, REPEAT_STEPS_PER_EPOCH)
        val_dataset   = RepeatSingleSampleDataset(base_dataset, source_row_index, 1)
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader    = DataLoader(val_dataset,   batch_size=1, shuffle=False)
        first_batch   = next(iter(val_loader))

    # ── Build model ───────────────────────────────────────────────────────
    model_cfg = GraphSdfModelConfig()
    if OCTREE_QUERY_NODES is not None:
        model_cfg = replace(model_cfg, octree_query_nodes=int(OCTREE_QUERY_NODES))
    model = GraphSdfPlanningModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    balancer  = EMALossBalancer(momentum=BALANCER_MOMENTUM) if USE_LOSS_BALANCER else None

    mode_str = "FullTrain" if full_train_mode else "Overfit-TransitionOnly"
    print(
        f"[{mode_str}] epochs={NUM_EPOCHS} lr={LEARNING_RATE} "
        f"octree_query_nodes={OCTREE_QUERY_NODES}\n"
        f"  pos_weight={OCTREE_POS_WEIGHT_FACTOR}  depth_weight_base={OCTREE_DEPTH_WEIGHT_BASE}  "
        f"monotonicity_weight={MONOTONICITY_WEIGHT}"
    )

    # ── Training loop ─────────────────────────────────────────────────────
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
        val_loss   = float(sum(val_losses)   / max(len(val_losses), 1))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS:
            pred = _evaluate_transition(model, first_batch, device)
            _print_eval(epoch, train_loss, val_loss, pred)

    print(f"[Done] {mode_str} run finished.")


if __name__ == "__main__":
    main()
