"""Overfit sanity check on a single parquet row.

Edit the constants below and run this file directly from VSCode/debug mode.
No CLI args are required.
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
from graph_sdf.training import transition_train_step, transition_validation_step


# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode.
# ---------------------------------------------------------------------------
PARQUET_PATH = r"Y:\04_개별폴더\22. 통합과정 오인욱\sdf_dataset_out\3dDataset0055_seed0_20260415_204610\3dDataset0055_seed0_process_skeleton_dataset.parquet"
USE_FIRST_CHOSEN_ROW = False
ROW_INDEX = 0                  # used only when USE_FIRST_CHOSEN_ROW=False
OCTREE_QUERY_NODES = None      # None = use full stored octree for deterministic overfit
REPEAT_STEPS_PER_EPOCH = 64    # repeated optimizer steps on the same sample per epoch
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
PRINT_EVERY = 10
SEED = 0


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


def _describe_row(dataset: ProcessSkeletonParquetDataset, row_index: int) -> None:
    row = dataset.df.iloc[int(row_index)]
    macro_id = int(row["macro_class_id"])
    tool_id = int(row["tool_choice_id"]) if "tool_choice_id" in row.index and not pd.isna(row["tool_choice_id"]) else -1
    tool_name = ID_TO_TOOL_CHOICE.get(tool_id, ("unknown", -1.0))
    octree_count = 0
    if "octree_centers" in row.index and row["octree_centers"] is not None and not (isinstance(row["octree_centers"], float) and np.isnan(row["octree_centers"])):
        octree_count = int(np.asarray(row["octree_centers"], dtype=object).size // 3)

    print("[Sample]")
    print(f"  source_row_index : {row_index}")
    print(f"  part_name        : {row.get('part_name', '?')}")
    print(f"  decision_step    : {row.get('decision_step', '?')}")
    print(f"  candidate_index  : {row.get('candidate_index', '?')}")
    print(f"  is_chosen        : {row.get('is_chosen', '?')}")
    print(f"  macro_class      : {ID_TO_MACRO_CLASS.get(macro_id, 'unknown')} ({macro_id})")
    print(f"  tool_choice      : {tool_name[0]}_{tool_name[1]} ({tool_id})")
    print(f"  action_face_id   : {row.get('action_face_id', row.get('target_node_id', '?'))}")
    print(f"  octree_leaf_count: {octree_count}")


@torch.no_grad()
def _evaluate_transition(model: GraphSdfPlanningModel, batch: dict, device: torch.device) -> dict:
    model.eval()
    state_embedding = model.encode_state(
        state_points=batch["state_points"].to(device),
        node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
        node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
    )
    oct_out = model.forward_octree(
        state_points=batch["state_points"].to(device),
        macro_class_id=batch["macro_class_id"].to(device),
        tool_choice_id=batch["tool_choice_id"].to(device),
        action_face_id=batch["action_face_id"].clamp_min(0).to(device),
        octree_centers=batch["octree_centers"].to(device),
        octree_depths=batch["octree_depths"].to(device),
        axis_visible=batch.get("axis_visible").to(device) if batch.get("axis_visible") is not None else None,
        node_process_state=batch.get("node_process_state").to(device) if batch.get("node_process_state") is not None else None,
        node_centrality=batch.get("node_centrality").to(device) if batch.get("node_centrality") is not None else None,
        spatial_pos=batch.get("spatial_pos").to(device) if batch.get("spatial_pos") is not None else None,
        face_area=batch.get("face_area").to(device) if batch.get("face_area") is not None else None,
        node_mask=batch.get("node_mask").to(device) if batch.get("node_mask") is not None else None,
        point_mask=batch.get("point_mask").to(device) if batch.get("point_mask") is not None else None,
        state_embedding=state_embedding,
    )
    occ_prob = torch.sigmoid(oct_out["occ_logits"])
    occ_pred = (occ_prob >= 0.5).float()
    occ_gt = batch["octree_occ_labels"].to(device)
    return {
        "octree_acc": float((occ_pred == occ_gt).float().mean().item()),
        "octree_pos_ratio_pred": float(occ_pred.mean().item()),
        "octree_pos_ratio_gt": float(occ_gt.mean().item()),
        "octree_prob_mean": float(occ_prob.mean().item()),
    }


def main() -> None:
    if not PARQUET_PATH:
        raise ValueError("Set PARQUET_PATH at the top of overfit_one_sample.py")

    parquet_path = str(Path(PARQUET_PATH).expanduser().resolve())
    set_seed(SEED)

    source_row_index = _resolve_row_index(parquet_path, USE_FIRST_CHOSEN_ROW, ROW_INDEX)
    base_dataset = ProcessSkeletonParquetDataset([parquet_path], octree_query_nodes=OCTREE_QUERY_NODES)
    _describe_row(base_dataset, source_row_index)

    train_dataset = RepeatSingleSampleDataset(base_dataset, source_row_index, REPEAT_STEPS_PER_EPOCH)
    val_dataset = RepeatSingleSampleDataset(base_dataset, source_row_index, 1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = GraphSdfModelConfig()
    if OCTREE_QUERY_NODES is not None:
        model_cfg = replace(model_cfg, octree_query_nodes=int(OCTREE_QUERY_NODES))
    model = GraphSdfPlanningModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    first_batch = next(iter(val_loader))
    print(f"[Device] {device}")
    print(
        f"[Train][TransitionOnly] epochs={NUM_EPOCHS} lr={LEARNING_RATE} "
        f"repeat_steps={REPEAT_STEPS_PER_EPOCH} octree_query_nodes={OCTREE_QUERY_NODES}"
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        train_losses = []
        for batch in train_loader:
            loss = transition_train_step(
                model,
                batch,
                optimizer,
                device,
                octree_pos_weight_factor=2.0,
            )
            train_losses.append(loss)

        val_losses = [transition_validation_step(model, batch, device) for batch in val_loader]
        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss = float(sum(val_losses) / max(len(val_losses), 1))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS:
            pred = _evaluate_transition(model, first_batch, device)
            print(f"[Epoch {epoch:04d}] train={train_loss:.6f} val={val_loss:.6f}")
            print(
                f"            octree_acc={pred['octree_acc']:.4f} "
                f"pred_pos={pred['octree_pos_ratio_pred']:.4f} "
                f"gt_pos={pred['octree_pos_ratio_gt']:.4f} "
                f"prob_mean={pred['octree_prob_mean']:.4f}"
            )

    print("[Done] one-sample overfit run finished.")


if __name__ == "__main__":
    main()
