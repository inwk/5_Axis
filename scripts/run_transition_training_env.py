"""Run transition-only training with environment-variable overrides.

This keeps train_transition_dataset.py usable from VSCode while allowing
scripted smoke/overfit/pilot runs without editing constants each time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_transition_dataset as train


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw.strip() else int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw.strip() else float(default)


def main() -> None:
    train.PARQUET_DIR = os.getenv("PARQUET_DIR", train.PARQUET_DIR)
    train.PARQUET_LIST_PATH = os.getenv("PARQUET_LIST_PATH", train.PARQUET_LIST_PATH)
    train.PARQUET_GLOB = os.getenv("PARQUET_GLOB", train.PARQUET_GLOB)
    train.EXPLICIT_PARQUET_PATHS = [
        item.strip()
        for item in os.getenv("EXPLICIT_PARQUET_PATHS", "").split(os.pathsep)
        if item.strip()
    ]

    train.VAL_RATIO = _env_float("VAL_RATIO", train.VAL_RATIO)
    train.SEED = _env_int("SEED", train.SEED)
    train.BATCH_SIZE = _env_int("BATCH_SIZE", train.BATCH_SIZE)
    train.NUM_WORKERS = _env_int("NUM_WORKERS", train.NUM_WORKERS)
    train.NUM_EPOCHS = _env_int("NUM_EPOCHS", train.NUM_EPOCHS)
    train.LEARNING_RATE = _env_float("LEARNING_RATE", train.LEARNING_RATE)
    train.PRINT_EVERY = _env_int("PRINT_EVERY", train.PRINT_EVERY)
    train.TRAIN_LOG_EVERY_BATCHES = _env_int("TRAIN_LOG_EVERY_BATCHES", train.TRAIN_LOG_EVERY_BATCHES)
    train.MAX_TRAIN_BATCHES_PER_EPOCH = _env_int(
        "MAX_TRAIN_BATCHES_PER_EPOCH",
        train.MAX_TRAIN_BATCHES_PER_EPOCH,
    )
    train.MAX_VAL_BATCHES = _env_int("MAX_VAL_BATCHES", train.MAX_VAL_BATCHES)
    train.MAX_TRAIN_FILES = _env_int("MAX_TRAIN_FILES", train.MAX_TRAIN_FILES)
    train.MAX_VAL_FILES = _env_int("MAX_VAL_FILES", train.MAX_VAL_FILES)
    train.LAZY_PARQUET_LOADING = _env_bool("LAZY_PARQUET_LOADING", train.LAZY_PARQUET_LOADING)
    train.SPLIT_BY_PART = _env_bool("SPLIT_BY_PART", train.SPLIT_BY_PART)
    train.SDF_RESAMPLE_EACH_EPOCH = _env_bool("SDF_RESAMPLE_EACH_EPOCH", train.SDF_RESAMPLE_EACH_EPOCH)
    train.USE_TARGET_TSDF_INPUT = _env_bool("USE_TARGET_TSDF_INPUT", train.USE_TARGET_TSDF_INPUT)
    train.OVERFIT_DEBUG_MODE = os.getenv("OVERFIT_DEBUG_MODE", train.OVERFIT_DEBUG_MODE)
    train.OVERFIT_PART_NAME = os.getenv("OVERFIT_PART_NAME", train.OVERFIT_PART_NAME)

    train.SDF_QUERY_NODES = _env_int("SDF_QUERY_NODES", train.SDF_QUERY_NODES)
    train.SDF_CHANGED_SAMPLE_FRACTION = _env_float(
        "SDF_CHANGED_SAMPLE_FRACTION",
        train.SDF_CHANGED_SAMPLE_FRACTION,
    )
    train.SAVE_CHECKPOINTS = _env_bool("SAVE_CHECKPOINTS", train.SAVE_CHECKPOINTS)
    train.RUN_NAME = os.getenv("RUN_NAME", train.RUN_NAME)
    train.CHECKPOINT_ROOT = os.getenv("CHECKPOINT_ROOT", train.CHECKPOINT_ROOT)
    train.RESUME_CHECKPOINT = os.getenv("RESUME_CHECKPOINT", train.RESUME_CHECKPOINT)
    train.RESUME_OPTIMIZER = _env_bool("RESUME_OPTIMIZER", train.RESUME_OPTIMIZER)

    train.main()


if __name__ == "__main__":
    main()
