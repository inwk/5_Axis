"""Removes incomplete axis-collection artifacts while preserving completed runs."""

import argparse
import glob
import json
import os
import shutil


DEFAULT_BASE_DIR = r"Y:\04_개별폴더\22. 통합과정 오인욱"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DIR, "sdf_dataset_out")


def _is_completed_run_dir(run_dir: str) -> bool:
    """Returns True only when a run directory has a finished episode record."""
    if not os.path.isdir(run_dir):
        return False
    episode_record_path = os.path.join(run_dir, "episode_record.json")
    if not os.path.isfile(episode_record_path):
        return False
    local_parquets = glob.glob(os.path.join(run_dir, "*_process_skeleton_dataset.parquet"))
    if not local_parquets:
        return False
    try:
        with open(episode_record_path, "r", encoding="utf-8") as f:
            episode_record = json.load(f)
    except Exception:
        return False
    termination = episode_record.get("termination")
    if not isinstance(termination, dict):
        return False
    return bool(termination.get("reason"))


def _extract_part_name_from_run_name(run_name: str, seed: int) -> str:
    token = f"_seed{int(seed)}_"
    idx = run_name.rfind(token)
    return run_name[:idx] if idx >= 0 else run_name


def _extract_part_name_from_log_name(filename: str, seed: int) -> str:
    suffix = f"_seed{int(seed)}.log"
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


def _global_parquet_run_name(filename: str) -> str:
    if filename.endswith("_chosen_only.parquet"):
        return filename[: -len("_chosen_only.parquet")]
    if filename.endswith(".parquet"):
        return filename[: -len(".parquet")]
    return filename


def _remove_path(path: str, dry_run: bool, is_dir: bool = False) -> bool:
    """Removes one file or directory unless running in dry-run mode."""
    if dry_run:
        return True
    try:
        if is_dir:
            shutil.rmtree(path)
        else:
            os.remove(path)
        return True
    except OSError as exc:
        print(f"[WARN] Failed to remove {path}: {exc}")
        return False


def cleanup_output_root(output_dir: str, seed: int, dry_run: bool = False) -> dict[str, int]:
    """Cleans incomplete run folders, stale global parquet files, and stale logs/locks."""
    output_dir = os.path.abspath(output_dir)
    run_pattern = os.path.join(output_dir, f"*_seed{int(seed)}_*")
    run_dirs = [path for path in glob.glob(run_pattern) if os.path.isdir(path)]

    completed_run_names: set[str] = set()
    completed_parts: set[str] = set()
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        if _is_completed_run_dir(run_dir):
            completed_run_names.add(run_name)
            completed_parts.add(_extract_part_name_from_run_name(run_name, seed))

    stats = {
        "removed_run_dirs": 0,
        "removed_global_parquets": 0,
        "removed_logs": 0,
        "removed_processing_files": 0,
    }

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        if run_name in completed_run_names:
            continue
        print(f"[CLEAN] removing incomplete run dir: {run_dir}")
        if _remove_path(run_dir, dry_run=dry_run, is_dir=True):
            stats["removed_run_dirs"] += 1

    global_dir = os.path.join(output_dir, "_ALL_PARQUET_FILES")
    if os.path.isdir(global_dir):
        for parquet_path in glob.glob(os.path.join(global_dir, f"*_seed{int(seed)}_*.parquet")):
            run_name = _global_parquet_run_name(os.path.basename(parquet_path))
            if run_name in completed_run_names:
                continue
            run_dir = os.path.join(output_dir, run_name)
            if _is_completed_run_dir(run_dir):
                continue
            print(f"[CLEAN] removing stale global parquet: {parquet_path}")
            if _remove_path(parquet_path, dry_run=dry_run, is_dir=False):
                stats["removed_global_parquets"] += 1

    for log_path in glob.glob(os.path.join(output_dir, f"*_seed{int(seed)}.log")):
        part_name = _extract_part_name_from_log_name(os.path.basename(log_path), seed)
        if part_name in completed_parts:
            continue
        print(f"[CLEAN] removing incomplete-part log: {log_path}")
        if _remove_path(log_path, dry_run=dry_run, is_dir=False):
            stats["removed_logs"] += 1

    for lock_path in glob.glob(os.path.join(output_dir, "*.processing")):
        print(f"[CLEAN] removing stale processing lock: {lock_path}")
        if _remove_path(lock_path, dry_run=dry_run, is_dir=False):
            stats["removed_processing_files"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean incomplete axis dataset output artifacts")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory that contains run folders and _ALL_PARQUET_FILES",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed suffix to target")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    args = parser.parse_args()

    stats = cleanup_output_root(args.output, seed=args.seed, dry_run=bool(args.dry_run))
    print(
        "[DONE] "
        f"removed_run_dirs={stats['removed_run_dirs']} "
        f"removed_global_parquets={stats['removed_global_parquets']} "
        f"removed_logs={stats['removed_logs']} "
        f"removed_processing_files={stats['removed_processing_files']}"
    )


if __name__ == "__main__":
    main()
