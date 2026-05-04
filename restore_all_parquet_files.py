"""Restores missing _ALL_PARQUET_FILES copies from completed run folders."""

import argparse
import glob
import json
import os
import shutil

try:
    from cleanup_incomplete_axis_outputs import DEFAULT_OUTPUT_DIR
except Exception:
    DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "sdf_dataset_out")


def _is_completed_run_dir(run_dir: str) -> bool:
    """Returns True only when a run directory has a finished episode record."""
    if not os.path.isdir(run_dir):
        return False
    episode_record_path = os.path.join(run_dir, "episode_record.json")
    if not os.path.isfile(episode_record_path):
        return False
    if not glob.glob(os.path.join(run_dir, "*_process_skeleton_dataset.parquet")):
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


def _part_name_from_run_name(run_name: str, seed: int) -> str:
    token = f"_seed{int(seed)}_"
    idx = run_name.rfind(token)
    return run_name[:idx] if idx >= 0 else run_name


def _global_parquet_stem_from_folder(run_name: str, seed: int) -> str:
    """Returns the PC-independent global parquet stem for a run folder."""
    token = f"_seed{int(seed)}_"
    idx = run_name.rfind(token)
    if idx < 0:
        return run_name
    prefix = run_name[: idx + len(token)]
    suffix = run_name[idx + len(token) :]
    parts = suffix.split("_")
    # create_run_output_dir uses YYYYMMDD_HHMMSS[_pc].
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return prefix + "_".join(parts[:2])
    return run_name


def _copy_if_missing(src_path: str, dst_path: str, dry_run: bool) -> str:
    if os.path.exists(dst_path):
        return "exists"
    if not os.path.isfile(src_path):
        return "missing_src"
    if not dry_run:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
    return "restored"


def restore_all_parquet_files(output_dir: str, seed: int, dry_run: bool = False) -> dict[str, int]:
    """Copies missing global parquet files from completed local run folders."""
    output_dir = os.path.abspath(output_dir)
    global_dir = os.path.join(output_dir, "_ALL_PARQUET_FILES")
    run_pattern = os.path.join(output_dir, f"*_seed{int(seed)}_*")

    stats = {
        "scanned_run_dirs": 0,
        "skipped_incomplete_run_dirs": 0,
        "restored_parquets": 0,
        "restored_chosen_parquets": 0,
        "skipped_existing_parquets": 0,
        "skipped_existing_chosen_parquets": 0,
        "missing_local_parquets": 0,
        "missing_local_chosen_parquets": 0,
    }

    for run_dir in sorted(path for path in glob.glob(run_pattern) if os.path.isdir(path)):
        stats["scanned_run_dirs"] += 1
        run_name = os.path.basename(run_dir)
        if not _is_completed_run_dir(run_dir):
            stats["skipped_incomplete_run_dirs"] += 1
            continue

        part_name = _part_name_from_run_name(run_name, seed)
        global_stem = _global_parquet_stem_from_folder(run_name, seed)
        local_parquet = os.path.join(run_dir, f"{part_name}_seed{int(seed)}_process_skeleton_dataset.parquet")
        global_parquet = os.path.join(global_dir, f"{global_stem}.parquet")

        status = _copy_if_missing(local_parquet, global_parquet, dry_run=dry_run)
        if status == "restored":
            print(f"[RESTORE] {local_parquet} -> {global_parquet}")
            stats["restored_parquets"] += 1
        elif status == "exists":
            stats["skipped_existing_parquets"] += 1
        else:
            print(f"[WARN] missing local parquet: {local_parquet}")
            stats["missing_local_parquets"] += 1

        local_chosen = os.path.join(run_dir, f"{part_name}_seed{int(seed)}_process_skeleton_dataset_chosen_only.parquet")
        global_chosen = os.path.join(global_dir, f"{global_stem}_chosen_only.parquet")
        if os.path.isfile(local_chosen):
            chosen_status = _copy_if_missing(local_chosen, global_chosen, dry_run=dry_run)
            if chosen_status == "restored":
                print(f"[RESTORE] {local_chosen} -> {global_chosen}")
                stats["restored_chosen_parquets"] += 1
            elif chosen_status == "exists":
                stats["skipped_existing_chosen_parquets"] += 1
        else:
            stats["missing_local_chosen_parquets"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore missing _ALL_PARQUET_FILES parquet copies from completed run folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory that contains run folders and _ALL_PARQUET_FILES",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed suffix to target")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying")
    args = parser.parse_args()

    stats = restore_all_parquet_files(args.output, seed=args.seed, dry_run=bool(args.dry_run))
    print(
        "[DONE] "
        f"scanned_run_dirs={stats['scanned_run_dirs']} "
        f"skipped_incomplete_run_dirs={stats['skipped_incomplete_run_dirs']} "
        f"restored_parquets={stats['restored_parquets']} "
        f"restored_chosen_parquets={stats['restored_chosen_parquets']} "
        f"skipped_existing_parquets={stats['skipped_existing_parquets']} "
        f"skipped_existing_chosen_parquets={stats['skipped_existing_chosen_parquets']} "
        f"missing_local_parquets={stats['missing_local_parquets']} "
        f"missing_local_chosen_parquets={stats['missing_local_chosen_parquets']}"
    )


if __name__ == "__main__":
    main()
