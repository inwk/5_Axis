"""Parallel runner for large-scale axis dataset collection on shared storage."""

import argparse
import glob
import multiprocessing
import os
import random
import subprocess
import sys
import time

# Use the current interpreter so jobs run in the active environment.
PYTHON_EXE = sys.executable
WORKER_SCRIPT = "collect_axis_dataset.py"

# Shared network paths for distributed collection machines.
SHARED_INPUT_DIR = r"\\165.132.180.130\04_媛쒕퀎?대뜑\22. ?듯빀怨쇱젙 ?ㅼ씤??prt_dataset"
SHARED_OUTPUT_DIR = r"\\165.132.180.130\04_媛쒕퀎?대뜑\22. ?듯빀怨쇱젙 ?ㅼ씤??axis_dataset_out"
CURRENT_SEED = 0


def _already_collected(part_name: str, output_dir: str) -> bool:
    """Checks whether a part already has a completed parquet dataset."""
    pattern = os.path.join(output_dir, f"{part_name}_seed{CURRENT_SEED}_*")
    for folder in glob.glob(pattern):
        if glob.glob(os.path.join(folder, "*.parquet")):
            return True
    return False


def process_file_safe(file_info: tuple[str, str]) -> None:
    """Processes one part with lock-file protection to avoid duplicate work."""
    prt_path, output_dir = file_info
    part_name = os.path.splitext(os.path.basename(prt_path))[0]

    if _already_collected(part_name, output_dir):
        return

    lock_file = os.path.join(output_dir, f"{part_name}.processing")
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL)
        os.close(fd)
    except (FileExistsError, OSError):
        return

    print(f"[Start] {part_name} (PID: {os.getpid()})")
    start_time = time.time()

    try:
        log_path = os.path.join(output_dir, f"{part_name}_seed{CURRENT_SEED}.log")
        cmd = [
            PYTHON_EXE,
            WORKER_SCRIPT,
            "--input",
            prt_path,
            "--output",
            output_dir,
            "--seed",
            str(CURRENT_SEED),
        ]
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(cmd, check=False, stdout=log_f, stderr=log_f)
        if result.returncode != 0:
            print(f"[Error] {part_name} failed (rc={result.returncode}), see {log_path}")
        else:
            print(f"[Done] {part_name} finished in {time.time() - start_time:.1f}s")
    except subprocess.CalledProcessError:
        print(f"[Error] Script failed for {part_name}")
    except Exception as exc:
        print(f"[Error] Unexpected error for {part_name}: {exc}")
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass


def main() -> None:
    """Parses arguments and executes multiprocessing workers."""
    parser = argparse.ArgumentParser(description="Parallel axis dataset collector runner")
    parser.add_argument("--cores", type=int, default=8, help="Number of local CPU cores to use")
    args = parser.parse_args()

    print(f"Using Python: {PYTHON_EXE}")
    print(f"Worker Script: {os.path.abspath(WORKER_SCRIPT)}")

    if not os.path.exists(SHARED_INPUT_DIR):
        print(f"[Error] Shared directory not found: {SHARED_INPUT_DIR}")
        return

    all_files = glob.glob(os.path.join(SHARED_INPUT_DIR, "*.prt"))
    random.shuffle(all_files)
    tasks = [(path, SHARED_OUTPUT_DIR) for path in all_files]

    print(f"Found {len(all_files)} files. Starting with {args.cores} cores.")
    with multiprocessing.Pool(processes=args.cores, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(process_file_safe, tasks, chunksize=1):
            pass
    print("All tasks processed (or skipped).")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
