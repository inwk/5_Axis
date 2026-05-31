"""Parallel Stage B runner: static embeddings -> synthetic training parquet."""

from __future__ import annotations

import glob
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time
from pathlib import Path


PYTHON_EXE = sys.executable
WORKER_SCRIPT = "generate_synthetic_dataset_from_embeddings_v2.py"

# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode/debug mode.
# ---------------------------------------------------------------------------
CORES = 1
PC_NAME = "615"
FORCE = False
CURRENT_SEED = 0

SHARED_BASE_DIR = r"Y:\04_개별폴더\22. 통합과정 오인욱"
STATIC_EMBEDDING_DIR = os.path.join(SHARED_BASE_DIR, "sdf_static_embeddings")
SYNTHETIC_DATASET_DIR = os.path.join(SHARED_BASE_DIR, "sdf_dataset_synthetic_v2")

# Synthetic generation controls.
# "auto" uses CUDA for C-space/Minkowski boolean masks when available, otherwise CPU.
SYNTHETIC_CSPACE_DEVICE = "auto"  # "auto", "cuda", or "cpu"
SYNTHETIC_CSPACE_GPU_MAX_PAIRS = 4_000_000
SYNTHETIC_GRID_RESOLUTION = 256


def _safe_name_component(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("._-")


def _load_static_manifest(static_dir: Path) -> dict | None:
    path = static_dir / "static_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_completed_static_dir(static_dir: Path) -> bool:
    manifest = _load_static_manifest(static_dir)
    return bool(manifest and manifest.get("status") == "completed")


def _part_name_from_static(static_dir: Path) -> str:
    manifest = _load_static_manifest(static_dir)
    if manifest and manifest.get("part_name"):
        return str(manifest["part_name"])
    return static_dir.name


def _dataset_dir(part_name: str, output_root: str, seed: int, pc_name: str = "") -> Path:
    pc_slug = _safe_name_component(pc_name)
    stem = f"{part_name}_seed{int(seed)}"
    if pc_slug:
        stem = f"{stem}_{pc_slug}"
    return Path(output_root).expanduser().resolve() / stem


def _already_processed(part_name: str, output_root: str, seed: int, pc_name: str = "") -> bool:
    out_dir = _dataset_dir(part_name, output_root, seed, pc_name)
    episode_path = out_dir / "episode_record.json"
    parquet_path = out_dir / f"{part_name}_seed{int(seed)}_process_skeleton_dataset.parquet"
    if not episode_path.exists() or not parquet_path.exists():
        return False
    try:
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    termination = episode.get("termination", {})
    return isinstance(termination, dict) and bool(termination.get("reason"))


def _resolve_static_dirs(root: Path) -> list[Path]:
    candidates = [Path(path).resolve() for path in glob.glob(str(root / "*")) if Path(path).is_dir()]
    return [path for path in candidates if _is_completed_static_dir(path)]


def process_static_dir_safe(task: tuple[str, str, str, bool]) -> None:
    static_dir_raw, output_root, pc_name, force = task
    static_dir = Path(static_dir_raw).expanduser().resolve()
    part_name = _part_name_from_static(static_dir)
    pc_slug = _safe_name_component(pc_name)
    output_root_path = Path(output_root).expanduser().resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    if not force and _already_processed(part_name, str(output_root_path), CURRENT_SEED, pc_slug):
        return

    lock_file = output_root_path / f"{part_name}.processing"
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL)
        os.close(fd)
    except (FileExistsError, OSError):
        return

    start_time = time.time()
    log_stem = f"{part_name}_seed{CURRENT_SEED}"
    if pc_slug:
        log_stem = f"{log_stem}_{pc_slug}"
    log_path = output_root_path / f"{log_stem}.log"
    cmd = [
        PYTHON_EXE,
        WORKER_SCRIPT,
        "--static-dir",
        str(static_dir),
        "--output",
        str(output_root_path),
        "--seed",
        str(CURRENT_SEED),
    ]
    if pc_slug:
        cmd.extend(["--pc-name", pc_slug])
    if force:
        cmd.append("--force")

    env = os.environ.copy()
    if pc_slug:
        env["PC_NAME"] = pc_slug
    env["SYNTHETIC_CSPACE_DEVICE"] = str(SYNTHETIC_CSPACE_DEVICE)
    env["SYNTHETIC_CSPACE_GPU_MAX_PAIRS"] = str(int(SYNTHETIC_CSPACE_GPU_MAX_PAIRS))
    env["SYNTHETIC_GRID_RESOLUTION"] = str(int(SYNTHETIC_GRID_RESOLUTION))

    try:
        print(f"[Start] synthetic {part_name} (PID: {os.getpid()} pc={pc_slug or '-'})")
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(cmd, check=False, stdout=log_f, stderr=log_f, env=env)
        if result.returncode != 0:
            print(f"[Error] synthetic {part_name} failed (rc={result.returncode}), see {log_path}")
        else:
            print(f"[Done] synthetic {part_name} finished in {time.time() - start_time:.1f}s")
    except Exception as exc:
        print(f"[Error] synthetic {part_name}: {exc!r}")
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    static_root = Path(STATIC_EMBEDDING_DIR).expanduser().resolve()
    output_root = Path(SYNTHETIC_DATASET_DIR).expanduser().resolve()
    if not static_root.exists():
        print(f"[Error] Static embedding directory not found: {static_root}")
        return
    output_root.mkdir(parents=True, exist_ok=True)

    static_dirs = _resolve_static_dirs(static_root)
    random.shuffle(static_dirs)
    tasks = [(str(path), str(output_root), PC_NAME, bool(FORCE)) for path in static_dirs]

    print(f"Using Python: {PYTHON_EXE}")
    print(f"Worker Script: {Path(WORKER_SCRIPT).resolve()}")
    print(f"Static embedding root: {static_root}")
    print(f"Synthetic dataset root: {output_root}")
    print(f"Found {len(static_dirs)} completed static dirs. Starting with {CORES} cores.")

    with multiprocessing.Pool(processes=max(1, int(CORES)), maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(process_static_dir_safe, tasks, chunksize=1):
            pass
    print("All synthetic dataset tasks processed or skipped.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
