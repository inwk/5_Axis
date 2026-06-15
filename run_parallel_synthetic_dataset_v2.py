"""Parallel Stage B runner: static embeddings -> synthetic training parquet."""

from __future__ import annotations

import glob
import json
import multiprocessing
import os
import random
import re
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
SYNTHETIC_GRID_RESOLUTION = 160
SYNTHETIC_FINISH_LOCAL_FINE_SDF = True
SYNTHETIC_FINISH_LOCAL_GRID_RESOLUTION = 160
SYNTHETIC_FINISH_LOCAL_PADDING_MM = 5.0
SYNTHETIC_HOLDER_CSPACE_RESOLUTION = 64
SYNTHETIC_SCENARIOS_PER_PART = 100
SYNTHETIC_STEPS_PER_ROLLOUT = 6
SYNTHETIC_ROUGH_STEPS = 4

_OUTPUT_NAME_RE = re.compile(r"^(?P<part>.+)_seed(?P<seed>\d+)(?:_(?P<suffix>.*))?$")


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


def _is_completed_dataset_dir(out_dir: Path, part_name: str, seed: int) -> bool:
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


def _parse_output_part_seed(name: str) -> tuple[str, int] | None:
    stem = Path(name).stem
    if stem.endswith("_chosen_only"):
        return None
    match = _OUTPUT_NAME_RE.match(stem)
    if not match:
        return None
    return str(match.group("part")), int(match.group("seed"))


def _scan_existing_part_seed_keys(output_root: Path, seed: int) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()

    global_dir = output_root / "_ALL_PARQUET_FILES"
    if global_dir.exists():
        for parquet_path in sorted(global_dir.glob("*.parquet")):
            key = _parse_output_part_seed(parquet_path.name)
            if key is not None and key[1] == int(seed) and parquet_path.stat().st_size > 0:
                keys.add(key)

    for out_dir in sorted(output_root.glob(f"*_seed{int(seed)}*")):
        if not out_dir.is_dir():
            continue
        key = _parse_output_part_seed(out_dir.name)
        if key is not None and key[1] == int(seed) and _is_completed_dataset_dir(out_dir, key[0], key[1]):
            keys.add(key)

    return keys


def _already_processed(part_name: str, output_root: str, seed: int, pc_name: str = "") -> bool:
    del pc_name
    out_root = Path(output_root).expanduser().resolve()
    stem = f"{part_name}_seed{int(seed)}"
    for out_dir in sorted(out_root.glob(f"{stem}*")):
        if out_dir.is_dir() and _is_completed_dataset_dir(out_dir, part_name, seed):
            return True

    global_dir = out_root / "_ALL_PARQUET_FILES"
    if global_dir.exists():
        for parquet_path in sorted(global_dir.glob(f"{stem}*.parquet")):
            key = _parse_output_part_seed(parquet_path.name)
            if key == (part_name, int(seed)) and parquet_path.is_file() and parquet_path.stat().st_size > 0:
                return True
    return False


def _resolve_static_dirs(root: Path) -> list[Path]:
    candidates = [Path(path).resolve() for path in glob.glob(str(root / "*")) if Path(path).is_dir()]
    return candidates


def process_static_dir_safe(task: tuple[str, str, str, bool]) -> None:
    task_start = time.time()
    static_dir_raw, output_root, pc_name, force = task
    static_dir = Path(static_dir_raw).expanduser().resolve()
    manifest = _load_static_manifest(static_dir)
    if not manifest or manifest.get("status") != "completed":
        print(f"[Skip] synthetic {static_dir.name} incomplete_static check={time.time() - task_start:.2f}s", flush=True)
        return
    part_name = str(manifest.get("part_name") or static_dir.name)
    pc_slug = _safe_name_component(pc_name)
    output_root_path = Path(output_root).expanduser().resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    if not force and _already_processed(part_name, str(output_root_path), CURRENT_SEED, pc_slug):
        print(f"[Skip] synthetic {part_name} already processed check={time.time() - task_start:.2f}s", flush=True)
        return

    lock_file = output_root_path / f"{part_name}.processing"
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL)
        os.close(fd)
    except (FileExistsError, OSError):
        print(f"[Skip] synthetic {part_name} lock exists check={time.time() - task_start:.2f}s", flush=True)
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
    env["SYNTHETIC_FINISH_LOCAL_FINE_SDF"] = "1" if bool(SYNTHETIC_FINISH_LOCAL_FINE_SDF) else "0"
    env["SYNTHETIC_FINISH_LOCAL_GRID_RESOLUTION"] = str(int(SYNTHETIC_FINISH_LOCAL_GRID_RESOLUTION))
    env["SYNTHETIC_FINISH_LOCAL_PADDING_MM"] = str(float(SYNTHETIC_FINISH_LOCAL_PADDING_MM))
    env["SYNTHETIC_HOLDER_CSPACE_RESOLUTION"] = str(int(SYNTHETIC_HOLDER_CSPACE_RESOLUTION))
    env["SYNTHETIC_SCENARIOS_PER_PART"] = str(int(SYNTHETIC_SCENARIOS_PER_PART))
    env["SYNTHETIC_STEPS_PER_ROLLOUT"] = str(int(SYNTHETIC_STEPS_PER_ROLLOUT))
    env["SYNTHETIC_ROUGH_STEPS"] = str(int(SYNTHETIC_ROUGH_STEPS))

    try:
        print(f"[Start] synthetic {part_name} (PID: {os.getpid()} pc={pc_slug or '-'}) prep={start_time - task_start:.2f}s log={log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log_f:
            launch_start = time.time()
            result = subprocess.run(cmd, check=False, stdout=log_f, stderr=log_f, env=env)
        run_elapsed = time.time() - launch_start
        if result.returncode != 0:
            print(f"[Error] synthetic {part_name} failed (rc={result.returncode}) run={run_elapsed:.1f}s total={time.time() - task_start:.1f}s, see {log_path}", flush=True)
        else:
            print(f"[Done] synthetic {part_name} run={run_elapsed:.1f}s total={time.time() - task_start:.1f}s log={log_path}", flush=True)
    except Exception as exc:
        print(f"[Error] synthetic {part_name}: {exc!r}")
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    total_start = time.time()
    static_root = Path(STATIC_EMBEDDING_DIR).expanduser().resolve()
    output_root = Path(SYNTHETIC_DATASET_DIR).expanduser().resolve()
    if not static_root.exists():
        print(f"[Error] Static embedding directory not found: {static_root}")
        return
    output_root.mkdir(parents=True, exist_ok=True)

    scan_start = time.time()
    static_dirs = _resolve_static_dirs(static_root)
    print(f"[Timing] resolve_static_dirs={time.time() - scan_start:.2f}s", flush=True)
    random.shuffle(static_dirs)

    existing_scan_start = time.time()
    existing_keys = set() if bool(FORCE) else _scan_existing_part_seed_keys(output_root, CURRENT_SEED)
    skipped_existing = 0
    tasks = []
    for path in static_dirs:
        part_name = path.name
        if not bool(FORCE) and (part_name, int(CURRENT_SEED)) in existing_keys:
            skipped_existing += 1
            continue
        tasks.append((str(path), str(output_root), PC_NAME, bool(FORCE)))
    print(
        f"[Timing] scan_existing_outputs={time.time() - existing_scan_start:.2f}s "
        f"existing_part_seed={len(existing_keys)} skipped_existing={skipped_existing} "
        f"scheduled={len(tasks)}",
        flush=True,
    )

    print(f"Using Python: {PYTHON_EXE}")
    print(f"Worker Script: {Path(WORKER_SCRIPT).resolve()}")
    print(f"Static embedding root: {static_root}")
    print(f"Synthetic dataset root: {output_root}")
    print(
        f"Found {len(static_dirs)} static dirs. Scheduled {len(tasks)} dirs. "
        f"Completed status is checked inside workers for scheduled dirs. Starting with {CORES} cores."
    )
    print(
        f"[Config] grid={SYNTHETIC_GRID_RESOLUTION} holder_cspace={SYNTHETIC_HOLDER_CSPACE_RESOLUTION} "
        f"finish_local={SYNTHETIC_FINISH_LOCAL_FINE_SDF} local_grid={SYNTHETIC_FINISH_LOCAL_GRID_RESOLUTION} "
        f"rows_per_prt={SYNTHETIC_SCENARIOS_PER_PART} "
        f"steps_per_rollout={SYNTHETIC_STEPS_PER_ROLLOUT} rough_steps={SYNTHETIC_ROUGH_STEPS} "
        f"cspace_device={SYNTHETIC_CSPACE_DEVICE} gpu_pairs={SYNTHETIC_CSPACE_GPU_MAX_PAIRS}",
        flush=True,
    )

    pool_start = time.time()
    print(f"[Timing] pool_start_begin elapsed={pool_start - total_start:.2f}s", flush=True)
    with multiprocessing.Pool(processes=max(1, int(CORES)), maxtasksperchild=1) as pool:
        print(f"[Timing] pool_created={time.time() - pool_start:.2f}s", flush=True)
        for _ in pool.imap_unordered(process_static_dir_safe, tasks, chunksize=1):
            pass
    print(f"All synthetic dataset tasks processed or skipped. total={time.time() - total_start:.1f}s", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
