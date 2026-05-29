"""Parallel Stage A runner: PRT -> NX-consistent static embeddings."""

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
WORKER_SCRIPT = "extract_prt_static_embeddings_v2.py"

# ---------------------------------------------------------------------------
# User config: edit these directly in VSCode/debug mode.
# ---------------------------------------------------------------------------
CORES = 1
PC_NAME = "615"
FORCE = False
CURRENT_SEED = 0

SHARED_BASE_DIR = r"Y:\04_개별폴더\22. 통합과정 오인욱"
PRT_INPUT_DIR = os.path.join(SHARED_BASE_DIR, "prt_dataset")
STATIC_EMBEDDING_DIR = os.path.join(SHARED_BASE_DIR, "sdf_static_embeddings")


REQUIRED_STATIC_FILES = (
    "target_body.obj",
    "graph_nx.json",
    "embed_centrality.npy",
    "embed_spatial_pos.npy",
    "embed_face_area.npy",
    "embed_face_type.npy",
    "embed_face_pc.npy",
    "embed_face_normal.npy",
    "embed_node_mask.npy",
    "embed_point_mask.npy",
    "face_index_manifest.json",
    "static_manifest.json",
)


def _safe_name_component(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("._-")


def _part_name(prt_path: str) -> str:
    return Path(prt_path).stem


def _static_dir(part_name: str) -> Path:
    return Path(STATIC_EMBEDDING_DIR).expanduser().resolve() / part_name


def _is_completed_static_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / name).exists() for name in REQUIRED_STATIC_FILES):
        return False
    try:
        manifest = json.loads((path / "static_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("status") == "completed"


def _already_processed(part_name: str) -> bool:
    return _is_completed_static_dir(_static_dir(part_name))


def process_file_safe(file_info: tuple[str, str, str, bool]) -> None:
    prt_path, output_root, pc_name, force = file_info
    part = _part_name(prt_path)
    pc_slug = _safe_name_component(pc_name)
    output_root_path = Path(output_root).expanduser().resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    if not force and _already_processed(part):
        return

    lock_file = output_root_path / f"{part}.processing"
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL)
        os.close(fd)
    except (FileExistsError, OSError):
        return

    start_time = time.time()
    log_stem = f"{part}_seed{CURRENT_SEED}"
    if pc_slug:
        log_stem = f"{log_stem}_{pc_slug}"
    log_path = output_root_path / f"{log_stem}.log"
    cmd = [
        PYTHON_EXE,
        WORKER_SCRIPT,
        "--input",
        prt_path,
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

    try:
        print(f"[Start] static {part} (PID: {os.getpid()} pc={pc_slug or '-'})")
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(cmd, check=False, stdout=log_f, stderr=log_f, env=env)
        if result.returncode != 0:
            print(f"[Error] static {part} failed (rc={result.returncode}), see {log_path}")
        else:
            print(f"[Done] static {part} finished in {time.time() - start_time:.1f}s")
    except Exception as exc:
        print(f"[Error] static {part}: {exc!r}")
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    input_dir = Path(PRT_INPUT_DIR).expanduser().resolve()
    output_dir = Path(STATIC_EMBEDDING_DIR).expanduser().resolve()
    if not input_dir.exists():
        print(f"[Error] PRT input directory not found: {input_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = glob.glob(str(input_dir / "*.prt"))
    random.shuffle(all_files)
    tasks = [(path, str(output_dir), PC_NAME, bool(FORCE)) for path in all_files]

    print(f"Using Python: {PYTHON_EXE}")
    print(f"Worker Script: {Path(WORKER_SCRIPT).resolve()}")
    print(f"Input PRT dir: {input_dir}")
    print(f"Static embedding dir: {output_dir}")
    print(f"Found {len(all_files)} PRT files. Starting with {CORES} cores.")

    with multiprocessing.Pool(processes=max(1, int(CORES)), maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(process_file_safe, tasks, chunksize=1):
            pass
    print("All static embedding tasks processed or skipped.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
