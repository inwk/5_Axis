"""Generates synthetic scenario parquet rows from extracted static embeddings.

This is Stage B of the split v2 pipeline:

    static_embedding_root/<part_name>/ -> synthetic_dataset_root/<part_name>_seed<seed>/

This script does not import NXOpen and can run on machines without NX.  It reads
the static embedding package and writes directly trainable synthetic TSDF parquet
rows using the Minkowski/C-space generator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from collect_axis_dataset_synthetic_v2 import (
    _generate_rows,
    _safe_filename,
    _write_rows_to_parquet,
)


def _load_static_manifest(static_dir: Path) -> dict:
    manifest_path = static_dir / "static_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing static_manifest.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _is_completed_dataset_dir(path: Path, part_name: str, seed: int) -> bool:
    episode_path = path / "episode_record.json"
    parquet_path = path / f"{part_name}_seed{int(seed)}_process_skeleton_dataset.parquet"
    if not episode_path.exists() or not parquet_path.exists():
        return False
    try:
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    termination = episode.get("termination", {})
    return isinstance(termination, dict) and bool(termination.get("reason"))


def generate_from_static_dir(
    static_dir: str,
    output_root: str,
    seed: int,
    pc_name: str = "",
    force: bool = False,
) -> dict:
    total_start = time.time()
    print(f"[Worker Timing] start static_dir={static_dir}", flush=True)
    static_path = Path(static_dir).expanduser().resolve()
    manifest = _load_static_manifest(static_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"Static manifest is not completed: {static_path}")
    print(f"[Worker Timing] load_manifest={time.time() - total_start:.2f}s part={manifest.get('part_name') or static_path.name}", flush=True)

    part_name = str(manifest.get("part_name") or static_path.name)
    prt_path = str(manifest.get("prt_file_path") or f"{part_name}.prt")
    pc_slug = _safe_filename(pc_name)

    out_root = Path(output_root).expanduser().resolve()
    out_dir = out_root / f"{part_name}_seed{int(seed)}"
    if pc_slug:
        out_dir = out_root / f"{part_name}_seed{int(seed)}_{pc_slug}"
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and _is_completed_dataset_dir(out_dir, part_name, seed):
        print(f"[Worker Timing] skipped_existing total={time.time() - total_start:.2f}s part={part_name}", flush=True)
        return {
            "status": "skipped_existing",
            "part_name": part_name,
            "static_feature_dir": str(static_path),
            "dataset_dir": str(out_dir),
        }

    max_rows = max(1, int(os.getenv("SYNTHETIC_SCENARIOS_PER_PART", "200")))
    rows_start = time.time()
    print(f"[Worker Timing] generate_rows_start part={part_name} max_rows={max_rows}", flush=True)
    rows = _generate_rows(
        prt_path=prt_path,
        out_dir=out_dir,
        static_dir=static_path,
        seed=seed,
        max_rows=max_rows,
    )
    print(f"[Worker Timing] generate_rows_done part={part_name} rows={len(rows)} elapsed={time.time() - rows_start:.2f}s total={time.time() - total_start:.2f}s", flush=True)
    if not rows:
        raise ValueError(f"No synthetic rows generated for {static_path}")

    parquet_path = out_dir / f"{part_name}_seed{int(seed)}_process_skeleton_dataset.parquet"
    chosen_path = out_dir / f"{part_name}_seed{int(seed)}_process_skeleton_dataset_chosen_only.parquet"
    write_start = time.time()
    print(f"[Worker Timing] write_parquet_start part={part_name}", flush=True)
    _write_rows_to_parquet(rows, parquet_path)
    _write_rows_to_parquet(rows, chosen_path)
    print(f"[Worker Timing] write_parquet_done part={part_name} elapsed={time.time() - write_start:.2f}s total={time.time() - total_start:.2f}s", flush=True)

    global_dir = out_root / "_ALL_PARQUET_FILES"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_stem = f"{part_name}_seed{int(seed)}"
    if pc_slug:
        global_stem = f"{global_stem}_{pc_slug}"
    global_parquet = global_dir / f"{global_stem}.parquet"
    global_chosen = global_dir / f"{global_stem}_chosen_only.parquet"
    copy_start = time.time()
    shutil.copy2(parquet_path, global_parquet)
    shutil.copy2(chosen_path, global_chosen)
    print(f"[Worker Timing] copy_global_done part={part_name} elapsed={time.time() - copy_start:.2f}s total={time.time() - total_start:.2f}s", flush=True)

    episode_record = {
        "part_name": part_name,
        "seed": int(seed),
        "row_unit": "one_synthetic_minkowski_transition",
        "stage": "synthetic_dataset_generation_from_static_embeddings",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "static_feature_dir": str(static_path),
        "prt_file_path": prt_path,
        "num_rows": int(len(rows)),
        "label_status": "synthetic_minkowski_cspace",
        "nx_required": False,
        "termination": {"reason": "synthetic_scenario_generation_complete"},
        "outputs": {
            "parquet_path": str(parquet_path),
            "chosen_parquet_path": str(chosen_path),
            "global_parquet_path": str(global_parquet),
            "global_chosen_parquet_path": str(global_chosen),
        },
    }
    (out_dir / "episode_record.json").write_text(
        json.dumps(episode_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Worker Timing] complete part={part_name} total={time.time() - total_start:.2f}s", flush=True)
    return episode_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic parquet from static embeddings")
    parser.add_argument("--static-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pc-name", default=os.getenv("PC_NAME", ""))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = generate_from_static_dir(
        static_dir=args.static_dir,
        output_root=args.output,
        seed=int(args.seed),
        pc_name=args.pc_name,
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
