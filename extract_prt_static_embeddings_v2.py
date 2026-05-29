"""Extracts NX-consistent static embeddings from one PRT file.

This is Stage A of the split v2 pipeline:

    PRT -> static_embedding_root/<part_name>/

NX is used only as a PRT/B-rep parser.  This script does not create CAM
operations, toolpaths, IPW simulations, or transition labels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from collect_axis_dataset_synthetic_v2 import (
    STATIC_FILES,
    _extract_static_features_with_nx,
    _part_name_from_prt,
)


def _is_completed_static_dir(path: Path) -> bool:
    manifest_path = path / "static_manifest.json"
    if not manifest_path.exists():
        return False
    if not all((path / name).exists() for name in STATIC_FILES):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("status") == "completed"


def extract_static_embeddings(
    prt_path: str,
    output_root: str,
    seed: int,
    pc_name: str = "",
    force: bool = False,
) -> dict:
    part_name = _part_name_from_prt(prt_path)
    output_dir = Path(output_root).expanduser().resolve() / part_name
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and _is_completed_static_dir(output_dir):
        return {
            "status": "skipped_existing",
            "part_name": part_name,
            "prt_file_path": str(Path(prt_path).expanduser().resolve()),
            "static_feature_dir": str(output_dir),
        }

    info = _extract_static_features_with_nx(prt_path, output_dir, seed)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "stage": "nx_static_embedding_extraction",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "part_name": part_name,
        "prt_file_path": str(Path(prt_path).expanduser().resolve()),
        "static_feature_dir": str(output_dir),
        "pc_name": str(pc_name),
        "seed": int(seed),
        "nx_usage": "prt_open_static_brep_feature_extraction_only",
        "required_files": list(STATIC_FILES),
        "normalization_center_xyz": info["normalization_center_xyz"].tolist(),
        "normalization_scale": float(info["normalization_scale"]),
        "bbox_extent_xyz": info["bbox_extent_xyz"].tolist(),
        "target_body_mesh_path": str(info["target_body_mesh_path"]),
        "raw_face_count": int(info["raw_face_count"]),
    }
    (output_dir / "static_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract static NX graph embeddings from one PRT")
    parser.add_argument("--input", required=True, help="Input .prt path")
    parser.add_argument("--output", required=True, help="Static embedding output root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pc-name", default=os.getenv("PC_NAME", ""))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = extract_static_embeddings(
        prt_path=args.input,
        output_root=args.output,
        seed=int(args.seed),
        pc_name=args.pc_name,
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
