"""Validates collected parquet + npy dataset files.

Usage:
    python validate_dataset.py --dir D:\axis_dataset_out
    python validate_dataset.py --dir D:\axis_dataset_out --run my_part_seed0_20240101_120000
    python validate_dataset.py --parquet path\to\file.parquet   # single parquet only
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# ── schema constants (duplicated to avoid NX-dependent imports) ──────────────
MACRO_CLASS_TO_ID = {
    "indexed_rough": 0,
    "indexed_finish": 1,
    "point_finish": 2,
    "flank_finish": 3,
    "stop": 4,
}
ID_TO_MACRO = {v: k for k, v in MACRO_CLASS_TO_ID.items()}
TOOL_LIBRARY = [
    ("flat", 20.0), ("flat", 16.0), ("flat", 12.0), ("flat", 10.0),
    ("flat", 8.0),  ("flat", 6.0),  ("flat", 4.0),
    ("ball", 8.0),  ("ball", 6.0),  ("ball", 4.0),
]

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  △"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def _fail(msg: str) -> None:
    print(f"{FAIL} {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"{WARN} {msg}")


def _load_array(row, col):
    """Safely converts a parquet list-column cell to numpy array."""
    val = row[col]
    if val is None:
        return None
    return np.asarray(val, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# NPY validation
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_NPY = {
    "embed_centrality.npy":    {"shape": (512,),        "dtype": np.int16},
    "embed_spatial_pos.npy":   {"shape": (512, 512),    "dtype": np.int16},
    "embed_face_area.npy":     {"shape": (512, 1),      "dtype": np.float32},
    "embed_face_pc.npy":       {"shape": (512, 100, 3), "dtype": np.float32},
}


def validate_npy(run_dir: str) -> int:
    """Returns number of failures."""
    _hdr(f"NPY files  —  {os.path.basename(run_dir)}")
    failures = 0
    for fname, spec in EXPECTED_NPY.items():
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            _fail(f"{fname}  NOT FOUND")
            failures += 1
            continue
        arr = np.load(path)
        shape_ok = arr.shape == spec["shape"]
        dtype_ok = np.issubdtype(arr.dtype, spec["dtype"])
        has_nan  = bool(np.isnan(arr.astype(np.float32)).any())
        has_inf  = bool(np.isinf(arr.astype(np.float32)).any())

        if shape_ok and dtype_ok and not has_nan and not has_inf:
            _ok(f"{fname}  shape={arr.shape}  dtype={arr.dtype}  "
                f"range=[{arr.min():.3f}, {arr.max():.3f}]")
        else:
            tag = FAIL
            if not shape_ok:
                _fail(f"{fname}  shape mismatch: got {arr.shape}, expected {spec['shape']}")
                failures += 1
            if not dtype_ok:
                _warn(f"{fname}  dtype mismatch: got {arr.dtype}, expected {spec['dtype']}")
            if has_nan:
                _fail(f"{fname}  contains NaN")
                failures += 1
            if has_inf:
                _fail(f"{fname}  contains Inf")
                failures += 1

    # meta.json
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        k = meta.get("K_raw_faces", meta.get("K_nodes_compressed", 0))
        scale = meta.get("note", {}).get("normalization", {}).get("reference_scale", 0)
        _ok(f"meta.json  K={k}  reference_scale={scale:.2f}")
    else:
        _warn("meta.json  NOT FOUND")

    # episode_record.json
    ep_path = os.path.join(run_dir, "episode_record.json")
    if os.path.exists(ep_path):
        with open(ep_path, encoding="utf-8") as f:
            ep = json.load(f)
        steps = ep.get("steps", [])
        committed = [s for s in steps if not s.get("stopped", False)]
        stopped   = [s for s in steps if s.get("stopped", False)]
        _ok(f"episode_record.json  committed_steps={len(committed)}  "
            f"stop_reason={stopped[-1].get('reason') if stopped else 'max_steps'}")
    else:
        _warn("episode_record.json  NOT FOUND")

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Parquet validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "part_name", "decision_step", "candidate_index", "is_chosen",
    "macro_class_id", "macro_class_name",
    "tool_choice_id", "tool_choice_valid",
    "action_face_id", "action_face_valid",
    "state_points", "node_mask", "point_mask",
    "octree_centers", "octree_depths", "octree_occ_labels",
    "octree_bbox_min", "octree_bbox_max",
    "node_process_state", "global_process_state",
    "macro_class_mask", "tool_choice_mask", "action_face_mask",
    "centrality_512", "spatial_pos_512x512", "face_area_512x1",
    "axis_visible_512",
]

SHAPE_CHECKS = {
    "state_points":       (512, 100, 7),
    "node_mask":          (512,),
    "point_mask":         (512, 100),
    "node_process_state": (512, 2),
    "global_process_state": (11,),
    "macro_class_mask":   (5,),
    "tool_choice_mask":   (10,),
    "action_face_mask":   (512,),
    "centrality_512":     (512,),
    "face_area_512x1":    (512, 1),
    "axis_visible_512":   (512,),
    "octree_bbox_min":    (3,),
    "octree_bbox_max":    (3,),
}


def validate_parquet(pq_path: str) -> int:
    """Validates one parquet file. Returns number of failures."""
    _hdr(f"Parquet  —  {os.path.basename(pq_path)}")
    failures = 0

    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:
        _fail(f"Cannot read parquet: {e}")
        return 1

    n_rows = len(df)
    _ok(f"Loaded  {n_rows} rows  x  {len(df.columns)} columns")

    # ── required columns ──────────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        for c in missing:
            _fail(f"Missing column: {c}")
        failures += len(missing)
    else:
        _ok(f"All {len(REQUIRED_COLS)} required columns present")

    if n_rows == 0:
        _fail("No rows — empty parquet")
        return failures + 1

    # ── is_chosen distribution ────────────────────────────────────────────────
    if "is_chosen" in df.columns:
        chosen_count = int(df["is_chosen"].sum())
        steps = df["decision_step"].nunique() if "decision_step" in df.columns else "?"
        _ok(f"is_chosen: {chosen_count} chosen / {n_rows} total  "
            f"({100*chosen_count/n_rows:.1f}%)  across {steps} decision steps")
        if chosen_count == 0:
            _fail("No is_chosen=1 rows — planner has nothing to learn from")
            failures += 1

        # Each decision_step should have exactly 1 chosen row
        if "decision_step" in df.columns:
            per_step = df.groupby("decision_step")["is_chosen"].sum()
            bad_steps = per_step[per_step != 1]
            if len(bad_steps) > 0:
                _warn(f"Steps with ≠1 chosen row: {bad_steps.to_dict()}")
            else:
                _ok("Every decision_step has exactly 1 chosen row")

    # ── macro class distribution ──────────────────────────────────────────────
    if "macro_class_id" in df.columns:
        dist = df["macro_class_id"].value_counts().sort_index()
        dist_str = "  ".join(
            f"{ID_TO_MACRO.get(int(i), i)}={v}" for i, v in dist.items()
        )
        _ok(f"macro_class distribution:  {dist_str}")

    # ── tool masking consistency ──────────────────────────────────────────────
    if "tool_choice_id" in df.columns and "tool_choice_mask" in df.columns:
        mask_violations = 0
        for _, row in df[["tool_choice_id", "tool_choice_mask"]].iterrows():
            tid = int(row["tool_choice_id"])
            mask = np.asarray(row["tool_choice_mask"], dtype=np.int16)
            if 0 <= tid < len(mask) and mask[tid] == 1:
                mask_violations += 1
        if mask_violations:
            _fail(f"tool_choice_id violates tool_choice_mask in {mask_violations} rows")
            failures += 1
        else:
            _ok("tool_choice_id respects tool_choice_mask in all rows")

    # ── shape checks (sample first 5 rows) ───────────────────────────────────
    shape_failures = 0
    sample = df.head(5)
    for col, expected_shape in SHAPE_CHECKS.items():
        if col not in df.columns:
            continue
        for _, row in sample.iterrows():
            arr = _load_array(row, col)
            if arr is None:
                continue
            if arr.shape != expected_shape:
                _fail(f"Column '{col}' shape: got {arr.shape}, expected {expected_shape}")
                shape_failures += 1
                failures += 1
                break
        else:
            _ok(f"'{col}'  shape={expected_shape}  ✓")

    # ── octree transition target checks ──────────────────────────────────────
    if {"octree_centers", "octree_depths", "octree_occ_labels"}.issubset(df.columns):
        octree_failures = 0
        node_counts = []
        occ_ratios = []
        for _, row in df.head(10).iterrows():
            centers = _load_array(row, "octree_centers")
            depths = _load_array(row, "octree_depths")
            labels = _load_array(row, "octree_occ_labels")
            if centers is None or depths is None or labels is None:
                _fail("octree target has missing value")
                octree_failures += 1
                continue
            centers = centers.reshape(-1, 3)
            depths = depths.reshape(-1)
            labels = labels.reshape(-1)
            if not (len(centers) == len(depths) == len(labels)):
                _fail(
                    "octree target length mismatch: "
                    f"centers={len(centers)} depths={len(depths)} labels={len(labels)}"
                )
                octree_failures += 1
                continue
            if np.isnan(centers).any() or np.isnan(labels).any():
                _fail("octree target contains NaN")
                octree_failures += 1
                continue
            if not np.all((labels >= 0.0) & (labels <= 1.0)):
                _fail("octree_occ_labels must be in [0, 1]")
                octree_failures += 1
                continue
            node_counts.append(len(labels))
            occ_ratios.append(float(labels.mean()) if len(labels) else 0.0)
        if octree_failures:
            failures += octree_failures
        elif node_counts:
            _ok(
                "octree targets "
                f"K range=[{min(node_counts)}, {max(node_counts)}] "
                f"mean_occ_ratio={float(np.mean(occ_ratios)):.3f}"
            )

    # ── SDF value range checks (legacy optional) ─────────────────────────────
    if "next_node_sdf" in df.columns:
        chosen = df[df["is_chosen"] == 1] if "is_chosen" in df.columns else df
        sdfs = np.concatenate([
            np.asarray(v, dtype=np.float32).reshape(-1) for v in chosen["next_node_sdf"]
        ])
        sdf_min, sdf_max = float(sdfs.min()), float(sdfs.max())
        has_nan = bool(np.isnan(sdfs).any())
        has_neg = bool((sdfs < -0.05).any())
        _ok(f"next_node_sdf (chosen rows)  range=[{sdf_min:.4f}, {sdf_max:.4f}]"
            f"{'  ⚠ NaN!' if has_nan else ''}"
            f"{'  △ negative values' if has_neg else ''}")
        if has_nan:
            _fail("next_node_sdf contains NaN")
            failures += 1

    # ── volume reduction sanity ───────────────────────────────────────────────
    if "state_volume" in df.columns and "next_state_volume" in df.columns:
        chosen = df[df["is_chosen"] == 1] if "is_chosen" in df.columns else df
        if len(chosen) > 0:
            vol_ok = (chosen["next_state_volume"] <= chosen["state_volume"] + 1e-3).all()
            avg_removal = float(chosen["out_removed_ratio"].mean()) if "out_removed_ratio" in df.columns else float("nan")
            if vol_ok:
                _ok(f"Volume monotonically decreasing  avg_removal_ratio={avg_removal:.3f}")
            else:
                _warn("Some rows show volume increase (measurement noise?)")

    # ── SDF progression (state done ratio should increase) ────────────────────
    if "state_done_ratio" in df.columns and "next_done_ratio" in df.columns:
        chosen = df[df["is_chosen"] == 1].sort_values("decision_step") if "is_chosen" in df.columns else df
        if len(chosen) > 1:
            ratios = chosen["state_done_ratio"].tolist()
            nexts  = chosen["next_done_ratio"].tolist()
            print(f"  ℹ  done_ratio progression (chosen):")
            for i, (r, n) in enumerate(zip(ratios, nexts)):
                bar = "█" * int(n * 20)
                print(f"       step {i}: {r:.3f} → {n:.3f}  |{bar:<20}|")

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Run-directory validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_run_dir(run_dir: str) -> int:
    failures = validate_npy(run_dir)
    parquet_files = glob.glob(os.path.join(run_dir, "*.parquet"))
    if not parquet_files:
        _fail(f"No parquet file found in {run_dir}")
        return failures + 1
    for pq in parquet_files:
        failures += validate_parquet(pq)
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate collected dataset files")
    parser.add_argument("--dir",     type=str, help="Root output dir (scans all run subdirs)")
    parser.add_argument("--run",     type=str, help="Single run subdir name inside --dir")
    parser.add_argument("--parquet", type=str, help="Validate a single parquet file directly")
    args = parser.parse_args()

    total_failures = 0

    if args.parquet:
        total_failures += validate_parquet(args.parquet)

    elif args.dir and args.run:
        run_dir = os.path.join(args.dir, args.run)
        if not os.path.isdir(run_dir):
            print(f"[Error] Run dir not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        total_failures += validate_run_dir(run_dir)

    elif args.dir:
        # Find all run directories (subdirs with at least one parquet or npy)
        run_dirs = sorted([
            d for d in glob.glob(os.path.join(args.dir, "*"))
            if os.path.isdir(d) and (
                glob.glob(os.path.join(d, "*.parquet")) or
                glob.glob(os.path.join(d, "*.npy"))
            )
        ])
        if not run_dirs:
            print(f"[Error] No run directories found under {args.dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(run_dirs)} run director{'y' if len(run_dirs)==1 else 'ies'}")
        for rd in run_dirs:
            total_failures += validate_run_dir(rd)
            print()

    else:
        parser.print_help()
        sys.exit(0)

    print(f"\n{'='*60}")
    if total_failures == 0:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {total_failures} CHECK(S) FAILED  — see ✗ lines above")
    print(f"{'='*60}\n")
    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
