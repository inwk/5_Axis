"""Benchmark CPU vs GPU C-space/Minkowski mask operations.

Edit constants below and run directly from VS Code/debug mode.
This benchmark measures only the synthetic C-space boolean operations:
    - holder forbidden tip mask: out[p - offset] |= obstacle[p]
    - cutter swept volume dilation: out[p + offset] |= config[p]
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import numpy as np

from collect_axis_dataset_synthetic_v2 import (
    _dilate_config_mask_cpu,
    _dilate_config_mask_gpu,
    _holder_forbidden_mask,
    _make_kernel_offsets,
    _shift_or_mask_cpu,
    _shift_or_mask_gpu,
    _torch_cuda_available,
)


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
DIMS = (256, 256, 256)
SPACING = (2.5, 2.5, 2.5)
SEED = 0
REPEATS = 5
WARMUP = 1

OBSTACLE_FRACTION = 0.08
CONFIG_FRACTION = 0.05

AXIS_DIR = (0.0, 0.0, 1.0)
TOOL_KIND = "flat"
TOOL_RADIUS_MM = 5.0
TOOL_LENGTH_MM = 40.0
HOLDER_RADIUS_MM = 25.0
HOLDER_LENGTH_MM = 60.0

# Lower this if CUDA memory is tight; raise it if GPU utilization is low.
GPU_MAX_PAIRS = 4_000_000
HOLDER_CSPACE_RESOLUTION = 64
RUN_FULLRES_HOLDER = False


def _time_call(fn: Callable[[], np.ndarray], repeats: int, warmup: int) -> tuple[float, np.ndarray]:
    out = None
    for _ in range(max(0, int(warmup))):
        out = fn()

    times: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    assert out is not None
    return float(sum(times) / len(times)), out


def _print_case(
    name: str,
    cpu_fn: Callable[[], np.ndarray],
    gpu_fn: Callable[[], np.ndarray] | None,
    gpu_skip_reason: str = "CUDA is not available",
) -> None:
    cpu_time, cpu_out = _time_call(cpu_fn, REPEATS, WARMUP)
    print(f"[{name}] CPU avg={cpu_time:.6f}s true_ratio={float(cpu_out.mean()):.6f}")

    if gpu_fn is None:
        print(f"[{name}] GPU skipped: {gpu_skip_reason}")
        return

    gpu_time, gpu_out = _time_call(gpu_fn, REPEATS, WARMUP)
    equal = bool(np.array_equal(cpu_out, gpu_out))
    speedup = cpu_time / max(gpu_time, 1e-12)
    print(
        f"[{name}] GPU avg={gpu_time:.6f}s "
        f"speedup={speedup:.2f}x equal={equal} true_ratio={float(gpu_out.mean()):.6f}"
    )
    if not equal:
        diff = np.logical_xor(cpu_out, gpu_out)
        print(f"[{name}] mismatch_voxels={int(diff.sum())}")


def _holder_forbidden_coarse(obstacle: np.ndarray, device: str) -> np.ndarray:
    old_device = os.environ.get("SYNTHETIC_CSPACE_DEVICE")
    os.environ["SYNTHETIC_CSPACE_DEVICE"] = device
    try:
        bbox_min = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        bbox_max = (np.asarray(DIMS, dtype=np.float32) - 1.0) * np.asarray(SPACING, dtype=np.float32)
        return _holder_forbidden_mask(
            obstacle=obstacle,
            axis_dir=np.asarray(AXIS_DIR, dtype=np.float32),
            tool_kind=TOOL_KIND,
            tool_radius=float(TOOL_RADIUS_MM),
            tool_length=float(TOOL_LENGTH_MM),
            holder_radius=float(HOLDER_RADIUS_MM),
            holder_length=float(HOLDER_LENGTH_MM),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
    finally:
        if old_device is None:
            os.environ.pop("SYNTHETIC_CSPACE_DEVICE", None)
        else:
            os.environ["SYNTHETIC_CSPACE_DEVICE"] = old_device


def main() -> None:
    os.environ["SYNTHETIC_CSPACE_GPU_MAX_PAIRS"] = str(int(GPU_MAX_PAIRS))
    os.environ["SYNTHETIC_HOLDER_CSPACE_RESOLUTION"] = str(int(HOLDER_CSPACE_RESOLUTION))

    rng = np.random.default_rng(SEED)
    obstacle = rng.random(DIMS) < float(OBSTACLE_FRACTION)
    config = rng.random(DIMS) < float(CONFIG_FRACTION)
    kernels = _make_kernel_offsets(
        axis_dir=np.asarray(AXIS_DIR, dtype=np.float32),
        spacing=SPACING,
        tool_kind=TOOL_KIND,
        tool_radius=float(TOOL_RADIUS_MM),
        tool_length=float(TOOL_LENGTH_MM),
        holder_radius=float(HOLDER_RADIUS_MM),
        holder_length=float(HOLDER_LENGTH_MM),
    )

    cuda = _torch_cuda_available()
    print(f"[Config] dims={DIMS} spacing={SPACING} cuda={cuda}")
    print(
        "[Kernel] "
        f"cutter_offsets={int(kernels['cutter'].shape[0])} "
        f"holder_offsets={int(kernels['holder'].shape[0])} "
        f"gpu_max_pairs={int(GPU_MAX_PAIRS)} "
        f"holder_cspace_resolution={int(HOLDER_CSPACE_RESOLUTION)}"
    )

    if RUN_FULLRES_HOLDER:
        _print_case(
            "holder_forbidden_fullres_shift",
            lambda: _shift_or_mask_cpu(obstacle, kernels["holder"]),
            (lambda: _shift_or_mask_gpu(obstacle, kernels["holder"])) if cuda else None,
        )
    else:
        print("[holder_forbidden_fullres_shift] skipped: set RUN_FULLRES_HOLDER=True to benchmark exact full-res holder C-space")
    _print_case(
        "holder_forbidden_coarse64",
        lambda: _holder_forbidden_coarse(obstacle, "cpu"),
        None,
        gpu_skip_reason="policy=CPU fixed; cutter full-res remains GPU-auto",
    )
    _print_case(
        "cutter_swept_dilation",
        lambda: _dilate_config_mask_cpu(config, kernels["cutter"]),
        (lambda: _dilate_config_mask_gpu(config, kernels["cutter"])) if cuda else None,
    )


if __name__ == "__main__":
    main()
