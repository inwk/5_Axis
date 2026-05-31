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
    _make_kernel_offsets,
    _shift_or_mask_cpu,
    _shift_or_mask_gpu,
    _torch_cuda_available,
)


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
DIMS = (160, 160, 160)
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
) -> None:
    cpu_time, cpu_out = _time_call(cpu_fn, REPEATS, WARMUP)
    print(f"[{name}] CPU avg={cpu_time:.6f}s true_ratio={float(cpu_out.mean()):.6f}")

    if gpu_fn is None:
        print(f"[{name}] GPU skipped: CUDA is not available")
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


def main() -> None:
    os.environ["SYNTHETIC_CSPACE_GPU_MAX_PAIRS"] = str(int(GPU_MAX_PAIRS))

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
        f"gpu_max_pairs={int(GPU_MAX_PAIRS)}"
    )

    _print_case(
        "holder_forbidden_shift",
        lambda: _shift_or_mask_cpu(obstacle, kernels["holder"]),
        (lambda: _shift_or_mask_gpu(obstacle, kernels["holder"])) if cuda else None,
    )
    _print_case(
        "cutter_swept_dilation",
        lambda: _dilate_config_mask_cpu(config, kernels["cutter"]),
        (lambda: _dilate_config_mask_gpu(config, kernels["cutter"])) if cuda else None,
    )


if __name__ == "__main__":
    main()
