"""Benchmark TSDF generation from dense solid masks.

Edit constants below and run directly from VS Code/debug mode.

This measures the current CPU exact Euclidean distance transform used by the
synthetic dataset generator. If CuPy is installed with CUDA support, it also
benchmarks cupyx.scipy.ndimage.distance_transform_edt as a GPU candidate.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# User config: edit these directly in VS Code.
# ---------------------------------------------------------------------------
DIMS_LIST = [
    (72, 72, 72),
    (96, 96, 96),
    (128, 128, 128),
    (160, 160, 160),
]
SPACING = (2.5, 2.5, 2.5)
TRUNCATION = 5.0
SEED = 0
REPEATS = 3
WARMUP = 1


def _make_solid(dims: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nx, ny, nz = dims
    xs = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")

    # A nontrivial stock-like solid: ellipsoid plus a few bumps/cavities.
    solid = (x / 0.75) ** 2 + (y / 0.55) ** 2 + (z / 0.65) ** 2 <= 1.0
    for _ in range(8):
        center = rng.uniform(-0.55, 0.55, size=3).astype(np.float32)
        radius = float(rng.uniform(0.12, 0.22))
        blob = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= radius ** 2
        if rng.random() < 0.5:
            solid |= blob
        else:
            solid &= ~blob
    return solid


def _tsdf_cpu(solid: np.ndarray) -> np.ndarray:
    outside = ndimage.distance_transform_edt(~solid, sampling=SPACING)
    inside = ndimage.distance_transform_edt(solid, sampling=SPACING)
    signed = outside - inside
    return np.clip(signed / float(TRUNCATION), -1.0, 1.0).astype(np.float32)


def _cupy_available() -> bool:
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cpx_ndimage
    except Exception:
        return False
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        _ = cpx_ndimage.distance_transform_edt
    except Exception:
        return False
    return True


def _tsdf_cupy(solid: np.ndarray) -> np.ndarray:
    import cupy as cp
    from cupyx.scipy import ndimage as cpx_ndimage

    solid_gpu = cp.asarray(solid)
    outside = cpx_ndimage.distance_transform_edt(~solid_gpu, sampling=SPACING)
    inside = cpx_ndimage.distance_transform_edt(solid_gpu, sampling=SPACING)
    signed = outside - inside
    tsdf = cp.clip(signed / float(TRUNCATION), -1.0, 1.0).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(tsdf)


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


def main() -> None:
    has_cupy = _cupy_available()
    print(f"[Config] spacing={SPACING} truncation={TRUNCATION} cupy_cuda={has_cupy}")
    for dims in DIMS_LIST:
        solid = _make_solid(dims, SEED)
        voxels = int(np.prod(dims))
        cpu_time, cpu_tsdf = _time_call(lambda: _tsdf_cpu(solid), REPEATS, WARMUP)
        print(
            f"[TSDF CPU] dims={dims} voxels={voxels:,} "
            f"avg={cpu_time:.6f}s solid_ratio={float(solid.mean()):.6f}"
        )

        if not has_cupy:
            print(f"[TSDF GPU] dims={dims} skipped: CuPy/CUDA EDT is not available")
            continue

        gpu_time, gpu_tsdf = _time_call(lambda: _tsdf_cupy(solid), REPEATS, WARMUP)
        mae = float(np.mean(np.abs(cpu_tsdf - gpu_tsdf)))
        max_abs = float(np.max(np.abs(cpu_tsdf - gpu_tsdf)))
        speedup = cpu_time / max(gpu_time, 1e-12)
        print(
            f"[TSDF GPU] dims={dims} avg={gpu_time:.6f}s "
            f"speedup={speedup:.2f}x mae={mae:.8f} max_abs={max_abs:.8f}"
        )


if __name__ == "__main__":
    main()
