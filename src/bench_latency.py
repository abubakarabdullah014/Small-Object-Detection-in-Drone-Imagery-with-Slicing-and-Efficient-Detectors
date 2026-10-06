"""
Latency benchmarking utilities for CPU inference.
Warmup + N repeats, optional thread pinning via OpenMP/MKL env vars.
"""
from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class LatencyStats:
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_repeats: int
    slices_per_image: float = 0.0
    forward_passes: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def configure_threads(num_threads: int = 4) -> None:
    """Pin BLAS / OpenMP / OpenVINO to a fixed thread count for stable latency."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[key] = str(num_threads)
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except Exception:
        pass


@contextmanager
def timer_ms():
    t0 = time.perf_counter()
    yield lambda: (time.perf_counter() - t0) * 1000.0


def measure_latency(
    fn: Callable[[], None],
    warmup: int = 3,
    repeats: int = 3,
    slices_per_image: float = 0.0,
    forward_passes: float = 0.0,
) -> LatencyStats:
    """Call fn() with warmup then measure repeats. fn should do one image end-to-end."""
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return LatencyStats(
        mean_ms=float(statistics.mean(times)),
        median_ms=float(statistics.median(times)),
        std_ms=float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        min_ms=float(min(times)),
        max_ms=float(max(times)),
        n_repeats=repeats,
        slices_per_image=slices_per_image,
        forward_passes=forward_passes,
    )


def aggregate_image_latencies(
    per_image_ms: List[float],
    slices: Optional[List[float]] = None,
    forwards: Optional[List[float]] = None,
) -> LatencyStats:
    if not per_image_ms:
        return LatencyStats(0, 0, 0, 0, 0, 0)
    return LatencyStats(
        mean_ms=float(statistics.mean(per_image_ms)),
        median_ms=float(statistics.median(per_image_ms)),
        std_ms=float(statistics.stdev(per_image_ms)) if len(per_image_ms) > 1 else 0.0,
        min_ms=float(min(per_image_ms)),
        max_ms=float(max(per_image_ms)),
        n_repeats=len(per_image_ms),
        slices_per_image=float(statistics.mean(slices)) if slices else 0.0,
        forward_passes=float(statistics.mean(forwards)) if forwards else 0.0,
    )


if __name__ == "__main__":
    configure_threads(4)

    def _noop():
        time.sleep(0.01)

    print(measure_latency(_noop, warmup=1, repeats=5).to_dict())
