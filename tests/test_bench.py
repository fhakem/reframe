"""Performance benchmarks (pytest-benchmark). Run: uv run pytest --benchmark-only."""

from __future__ import annotations

import numpy as np
import pytest

from reframe.config import ReframeConfig
from reframe.core.energy import forward_costs
from reframe.core.pipeline import plan_frame, reframe_frame
from reframe.core.seams import find_seam


@pytest.fixture(scope="module")
def gray540() -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 255, (540, 405))


def test_bench_dp_numba_540p(benchmark, gray540):
    costs = forward_costs(gray540)
    base = np.zeros_like(gray540)
    benchmark(find_seam, base, costs)


@pytest.mark.gpu
def test_bench_dp_torch_540p(benchmark, gray540):
    pytest.importorskip("torch")
    from reframe.core.seams.dp_torch import default_device, find_seam_torch

    costs = forward_costs(gray540)
    base = np.zeros_like(gray540)
    benchmark(find_seam_torch, base, costs, device=default_device())


def test_bench_frame_1080p(benchmark):
    cfg = ReframeConfig(seam_ratio=0.25)
    plan = plan_frame(1920, 1080, cfg)
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    out = benchmark.pedantic(reframe_frame, args=(frame, plan, cfg), rounds=2, iterations=1)
    assert out.shape == (1080, 608, 3)
