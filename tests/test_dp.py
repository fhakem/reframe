"""DP correctness: brute force, connectivity, CPU/GPU parity."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from reframe.core.energy import backward_energy, forward_costs
from reframe.core.seams import find_seam, remove_seam, seam_cost


def all_seams(h: int, w: int):
    for start in range(w):
        for steps in itertools.product((-1, 0, 1), repeat=h - 1):
            seam = [start]
            for s in steps:
                seam.append(seam[-1] + s)
            if all(0 <= c < w for c in seam):
                yield np.array(seam, dtype=np.int64)


def brute_force(base, costs=None) -> float:
    h, w = base.shape
    return min(seam_cost(s, base, costs) for s in all_seams(h, w))


small_images = st.tuples(st.integers(1, 6), st.integers(1, 6)).flatmap(
    lambda hw: arrays(np.float64, hw, elements=st.floats(0, 255, allow_nan=False))
)


@given(small_images)
def test_backward_dp_matches_brute_force(gray):
    base = backward_energy(gray)
    seam, cost = find_seam(base)
    assert cost == pytest.approx(brute_force(base), rel=1e-9, abs=1e-9)
    assert cost == pytest.approx(seam_cost(seam, base), rel=1e-9, abs=1e-9)


@given(small_images, st.data())
def test_forward_dp_matches_brute_force(gray, data):
    # Forward energy: same path-dependent cost model on both sides.
    extra = data.draw(arrays(np.float64, gray.shape, elements=st.floats(0, 50)))
    costs = forward_costs(gray)
    seam, cost = find_seam(extra, costs)
    assert cost == pytest.approx(brute_force(extra, costs), rel=1e-9, abs=1e-9)
    assert cost == pytest.approx(seam_cost(seam, extra, costs), rel=1e-9, abs=1e-9)


@given(st.integers(2, 40), st.integers(2, 40), st.integers(0, 2**32 - 1))
def test_seam_is_8_connected_one_pixel_per_row(h, w, seed):
    gray = np.random.default_rng(seed).uniform(0, 255, (h, w))
    seam, _ = find_seam(np.zeros_like(gray), forward_costs(gray))
    assert seam.shape == (h,)
    assert seam.min() >= 0 and seam.max() < w
    assert np.all(np.abs(np.diff(seam)) <= 1)
    out = remove_seam(gray, seam)
    assert out.shape == (h, w - 1)
    for i in range(h):
        np.testing.assert_array_equal(out[i], np.delete(gray[i], seam[i]))


def test_known_seam_follows_dark_column():
    base = np.full((8, 9), 100.0)
    base[:, 4] = 0.0
    seam, cost = find_seam(base)
    assert np.all(seam == 4)
    assert cost == 0.0


@pytest.mark.parametrize("mode", ["backward", "forward"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("accel", marks=pytest.mark.gpu)])
def test_torch_parity(mode, device, rng):
    torch = pytest.importorskip("torch")
    from reframe.core.seams.dp_torch import default_device, find_seam_torch

    if device == "accel":
        device = default_device()
        if device == "cpu":
            pytest.skip("no CUDA/MPS device")
    for _ in range(5):
        gray = rng.uniform(0, 255, (48, 64))
        if mode == "backward":
            base, costs = backward_energy(gray), None
        else:
            base, costs = rng.uniform(0, 10, gray.shape), forward_costs(gray)
        s_np, c_np = find_seam(base, costs)
        s_t, c_t = find_seam_torch(base, costs, device=device)
        fp32 = torch.device(device).type == "mps"
        tol = 1e-5 if fp32 else 1e-12
        assert c_t == pytest.approx(c_np, rel=tol)
        # Paths may differ only on (near-)ties: both must have the optimal cost.
        assert seam_cost(s_t, base, costs) == pytest.approx(c_np, rel=tol)
        if not fp32:
            np.testing.assert_array_equal(s_t, s_np)
