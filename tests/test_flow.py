"""Optical flow direction conventions, seam propagation and backward warping."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from reframe.core.flow import available_flow_methods, estimate_flow, warp_columns, warp_image


def _texture(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    n = np.random.default_rng(seed).normal(size=(h, w))
    t = cv2.GaussianBlur(n, (0, 0), 2.0)
    return np.clip(128 + 50 * t / t.std(), 0, 255).astype(np.uint8)


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


@pytest.mark.parametrize("method", available_flow_methods())
def test_flow_recovers_integer_translation(method):
    a = _texture()
    b = _shift(a, 3, -2)  # content moves right 3, up 2
    flow = estimate_flow(a, b, method=method)
    assert flow.shape == (*a.shape, 2) and flow.dtype == np.float32
    inner = flow[20:-20, 20:-20]
    assert np.median(inner[..., 0]) == pytest.approx(3.0, abs=0.25)
    assert np.median(inner[..., 1]) == pytest.approx(-2.0, abs=0.25)


def test_flow_accepts_rgb_and_rejects_bad_input():
    a = _texture()
    rgb = np.repeat(a[..., None], 3, axis=2)
    assert estimate_flow(rgb, rgb).shape == (*a.shape, 2)
    with pytest.raises(ValueError):
        estimate_flow(a, a[:, :-1])
    with pytest.raises(ValueError):
        estimate_flow(a, a, method="nope")


def test_warp_image_is_backward_warp():
    a = _texture()
    b = _shift(a, 4, 1)
    flow = estimate_flow(a, b)
    back = warp_image(b, flow)  # ~ a
    err = np.abs(back.astype(float) - a.astype(float))[16:-16, 16:-16].mean()
    raw = np.abs(b.astype(float) - a.astype(float))[16:-16, 16:-16].mean()
    assert err < 0.15 * raw
    exact = np.zeros((*a.shape, 2), np.float32)
    exact[..., 0], exact[..., 1] = 4.0, 1.0
    np.testing.assert_array_equal(warp_image(b, exact)[:-1, :-4], a[:-1, :-4])


def test_warp_columns_constant_and_offset():
    h, w = 10, 50
    flow = np.zeros((h, w, 2), np.float32)
    flow[..., 0] = 2.5
    seam = np.full(h, 10.0)
    np.testing.assert_allclose(warp_columns(seam, flow), 12.5)
    seams = np.stack([seam, seam + 5])  # (n, h)
    np.testing.assert_allclose(warp_columns(seams, flow), seams + 2.5)
    # Positions relative to a crop starting at column 20 of the flow grid.
    flow[:, 30:, 0] = -1.0
    np.testing.assert_allclose(warp_columns(seam, flow, x_offset=20.0), 9.0)
    # Clamped to the flow grid.
    np.testing.assert_allclose(warp_columns(np.full(h, 48.0), flow), 47.0)
    with pytest.raises(ValueError):
        warp_columns(np.zeros(h + 1), flow)


def test_warp_columns_linear_interpolation_and_real_flow():
    h, w = 4, 10
    flow = np.zeros((h, w, 2), np.float32)
    flow[..., 0] = np.arange(w, dtype=np.float32)  # dx = x
    np.testing.assert_allclose(warp_columns(np.full(h, 2.25), flow), 4.5)
    a = _texture()
    b = _shift(a, 3, 0)
    seam = np.linspace(40, 60, a.shape[0])
    moved = warp_columns(seam, estimate_flow(a, b))
    assert np.median(moved - seam) == pytest.approx(3.0, abs=0.25)
