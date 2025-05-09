"""Carving, masks, insertion, forward vs backward, order-map upsampling."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from reframe.core.energy import to_gray
from reframe.core.metrics import ssim
from reframe.core.seams import apply_removal_order, carve_width, insert_seams, remove_seam


@st.composite
def image_and_n(draw, max_h=24, max_w=24, channels=3):
    h = draw(st.integers(2, max_h))
    w = draw(st.integers(3, max_w))
    n = draw(st.integers(0, w - 1))
    seed = draw(st.integers(0, 2**32 - 1))
    img = np.random.default_rng(seed).integers(0, 256, (h, w, channels), dtype=np.uint8)
    return img, n


@given(image_and_n(), st.sampled_from(["forward", "backward"]))
def test_output_shape_and_order_map(data, mode):
    img, n = data
    h, w, c = img.shape
    res = carve_width(img, n, mode=mode)
    assert res.image.shape == (h, w - n, c)
    assert res.image.dtype == img.dtype
    # Exactly one pixel per row per seam, each seam index used once per row.
    assert np.all((res.order >= 0).sum(axis=1) == n)
    for k in range(n):
        assert np.all((res.order == k).sum(axis=1) == 1)
    # Surviving pixels are exactly the unmarked ones, in order.
    np.testing.assert_array_equal(res.image, img[res.order < 0].reshape(h, w - n, c))


@given(image_and_n(max_w=16))
def test_insert_then_width(data):
    img, n = data
    out = insert_seams(img, n)
    assert out.shape == (img.shape[0], img.shape[1] + n, img.shape[2])
    # Removing n seams from the widened image gets back to the original width.
    assert carve_width(out, n).image.shape == img.shape


def test_insert_duplicates_low_energy_region():
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    img[:, 6:] = 255  # a single strong vertical edge
    out = insert_seams(img, 3)
    # The edge must survive intact: new columns land in flat areas.
    g = to_gray(out)
    edges = np.abs(np.diff(g, axis=1)) > 1
    assert np.all(edges.sum(axis=1) == 1)


def free_path_exists(protected: np.ndarray) -> bool:
    """Is there an 8-connected top-to-bottom path avoiding every protected pixel?"""
    reach = ~protected[0]
    for row in protected[1:]:
        spread = reach.copy()
        spread[1:] |= reach[:-1]
        spread[:-1] |= reach[1:]
        reach = spread & ~row
    return bool(reach.any())


@st.composite
def protect_case(draw):
    """Arbitrary random protect mask (any density, no structural guarantee)."""
    h = draw(st.integers(2, 20))
    w = draw(st.integers(3, 20))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    mask = rng.random((h, w)) < draw(st.floats(0.0, 0.9))
    n = draw(st.integers(1, w - 1))
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return img, mask, n


@given(protect_case(), st.sampled_from(["forward", "backward"]))
def test_protect_mask_avoided_whenever_possible(case, mode):
    """Replay the carve: each seam must avoid protected pixels iff a free path exists.

    ("No protected pixel removed while n <= free columns" is false in general:
    a seam may consume the bridge pixels another free path needed.)
    """
    img, mask, n = case
    res = carve_width(img, n, mode=mode, protect=mask)
    m = mask.copy()
    rows = np.arange(m.shape[0])
    for seam in res.seams:
        if free_path_exists(m):
            assert not m[rows, seam].any()
        m = remove_seam(m, seam)


def test_protect_mask_crossed_only_when_unavoidable():
    mask = np.ones((5, 6), dtype=bool)
    mask[:, 2] = False  # one free column
    img = np.random.default_rng(0).integers(0, 256, (5, 6, 3), dtype=np.uint8)
    res = carve_width(img, 2, protect=mask)
    assert np.all(res.seams[0] == 2)  # free path taken first
    assert res.image.shape == (5, 4, 3)  # second seam has to cross: still valid


@given(st.integers(2, 20), st.integers(4, 20), st.integers(0, 2**32 - 1), st.data())
def test_remove_mask_goes_first(h, w, seed, data):
    rng = np.random.default_rng(seed)
    k = data.draw(st.integers(1, w - 2))
    cols = rng.choice(w, k, replace=False)
    remove = np.zeros((h, w), dtype=bool)
    remove[:, cols] = True
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    res = carve_width(img, k, remove=remove)
    np.testing.assert_array_equal(res.order >= 0, remove)


def test_mask_shape_validation():
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        carve_width(img, 1, protect=np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError):
        carve_width(img, 5)


def _sines(h: int = 64, w: int = 96) -> np.ndarray:
    """Smooth periodic texture: no flat region a seam can hide in."""
    y, x = np.mgrid[0:h, 0:w]
    return (128 + 60 * np.sin(x / 7.0) + 40 * np.cos(y / 5.0)).astype(np.uint8)


def _fixed_noise(h: int = 64, w: int = 96) -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 255, (h, w)).astype(np.uint8)


@pytest.mark.parametrize("make", [_sines, _fixed_noise])
def test_forward_energy_creates_fewer_edges(make):
    img = make()
    n = 24
    fwd = carve_width(img, n, mode="forward")
    bwd = carve_width(img, n, mode="backward")
    # Fixed images where the two genuinely differ (on flat/structured synthetic
    # images both can find zero-cost seams). Measured ratios: ~0.82 and ~0.57.
    assert fwd.created_energy < 0.9 * bwd.created_energy


def _smooth_scene(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    img = np.stack(
        [
            128 + 100 * np.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.02),
            128 + 100 * np.exp(-((x - 0.75) ** 2 + (y - 0.4) ** 2) / 0.01),
            80 + 60 * np.sin(6 * x) * np.cos(4 * y),
        ],
        axis=-1,
    )
    return np.clip(img, 0, 255).astype(np.uint8)


def test_order_map_upsampled_matches_direct_carve():
    big = _smooth_scene(96, 160)
    small = big.reshape(48, 2, 80, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
    n_small = 16
    low = carve_width(small, n_small)
    via_map = apply_removal_order(big, low.order, n_small)
    direct = carve_width(big, 2 * n_small).image
    assert via_map.shape == direct.shape == (96, 128, 3)
    assert ssim(to_gray(via_map), to_gray(direct)) > 0.9


@given(st.integers(4, 40), st.integers(5, 40), st.integers(2, 120), st.integers(0, 2**32 - 1))
def test_order_map_any_scale_keeps_rows_equal(h, w, big_w, seed):
    assume(big_w > 1)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    n = int(rng.integers(1, w))
    res = carve_width(img, n)
    big = rng.integers(0, 256, (2 * h + 1, big_w, 3), dtype=np.uint8)
    target = int(rng.integers(1, big_w + 1))
    out = apply_removal_order(big, res.order, n, target_width=target)
    assert out.shape == (2 * h + 1, target, 3)
