"""carve_width hook / seam_costs, TemporalSeamCarver and stability metrics."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from reframe.core.energy.gradient import forward_costs, to_gray
from reframe.core.flow import estimate_flow
from reframe.core.metrics import flow_warping_error, seam_jitter, seam_positions
from reframe.core.seams.carve import carve_width
from reframe.core.seams.dp import find_seam
from reframe.core.seams.temporal import TemporalSeamCarver

H, W, N, T = 96, 128, 24, 12


def _scene(h: int = H, w: int = 420, seed: int = 0) -> np.ndarray:
    """Multi-scale smooth texture (no flat region where seams could hide for free)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w))
    for sigma, amp in [(3, 60), (8, 80), (20, 80)]:
        b = cv2.GaussianBlur(rng.normal(size=(h, w)), (0, 0), sigma)
        acc += amp * b / b.std()
    return np.clip(128 + acc / 3, 0, 255)


def _pan_sequence(speed: int, *, seed: int = 1) -> list[np.ndarray]:
    big = _scene()
    rng = np.random.default_rng(seed)
    out = []
    for t in range(T):
        f = big[:, 50 + speed * t : 50 + speed * t + W] + rng.normal(0, 2, (H, W))
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return out


def _assert_valid(res, h: int, w: int, n: int) -> None:
    assert res.image.shape[:2] == (h, w - n)
    assert np.all((res.order >= 0).sum(axis=1) == n)
    for k in range(n):
        assert np.all((res.order == k).sum(axis=1) == 1)


# --- carve_width hook / seam_costs ------------------------------------------------


def test_seam_costs_match_dp_and_zero_hook_is_identity():
    img = _pan_sequence(0)[0]
    plain = carve_width(img, 6)
    assert len(plain.seam_costs) == 6
    _, c0 = find_seam(np.zeros((H, W)), forward_costs(to_gray(img)))
    assert plain.seam_costs[0] == pytest.approx(c0)
    calls: list[tuple[int, tuple[int, ...]]] = []

    def hook(k, cols):
        calls.append((k, cols.shape))
        return np.full(cols.shape, 50.0)  # constant: same seams, excluded from costs

    hooked = carve_width(img, 6, seam_hook=hook)
    np.testing.assert_array_equal(hooked.order, plain.order)
    np.testing.assert_allclose(hooked.seam_costs, plain.seam_costs)
    assert calls == [(k, (H, W - k)) for k in range(6)]
    back = carve_width(img, 3, mode="backward", seam_hook=lambda k, c: np.zeros(c.shape))
    np.testing.assert_array_equal(back.order, carve_width(img, 3, mode="backward").order)


def test_hook_cols_are_input_columns():
    img = _pan_sequence(0)[0]
    targets = [100, 20, 64, 101]  # input-image columns, any order

    def hook(k, cols):
        return np.where(cols == targets[k], 0.0, 1e6)

    res = carve_width(img, len(targets), seam_hook=hook)
    for k, c in enumerate(targets):
        assert np.all(res.order[:, c] == k)


def test_hook_shape_checked():
    img = np.zeros((8, 10), np.uint8)
    with pytest.raises(ValueError):
        carve_width(img, 2, seam_hook=lambda k, c: np.zeros((8, 3)))


# --- metrics -----------------------------------------------------------------------


def test_seam_positions_and_jitter():
    img = _pan_sequence(0)[0]
    res = carve_width(img, 5)
    pos = seam_positions(res.order, crop_x=10.0)
    assert pos.shape == (5, H)
    rows = np.arange(H)
    for k in range(5):
        assert np.all(res.order[rows, (pos[k] - 10).astype(int)] == k)
    a = np.zeros((2, 4))
    b = np.array([[1.0] * 4, [3.0] * 4])
    assert seam_jitter([a, b]) == pytest.approx(2.0)
    assert seam_jitter([b, b[::-1]]) == pytest.approx(2.0)
    assert seam_jitter([b, b[::-1]], match="sorted") == 0.0
    assert seam_jitter([a]) == 0.0
    with pytest.raises(ValueError):
        seam_positions(np.array([[0, -1], [-1, -1]]))  # row 1 misses seam 0


def test_flow_warping_error_translation():
    f = _pan_sequence(2)
    still = flow_warping_error(f[0], f[0])
    moved = flow_warping_error(f[0], f[1])  # explained by flow
    assert still < 0.5
    assert moved < 0.3 * np.abs(f[1].astype(float) - f[0].astype(float)).mean()


# --- TemporalSeamCarver ------------------------------------------------------------


def test_first_frame_and_reset_equal_plain_carve():
    frames = _pan_sequence(1)
    c = TemporalSeamCarver(N)
    r0 = c.carve(frames[0], crop_x=0.0)
    np.testing.assert_array_equal(r0.order, carve_width(frames[0], N).order)
    c.carve(frames[1], crop_x=0.0)
    c.reset()
    assert c.prev_seams_src is None
    r = c.carve(frames[5], crop_x=0.0)
    np.testing.assert_array_equal(r.order, carve_width(frames[5], N).order)


@pytest.mark.parametrize("speed", [0, 1])
def test_temporal_reduces_jitter(speed):
    frames = _pan_sequence(speed)
    indep = [seam_positions(carve_width(f, N).order) for f in frames]
    c = TemporalSeamCarver(N)
    temp, prev = [], None
    for f in frames:
        flow = None if prev is None else estimate_flow(prev, f)
        res = c.carve(f, crop_x=0.0, flow=flow)
        _assert_valid(res, H, W, N)
        temp.append(seam_positions(res.order))
        prev = f
    for match in ("index", "sorted"):
        j_ind, j_tmp = seam_jitter(indep, match=match), seam_jitter(temp, match=match)
        print(f"speed={speed} {match}: per-frame {j_ind:.2f} temporal {j_tmp:.2f}")
        # Measured (speed 0 / 1): index 36.4 -> 0.13 / 37.7 -> 0.99,
        # sorted 4.0 -> 0.08 / 4.7 -> 0.97
        # (with a pan, seams follow the content at ~speed px/frame, as they should).
        assert j_tmp <= 0.5 * j_ind
        assert j_tmp <= speed + 0.5


def test_crop_shift_is_honoured():
    big = _scene().astype(np.uint8)  # static content, moving crop window
    d, w = 3, 160
    border = np.zeros((H, w))
    border[:, :16] = border[:, -16:] = 1e4  # keep seams in the overlap of all windows

    def run(honour: bool) -> list[np.ndarray]:
        c = TemporalSeamCarver(N)
        out = []
        for t in range(6):
            x = 40 + d * t
            res = c.carve(big[:, x : x + w], crop_x=float(x if honour else 40), extra_energy=border)
            _assert_valid(res, H, w, N)
            out.append(seam_positions(res.order, crop_x=x))  # true source columns
        return out

    honoured = seam_jitter(run(True))
    ignored = seam_jitter(run(False))
    # Measured: 0.21 vs 2.36 (seams dragged along with the window, ~d px/frame);
    # independent per-frame carving gives 23.0.
    assert honoured < 0.5
    assert ignored > 4 * honoured


def test_reset_removes_temporal_influence():
    a = _pan_sequence(0, seed=3)[0]
    b = _scene(seed=9)[:, :W].astype(np.uint8)  # unrelated shot
    c = TemporalSeamCarver(N, lam=50.0)
    c.carve(a, crop_x=0.0)
    with_state = c.carve(b, crop_x=0.0)
    c.carve(a, crop_x=0.0)
    c.reset()
    after_reset = c.carve(b, crop_x=0.0)
    plain = carve_width(b, N)
    np.testing.assert_array_equal(after_reset.order, plain.order)
    assert not np.array_equal(with_state.order, plain.order)


def test_appearance_term_and_argument_checks():
    frames = _pan_sequence(1)
    rgb = [np.repeat(f[..., None], 3, axis=2) for f in frames[:3]]
    c = TemporalSeamCarver(N, appearance_weight=300.0, mode="backward")
    for f in rgb:
        _assert_valid(c.carve(f, crop_x=5.0), H, W, N)
    with pytest.raises(ValueError):
        c.carve(rgb[0], crop_x=5.0, flow=np.zeros((H + 1, W, 2), np.float32))
    # Shape change (new geometry) silently restarts the shot.
    small = frames[0][:50]
    np.testing.assert_array_equal(
        c.carve(small, crop_x=0.0).order, carve_width(small, N, mode="backward").order
    )
    with pytest.raises(ValueError):
        TemporalSeamCarver(-1)


def test_flow_with_moving_crop_tracks_content():
    """M4 path: content pans left by s px/frame, crop follows it, flow on full frames."""
    big = _scene()
    s, full_w, w = 2, 300, W
    rng = np.random.default_rng(5)
    full = [
        np.clip(
            big[:, 60 + s * t : 60 + s * t + full_w] + rng.normal(0, 2, (H, full_w)), 0, 255
        ).astype(np.uint8)
        for t in range(8)
    ]

    def run(use_flow: bool) -> tuple[float, float]:
        c = TemporalSeamCarver(N)
        work_pos, src_pos, prev = [], [], None
        for t, f in enumerate(full):
            x = 100 - s * t  # crop window moves with the content
            flow: np.ndarray | None = (
                estimate_flow(prev, f) if use_flow and prev is not None else None
            )
            res = c.carve(f[:, x : x + w], crop_x=float(x), flow=flow)
            _assert_valid(res, H, w, N)
            work_pos.append(seam_positions(res.order))
            src_pos.append(seam_positions(res.order, crop_x=x))
            prev = f
        return seam_jitter(work_pos), seam_jitter(src_pos)

    work_j, src_j = run(True)
    assert work_j < 0.3  # same content in the window -> seams stay put in work columns
    assert src_j == pytest.approx(s, abs=0.3)  # ...and move with the content in source columns
    work_nf, _ = run(False)
    assert work_nf > 3 * work_j  # without flow seams are pinned to stale source columns


def test_zero_flow_equals_no_flow():
    big = _scene().astype(np.uint8)
    zero = np.zeros((H, 300, 2), np.float32)  # full-frame width, source coordinates
    a, b = TemporalSeamCarver(N), TemporalSeamCarver(N)
    for x in (40, 43, 47):
        work = big[:, x : x + W]
        ra = a.carve(work, crop_x=float(x), flow=zero)
        rb = b.carve(work, crop_x=float(x))
        np.testing.assert_array_equal(ra.order, rb.order)
