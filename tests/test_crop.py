from __future__ import annotations

import numpy as np
import pytest

from reframe.core.contracts import Box, FrameSummary, uniform_profile
from reframe.core.crop.budget import (
    BudgetConfig,
    choose_seam_count,
    max_seams_for,
    seam_budget_for_shot,
)
from reframe.core.crop.trajectory import (
    TrajectoryConfig,
    enforce_limits,
    plan_crop_path,
    smooth_path,
)

W = 960
CROP = 400.0


def _box(cx: float, w: float = 80.0, label: str = "face", score: float = 1.0) -> Box:
    return Box(cx - w / 2, 200, cx + w / 2, 300, score, label)  # type: ignore[arg-type]


def _summ(i: int, boxes: list[Box] | None = None, prof: np.ndarray | None = None) -> FrameSummary:
    return FrameSummary(i, boxes or [], uniform_profile(W) if prof is None else prof)


def _shot(centres: np.ndarray) -> list[FrameSummary]:
    return [_summ(i, [_box(float(c))]) for i, c in enumerate(centres)]


def _check_limits(x: np.ndarray, cfg: TrajectoryConfig, hi: float) -> None:
    assert x.min() >= 0.0 and x.max() <= hi + 1e-9
    assert np.abs(np.diff(x)).max() <= cfg.max_velocity + 1e-6
    assert np.abs(np.diff(x, 2)).max() <= cfg.max_acceleration + 1e-6


def test_empty_summaries_centred() -> None:
    path = plan_crop_path([_summ(i) for i in range(30)], W, CROP)
    assert path.mode == "stationary"
    np.testing.assert_allclose(path.x, (W - CROP) / 2)
    assert plan_crop_path([], W, CROP).x.shape == (0,)


def test_static_subject_stationary() -> None:
    path = plan_crop_path(_shot(np.full(60, 300.0)), W, CROP)
    assert path.mode == "stationary"
    assert np.ptp(path.x) == 0.0
    assert path.x[0] <= 300 - 40 and path.x[0] + CROP >= 300 + 40  # subject inside


def test_constant_velocity_pan() -> None:
    centres = 250.0 + 3.0 * np.arange(120)
    path = plan_crop_path(_shot(centres), W, CROP)
    assert path.mode == "pan"
    assert np.abs(path.x + CROP / 2 - centres)[10:-10].max() < 15.0
    _check_limits(path.x, TrajectoryConfig(), W - CROP)


def test_pan_parking_at_edge_keeps_limits() -> None:
    cfg = TrajectoryConfig()
    centres = np.minimum(400.0 + 4.0 * np.arange(150), W - 60.0)  # parks for the last ~30
    path = plan_crop_path(_shot(centres), W, CROP, cfg=cfg)
    _check_limits(path.x, cfg, W - CROP)
    assert path.x[-1] == pytest.approx(W - CROP, abs=2.0)


def test_enforce_limits_randomised() -> None:
    rng = np.random.default_rng(3)
    lo, hi, v_max, a_max = 0.0, 560.0, 6.0, 0.5
    for k in range(300):
        n = int(rng.integers(5, 200))
        t = np.arange(n, dtype=np.float64)
        if k % 3 == 0:
            y = rng.uniform(-200, 800) + rng.uniform(-8, 8) * t
        elif k % 3 == 1:
            y = np.where(t < rng.integers(1, n), rng.uniform(-100, 700), rng.uniform(-100, 700))
        else:
            y = np.cumsum(rng.normal(0, 5, n)) + rng.uniform(0, 560)
        x = enforce_limits(y, lo, hi, v_max, a_max)
        assert x.min() >= lo and x.max() <= hi
        assert np.abs(np.diff(x)).max() <= v_max + 1e-9
        if n > 2:
            assert np.abs(np.diff(x, 2)).max() <= a_max + 1e-9


def test_follows_moving_box_and_limits() -> None:
    cfg = TrajectoryConfig()
    t = np.arange(300)
    centres = W / 2 + 200 * np.sin(2 * np.pi * t / 200)
    path = plan_crop_path(_shot(centres), W, CROP, cfg=cfg)
    assert path.mode == "track"
    _check_limits(path.x, cfg, W - CROP)
    # subject box always inside the window
    assert np.all(path.x <= centres - 40) and np.all(path.x + CROP >= centres + 40)


def test_jump_respects_velocity_and_acceleration() -> None:
    cfg = TrajectoryConfig()
    centres = np.where(np.arange(200) < 100, 220.0, 760.0)
    path = plan_crop_path(_shot(centres), W, CROP, cfg=cfg)
    _check_limits(path.x, cfg, W - CROP)
    assert path.x[-1] + CROP / 2 > 700 and path.x[0] + CROP / 2 < 300


def test_bounds_clamped_for_edge_subject() -> None:
    cfg = TrajectoryConfig()
    centres = np.linspace(20, 940, 250)
    path = plan_crop_path(_shot(centres), W, CROP, cfg=cfg)
    _check_limits(path.x, cfg, W - CROP)
    assert path.x[0] == pytest.approx(0.0, abs=1.0)
    assert path.x[-1] == pytest.approx(W - CROP, abs=1.0)


def test_smooth_on_noisy_targets() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(300)
    clean = 300 + 1.2 * t
    noisy = clean + rng.normal(0, 25, t.size)
    path = plan_crop_path(_shot(noisy), W, CROP)
    raw_d2 = np.abs(np.diff(noisy, 2)).mean()
    assert np.abs(np.diff(path.x, 2)).mean() < raw_d2 / 50
    assert np.abs(path.x + CROP / 2 - clean)[20:-20].max() < 30


def test_saliency_only_target() -> None:
    prof = np.full(W, 1e-4)
    prof[700:800] = 1.0
    prof /= prof.sum()
    path = plan_crop_path([_summ(i, prof=prof) for i in range(40)], W, CROP)
    assert path.mode == "stationary"
    assert path.x[0] <= 700 and path.x[0] + CROP >= 800


def test_face_beats_salient() -> None:
    boxes = [_box(150, label="face", score=0.6), _box(800, w=300, label="salient")]
    path = plan_crop_path([_summ(i, boxes) for i in range(30)], W, CROP)
    assert path.x[0] <= 110 and path.x[0] + CROP >= 190


def test_crop_wider_than_frame() -> None:
    path = plan_crop_path(_shot(np.full(5, 300.0)), W, W + 10)
    np.testing.assert_array_equal(path.x, 0.0)


def test_smooth_path_matches_dense_solve() -> None:
    rng = np.random.default_rng(1)
    n = 40
    c = rng.normal(0, 10, n)
    w = rng.uniform(0, 2, n)
    alpha, beta = 3.0, 7.0
    d1 = np.diff(np.eye(n), axis=0)
    d2 = np.diff(np.eye(n), 2, axis=0)
    wf = np.maximum(w, 1e-4)
    a = np.diag(wf) + alpha * d1.T @ d1 + beta * d2.T @ d2
    np.testing.assert_allclose(smooth_path(c, w, alpha, beta), np.linalg.solve(a, wf * c))


def test_enforce_limits_identity_when_feasible() -> None:
    y = 100 + 2.0 * np.arange(50, dtype=np.float64)
    np.testing.assert_allclose(enforce_limits(y, 0, 500, 6, 0.5), y)


# ---------------------------------------------------------------- budget


def test_seam_count_knee() -> None:
    curve = [10.0] * 40 + [10.0 * 1.5**k for k in range(1, 40)]
    noisy = [c * 1.05 for c in curve]
    assert choose_seam_count([curve, noisy, curve], 100) == pytest.approx(42, abs=2)


def test_seam_count_rises_immediately() -> None:
    # Steady geometric rise from the first seam: no plateau -> 0.
    assert choose_seam_count([[500.0 * 1.3**k for k in range(50)]], 50) == 0
    # One cheap seam then a jump: still no plateau -> 0.
    assert choose_seam_count([[10.0] + [5000.0] * 49], 50) == 0
    # Expensive from the first seam in absolute terms (letters everywhere).
    assert choose_seam_count([[6000.0] * 50], 50, rows=100) == 0
    assert choose_seam_count([[6000.0] * 50], 50) == 50  # no rows: no absolute cap


def test_seam_count_flat_returns_max() -> None:
    assert choose_seam_count([[5.0] * 200, [6.0] * 200], 120) == 120
    assert choose_seam_count([[5.0] * 80], 120) == 80


def test_seam_count_robust_to_one_bad_keyframe() -> None:
    flat = [10.0] * 100
    bad = [10.0] * 5 + [1000.0] * 95
    assert choose_seam_count([flat, flat, bad], 100) == 100


def test_seam_count_zero_cost_background() -> None:
    # Pure white then letters: tiny absolute wiggles must not stop the count.
    curve = [0.0, 0.0, 1.0, 0.5, 2.0] * 6 + [5000.0] * 10
    assert choose_seam_count([curve], 40, rows=100) == 30
    assert choose_seam_count([], 40) == 0
    assert choose_seam_count([curve], 0) == 0


def test_seam_count_cfg() -> None:
    curve = [10.0] * 10 + [25.0] * 10
    assert choose_seam_count([curve], 20, cfg=BudgetConfig(rise_ratio=2.0)) == 10
    assert choose_seam_count([curve], 20, cfg=BudgetConfig(rise_ratio=3.0)) == 20


def test_seam_budget_for_shot() -> None:
    target = 303.0
    assert max_seams_for(target, 960, 0.25) == 101
    assert seam_budget_for_shot(target, 960, 0.25, 40) == (343.0, 40)
    assert seam_budget_for_shot(target, 960, 0.25, 500) == (404.0, 101)
    assert seam_budget_for_shot(target, 320, 0.25, 500) == (320.0, 17)
    assert seam_budget_for_shot(target, 960, 0.25, -3) == (303.0, 0)
