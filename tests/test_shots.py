"""Streaming shot detection on synthetic sequences."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from reframe.core.contracts import Shot, ShotDetector
from reframe.core.shots import HistogramShotDetector, shots_from_cuts

H, W = 90, 160


def _scene_a(t: int) -> np.ndarray:
    """Warm gradient with a moving bright disc."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([180 + 0 * x, 60 + 0.5 * x, 40 + 0.3 * y], axis=-1).astype(np.float64)
    cx = 30 + 2 * t
    img[(x - cx) ** 2 + (y - 45) ** 2 < 15**2] = (250, 230, 120)
    return np.clip(img, 0, 255).astype(np.uint8)


def _scene_b(t: int) -> np.ndarray:
    """Cool, slowly panning texture."""
    rng = np.random.default_rng(7)
    tex = cv2.GaussianBlur(rng.normal(size=(H, W + 200)), (0, 0), 3.0)
    tex = (tex - tex.min()) / np.ptp(tex)
    win = tex[:, t : t + W]
    img = np.stack([30 + 40 * win, 90 + 60 * win, 160 + 80 * win], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _run(frames: list[np.ndarray], det: ShotDetector | None = None) -> list[int]:
    det = det or HistogramShotDetector()
    return [i for i, f in enumerate(frames) if det.push(f)]


def test_shots_from_cuts():
    assert shots_from_cuts([30], 60) == [Shot(0, 30), Shot(30, 60)]
    assert shots_from_cuts([], 5) == [Shot(0, 5)]
    assert shots_from_cuts([40, 0, 10, 10, 60, 99, -3], 60) == [
        Shot(0, 10),
        Shot(10, 40),
        Shot(40, 60),
    ]
    assert shots_from_cuts(np.array([2]), 0) == []
    shots = shots_from_cuts([3, 7], 9)
    assert sum(len(s) for s in shots) == 9


def test_hard_cut_detected_exactly_once():
    frames = [_scene_a(t) for t in range(30)] + [_scene_b(t) for t in range(30)]
    assert _run(frames) == [30]
    frames = [_scene_b(t) for t in range(25)] + [_scene_a(t) for t in range(25)]
    assert _run(frames) == [25]


def test_first_frame_is_not_a_cut_and_reset():
    det = HistogramShotDetector()
    assert det.push(_scene_a(0)) is False
    det.reset()
    assert det.push(_scene_b(0)) is False  # first frame after reset


@pytest.mark.parametrize("scene", [_scene_a, _scene_b])
def test_gradual_brightness_change_does_not_cut(scene):
    frames = []
    for t in range(60):
        gain = 0.4 + 0.02 * t  # 0.4 -> 1.6 over 60 frames (fade in + over-exposure)
        frames.append(np.clip(scene(t).astype(np.float64) * gain, 0, 255).astype(np.uint8))
    assert _run(frames) == []


def test_fade_to_black_does_not_cut():
    frames = [(_scene_a(t).astype(np.float64) * (1 - t / 30)).astype(np.uint8) for t in range(30)]
    assert _run(frames) == []


def test_min_shot_length_suppresses_flash_cuts():
    frames = [_scene_a(t) for t in range(20)]
    frames += [_scene_b(t) for t in range(3)]  # 3-frame insert
    frames += [_scene_a(t) for t in range(20, 40)]
    cuts = _run(frames, HistogramShotDetector(min_shot_len=5))
    assert cuts == [20]
    assert _run(frames, HistogramShotDetector(min_shot_len=2)) == [20, 23]


def test_grayscale_and_float_input():
    a = [_scene_a(t).mean(axis=2) for t in range(15)]
    b = [_scene_b(t).mean(axis=2) * 0.3 for t in range(15)]
    assert _run([*a, *b]) == [15]
