from __future__ import annotations

import numpy as np

from reframe.core.contracts import Box, FrameSummary, uniform_profile
from reframe.core.fallback import (
    FallbackConfig,
    blur_pad,
    decide_fallback,
    split_screen,
    subject_pair,
)

W = 960
CROP = 300.0


def _summ(i: int, boxes: list[Box] | None = None, prof: np.ndarray | None = None) -> FrameSummary:
    return FrameSummary(i, boxes or [], uniform_profile(W) if prof is None else prof)


def _face(cx: float, label: str = "face") -> Box:
    return Box(cx - 40, 150, cx + 40, 250, 0.9, label)  # type: ignore[arg-type]


def test_blur_pad_shape_and_content() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :960] = (255, 0, 0)
    frame[:, 960:] = (0, 0, 255)
    out = blur_pad(frame, 1080, 1920)
    assert out.shape == (1920, 1080, 3) and out.dtype == np.uint8
    mid = 1920 // 2
    # foreground fits width-wise in the middle band, unblurred colours present
    assert tuple(out[mid, 10]) == (255, 0, 0)
    assert tuple(out[mid, 1070]) == (0, 0, 255)
    # background is darker than the foreground
    assert out[50].max() < 200


def test_split_screen_shape_and_halves() -> None:
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    frame[400:600, 200:400] = (255, 0, 0)  # subject A (left)
    frame[400:600, 1500:1700] = (0, 255, 0)  # subject B (right)
    a = Box(100, 200, 200, 300, 1.0, "face")  # analysis px (scale 2)
    b = Box(750, 200, 850, 300, 1.0, "face")
    out = split_screen(frame, a, b, 1080, 1920, scale=2.0, separator=6)
    assert out.shape == (1920, 1080, 3) and out.dtype == np.uint8
    top, bot = out[:957], out[963:]
    red = (top[..., 0] > 200) & (top[..., 1] < 50)
    green = (bot[..., 1] > 200) & (bot[..., 0] < 50)
    assert red.mean() > 0.05 and green.mean() > 0.05
    assert not ((top[..., 1] > 200) & (top[..., 0] < 50)).any()
    assert np.all(out[957:963] == 0)


def test_decide_none_when_empty() -> None:
    assert decide_fallback([], W, CROP) == "none"
    assert decide_fallback([_summ(i) for i in range(10)], W, CROP) == "none"


def test_decide_split_for_two_far_subjects() -> None:
    shot = [_summ(i, [_face(150), _face(800, "person")]) for i in range(20)]
    assert decide_fallback(shot, W, CROP) == "split"


def test_decide_no_split_when_close_or_transient() -> None:
    close = [_summ(i, [_face(400), _face(550)]) for i in range(20)]
    assert decide_fallback(close, W, CROP) == "none"
    transient = [_summ(i, [_face(150), _face(800)] if i < 5 else [_face(150)]) for i in range(20)]
    assert decide_fallback(transient, W, CROP) == "none"


def test_decide_blur_pad_for_spread_saliency() -> None:
    prof = np.full(W, 1e-4)
    prof[0:120] = 1.0
    prof[420:540] = 1.0
    prof[840:960] = 1.0
    prof /= prof.sum()
    shot = [_summ(i, prof=prof) for i in range(10)]
    assert decide_fallback(shot, W, CROP) == "blur_pad"
    assert decide_fallback(shot, W, CROP, cfg=FallbackConfig(blur_min_coverage=0.3)) == "none"


def test_decide_none_for_concentrated_saliency() -> None:
    prof = np.full(W, 1e-4)
    prof[300:500] = 1.0
    prof /= prof.sum()
    assert decide_fallback([_summ(i, prof=prof) for i in range(10)], W, CROP) == "none"


def test_no_split_for_face_inside_own_person_box() -> None:
    person = Box(200, 50, 200 + CROP + 150, 500, 0.95, "person")  # close-up, wider than crop
    face = Box(380, 80, 460, 180, 0.9, "face")
    shot = [_summ(i, [person, face]) for i in range(20)]
    assert subject_pair([person, face]) is None
    assert decide_fallback(shot, W, CROP) == "none"


def test_split_pairs_two_people_when_one_face() -> None:
    boxes = [
        _face(150),
        Box(60, 100, 240, 500, 0.9, "person"),
        Box(720, 100, 900, 500, 0.9, "person"),
    ]
    pair = subject_pair(boxes)
    assert pair is not None and pair[0].cx < pair[1].cx and pair[1].cx > 700
    assert decide_fallback([_summ(i, boxes) for i in range(10)], W, CROP) == "split"


def test_moving_compact_blob_is_not_blur_pad() -> None:
    shot = []
    for i in range(40):
        prof = np.full(W, 1e-4)
        x = 20 * i
        prof[x : x + 100] = 1.0
        shot.append(_summ(i, prof=prof / prof.sum()))
    assert decide_fallback(shot, W, CROP) == "none"


def test_face_with_diffuse_saliency_is_none() -> None:
    prof = np.full(W, 1e-4)
    prof[0:120] = 1.0
    prof[840:960] = 1.0
    prof /= prof.sum()
    shot = [_summ(i, [_face(480)], prof=prof) for i in range(10)]
    assert decide_fallback(shot, W, CROP) == "none"
