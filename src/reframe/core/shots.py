"""Streaming shot (hard cut) detection.

Similar in spirit to PySceneDetect's ``ContentDetector`` / ``AdaptiveDetector``
(not a dependency: it conflicts with our OpenCV wheel). Each frame is reduced
to a small HSV thumbnail and compared with the previous one:

* ``content`` = weighted mean |Δ| of the H, S, V channels (0..255 scale, H wraps);
* ``hist``    = 1 - correlation of normalised 2-D H-S histograms (0..2, ~0..1).

The frame score is ``content + hist_weight * hist``. A frame starts a new shot
when the score is above the absolute ``threshold``, *or* above ``min_score``
and ``adaptive_ratio`` times the mean score of the last ``window`` frames, and
at least ``min_shot_len`` frames have passed since the previous cut. Hue and
saturation are weighted above value so that global brightness ramps (fades,
exposure changes) do not fire; hue differences are gated by saturation and
saturation differences by value, since both are noise near grey / black.
"""

from __future__ import annotations

import itertools
from collections import deque

import cv2
import numpy as np
from numpy.typing import NDArray

from reframe.core.contracts import Shot
from reframe.core.types import ImageArray


def shots_from_cuts(cut_indices: list[int] | NDArray[np.int64], n_frames: int) -> list[Shot]:
    """Half-open shots covering ``[0, n_frames)``; each cut index starts a new shot.

    Unsorted / duplicate cuts, a cut at 0 and cuts outside ``(0, n_frames)`` are tolerated.
    """
    if n_frames <= 0:
        return []
    bounds = sorted({int(c) for c in cut_indices if 0 < int(c) < n_frames})
    edges = [0, *bounds, n_frames]
    return [Shot(a, b) for a, b in itertools.pairwise(edges)]


class HistogramShotDetector:
    """Implements :class:`reframe.core.contracts.ShotDetector` (streaming ``push``)."""

    def __init__(
        self,
        *,
        threshold: float = 30.0,
        min_score: float = 12.0,
        adaptive_ratio: float = 3.0,
        window: int = 8,
        min_shot_len: int = 8,
        weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
        hist_weight: float = 30.0,
        thumb_width: int = 128,
    ) -> None:
        self.threshold = threshold
        self.min_score = min_score
        self.adaptive_ratio = adaptive_ratio
        self.min_shot_len = min_shot_len
        self.weights = np.asarray(weights, dtype=np.float64) / float(sum(weights))
        self.hist_weight = hist_weight
        self.thumb_width = thumb_width
        self._scores: deque[float] = deque(maxlen=window)
        self._prev_hsv: NDArray[np.float64] | None = None
        self._prev_hist: NDArray[np.float32] | None = None
        self._since_cut = 0
        self.last_score = 0.0

    def reset(self) -> None:
        self._scores.clear()
        self._prev_hsv = None
        self._prev_hist = None
        self._since_cut = 0
        self.last_score = 0.0

    def _features(self, frame: ImageArray) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        h, w = arr.shape[:2]
        if w > self.thumb_width:
            tw = self.thumb_width
            th = max(1, round(h * tw / w))
            arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(np.ascontiguousarray(arr[..., :3]), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        return hsv.astype(np.float64), np.asarray(hist, dtype=np.float32)

    def _score(self, hsv: NDArray[np.float64], hist: NDArray[np.float32]) -> float:
        assert self._prev_hsv is not None and self._prev_hist is not None
        prev = self._prev_hsv
        d = np.abs(hsv - prev)
        dh = np.minimum(d[..., 0], 180.0 - d[..., 0]) * (255.0 / 90.0)  # wrap, rescale
        # Hue is meaningless for unsaturated pixels and saturation for dark ones
        # (quantisation noise near black would fire on fades): gate them.
        dh *= np.minimum(hsv[..., 1], prev[..., 1]) / 255.0
        ds = d[..., 1] * np.minimum(np.minimum(hsv[..., 2], prev[..., 2]) / 64.0, 1.0)
        means = np.array([dh.mean(), ds.mean(), d[..., 2].mean()])
        content = float(self.weights @ means)
        corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
        # H-S histograms of (near-)black frames are noise: fade them out too.
        dark = min(float(min(hsv[..., 2].mean(), prev[..., 2].mean())) / 64.0, 1.0)
        return content + self.hist_weight * (1.0 - corr) * dark

    def push(self, frame: ImageArray) -> bool:
        """Return True if ``frame`` starts a new shot (the first frame returns False)."""
        hsv, hist = self._features(frame)
        if self._prev_hsv is None or self._prev_hsv.shape != hsv.shape:
            self._prev_hsv, self._prev_hist = hsv, hist
            self._since_cut = 1
            return False
        score = self._score(hsv, hist)
        self.last_score = score
        self._prev_hsv, self._prev_hist = hsv, hist
        self._since_cut += 1
        baseline = float(np.mean(self._scores)) if self._scores else 0.0
        is_cut = score >= self.threshold or (
            score >= self.min_score and score >= self.adaptive_ratio * baseline
        )
        if is_cut and self._since_cut > self.min_shot_len:
            self._since_cut = 1
            self._scores.clear()
            return True
        self._scores.append(score)
        return False
