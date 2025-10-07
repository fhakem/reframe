"""Straight-structure map in [0, 1].

A seam crossing a straight line (horizon, building edge, table, text baseline)
leaves a visible kink. We detect segments with OpenCV's Fast Line Detector
(``ximgproc``) or LSD, rasterise them with an intensity that grows with segment
length (long lines are the ones whose breaks are noticeable), and dilate
slightly so a seam cannot slip between the line and its anti-aliased border.

A learned detector (DeepLSD, M-LSD...) can replace it via :class:`SegmentDetector`.
"""

from __future__ import annotations

from typing import Literal, Protocol

import cv2
import numpy as np

from reframe.core.energy.saliency import to_gray_u8
from reframe.core.types import FloatArray, ImageArray


class SegmentDetector(Protocol):
    """Return segments as an (n, 4) float array of ``x0, y0, x1, y1`` on a gray uint8 image."""

    def __call__(self, gray: np.ndarray) -> np.ndarray: ...


def _fld() -> SegmentDetector:
    det = cv2.ximgproc.createFastLineDetector(
        length_threshold=15, distance_threshold=1.41, canny_th1=50, canny_th2=50, do_merge=False
    )

    def run(gray: np.ndarray) -> np.ndarray:
        segs = det.detect(gray)
        return np.zeros((0, 4)) if segs is None else np.asarray(segs, np.float64).reshape(-1, 4)

    return run


def _lsd() -> SegmentDetector:
    det = cv2.createLineSegmentDetector()

    def run(gray: np.ndarray) -> np.ndarray:
        segs = det.detect(gray)[0]
        return np.zeros((0, 4)) if segs is None else np.asarray(segs, np.float64).reshape(-1, 4)

    return run


class LineMap:
    """Rasterised, length-weighted straight-segment map in [0, 1]."""

    def __init__(
        self,
        *,
        method: Literal["fld", "lsd"] = "fld",
        min_len_frac: float = 0.04,
        full_len_frac: float = 0.25,
        thickness: int = 3,
        blur_sigma: float = 1.5,
        detector: SegmentDetector | None = None,
    ) -> None:
        self.min_len_frac = min_len_frac
        self.full_len_frac = full_len_frac
        self.thickness = thickness
        self.blur_sigma = blur_sigma
        self._det = detector if detector is not None else (_fld() if method == "fld" else _lsd())

    def segments(self, frame: ImageArray) -> np.ndarray:
        gray = to_gray_u8(frame)
        if gray.size == 0 or int(gray.max()) == int(gray.min()):
            return np.zeros((0, 4))
        return self._det(gray)

    def __call__(self, frame: ImageArray) -> FloatArray:
        h, w = np.asarray(frame).shape[:2]
        out = np.zeros((h, w), dtype=np.float32)
        segs = self.segments(frame)
        if len(segs):
            diag = float(np.hypot(h, w))
            lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
            lo, hi = self.min_len_frac * diag, self.full_len_frac * diag
            strength = np.clip((lengths - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            # Paint weak segments first so strong ones win where they overlap.
            keep = np.flatnonzero(strength > 0)
            for i in keep[np.argsort(strength[keep])]:
                x0, y0, x1, y1 = (round(float(v)) for v in segs[i])
                v = 0.3 + 0.7 * float(strength[i])
                cv2.line(out, (x0, y0), (x1, y1), v, self.thickness, cv2.LINE_8)
            if self.blur_sigma > 0:
                peak = out.copy()
                soft = np.asarray(cv2.GaussianBlur(out, (0, 0), self.blur_sigma), np.float32)
                out = np.maximum(soft, peak)
        return np.clip(out.astype(np.float64), 0.0, 1.0)


__all__ = ["LineMap", "SegmentDetector"]
