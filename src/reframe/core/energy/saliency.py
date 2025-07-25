"""Weight-free visual saliency in [0, 1].

Two classical OpenCV estimators are blended:

* **spectral residual** (Hou & Zhang 2007): very fast, coarse, good at "where
  is the odd thing in this frame";
* **fine-grained** (Montabone & Soto 2010): centre-surround on intensity,
  sharper object outlines.

Both run on a small working copy (``work_width`` columns); the blend is
blurred and robustly normalised (percentiles, not min/max) so a few hot pixels
cannot squash the map, then upsampled once. Flat frames return all zeros,
never NaN.

Deep estimators (U^2-Net, depth-aware saliency...) can replace this by
implementing :class:`SaliencyEstimator`.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from reframe.core.types import FloatArray, ImageArray


class SaliencyEstimator(Protocol):
    """Anything that turns an RGB uint8 frame into an (h, w) map in [0, 1]."""

    def __call__(self, frame: ImageArray) -> FloatArray: ...


def robust_normalize(
    m: FloatArray, lo_pct: float = 2.0, hi_pct: float = 99.5, eps: float = 1e-6
) -> FloatArray:
    """Map ``m`` to [0, 1] using percentiles; constant or non-finite input -> zeros."""
    a = np.nan_to_num(np.asarray(m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = (float(v) for v in np.percentile(a, [lo_pct, hi_pct]))
    if not hi - lo > eps:
        # Percentiles collapsed (tiny salient spot): fall back to min/max, or give up.
        lo, hi = float(a.min()), float(a.max())
        if not hi - lo > eps:
            return np.zeros(a.shape, dtype=np.float64)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def to_gray_u8(frame: ImageArray) -> np.ndarray:
    """RGB (or already gray) uint8 frame -> contiguous gray uint8."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(arr[..., :3], dtype=np.uint8), cv2.COLOR_RGB2GRAY)


class ClassicalSaliency:
    """Spectral residual + fine-grained saliency, blurred and normalised to [0, 1]."""

    def __init__(
        self,
        *,
        spectral_weight: float = 0.5,
        work_width: int = 240,
        blur_frac: float = 0.02,
    ) -> None:
        if not 0.0 <= spectral_weight <= 1.0:
            raise ValueError("spectral_weight must be in [0, 1]")
        self.spectral_weight = spectral_weight
        self.work_width = work_width
        self.blur_frac = blur_frac
        self._sr = cv2.saliency.StaticSaliencySpectralResidual.create()
        self._fg = cv2.saliency.StaticSaliencyFineGrained.create()

    def __call__(self, frame: ImageArray) -> FloatArray:
        gray = to_gray_u8(frame)
        h, w = gray.shape
        if h < 2 or w < 2 or int(gray.max()) == int(gray.min()):
            return np.zeros((h, w), dtype=np.float64)
        sw = min(self.work_width, w)
        sh = max(2, round(h * sw / w))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        acc = np.zeros((sh, sw), dtype=np.float64)
        if self.spectral_weight > 0:
            ok, sr = self._sr.computeSaliency(small)
            if ok:
                sr_small = cv2.resize(np.asarray(sr, np.float32), (sw, sh))
                acc += self.spectral_weight * robust_normalize(sr_small.astype(np.float64))
        if self.spectral_weight < 1:
            ok, fg = self._fg.computeSaliency(small)
            if ok:
                acc += (1.0 - self.spectral_weight) * robust_normalize(
                    np.asarray(fg, dtype=np.float64)
                )
        sigma = max(0.5, self.blur_frac * max(sh, sw))
        blurred = np.asarray(cv2.GaussianBlur(acc, (0, 0), sigma), dtype=np.float64)
        out = cv2.resize(robust_normalize(blurred), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


__all__ = ["ClassicalSaliency", "SaliencyEstimator", "robust_normalize", "to_gray_u8"]
