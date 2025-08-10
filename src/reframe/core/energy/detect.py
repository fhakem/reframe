"""Face / person detection without downloaded weights, plus box utilities.

Only detectors shipped *inside* OpenCV are used: Haar cascades (frontal and
profile faces, upper body) and the HOG + linear-SVM people detector. They are
weak on cartoons, small or occluded subjects, but cost nothing to install.

Any learned detector (YOLO, RetinaFace, SAM-prompted boxes...) plugs in by
implementing :class:`BoxDetector`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import cv2
import numpy as np

from reframe.core.contracts import Box, BoxLabel
from reframe.core.energy.saliency import to_gray_u8
from reframe.core.types import BoolArray, FloatArray, ImageArray


class BoxDetector(Protocol):
    """Detect boxes on an RGB uint8 frame (analysis resolution, source coords)."""

    def detect(self, frame: ImageArray) -> list[Box]: ...


# --------------------------------------------------------------------------- utils


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.width * (a.y1 - a.y0) + b.width * (b.y1 - b.y0) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: Sequence[Box], iou_thresh: float = 0.4) -> list[Box]:
    """Greedy non-max suppression, per label, highest score first."""
    keep: list[Box] = []
    for b in sorted(boxes, key=lambda x: -x.score):
        if all(k.label != b.label or iou(k, b) <= iou_thresh for k in keep):
            keep.append(b)
    return keep


def boxes_to_mask(
    boxes: Iterable[Box],
    shape: tuple[int, int],
    *,
    margin: float = 0.1,
    margin_px: float = 2.0,
) -> BoolArray:
    """Rasterise boxes into a bool mask, each grown by ``margin`` x its size + ``margin_px``."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        mx = margin * b.width + margin_px
        my = margin * (b.y1 - b.y0) + margin_px
        x0 = max(0, int(np.floor(b.x0 - mx)))
        y0 = max(0, int(np.floor(b.y0 - my)))
        x1 = min(w, int(np.ceil(b.x1 + mx)))
        y1 = min(h, int(np.ceil(b.y1 + my)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def boxes_to_soft(
    boxes: Iterable[Box], shape: tuple[int, int], *, margin: float = 0.25
) -> FloatArray:
    """Soft [0, 1] map: each box (grown by ``margin``) filled with its score, feathered."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    sizes: list[float] = []
    for b in boxes:
        mx, my = margin * b.width, margin * (b.y1 - b.y0)
        x0, y0 = max(0, int(b.x0 - mx)), max(0, int(b.y0 - my))
        x1, y1 = min(w, int(np.ceil(b.x1 + mx))), min(h, int(np.ceil(b.y1 + my)))
        if x1 > x0 and y1 > y0:
            region = out[y0:y1, x0:x1]
            np.maximum(region, np.float32(min(max(b.score, 0.0), 1.0)), out=region)
            sizes.append(min(b.width, b.y1 - b.y0))
    if sizes:
        sigma = max(1.0, 0.15 * float(np.median(sizes)))
        return np.asarray(cv2.GaussianBlur(out, (0, 0), sigma), dtype=np.float64)
    return out.astype(np.float64)


def _rects_to_boxes(
    rects: Sequence[Sequence[int]] | np.ndarray,
    scores: Sequence[float] | np.ndarray | None,
    label: BoxLabel,
    scale: float,
    flip_w: int | None = None,
) -> list[Box]:
    out: list[Box] = []
    for i, r in enumerate(np.asarray(rects, dtype=np.float64).reshape(-1, 4)):
        x, y, rw, rh = (float(v) for v in r)
        if flip_w is not None:
            x = flip_w - x - rw
        s = 1.0 if scores is None else float(np.asarray(scores).reshape(-1)[i])
        out.append(Box(x * scale, y * scale, (x + rw) * scale, (y + rh) * scale, s, label))
    return out


# ------------------------------------------------------------------- detectors


def _cascade(name: str) -> cv2.CascadeClassifier:
    c = cv2.CascadeClassifier(cv2.data.haarcascades + name)  # type: ignore[attr-defined]
    if c.empty():
        raise RuntimeError(f"cannot load Haar cascade {name}")
    return c


def _score(levels: np.ndarray | Sequence[float], n: int) -> list[float]:
    """Map cascade level weights (unbounded, ~0..10) to (0, 1)."""
    lw = np.asarray(levels, dtype=np.float64).reshape(-1)
    if lw.size != n:
        return [0.8] * n
    return [float(v) for v in 1.0 / (1.0 + np.exp(-0.5 * lw))]


class HaarFaceDetector:
    """Frontal + profile (both directions) Haar faces, label ``"face"``."""

    def __init__(
        self,
        *,
        work_width: int = 480,
        min_size_frac: float = 0.06,
        min_neighbors: int = 5,
        scale_factor: float = 1.2,
    ) -> None:
        self.work_width = work_width
        self.min_size_frac = min_size_frac
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        self._frontal = _cascade("haarcascade_frontalface_default.xml")
        self._profile = _cascade("haarcascade_profileface.xml")

    def _run(
        self, c: cv2.CascadeClassifier, gray: np.ndarray, flip: bool, scale: float
    ) -> list[Box]:
        g = cv2.flip(gray, 1) if flip else gray
        ms = max(12, int(self.min_size_frac * min(g.shape)))
        rects, _, levels = c.detectMultiScale3(
            g,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(ms, ms),
            outputRejectLevels=True,
        )
        n = len(rects)
        if n == 0:
            return []
        flip_w = g.shape[1] if flip else None
        return _rects_to_boxes(rects, _score(levels, n), "face", scale, flip_w)

    def detect(self, frame: ImageArray) -> list[Box]:
        gray = to_gray_u8(frame)
        h, w = gray.shape
        s = min(1.0, self.work_width / w)
        if s < 1.0:
            gray = cv2.resize(
                gray, (self.work_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA
            )
        gray = cv2.equalizeHist(gray)
        boxes = self._run(self._frontal, gray, False, 1.0 / s)
        boxes += self._run(self._profile, gray, False, 1.0 / s)
        boxes += self._run(self._profile, gray, True, 1.0 / s)
        return nms(boxes, 0.3)


class PersonDetector:
    """HOG people detector (+ optional upper-body cascade), label ``"person"``.

    HOG runs on a frame downscaled to ``hog_width`` columns: its 64x128 window
    then matches people of roughly a quarter of the frame height and taller.
    """

    def __init__(
        self,
        *,
        hog_width: int = 480,
        hog_threshold: float = 0.5,
        use_upper_body: bool = True,
        min_neighbors: int = 4,
    ) -> None:
        self.hog_width = hog_width
        self.hog_threshold = hog_threshold
        self.min_neighbors = min_neighbors
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(
            np.asarray(cv2.HOGDescriptor.getDefaultPeopleDetector(), dtype=np.float32)
        )
        self._upper = _cascade("haarcascade_upperbody.xml") if use_upper_body else None

    def detect(self, frame: ImageArray) -> list[Box]:
        gray = to_gray_u8(frame)
        h, w = gray.shape
        boxes: list[Box] = []
        s = min(1.0, self.hog_width / w)
        small = gray if s == 1.0 else cv2.resize(gray, (self.hog_width, max(1, round(h * s))))
        if small.shape[0] >= 128 and small.shape[1] >= 64:
            rects, weights = self._hog.detectMultiScale(
                small, winStride=(8, 8), padding=(8, 8), scale=1.1
            )
            w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
            for r, wt in zip(np.asarray(rects).reshape(-1, 4), w_arr, strict=False):
                if wt >= self.hog_threshold:
                    sc = float(1.0 / (1.0 + np.exp(-2.0 * (wt - 0.5))))
                    boxes += _rects_to_boxes([r.tolist()], [sc], "person", 1.0 / s)
        if self._upper is not None:
            ms = max(32, int(0.12 * min(h, w)))
            rects, _, levels = self._upper.detectMultiScale3(
                gray,
                scaleFactor=1.15,
                minNeighbors=self.min_neighbors,
                minSize=(ms, ms),
                outputRejectLevels=True,
            )
            if len(rects):
                boxes += _rects_to_boxes(rects, _score(levels, len(rects)), "person", 1.0)
        return nms(boxes, 0.4)


__all__ = [
    "BoxDetector",
    "HaarFaceDetector",
    "PersonDetector",
    "boxes_to_mask",
    "boxes_to_soft",
    "iou",
    "nms",
]
