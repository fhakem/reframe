"""Fallback layouts when a single tracked crop cannot frame the shot (M4).

* ``split``: two important subjects (faces/people) persistently farther apart
  than one crop window can cover -> two stacked crops.
* ``blur_pad``: salient content spread so wide that the best window covers
  too little of it -> the whole frame fit width-wise over a blurred copy.

Boxes and summaries are analysis px (source coords); frames are full
resolution uint8 RGB, so rendering functions take ``scale`` (analysis -> full).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from reframe.core.contracts import Box, FrameSummary
from reframe.core.crop.trajectory import window_masses
from reframe.core.types import ImageArray

FallbackKind = Literal["none", "split", "blur_pad"]


@dataclass(frozen=True)
class FallbackConfig:
    """Fallback decision parameters (analysis px)."""

    subject_tiers: tuple[tuple[str, ...], ...] = (("face", "manual"), ("person",))
    """Important-subject labels for split screen, in priority order."""
    split_min_score: float = 0.5
    split_span_factor: float = 1.0
    """A frame needs split if its two main subjects span more than this x ``crop_w``."""
    split_persist: float = 0.6
    """...in at least this fraction of the shot's frames."""
    blur_min_coverage: float = 0.55
    """blur_pad when the median per-frame best-window saliency coverage is below this."""
    blur_min_peak: float = 1.5
    """A frame's profile is informative if ``max(profile) * w >= this``; flat
    frames and frames with important subjects count as fully covered (the
    tracked crop handles them)."""


def subject_pair(boxes: list[Box], cfg: FallbackConfig | None = None) -> tuple[Box, Box] | None:
    """The two main subjects (left, right) of a frame, or ``None``.

    Boxes are ranked by tier (faces before people), then score x area; the pair
    is the top box and the best-ranked box *horizontally disjoint* from it, so
    a face is never paired with its own body box.
    """
    cfg = cfg or FallbackConfig()
    rank = {lab: i for i, tier in enumerate(cfg.subject_tiers) for lab in tier}
    cand = [b for b in boxes if b.label in rank and b.score >= cfg.split_min_score]
    cand.sort(key=lambda b: (rank[b.label], -b.score * b.width * max(b.y1 - b.y0, 1.0)))
    if not cand:
        return None
    a = cand[0]
    for b in cand[1:]:
        if b.x0 >= a.x1 or b.x1 <= a.x0:
            return (a, b) if a.cx <= b.cx else (b, a)
    return None


def _has_subject(boxes: list[Box], cfg: FallbackConfig) -> bool:
    labels = {lab for tier in cfg.subject_tiers for lab in tier}
    return any(b.label in labels and b.score >= cfg.split_min_score for b in boxes)


def decide_fallback(
    summaries: list[FrameSummary],
    frame_w: float,
    crop_w: float,
    *,
    cfg: FallbackConfig | None = None,
) -> FallbackKind:
    """Choose the shot layout: ``"split"`` > ``"blur_pad"`` > ``"none"``."""
    cfg = cfg or FallbackConfig()
    if not summaries or crop_w >= frame_w:
        return "none"
    n_split = 0
    for s in summaries:
        pair = subject_pair(s.boxes, cfg)
        if pair is not None:
            span = max(pair[0].x1, pair[1].x1) - min(pair[0].x0, pair[1].x0)
            n_split += span > cfg.split_span_factor * crop_w
    if n_split >= cfg.split_persist * len(summaries):
        return "split"

    coverage = np.ones(len(summaries))
    for i, s in enumerate(summaries):
        p = np.asarray(s.column_saliency, dtype=np.float64)
        total = float(p.sum())
        if _has_subject(s.boxes, cfg) or p.size == 0 or total <= 0:
            continue
        p = p / total
        if float(p.max()) * p.size < cfg.blur_min_peak:
            continue
        coverage[i] = float(window_masses(p, crop_w * p.size / frame_w).max())
    return "blur_pad" if float(np.median(coverage)) < cfg.blur_min_coverage else "none"


def _resize(img: ImageArray, w: int, h: int) -> ImageArray:
    interp = cv2.INTER_AREA if w * h < img.shape[0] * img.shape[1] else cv2.INTER_LINEAR
    return np.asarray(cv2.resize(img, (w, h), interpolation=interp))


def blur_pad(
    frame: ImageArray,
    out_w: int,
    out_h: int,
    *,
    blur_frac: float = 0.04,
    darken: float = 0.55,
) -> ImageArray:
    """Fit ``frame`` width-wise into an ``out_w x out_h`` canvas over a blurred,
    darkened, cover-scaled copy of itself.

    ``blur_frac`` is the Gaussian sigma as a fraction of ``out_w``; the blur is
    computed at 1/4 scale for speed. Returns uint8 RGB ``(out_h, out_w, 3)``.
    """
    h, w = frame.shape[:2]
    # Background: cover-scale then centre-crop.
    s = max(out_w / w, out_h / h)
    cw, ch = max(round(w * s), out_w), max(round(h * s), out_h)
    cover = _resize(frame, cw, ch)
    x0, y0 = (cw - out_w) // 2, (ch - out_h) // 2
    bg = cover[y0 : y0 + out_h, x0 : x0 + out_w]
    sw, sh = max(out_w // 4, 1), max(out_h // 4, 1)
    small = _resize(bg, sw, sh)
    sigma = max(blur_frac * out_w / 4.0, 0.5)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = _resize(small, out_w, out_h).astype(np.float32) * darken
    canvas = np.clip(bg, 0, 255).astype(np.uint8)
    # Foreground: fit (width-wise, or height-wise if the source is taller).
    f = min(out_w / w, out_h / h)
    fw, fh = max(min(round(w * f), out_w), 1), max(min(round(h * f), out_h), 1)
    fg = _resize(frame, fw, fh)
    fx, fy = (out_w - fw) // 2, (out_h - fh) // 2
    canvas[fy : fy + fh, fx : fx + fw] = fg
    return canvas


def _subject_crop(
    frame: ImageArray, box: Box, aspect: float, zoom: float, min_h_frac: float
) -> ImageArray:
    """Crop of aspect ``aspect`` (w/h) centred on ``box`` (full-res px), clamped in frame."""
    fh, fw = frame.shape[:2]
    ch = float(np.clip(max((box.y1 - box.y0) * zoom, min_h_frac * fh), 2.0, fh))
    cw = ch * aspect
    if cw > fw:
        cw, ch = float(fw), fw / aspect
    cx, cy = box.cx, (box.y0 + box.y1) / 2.0
    x0 = round(float(np.clip(cx - cw / 2.0, 0, fw - cw)))
    y0 = round(float(np.clip(cy - ch / 2.0, 0, fh - ch)))
    return frame[y0 : y0 + max(int(ch), 1), x0 : x0 + max(int(cw), 1)]


def split_screen(
    frame: ImageArray,
    box_a: Box,
    box_b: Box,
    out_w: int,
    out_h: int,
    *,
    scale: float = 1.0,
    separator: int = 4,
    separator_color: tuple[int, int, int] = (0, 0, 0),
    zoom: float = 2.5,
    min_h_frac: float = 0.4,
) -> ImageArray:
    """Two vertically stacked crops: ``box_a`` on top, ``box_b`` below.

    Boxes are analysis px and mapped to ``frame`` with ``scale``. Each half has
    aspect ``out_w : (out_h - separator) / 2`` and shows its subject at
    ``zoom`` x the box height (at least ``min_h_frac`` of the frame height).
    Returns uint8 RGB ``(out_h, out_w, 3)``.
    """
    sep = max(min(separator, out_h - 2), 0)
    h_top = (out_h - sep) // 2
    h_bot = out_h - sep - h_top
    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[h_top : h_top + sep] = separator_color
    for box, y, hh in ((box_a, 0, h_top), (box_b, h_top + sep, h_bot)):
        crop = _subject_crop(frame, box.scaled(scale), out_w / hh, zoom, min_h_frac)
        canvas[y : y + hh] = _resize(np.ascontiguousarray(crop), out_w, hh)
    return canvas


__all__ = [
    "FallbackConfig",
    "FallbackKind",
    "blur_pad",
    "decide_fallback",
    "split_screen",
    "subject_pair",
]
