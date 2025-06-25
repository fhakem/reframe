"""Frame pipeline — **M1 naive baseline**.

Per frame: centre crop to an intermediate width, then per-frame seam carving
down to the 9:16 width, then ffmpeg upscales to the preset size.

This is deliberately *not* the target pipeline: there is no shot detection,
no subject tracking (the crop is centred) and no temporal coherence, so seams
jitter from frame to frame. It exists to exercise I/O + seam code end to end
and to serve as the jitter baseline for M3 (Grundmann temporal term) and M4
(tracked smart crop).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from reframe.config import ReframeConfig
from reframe.core.io.video import VideoInfo, VideoWriter, probe, read_frames
from reframe.core.seams.apply import apply_removal_order
from reframe.core.seams.carve import EnergyMode, carve_width
from reframe.core.types import ImageArray


def _even(x: float) -> int:
    return max(2, round(x / 2.0) * 2)


@dataclass(frozen=True)
class FramePlan:
    """Geometry shared by every frame of a clip."""

    src_w: int
    src_h: int
    """Source height actually used (even)."""
    crop_y: int
    """Top of the used rows (non-zero only for sources narrower than the target ratio)."""
    crop_x: int
    crop_w: int
    target_w: int
    """Width after seam carving; aspect target_w / src_h == out_w / out_h."""
    factor: int
    """Integer downscale factor for the seam computation."""

    @property
    def n_seams(self) -> int:
        return self.crop_w - self.target_w


def plan_frame(width: int, height: int, cfg: ReframeConfig) -> FramePlan:
    out_w, out_h = cfg.out_size
    w_even = width - (width % 2)
    h = height - (height % 2)
    target_w = _even(h * out_w / out_h)
    if target_w > w_even:
        # Source already narrower than the target ratio: crop height (centred)
        # instead of letting the encoder stretch the picture.
        target_w = w_even
        h = min(h, max(2, int(w_even * out_h / out_w) // 2 * 2))
    crop_y = (height - h) // 2
    crop_w = min(_even(target_w / (1.0 - cfg.seam_ratio)), w_even)
    crop_w = max(crop_w, target_w)
    crop_x = (width - crop_w) // 2
    factor = max(1, h // cfg.work_height)
    return FramePlan(width, h, crop_y, crop_x, crop_w, target_w, factor)


def _downscale(img: ImageArray, f: int) -> ImageArray:
    if f == 1:
        return img
    h, w = img.shape[0] // f, img.shape[1] // f
    small = img[: h * f, : w * f].reshape(h, f, w, f, -1).mean(axis=(1, 3))
    return cast(ImageArray, small)


def reframe_frame(frame: ImageArray, plan: FramePlan, cfg: ReframeConfig) -> ImageArray:
    crop = frame[plan.crop_y : plan.crop_y + plan.src_h, plan.crop_x : plan.crop_x + plan.crop_w]
    if plan.n_seams == 0:
        return np.ascontiguousarray(crop)
    small = _downscale(crop, plan.factor)
    # Ceil so the upsampled map always marks >= n_seams pixels per row for removal.
    n_small = max(1, math.ceil(plan.n_seams * small.shape[1] / plan.crop_w))
    n_small = min(n_small, small.shape[1] - 1)
    res = carve_width(small, n_small, mode=cast(EnergyMode, cfg.energy))
    out = apply_removal_order(crop, res.order, n_small, target_width=plan.target_w)
    return np.ascontiguousarray(out)


def reframe_video(
    src: str | Path,
    dst: str | Path,
    cfg: ReframeConfig,
    *,
    progress: Callable[[int, int | None], None] | None = None,
) -> VideoInfo:
    """Reframe ``src`` to ``dst``; returns the source info."""
    info = probe(src)
    plan = plan_frame(info.width, info.height, cfg)
    with VideoWriter(
        dst,
        frame_size=(plan.target_w, plan.src_h),
        fps=info.fps,
        out_size=cfg.out_size,
        audio_from=info,
        crf=cfg.crf,
        preset=cfg.x264_preset,
    ) as writer:
        for i, frame in enumerate(read_frames(src, info)):
            writer.write(reframe_frame(frame, plan, cfg))
            if progress is not None:
                progress(i + 1, info.n_frames)
    return info
