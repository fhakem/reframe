"""Smart pipeline (M2-M4 integrated, M6 parallel render).

Two passes over the video:

1. **Analysis** (decoded directly at analysis resolution by ffmpeg): streaming
   shot detection, semantic analysis per frame kept as a small
   :class:`~reframe.core.contracts.FrameSummary`, and a few keyframes per shot.
2. **Planning** per shot: crop path at the widest candidate crop, seam cost
   curves on keyframes -> automatic seam budget (fewer seams where they would
   cut structure), final crop path, fallback decision.
3. **Render** (full resolution, decoded again): per shot a fresh
   :class:`~reframe.core.seams.temporal.TemporalSeamCarver`; per frame the crop
   window moves along the path, energy is recomputed on the work crop, seams
   are propagated with optical flow, and the removal-order map is applied at
   full resolution. Shots are grouped into contiguous chunks rendered in
   parallel processes as video-only segments, then concatenated and muxed with
   the source audio once.

Memory: the analysis pass keeps 1-D summaries plus <= ``KEYFRAMES_PER_SHOT``
analysis-resolution keyframes per shot; render processes hold one frame.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from reframe.config import ReframeConfig
from reframe.core.contracts import Box, CropPath, FrameSummary, Shot
from reframe.core.crop.budget import choose_seam_count, max_seams_for, seam_budget_for_shot
from reframe.core.crop.trajectory import plan_crop_path
from reframe.core.energy.combine import ClassicalAnalyzer
from reframe.core.fallback import (
    FallbackConfig,
    blur_pad,
    decide_fallback,
    split_screen,
    subject_pair,
)
from reframe.core.flow import estimate_flow
from reframe.core.io.video import VideoInfo, VideoWriter, concat_and_mux, probe, read_frames
from reframe.core.pipeline import FramePlan, plan_frame
from reframe.core.seams.apply import apply_removal_order
from reframe.core.seams.carve import EnergyMode, carve_width
from reframe.core.seams.temporal import TemporalSeamCarver
from reframe.core.shots import HistogramShotDetector, shots_from_cuts
from reframe.core.types import ImageArray

#: Classical (spectral-residual) saliency is diffuse, so the blur-pad trigger is
#: kept for genuinely spread-out content only (calibrated on Big Buck Bunny).
FALLBACK_CFG = FallbackConfig(blur_min_coverage=0.2)
KEYFRAMES_PER_SHOT = 3
KEYFRAME_EVERY = 12

Progress = Callable[[int, int | None], None]
Fallback = Literal["none", "split", "blur_pad"]


@dataclass(frozen=True)
class Geometry:
    """Full-resolution plan plus the matching analysis-resolution sizes."""

    plan: FramePlan
    wa: int
    """Analysis width of the whole source frame."""
    ha: int
    """Analysis height (of the used rows)."""
    target_wa: int

    @property
    def f(self) -> int:
        return self.plan.factor


def geometry(info: VideoInfo, cfg: ReframeConfig) -> Geometry:
    # Seam budget is decided per shot; plan with the max ratio for the bounds.
    plan = plan_frame(info.width, info.height, cfg)
    f = plan.factor
    return Geometry(plan, info.width // f, plan.src_h // f, plan.target_w // f)


@dataclass
class ShotPlan:
    shot: Shot
    n_seams: int
    """Seams at analysis resolution (full-res crop = target_w + n_seams * f)."""
    path: CropPath
    fallback: Fallback = "none"
    pair: tuple[Box, Box] | None = None
    seam_curve_max: list[float] = field(default_factory=list)


@dataclass
class Analysis:
    n_frames: int
    shots: list[Shot]
    summaries: list[FrameSummary]
    keyframes: dict[int, ImageArray]


# --------------------------------------------------------------------------- pass 1


def analyze(
    src: str | Path,
    info: VideoInfo,
    geo: Geometry,
    cfg: ReframeConfig,
    progress: Progress | None = None,
    total: int | None = None,
) -> Analysis:
    detector = HistogramShotDetector()
    analyzer = ClassicalAnalyzer(cfg.semantic)
    summaries: list[FrameSummary] = []
    keyframes: dict[int, ImageArray] = {}
    cuts: list[int] = []
    since_cut = 0
    full_h = info.height // geo.f
    for i, frame in enumerate(read_frames(src, info, size=(geo.wa, full_h))):
        fr = frame[: geo.ha]
        if detector.push(fr):
            cuts.append(i)
            since_cut = 0
        maps = analyzer.analyze(fr)
        summaries.append(analyzer.summarize(i, maps))
        if since_cut % KEYFRAME_EVERY == 0:
            keyframes[i] = fr.copy()
        since_cut += 1
        if progress is not None:
            progress(i + 1, total)
    n = len(summaries)
    return Analysis(n, shots_from_cuts(cuts, n), summaries, keyframes)


# --------------------------------------------------------------------------- planning


def _pick_keyframes(shot: Shot, keyframes: dict[int, ImageArray]) -> list[int]:
    idx = [i for i in sorted(keyframes) if shot.start <= i < shot.end]
    if len(idx) <= KEYFRAMES_PER_SHOT:
        return idx
    pos = np.linspace(0, len(idx) - 1, KEYFRAMES_PER_SHOT).round().astype(int)
    return [idx[p] for p in pos]


def text_width(boxes: list[Box], w: int) -> int:
    """Number of columns covered by the union of text boxes."""
    cols = np.zeros(w, dtype=bool)
    for b in boxes:
        if b.label == "text":
            cols[max(int(b.x0), 0) : min(math.ceil(b.x1), w)] = True
    return int(cols.sum())


def with_center_prior(s: FrameSummary, sigma_frac: float = 0.3) -> FrameSummary:
    """Multiply the column saliency by a centred Gaussian (cinematographic centre bias).

    Classical spectral saliency fires on any high-contrast detail (a leaf in a
    corner); learned saliency models embed this prior, so we add it explicitly.
    """
    w = s.column_saliency.shape[0]
    x = (np.arange(w) - (w - 1) / 2.0) / max(sigma_frac * w, 1.0)
    prof = s.column_saliency * np.exp(-0.5 * x * x)
    total = prof.sum()
    prof = prof / total if total > 0 else np.full(w, 1.0 / w)
    return FrameSummary(s.index, s.boxes, prof)


def plan_shots(analysis: Analysis, geo: Geometry, cfg: ReframeConfig) -> list[ShotPlan]:
    analyzer = ClassicalAnalyzer(cfg.semantic)
    mode = cast(EnergyMode, cfg.energy)
    plans: list[ShotPlan] = []
    for shot in analysis.shots:
        sums = [with_center_prior(x) for x in analysis.summaries[shot.start : shot.end]]
        # Text (title / intertitle card) wider than the output cannot survive
        # either crop or seams: show the whole frame over a blurred background.
        widths = [text_width(x.boxes, geo.wa) for x in sums]
        if cfg.fallback and widths and float(np.median(widths)) > geo.target_wa:
            path = CropPath(
                np.full(len(shot), (geo.wa - geo.target_wa) / 2.0), geo.target_wa, "stationary"
            )
            plans.append(ShotPlan(shot, 0, path, "blur_pad"))
            continue
        max_n = max_seams_for(geo.target_wa, geo.wa, cfg.seam_ratio)
        n_seams, curves = max_n, []
        if cfg.auto_budget and max_n > 0:
            path_max = plan_crop_path(sums, geo.wa, float(geo.target_wa + max_n))
            for i in _pick_keyframes(shot, analysis.keyframes):
                x = int(path_max.x[i - shot.start])
                crop = analysis.keyframes[i][:, x : x + geo.target_wa + max_n]
                maps = analyzer.analyze(crop)
                # Budget on *damage* only: forward step costs (structure cut) plus
                # protected pixels crossed. Semantic energy steers seams at render
                # time but would swamp the cost-knee detection here.
                res = carve_width(crop, max_n, mode=mode, protect=maps.protect)
                curves.append(res.seam_costs)
            if curves:
                n_seams = choose_seam_count(curves, max_n, rows=geo.ha)
        crop_w, n_seams = seam_budget_for_shot(geo.target_wa, geo.wa, cfg.seam_ratio, n_seams)
        path = plan_crop_path(sums, geo.wa, crop_w)
        fb: Fallback = "none"
        pair = None
        if cfg.fallback:
            fb = decide_fallback(sums, geo.wa, crop_w, cfg=FALLBACK_CFG)
            if fb == "split":
                pair = _shot_pair(sums)
                if pair is None:
                    fb = "none"
        plans.append(ShotPlan(shot, n_seams, path, fb, pair, curves[0] if curves else []))
    return plans


def _shot_pair(sums: list[FrameSummary]) -> tuple[Box, Box] | None:
    """Median pair of subjects over the shot (stable split-screen framing)."""
    pairs = [p for s in sums if (p := subject_pair(s.boxes)) is not None]
    if not pairs:
        return None

    def med(boxes: list[Box]) -> Box:
        a = np.median([[b.x0, b.y0, b.x1, b.y1] for b in boxes], axis=0)
        return Box(float(a[0]), float(a[1]), float(a[2]), float(a[3]), label=boxes[0].label)

    left = [min(p, key=lambda b: b.cx) for p in pairs]
    right = [max(p, key=lambda b: b.cx) for p in pairs]
    return med(left), med(right)


# --------------------------------------------------------------------------- pass 2


def render_shots(
    src: str | Path,
    info: VideoInfo,
    geo: Geometry,
    cfg: ReframeConfig,
    shots: list[ShotPlan],
    dst: str | Path,
    *,
    audio: bool,
    progress: Progress | None = None,
    done_offset: int = 0,
    total: int | None = None,
) -> int:
    """Render a contiguous run of shots to ``dst``. Returns frames written."""
    plan = geo.plan
    analyzer = ClassicalAnalyzer(cfg.semantic)
    mode = cast(EnergyMode, cfg.energy)
    start = shots[0].shot.start
    end = shots[-1].shot.end
    frames = read_frames(src, info, start_frame=start)
    written = 0
    with VideoWriter(
        dst,
        frame_size=(plan.target_w, plan.src_h),
        fps=info.fps,
        out_size=cfg.out_size,
        audio_from=info if audio else None,
        crf=cfg.crf,
        preset=cfg.x264_preset,
    ) as writer:
        for sp in shots:
            carver = TemporalSeamCarver(sp.n_seams, mode, lam=cfg.temporal_lambda)
            prev_gray: ImageArray | None = None
            for t in range(len(sp.shot)):
                frame = next(frames, None)
                if frame is None:
                    break
                full = frame[plan.crop_y : plan.crop_y + plan.src_h]
                small = gray = None
                if sp.fallback == "none" and sp.n_seams > 0:
                    small = cv2.resize(full, (geo.wa, geo.ha), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                flow = None
                if gray is not None and prev_gray is not None and cfg.flow_method != "none":
                    flow = estimate_flow(prev_gray, gray, method=cfg.flow_method)
                writer.write(_render_frame(full, small, flow, t, sp, geo, analyzer, carver))
                prev_gray = gray
                written += 1
                if progress is not None:
                    progress(done_offset + written, total)
    frames.close()
    assert start + written <= end
    return written


def _render_frame(
    full: ImageArray,
    small: ImageArray | None,
    flow: ImageArray | None,
    t: int,
    sp: ShotPlan,
    geo: Geometry,
    analyzer: ClassicalAnalyzer,
    carver: TemporalSeamCarver,
) -> ImageArray:
    plan = geo.plan
    f = geo.f
    if sp.fallback == "blur_pad":
        return blur_pad(full, plan.target_w, plan.src_h)
    if sp.fallback == "split" and sp.pair is not None:
        a, b = sp.pair
        return split_screen(full, a, b, plan.target_w, plan.src_h, scale=float(f))

    crop_wa = geo.target_wa + sp.n_seams
    xa = int(min(max(round(float(sp.path.x[t])), 0), geo.wa - crop_wa))
    crop_w_full = plan.target_w + sp.n_seams * f
    x_full = min(xa * f, plan.src_w - crop_w_full)
    crop = full[:, x_full : x_full + crop_w_full]
    if sp.n_seams == 0 or small is None:
        return np.ascontiguousarray(crop)

    work = np.ascontiguousarray(small[:, xa : xa + crop_wa])
    maps = analyzer.analyze(work)
    res = carver.carve(
        work, crop_x=float(xa), extra_energy=maps.energy, protect=maps.protect, flow=flow
    )
    out = apply_removal_order(crop, res.order, sp.n_seams, target_width=plan.target_w)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- orchestration


def _chunks(plans: list[ShotPlan], k: int) -> list[list[ShotPlan]]:
    """Split shots into <= k contiguous groups of roughly equal frame counts."""
    total = sum(len(p.shot) for p in plans)
    target = total / max(k, 1)
    groups: list[list[ShotPlan]] = [[]]
    acc = 0
    for p in plans:
        if groups[-1] and acc >= target * len(groups) and len(groups) < k:
            groups.append([])
        groups[-1].append(p)
        acc += len(p.shot)
    return [g for g in groups if g]


def _render_chunk(
    args: tuple[str, VideoInfo, Geometry, ReframeConfig, list[ShotPlan], str],
) -> int:
    src, info, geo, cfg, shots, dst = args
    return render_shots(src, info, geo, cfg, shots, dst, audio=False)


def auto_workers(cfg: ReframeConfig) -> int:
    if cfg.workers:
        return cfg.workers
    # ~0.5-1 GB per render process (Numba + OpenCV + buffers); stay safe on 8 GB.
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def reframe_video_smart(
    src: str | Path,
    dst: str | Path,
    cfg: ReframeConfig,
    *,
    progress: Progress | None = None,
) -> VideoInfo:
    info = probe(src)
    geo = geometry(info, cfg)
    est = info.n_frames or max(1, math.ceil(info.duration * float(info.fps)))
    total = 2 * est

    analysis = analyze(src, info, geo, cfg, progress, total)
    plans = plan_shots(analysis, geo, cfg)
    if not plans:
        raise ValueError(f"no frames decoded from {src}")
    total = 2 * analysis.n_frames
    offset = analysis.n_frames

    workers = min(auto_workers(cfg), len(plans))
    if workers <= 1:
        render_shots(
            src, info, geo, cfg, plans, dst, audio=True,
            progress=progress, done_offset=offset, total=total,
        )  # fmt: skip
        return info

    groups = _chunks(plans, workers)
    tmp = Path(tempfile.mkdtemp(prefix="reframe-"))
    try:
        segs = [str(tmp / f"seg{i:04d}.mp4") for i in range(len(groups))]
        jobs = [(str(src), info, geo, cfg, g, s) for g, s in zip(groups, segs, strict=True)]
        done = offset
        with ProcessPoolExecutor(len(groups), mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_render_chunk, j): j for j in jobs}
            for fut in as_completed(futures):
                done += fut.result()
                if progress is not None:
                    progress(done, total)
        concat_and_mux(segs, dst, audio_from=info)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return info
