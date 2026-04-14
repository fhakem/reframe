"""Smart-crop trajectory for one shot (M4), AutoFlip-style.

All quantities are analysis-resolution pixels in source-frame coordinates; the
path is the crop window's *left edge* per frame (see :class:`CropPath`).

Steps
-----
1. Per-frame target centre ``c_t`` and confidence ``w_t``: boxes of the most
   important label tier present (manual/face > person > text > salient),
   weighted by ``score * area``; otherwise the centre of the crop-wide window
   that holds the most column saliency (confidence grows with how concentrated
   that saliency is; a uniform profile gives zero confidence).
2. Smooth path: ``min sum w_t (y_t - c_t)^2 + alpha sum (dy)^2 + beta sum (d2y)^2``,
   a symmetric positive-definite pentadiagonal system solved with a banded
   Cholesky factorisation (O(n), no scipy).
3. Mode: **stationary** if the smoothed path barely moves (one fixed x that
   maximises mean covered importance), **pan** if it is well fit by a constant
   velocity line, else **track** (the smoothed path).
4. Clamp to ``[0, frame_w - crop_w]`` and enforce bounded velocity and
   acceleration (px/frame, px/frame^2) with a forward/backward rate-limited
   follower whose average keeps both bounds and has no net lag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from reframe.core.contracts import Box, BoxLabel, CropMode, CropPath, FrameSummary
from reframe.core.types import FloatArray

_TIERS: tuple[tuple[BoxLabel, ...], ...] = (
    ("manual", "face"),
    ("person",),
    ("text",),
    ("salient",),
)


def _default_label_confidence() -> dict[str, float]:
    return {"manual": 4.0, "face": 2.0, "person": 1.5, "text": 1.0, "salient": 0.8}


@dataclass(frozen=True)
class TrajectoryConfig:
    """Smart-crop trajectory parameters (analysis px, frames).

    Defaults target ~540p analysis frames at 24-30 fps.
    """

    label_confidence: dict[str, float] = field(default_factory=_default_label_confidence)
    """Data-term weight ``w_t`` of a frame whose target comes from boxes of that tier."""
    saliency_confidence: float = 0.5
    """Max data weight of a saliency-only frame (scaled by saliency concentration)."""
    alpha: float = 50.0
    """Velocity penalty; with ``beta`` gives a smoothing time constant of ~7 frames."""
    beta: float = 2000.0
    """Acceleration penalty (dominant for jitter removal)."""
    stationary_frac: float = 0.08
    """Stationary if the smoothed path's range <= this fraction of ``crop_w``."""
    pan_residual_frac: float = 0.03
    """Pan if the RMS residual of a line fit to the smoothed path <= this x ``crop_w``."""
    max_velocity: float = 6.0
    """px/frame (~150 px/s at 25 fps on a 960 px wide analysis frame)."""
    max_acceleration: float = 0.5
    """px/frame^2: reaching max velocity takes ~12 frames (no whip pans)."""
    tie_tolerance: float = 1e-3
    """Stationary x: windows within this relative mass of the best are ties,
    broken toward the median target."""


# --------------------------------------------------------------------------- helpers


def window_masses(profile: FloatArray, crop_w: float) -> FloatArray:
    """Mass inside each integer-aligned window of width ``round(crop_w)``.

    Element ``s`` is the mass of columns ``[s, s + win)`` (cumulative sum).
    """
    n = profile.shape[0]
    win = int(min(max(round(crop_w), 1), n))
    cs = np.concatenate(([0.0], np.cumsum(profile, dtype=np.float64)))
    return cs[win:] - cs[: n - win + 1]


def best_window(profile: FloatArray, crop_w: float, prefer_x: float, tol: float = 1e-3) -> float:
    """Left edge of the crop window covering most of ``profile``.

    Near-ties (within ``tol`` of the best mass, relative) are broken toward
    ``prefer_x`` so plateaus (uniform profiles, narrow subjects) are stable.
    """
    masses = window_masses(profile, crop_w)
    best = float(masses.max())
    cand = np.flatnonzero(masses >= best - tol * max(abs(best), 1e-12))
    return float(cand[np.argmin(np.abs(cand - prefer_x))])


def box_profile(boxes: list[Box], w: int) -> FloatArray:
    """Column importance profile of boxes (each ``score`` spread over its width), sum 1."""
    prof = np.zeros(w, dtype=np.float64)
    for b in boxes:
        lo = int(np.clip(math.floor(b.x0), 0, w - 1))
        hi = int(np.clip(math.ceil(b.x1), lo + 1, w))
        prof[lo:hi] += max(b.score, 0.0) / (hi - lo)
    s = prof.sum()
    return prof / s if s > 0 else prof


def important_boxes(boxes: list[Box]) -> list[Box]:
    """Boxes of the highest-priority label tier present (score > 0, non-empty)."""
    for tier in _TIERS:
        sel = [b for b in boxes if b.label in tier and b.score > 0 and b.width > 0]
        if sel:
            return sel
    return []


def frame_target(
    s: FrameSummary, frame_w: float, crop_w: float, cfg: TrajectoryConfig
) -> tuple[float, float]:
    """Target crop centre and data confidence for one frame."""
    sel = important_boxes(s.boxes)
    if sel:
        wts = np.array([b.score * max(b.width * (b.y1 - b.y0), 1.0) for b in sel])
        cx = float(np.dot(wts, [b.cx for b in sel]) / wts.sum())
        conf = max(cfg.label_confidence.get(b.label, 1.0) for b in sel)
        return cx, conf
    prof = np.asarray(s.column_saliency, dtype=np.float64)
    total = float(prof.sum())
    if prof.size == 0 or total <= 0:
        return frame_w / 2.0, 0.0
    prof = prof / total
    masses = window_masses(prof, crop_w)
    uniform = min(round(crop_w), prof.size) / prof.size
    concentration = (float(masses.max()) - uniform) / max(1.0 - uniform, 1e-9)
    if concentration <= 1e-6:
        return frame_w / 2.0, 0.0
    x = best_window(prof, crop_w, prefer_x=(prof.size - crop_w) / 2.0)
    return x + crop_w / 2.0, cfg.saliency_confidence * min(concentration, 1.0)


def smooth_path(
    c: FloatArray, w: FloatArray, alpha: float, beta: float, eps: float = 1e-4
) -> FloatArray:
    """Solve ``(W + alpha D1'D1 + beta D2'D2) y = W c`` (pentadiagonal SPD, banded Cholesky).

    Weights are floored at ``eps`` so the system stays positive definite; give
    zero-confidence frames an interpolated ``c`` before calling.
    """
    n = c.shape[0]
    wf = np.maximum(w, eps)
    d0 = wf.copy()
    d1 = np.zeros(max(n - 1, 0))
    d2 = np.zeros(max(n - 2, 0))
    if n >= 2:
        d0[:-1] += alpha
        d0[1:] += alpha
        d1 -= alpha
    if n >= 3:
        d0[:-2] += beta
        d0[1:-1] += 4.0 * beta
        d0[2:] += beta
        d1[:-1] -= 2.0 * beta
        d1[1:] -= 2.0 * beta
        d2 += beta
    b = wf * c
    l0 = np.zeros(n)
    l1 = np.zeros(n)  # L[i, i-1]
    l2 = np.zeros(n)  # L[i, i-2]
    for i in range(n):
        if i >= 2:
            l2[i] = d2[i - 2] / l0[i - 2]
        if i >= 1:
            l1[i] = (d1[i - 1] - l2[i] * l1[i - 1]) / l0[i - 1]
        l0[i] = math.sqrt(d0[i] - l1[i] ** 2 - l2[i] ** 2)
    z = np.zeros(n)
    for i in range(n):
        acc = b[i]
        if i >= 1:
            acc -= l1[i] * z[i - 1]
        if i >= 2:
            acc -= l2[i] * z[i - 2]
        z[i] = acc / l0[i]
    y = np.zeros(n)
    for i in range(n - 1, -1, -1):
        acc = z[i]
        if i + 1 < n:
            acc -= l1[i + 1] * y[i + 1]
        if i + 2 < n:
            acc -= l2[i + 2] * y[i + 2]
        y[i] = acc / l0[i]
    return y


def _brake(dist: float, a_max: float) -> float:
    """Largest speed from which a discrete stop within ``dist`` px is possible.

    Exact inverse of the stopping distance of steps ``v, v - a, ..., v - k a``
    (``k = floor(v / a)``): ``D(v) = (k + 1) v - a k (k + 1) / 2``.
    """
    d = max(dist, 0.0)
    est = int((-a_max + math.sqrt(a_max * a_max + 8.0 * a_max * d)) / (2.0 * a_max))
    for k in range(est + 1, max(est - 2, -1), -1):
        v = (d + a_max * k * (k + 1) / 2.0) / (k + 1)
        if k * a_max <= v <= (k + 1) * a_max:
            return v
    return min(d, a_max)


def _follow(y: FloatArray, lo: float, hi: float, v_max: float, a_max: float) -> FloatArray:
    """Rate-limited follower of ``y``: |v| <= v_max, |dv| <= a_max, x in [lo, hi].

    Tracks ``y`` exactly while ``y`` itself respects the limits; otherwise
    catches up with a braking profile (in the frame moving with ``y``) so it
    does not overshoot. Speed toward a wall is capped by the braking distance,
    so the bounds never need a hard (acceleration-breaking) stop.
    """
    n = y.shape[0]
    x = np.empty(n)
    x[0] = y[0]
    v = float(np.clip(y[1] - y[0], -v_max, v_max)) if n > 1 else 0.0
    v = min(max(v, -_brake(x[0] - lo, a_max)), _brake(hi - x[0], a_max))
    for t in range(1, n):
        xp = x[t - 1]
        vy = y[t] - y[t - 1]
        e = y[t - 1] - xp
        v_des = vy + math.copysign(_brake(abs(e), a_max), e)
        v_des = min(max(v_des, -v_max, -_brake(xp - lo, a_max)), v_max, _brake(hi - xp, a_max))
        v_new = min(max(v_des, v - a_max), v + a_max)
        xt = min(max(xp + v_new, lo), hi)
        v = xt - xp
        x[t] = xt
    return x


def enforce_limits(y: FloatArray, lo: float, hi: float, v_max: float, a_max: float) -> FloatArray:
    """Clamp to ``[lo, hi]`` with bounded velocity/acceleration and no net lag.

    Average of a forward and a backward follower: each respects the limits, so
    (both being linear constraints) does their mean; the lag of one cancels the
    lead of the other.
    """
    y = np.clip(y, lo, hi)
    if y.shape[0] < 2:
        return y
    fwd = _follow(y, lo, hi, v_max, a_max)
    bwd = _follow(y[::-1].copy(), lo, hi, v_max, a_max)[::-1]
    out: FloatArray = np.clip((fwd + bwd) / 2.0, lo, hi)
    return out


def _weighted_median(v: FloatArray, w: FloatArray) -> float:
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw, cw[-1] / 2.0)])


# --------------------------------------------------------------------------- API


def plan_crop_path(
    summaries: list[FrameSummary],
    frame_w: float,
    crop_w: float,
    *,
    cfg: TrajectoryConfig | None = None,
) -> CropPath:
    """Crop path for one shot (``summaries`` in frame order, analysis px)."""
    cfg = cfg or TrajectoryConfig()
    n = len(summaries)
    hi = max(frame_w - crop_w, 0.0)
    centred = hi / 2.0
    if n == 0:
        return CropPath(np.zeros(0), crop_w, "stationary")
    if hi <= 0:
        return CropPath(np.zeros(n), crop_w, "stationary")

    targets = [frame_target(s, frame_w, crop_w, cfg) for s in summaries]
    c = np.array([t[0] for t in targets], dtype=np.float64)
    w = np.array([t[1] for t in targets], dtype=np.float64)
    known = w > 0
    if not known.any():
        return CropPath(np.full(n, centred), crop_w, "stationary")
    t_idx = np.arange(n, dtype=np.float64)
    c = np.interp(t_idx, t_idx[known], c[known])
    c = np.clip(c, crop_w / 2.0, frame_w - crop_w / 2.0)

    y = smooth_path(c, w, cfg.alpha, cfg.beta) - crop_w / 2.0
    y = np.clip(y, 0.0, hi)

    mode: CropMode
    if float(np.ptp(y)) <= cfg.stationary_frac * crop_w:
        mode = "stationary"
        w_an = len(summaries[0].column_saliency) or int(frame_w)
        prof = np.zeros(w_an)
        for s in summaries:
            sel = important_boxes(s.boxes)
            if sel:
                prof += box_profile(sel, w_an)
            else:
                cs = np.asarray(s.column_saliency, dtype=np.float64)
                if cs.shape[0] == w_an and cs.sum() > 0:
                    prof += cs / cs.sum()
        prefer = _weighted_median(c, w) - crop_w / 2.0
        x0 = best_window(prof, crop_w, prefer, cfg.tie_tolerance) if prof.sum() > 0 else prefer
        return CropPath(np.full(n, float(np.clip(x0, 0.0, hi))), crop_w, mode)

    slope, icpt = np.polyfit(t_idx, y, 1)
    line = icpt + slope * t_idx
    rms = float(np.sqrt(np.mean((y - line) ** 2)))
    if rms <= cfg.pan_residual_frac * crop_w and abs(slope) <= cfg.max_velocity:
        mode, raw = "pan", line
    else:
        mode, raw = "track", y
    x = enforce_limits(raw, 0.0, hi, cfg.max_velocity, cfg.max_acceleration)
    return CropPath(x, crop_w, mode)


__all__ = [
    "TrajectoryConfig",
    "best_window",
    "box_profile",
    "enforce_limits",
    "frame_target",
    "important_boxes",
    "plan_crop_path",
    "smooth_path",
    "window_masses",
]
