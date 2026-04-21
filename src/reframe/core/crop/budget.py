"""Automatic crop/seam split (M4).

The crop window is ``target_w + n_seams`` wide and seam carving removes the
``n_seams`` extra columns. A fixed seam ratio (M1: 25%) cuts through subjects
when the content has no cheap seams (title cards, flat cartoons with a few
strong shapes). Instead we carve the *widest* candidate crop on a few
keyframes, look at the DP cost of seam ``k`` as ``k`` grows, and keep only the
seams before the cost knee; the rest of the width is taken by the crop.

Costs are in forward-energy units (see ``contracts``), one value per seam in
removal order (``CarveResult.seam_costs``), at analysis resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BudgetConfig:
    """Seam-count selection parameters."""

    ref_seams: int = 5
    """Baseline = cheapest (smoothed, keyframe-median) cost among the first
    ``ref_seams`` seams, so a jump inside that window is still detected."""
    rise_ratio: float = 3.0
    """A seam is expensive once its (smoothed) cost exceeds ``rise_ratio * baseline``."""
    min_cost_per_row: float = 2.0
    """Baseline floor per seam row (forward units), so near-zero costs on flat
    backgrounds do not turn tiny absolute increases into huge ratios. Applied
    only when ``rows`` is given; otherwise the floor is ``abs_floor``."""
    max_cost_per_row: float | None = 40.0
    """Absolute cap per row: a seam averaging more than this |dI| per row is cutting
    structure regardless of the curve's shape (e.g. letters on white). Only
    applied when ``rows`` is given; ``None`` disables it."""
    plateau_ratio: float = 2.0
    """The first ``ref_seams`` seams must form a plateau (all within
    ``plateau_ratio x baseline``); a curve already rising there has no cheap
    seams and yields 0."""
    abs_floor: float = 10.0
    """Baseline floor (total seam cost, forward units) when ``rows`` is unknown;
    10 units over a ~540-row seam is visually free."""
    smooth: int = 3
    """Running-median window over seam index to ignore single-seam spikes."""


def _running_median(v: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or v.shape[0] < k:
        return v
    pad = k // 2
    vp = np.pad(v, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(vp, k)
    out: np.ndarray = np.median(win, axis=1)
    return out


def choose_seam_count(
    cost_curves: list[list[float]],
    max_seams: int,
    *,
    cfg: BudgetConfig | None = None,
    rows: int | None = None,
) -> int:
    """Number of cheap seams (0..``max_seams``) before the cost knee.

    ``cost_curves[j][k]`` is the DP cost of seam ``k`` on keyframe ``j`` when
    carving the widest candidate crop. Curves are truncated to the shortest one
    and to ``max_seams``; the per-seam median across keyframes is used (robust
    to one odd keyframe). ``rows`` is the seam length (work-image height) and
    enables the per-row floor and absolute cap.

    Returns 0 if the first ``ref_seams`` seams are not a plateau (costs rise
    immediately), else the count ``n`` such that seams ``0..n-1`` all stay
    below ``rise_ratio x baseline`` (and the absolute cap when ``rows`` is given).
    """
    cfg = cfg or BudgetConfig()
    curves = [np.asarray(c, dtype=np.float64) for c in cost_curves if len(c) > 0]
    if max_seams <= 0 or not curves:
        return 0
    m_len = min(min(c.shape[0] for c in curves), max_seams)
    med = np.median(np.stack([c[:m_len] for c in curves]), axis=0)
    med = _running_median(med, cfg.smooth)

    floor = cfg.min_cost_per_row * rows if rows else cfg.abs_floor
    baseline = max(float(np.min(med[: max(cfg.ref_seams, 1)])), floor)
    n_ref = min(max(cfg.ref_seams, 1), m_len)
    if float(med[:n_ref].max()) > cfg.plateau_ratio * baseline:
        return 0
    limit = cfg.rise_ratio * baseline
    if rows and cfg.max_cost_per_row is not None:
        limit = min(limit, cfg.max_cost_per_row * rows)
    over = np.flatnonzero(med > limit)
    return int(over[0]) if over.size else int(m_len)


def max_seams_for(target_w: float, frame_w: float, max_seam_ratio: float) -> int:
    """Widest candidate seam count: crop ``= target_w / (1 - ratio)``, capped by the frame."""
    ratio = min(max(max_seam_ratio, 0.0), 0.95)
    widest = min(target_w / (1.0 - ratio), frame_w)
    return max(math.floor(widest - target_w + 1e-9), 0)


def seam_budget_for_shot(
    target_w: float, frame_w: float, max_seam_ratio: float, n_cheap: int
) -> tuple[float, int]:
    """``(crop_w, n_seams)`` for a shot, analysis px.

    ``n_seams = min(n_cheap, max_seams_for(...))`` and ``crop_w = target_w +
    n_seams`` (float, analysis px). Rounding to even full-resolution widths is
    left to the integrator (multiply by ``FramePlan.factor``).
    """
    n = max(min(int(n_cheap), max_seams_for(target_w, frame_w, max_seam_ratio)), 0)
    return float(target_w + n), n


__all__ = ["BudgetConfig", "choose_seam_count", "max_seams_for", "seam_budget_for_shot"]
