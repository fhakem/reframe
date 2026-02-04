"""Image quality metrics used by tests and (later) regression benchmarks."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np

from reframe.core.energy.gradient import to_gray
from reframe.core.types import FloatArray, ImageArray, IntArray


def _box(a: FloatArray, r: int) -> FloatArray:
    """Mean filter of radius r (window 2r+1) with edge replication, via integral image."""
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def ssim(a: FloatArray, b: FloatArray, *, data_range: float = 255.0, radius: int = 3) -> float:
    """Mean SSIM (Wang et al. 2004) with a box window, for single-channel images."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mx, my = _box(x, radius), _box(y, radius)
    vx = _box(x * x, radius) - mx * mx
    vy = _box(y * y, radius) - my * my
    cxy = _box(x * y, radius) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


# --- temporal stability (M3) -------------------------------------------------


def seam_positions(order: IntArray, crop_x: float = 0.0) -> FloatArray:
    """(n, h) column of the k-th seam in each row, from a removal-order map.

    ``crop_x`` is added so positions are in source-frame columns (analysis
    resolution) when ``order`` was computed on a crop window starting there.
    """
    o = np.asarray(order)
    h = o.shape[0]
    n = int(o.max()) + 1 if o.size else 0
    pos = np.full((n, h), np.nan, dtype=np.float64)
    r, c = np.nonzero(o >= 0)
    pos[o[r, c], r] = c
    if np.isnan(pos).any():
        raise ValueError("order map must remove exactly one pixel per row per seam")
    return pos + float(crop_x)


def seam_jitter(
    seams: Sequence[FloatArray | IntArray],
    *,
    match: Literal["index", "sorted"] = "index",
) -> float:
    """Mean |Δ position| of seams between consecutive frames (source columns).

    ``seams`` holds one (n, h) array per frame (see :func:`seam_positions`), all
    in source-frame columns. ``match="index"`` compares the k-th seam of frame t
    with the k-th seam of frame t-1 (Grundmann-style seam identity; penalises
    reordering too). ``match="sorted"`` compares, row by row, the sorted removed
    columns: it measures how much the *set* of removed pixels moves, which is
    what is visible, regardless of seam order. Returns 0 for < 2 frames.
    """
    if len(seams) < 2:
        return 0.0
    arrs = [np.asarray(s, dtype=np.float64) for s in seams]
    if match == "sorted":
        arrs = [np.sort(a, axis=0) for a in arrs]
    diffs = [np.abs(b - a).mean() for a, b in itertools.pairwise(arrs)]
    return float(np.mean(diffs))


def flow_warping_error(
    prev_out: ImageArray,
    out: ImageArray,
    flow: FloatArray | None = None,
) -> float:
    """Mean abs luma difference between ``out`` and ``prev_out`` warped onto it.

    ``flow`` must be the flow from ``out`` to ``prev_out`` computed *on the
    outputs*, i.e. ``estimate_flow(out, prev_out)`` (computed if ``None``), so
    that ``warp_image(prev_out, flow)`` ~ ``out`` where motion is explained by
    the flow. Residual differences are temporal artefacts (seam flicker). No
    occlusion masking; values are on 0..255 luma, lower is more stable.
    """
    from reframe.core.flow import estimate_flow, warp_image

    a = to_gray(np.asarray(prev_out))
    b = to_gray(np.asarray(out))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    f = estimate_flow(b, a) if flow is None else np.asarray(flow, dtype=np.float32)
    warped = warp_image(a.astype(np.float32), f).astype(np.float64)
    return float(np.abs(b - warped).mean())
