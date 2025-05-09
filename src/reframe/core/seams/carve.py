"""Seam removal / insertion with protect & remove masks and a removal-order map.

Every carve tracks, for each row, the *original* column index of each surviving
pixel. That single bookkeeping array gives us, for free:

* the **removal order map** (a.k.a. "seam index map"): ``order[y, x] = k`` if the
  original pixel ``(y, x)`` was removed by the k-th seam, ``-1`` if it was kept;
* mask tracking (masks are carved alongside the image);
* seam insertion (find the N seams by removal, then duplicate those pixels).

The order map can be computed on a downscaled frame and applied to the full
resolution frame with :func:`reframe.core.seams.apply.apply_removal_order`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from reframe.core.energy.gradient import backward_energy, forward_costs, to_gray
from reframe.core.seams.dp import find_seam, remove_seam
from reframe.core.types import BoolArray, FloatArray, ImageArray, IntArray

EnergyMode = Literal["forward", "backward"]

#: Penalty per protected pixel. Large but finite so the DP stays well-defined
#: (and a seam *can* cross a protected area if nothing else is possible).
PROTECT_COST = 1e8
#: Bonus (negative cost) per pixel flagged for removal.
REMOVE_COST = -1e8


@dataclass(frozen=True)
class CarveResult:
    image: ImageArray
    """Carved image, shape (H, W - n, C) or (H, W - n)."""
    order: IntArray
    """(H, W) int64: index of the seam that removed each original pixel, -1 if kept."""
    seams: list[IntArray]
    """Each seam in the coordinates of the image it was removed from."""
    created_energy: float
    """Sum over seams of the forward-energy cost of the chosen seam (excluding masks).

    This is the luminance discontinuity introduced by the seams, which is what
    forward energy minimises; it is reported for both modes so they can be compared.
    """


def _check_mask(mask: BoolArray | None, shape: tuple[int, int], name: str) -> BoolArray | None:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"{name} mask shape {m.shape} != image shape {shape}")
    return np.ascontiguousarray(m)


def carve_width(
    image: ImageArray,
    n: int,
    *,
    mode: EnergyMode = "forward",
    protect: BoolArray | None = None,
    remove: BoolArray | None = None,
    extra_energy: FloatArray | None = None,
) -> CarveResult:
    """Remove ``n`` vertical seams from ``image``.

    ``extra_energy`` (H, W) is added to the per-pixel base energy; it is the
    hook for semantic energies (saliency, faces, lines...) in later milestones.
    """
    img = np.ascontiguousarray(image)
    h, w = img.shape[:2]
    if not 0 <= n < w:
        raise ValueError(f"cannot remove {n} seams from width {w}")
    protect_m = _check_mask(protect, (h, w), "protect")
    remove_m = _check_mask(remove, (h, w), "remove")

    gray = to_gray(img)
    extra = (
        np.zeros((h, w), dtype=np.float64)
        if extra_energy is None
        else np.ascontiguousarray(extra_energy, dtype=np.float64)
    )
    if extra.shape != (h, w):
        raise ValueError(f"extra_energy shape {extra.shape} != image shape {(h, w)}")
    if protect_m is not None:
        extra = extra + protect_m * PROTECT_COST
    if remove_m is not None:
        extra = extra + remove_m * REMOVE_COST

    cols = np.broadcast_to(np.arange(w, dtype=np.int64), (h, w)).copy()
    order = np.full((h, w), -1, dtype=np.int64)
    rows = np.arange(h)
    seams: list[IntArray] = []
    created = 0.0

    for k in range(n):
        costs = forward_costs(gray)
        cl, cu, cr = costs
        if mode == "forward":
            seam, _ = find_seam(extra, costs)
        else:
            seam, _ = find_seam(backward_energy(gray) + extra)
        created += _forward_cost_of(seam, cl, cu, cr)
        order[rows, cols[rows, seam]] = k
        seams.append(seam)
        img = remove_seam(img, seam)
        gray = remove_seam(gray, seam)
        extra = remove_seam(extra, seam)
        cols = remove_seam(cols, seam)

    return CarveResult(image=img, order=order, seams=seams, created_energy=created)


def _forward_cost_of(seam: IntArray, cl: FloatArray, cu: FloatArray, cr: FloatArray) -> float:
    rows = np.arange(seam.shape[0])
    step = np.zeros_like(seam)
    step[1:] = seam[:-1] - seam[1:]
    pick = np.where(step == -1, cl[rows, seam], np.where(step == 1, cr[rows, seam], cu[rows, seam]))
    return float(pick.sum())


def insert_seams(
    image: ImageArray,
    n: int,
    *,
    mode: EnergyMode = "forward",
    protect: BoolArray | None = None,
) -> ImageArray:
    """Widen ``image`` by ``n`` columns.

    The ``n`` lowest-energy seams are found by removal on a copy (so they are
    distinct), then every pixel on those seams is duplicated as the average of
    itself and its right neighbour. Requires ``n < W``.
    """
    img = np.asarray(image)
    h, w = img.shape[:2]
    res = carve_width(img, n, mode=mode, protect=protect)
    dup = res.order >= 0  # (H, W): pixels to duplicate, exactly n per row
    src = img.astype(np.float64)
    right = np.concatenate([src[:, 1:], src[:, -1:]], axis=1)
    mid = (src + right) / 2.0

    # Output position of each original pixel: x + (#duplicated pixels strictly before x).
    shift = np.cumsum(dup, axis=1) - dup
    pos = np.arange(w)[None, :] + shift
    out = np.empty((h, w + n, *img.shape[2:]), dtype=np.float64)
    r = np.repeat(np.arange(h)[:, None], w, axis=1)
    out[r, pos] = src
    dr, dc = np.nonzero(dup)
    out[dr, pos[dr, dc] + 1] = mid[dr, dc]
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(img.dtype)
    return out.astype(img.dtype)
