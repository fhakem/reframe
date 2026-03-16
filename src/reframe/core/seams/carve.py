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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from reframe.core.energy.gradient import to_gray
from reframe.core.seams.fused import remove_inplace, seam_step
from reframe.core.types import BoolArray, FloatArray, ImageArray, IntArray

EnergyMode = Literal["forward", "backward"]

SeamHookFn = Callable[[int, IntArray], FloatArray]
"""Per-seam energy hook, see :data:`reframe.core.contracts.SeamHook`."""

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
    seam_costs: list[float] = field(default_factory=list)
    """DP objective of each chosen seam **excluding the seam_hook term**.

    It includes the mode's energy (forward step costs or backward gradient),
    ``extra_energy`` and the protect/remove mask penalties.
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
    seam_hook: SeamHookFn | None = None,
) -> CarveResult:
    """Remove ``n`` vertical seams from ``image``.

    ``extra_energy`` (H, W) is added to the per-pixel base energy; it is the
    hook for semantic energies (saliency, faces, lines...) in later milestones.

    ``seam_hook(k, cols)`` is called once per seam ``k`` with ``cols`` (h, w_cur):
    the column in the *input* image of every pixel of the current, partially
    carved image. Its (h, w_cur) result is added to the base energy for seam
    ``k`` only (forward units), e.g. a temporal-coherence term.
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
    seam_costs: list[float] = []
    created = 0.0

    # Fixed-capacity buffers; the live region is [:, :wc] and shrinks in place.
    gray = gray.copy()
    extra = extra.copy()
    cost = np.empty((h, w), dtype=np.float64)
    back = np.empty((h, w), dtype=np.int8)
    no_hook = np.empty((1, 1), dtype=np.float64)
    forward = mode == "forward"

    for k in range(n):
        wc = w - k
        hook_term = no_hook
        if seam_hook is not None:
            hook_term = np.ascontiguousarray(seam_hook(k, cols[:, :wc]), dtype=np.float64)
            if hook_term.shape != (h, wc):
                raise ValueError(f"seam_hook returned {hook_term.shape}, expected {(h, wc)}")
        seam, total, hook_sum, seam_created = seam_step(
            gray, extra, hook_term, seam_hook is not None, wc, forward, cost, back
        )
        seam_costs.append(float(total - hook_sum))
        created += float(seam_created)
        order[rows, cols[rows, seam]] = k
        seams.append(seam)
        remove_inplace(gray, extra, cols, seam, wc)

    carved = img[order < 0].reshape(h, w - n, *img.shape[2:])
    return CarveResult(
        image=carved, order=order, seams=seams, created_energy=created, seam_costs=seam_costs
    )


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
