"""Apply a removal-order map computed at low resolution to a full-resolution frame.

Seams are expensive at 1080p and the DP is O(W·H) per seam, so we compute them
on a downscaled frame (e.g. 540p) and upsample the resulting order map. For a
low-res map of width ``w`` with ``n`` seams applied to a frame of width ``W``,
each full-res row loses exactly ``round(n * W / w)`` pixels: the ones whose
(nearest-neighbour) low-res source was removed earliest. Selecting a fixed
count per row (top-K by order) rather than thresholding keeps every row the
same width at non-integer scales.
"""

from __future__ import annotations

import numpy as np

from reframe.core.types import ImageArray, IntArray


def upsample_order(order: IntArray, height: int, width: int) -> IntArray:
    """Nearest-neighbour resize of a removal-order map to (height, width)."""
    h, w = order.shape
    ys = np.minimum((np.arange(height) * h) // height, h - 1)
    xs = np.minimum((np.arange(width) * w) // width, w - 1)
    return order[ys[:, None], xs[None, :]]


def apply_removal_order(
    image: ImageArray, order: IntArray, n_seams: int, *, target_width: int | None = None
) -> ImageArray:
    """Remove pixels from ``image`` following a (possibly lower-res) order map.

    ``n_seams`` is how many seams the map encodes (orders ``0..n_seams-1``).
    ``target_width`` overrides the output width (default: scaled from the map).
    """
    img = np.asarray(image)
    big_h, big_w = img.shape[:2]
    small_w = order.shape[1]
    k = round(n_seams * big_w / small_w) if target_width is None else big_w - int(target_width)
    if not 0 <= k < big_w:
        raise ValueError(f"cannot remove {k} columns from width {big_w}")
    if k == 0:
        return img.copy()

    up = upsample_order(order, big_h, big_w)
    # Kept pixels (-1) sort last; earlier seams first. Ties (one low-res column
    # spread over several full-res columns) are broken by position, stably.
    key = np.where(up < 0, np.iinfo(np.int64).max, up)
    idx = np.argsort(key, axis=1, kind="stable")[:, :k]
    keep = np.ones((big_h, big_w), dtype=bool)
    keep[np.arange(big_h)[:, None], idx] = False
    out = img[keep]
    return out.reshape(big_h, big_w - k, *img.shape[2:])
