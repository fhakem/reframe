"""Vertical-seam dynamic programming (CPU, Numba).

The DP is written once in its generic form::

    M[0, j] = base[0, j] + C_U[0, j]
    M[i, j] = base[i, j] + min(M[i-1, j-1] + C_L[i, j],
                               M[i-1, j  ] + C_U[i, j],
                               M[i-1, j+1] + C_R[i, j])

``base`` holds per-pixel energy (backward gradient, mask penalties, saliency...);
``C_L, C_U, C_R`` hold the path-dependent forward-energy step costs (all zeros
for pure backward energy). A seam is an 8-connected path: one column index per
row with ``|seam[i] - seam[i-1]| <= 1``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from reframe.core.types import FloatArray, ImageArray, IntArray


@njit(cache=True)
def _dp_generic(
    base: FloatArray, cl: FloatArray, cu: FloatArray, cr: FloatArray
) -> tuple[IntArray, float]:
    h, w = base.shape
    cost = np.empty((h, w), dtype=np.float64)
    back = np.zeros((h, w), dtype=np.int8)
    for j in range(w):
        cost[0, j] = base[0, j] + cu[0, j]
    for i in range(1, h):
        for j in range(w):
            best = cost[i - 1, j] + cu[i, j]
            step = 0
            if j > 0:
                c = cost[i - 1, j - 1] + cl[i, j]
                if c < best:
                    best = c
                    step = -1
            if j < w - 1:
                c = cost[i - 1, j + 1] + cr[i, j]
                if c < best:
                    best = c
                    step = 1
            cost[i, j] = base[i, j] + best
            back[i, j] = step
    seam = np.empty(h, dtype=np.int64)
    j = 0
    best = cost[h - 1, 0]
    for k in range(1, w):
        if cost[h - 1, k] < best:
            best = cost[h - 1, k]
            j = k
    seam[h - 1] = j
    for i in range(h - 1, 0, -1):
        j = j + back[i, j]
        seam[i - 1] = j
    return seam, best


def find_seam(
    base: FloatArray,
    costs: tuple[FloatArray, FloatArray, FloatArray] | None = None,
) -> tuple[IntArray, float]:
    """Minimum vertical seam and its total cost.

    ``costs`` are the forward-energy step costs ``(C_L, C_U, C_R)``; ``None``
    means pure backward energy (zero step costs).
    """
    base = np.ascontiguousarray(base, dtype=np.float64)
    if costs is None:
        zero = np.zeros_like(base)
        cl = cu = cr = zero
    else:
        cl, cu, cr = (np.ascontiguousarray(c, dtype=np.float64) for c in costs)
    seam, total = _dp_generic(base, cl, cu, cr)
    return seam, float(total)


def seam_cost(
    seam: IntArray,
    base: FloatArray,
    costs: tuple[FloatArray, FloatArray, FloatArray] | None = None,
) -> float:
    """Cost of an arbitrary 8-connected seam under the same model as :func:`find_seam`."""
    h = base.shape[0]
    total = float(base[0, seam[0]])
    if costs is not None:
        cl, cu, cr = costs
        total += float(cu[0, seam[0]])
    for i in range(1, h):
        j, p = int(seam[i]), int(seam[i - 1])
        total += float(base[i, j])
        if costs is not None:
            step = p - j  # -1: came from up-left, +1: from up-right
            total += float((cl if step == -1 else cr if step == 1 else cu)[i, j])
    return total


@njit(cache=True)
def _remove_seam_2d(arr: ImageArray, seam: IntArray) -> ImageArray:
    h, w = arr.shape
    out = np.empty((h, w - 1), dtype=arr.dtype)
    for i in range(h):
        s = seam[i]
        out[i, :s] = arr[i, :s]
        out[i, s:] = arr[i, s + 1 :]
    return out


@njit(cache=True)
def _remove_seam_3d(arr: ImageArray, seam: IntArray) -> ImageArray:
    h, w, c = arr.shape
    out = np.empty((h, w - 1, c), dtype=arr.dtype)
    for i in range(h):
        s = seam[i]
        out[i, :s] = arr[i, :s]
        out[i, s:] = arr[i, s + 1 :]
    return out


def remove_seam(arr: ImageArray, seam: IntArray) -> ImageArray:
    """Drop one pixel per row (the seam) from a (H, W) or (H, W, C) array."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        return _remove_seam_2d(arr, seam)
    return _remove_seam_3d(arr, seam)
