"""Fused, allocation-free seam step kernels (M6 performance).

The reference loop in :mod:`reframe.core.seams.carve` used to recompute the
forward-cost arrays, run the DP, then reallocate four arrays per seam. Here
one Numba kernel computes forward (or backward) energy on the fly inside the
DP, and a second one removes the seam *in place* from fixed-capacity buffers
whose live width shrinks by one per seam. Results are bit-identical to the
reference: same cost formulas, same tie-breaking (up, then left, then right,
strict ``<``; leftmost minimum on the last row).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from reframe.core.types import FloatArray, IntArray


@njit(cache=True, nogil=True)
def seam_step(
    gray: FloatArray,
    base: FloatArray,
    hook: FloatArray,
    use_hook: bool,
    wc: int,
    forward: bool,
    cost: FloatArray,
    back: np.ndarray,
) -> tuple[IntArray, float, float, float]:
    """One DP pass over the live region ``[:, :wc]``.

    ``cost`` (h, capacity) float64 and ``back`` (h, capacity) int8 are scratch
    buffers. Returns ``(seam, total, hook_sum, created)`` where ``total``
    includes the hook term, ``hook_sum`` is the hook's share along the seam and
    ``created`` the forward step costs along the seam.
    """
    h = gray.shape[0]
    for i in range(h):
        up = max(i - 1, 0)
        down = min(i + 1, h - 1)
        for j in range(wc):
            left = max(j - 1, 0)
            right = min(j + 1, wc - 1)
            u = abs(gray[i, right] - gray[i, left])
            b = base[i, j]
            if not forward:
                b = (u + abs(gray[down, j] - gray[up, j])) + b
            if use_hook:
                b += hook[i, j]
            if forward:
                if i == 0:
                    cost[i, j] = b + u
                    back[i, j] = 0
                    continue
                cl = u + abs(gray[up, j] - gray[i, left])
                cr = u + abs(gray[up, j] - gray[i, right])
                best = cost[i - 1, j] + u
                step = 0
                if j > 0:
                    c = cost[i - 1, j - 1] + cl
                    if c < best:
                        best = c
                        step = -1
                if j < wc - 1:
                    c = cost[i - 1, j + 1] + cr
                    if c < best:
                        best = c
                        step = 1
            else:
                if i == 0:
                    cost[i, j] = b
                    back[i, j] = 0
                    continue
                best = cost[i - 1, j]
                step = 0
                if j > 0 and cost[i - 1, j - 1] < best:
                    best = cost[i - 1, j - 1]
                    step = -1
                if j < wc - 1 and cost[i - 1, j + 1] < best:
                    best = cost[i - 1, j + 1]
                    step = 1
            cost[i, j] = b + best
            back[i, j] = step

    seam = np.empty(h, dtype=np.int64)
    j = 0
    total = cost[h - 1, 0]
    for k in range(1, wc):
        if cost[h - 1, k] < total:
            total = cost[h - 1, k]
            j = k
    seam[h - 1] = j
    for i in range(h - 1, 0, -1):
        j = j + back[i, j]
        seam[i - 1] = j

    # Hook share and forward step costs along the chosen seam.
    hook_sum = 0.0
    created = 0.0
    for i in range(h):
        j = seam[i]
        if use_hook:
            hook_sum += hook[i, j]
        left = max(j - 1, 0)
        right = min(j + 1, wc - 1)
        u = abs(gray[i, right] - gray[i, left])
        if i == 0:
            created += u
            continue
        step = seam[i - 1] - j
        up = i - 1
        if step == -1:
            created += u + abs(gray[up, j] - gray[i, left])
        elif step == 1:
            created += u + abs(gray[up, j] - gray[i, right])
        else:
            created += u
    return seam, total, hook_sum, created


@njit(cache=True, nogil=True)
def remove_inplace(
    gray: FloatArray, base: FloatArray, cols: IntArray, seam: IntArray, wc: int
) -> None:
    """Shift pixels right of the seam one column left, for the live width ``wc``."""
    h = gray.shape[0]
    for i in range(h):
        for j in range(seam[i], wc - 1):
            gray[i, j] = gray[i, j + 1]
            base[i, j] = base[i, j + 1]
            cols[i, j] = cols[i, j + 1]
