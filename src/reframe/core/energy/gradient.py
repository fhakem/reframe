"""Low-level energies used by the seam DP.

Two families, both computed on a single-channel float64 luminance image:

* **Backward energy** (Avidan & Shamir 2007): ``e = |dI/dx| + |dI/dy|``. The DP
  minimises the energy *removed* by the seam.
* **Forward energy** (Rubinstein, Shamir & Avidan 2008): the DP minimises the
  energy *created* by joining pixels that become neighbours once the seam is
  removed. For pixel ``(i, j)`` with replicate padding:

  - ``C_U = |I(i, j+1) - I(i, j-1)|``                     (seam comes from above)
  - ``C_L = C_U + |I(i-1, j) - I(i, j-1)|``               (seam comes from up-left)
  - ``C_R = C_U + |I(i-1, j) - I(i, j+1)|``               (seam comes from up-right)

  Row 0 only pays ``C_U``.

Every DP in :mod:`reframe.core.seams` consumes the generic form
``(base, cost_left, cost_up, cost_right)``; backward energy is the special case
where the three step costs are zero.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from reframe.core.types import FloatArray, ImageArray

# ITU-R BT.709 luma weights (HD content).
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def to_gray(image: ImageArray) -> FloatArray:
    """Return a contiguous float64 luminance image in [0, 255] from (H, W) or (H, W, C)."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.ascontiguousarray(arr[..., :3].astype(np.float64) @ _LUMA)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.ascontiguousarray(arr[..., 0], dtype=np.float64)
    raise ValueError(f"unsupported image shape {arr.shape}")


@njit(cache=True)
def _backward_energy(gray: FloatArray) -> FloatArray:
    h, w = gray.shape
    out = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        up = max(i - 1, 0)
        down = min(i + 1, h - 1)
        for j in range(w):
            left = max(j - 1, 0)
            right = min(j + 1, w - 1)
            dx = abs(gray[i, right] - gray[i, left])
            dy = abs(gray[down, j] - gray[up, j])
            out[i, j] = dx + dy
    return out


@njit(cache=True)
def _forward_costs(gray: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    h, w = gray.shape
    cl = np.empty((h, w), dtype=np.float64)
    cu = np.empty((h, w), dtype=np.float64)
    cr = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        up = max(i - 1, 0)
        for j in range(w):
            left = max(j - 1, 0)
            right = min(j + 1, w - 1)
            u = abs(gray[i, right] - gray[i, left])
            cu[i, j] = u
            cl[i, j] = u + abs(gray[up, j] - gray[i, left])
            cr[i, j] = u + abs(gray[up, j] - gray[i, right])
    return cl, cu, cr


def backward_energy(gray: FloatArray) -> FloatArray:
    """Classic gradient-magnitude (L1) energy, replicate borders."""
    return _backward_energy(np.ascontiguousarray(gray, dtype=np.float64))


def forward_costs(gray: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Forward-energy step costs ``(C_L, C_U, C_R)``, each shaped like ``gray``."""
    return _forward_costs(np.ascontiguousarray(gray, dtype=np.float64))
