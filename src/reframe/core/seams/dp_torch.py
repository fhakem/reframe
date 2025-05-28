"""Torch implementation of the generic seam DP (CUDA / MPS / CPU).

Rows are processed sequentially (the recurrence is inherently row-serial) but
each row is fully vectorised over the width. The backtrack runs on the CPU.
MPS has no float64, so float32 is used there; costs then match the Numba DP
to ~1e-5 relative, and seams may differ on near-ties.

Requires the optional ``gpu`` extra (``uv sync --extra gpu``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reframe.core.types import FloatArray, IntArray


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_seam_torch(
    base: FloatArray,
    costs: tuple[FloatArray, FloatArray, FloatArray] | None = None,
    *,
    device: str | None = None,
) -> tuple[IntArray, float]:
    """Same contract as :func:`reframe.core.seams.dp.find_seam`."""
    import torch

    dev = torch.device(device or default_device())
    dtype = torch.float32 if dev.type == "mps" else torch.float64

    def t(a: FloatArray) -> Any:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)

    b = t(base)
    h, w = b.shape
    if costs is None:
        cl = cu = cr = torch.zeros_like(b)
    else:
        cl, cu, cr = (t(c) for c in costs)
    inf = torch.tensor(float("inf"), dtype=dtype, device=dev)

    back = torch.empty((h, w), dtype=torch.int8, device=dev)
    prev = b[0] + cu[0]
    back[0] = 0
    for i in range(1, h):
        up = prev + cu[i]
        left = torch.cat([inf.view(1), prev[:-1]]) + cl[i]
        right = torch.cat([prev[1:], inf.view(1)]) + cr[i]
        # Same tie-breaking as the Numba DP: up, then left, then right (strict <).
        stacked = torch.stack([up, left, right])
        best, arg = stacked.min(dim=0)
        # torch.min returns the first minimum on ties, matching up > left > right.
        back[i] = torch.where(arg == 1, -1, torch.where(arg == 2, 1, 0)).to(torch.int8)
        prev = b[i] + best

    total, j = prev.min(dim=0)
    back_np = back.cpu().numpy()
    seam = np.empty(h, dtype=np.int64)
    col = int(j.item())
    seam[h - 1] = col
    for i in range(h - 1, 0, -1):
        col += int(back_np[i, col])
        seam[i - 1] = col
    return seam, float(total.item())
