"""Dense optical flow (OpenCV DIS / Farneback) and seam / image warping.

Direction convention (same as OpenCV): ``flow = estimate_flow(prev, cur)`` is
*forward* flow on the grid of ``prev``::

    prev(y, x)  ~  cur(y + flow[y, x, 1], x + flow[y, x, 0])

Hence:

* :func:`warp_columns` pushes positions of ``prev`` forward into ``cur``;
* :func:`warp_image` is a *backward* warp: ``warp_image(cur, estimate_flow(prev, cur)) ~ prev``.

Estimators are pluggable through :func:`register_flow_estimator` (e.g. RAFT /
SEA-RAFT later); the built-ins need no weights or downloads.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from reframe.core.energy.gradient import to_gray
from reframe.core.types import FloatArray, ImageArray

FlowArray = NDArray[np.float32]
"""(h, w, 2) float32, channel 0 = dx, channel 1 = dy (pixels)."""
FlowEstimator = Callable[[ImageArray, ImageArray], FlowArray]
"""``fn(prev_gray_u8, gray_u8) -> forward flow`` on the grid of ``prev``."""


def _as_u8_gray(img: ImageArray) -> NDArray[np.uint8]:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = to_gray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@cache
def _dis(preset: int) -> Any:
    # Not in every cv2 stub set; instances are cached (DIS is not thread-safe:
    # use one process per worker, not threads).
    return getattr(cv2, "DISOpticalFlow_create")(preset)  # noqa: B009


def _dis_fast(prev: ImageArray, cur: ImageArray) -> FlowArray:
    return np.asarray(_dis(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(prev, cur, None), np.float32)


def _dis_medium(prev: ImageArray, cur: ImageArray) -> FlowArray:
    return np.asarray(_dis(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(prev, cur, None), np.float32)


def _farneback(prev: ImageArray, cur: ImageArray) -> FlowArray:
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 4, 15, 3, 5, 1.2, 0)
    return np.asarray(flow, np.float32)


_ESTIMATORS: dict[str, FlowEstimator] = {
    "farneback": _farneback,
}
if hasattr(cv2, "DISOpticalFlow_create"):
    _ESTIMATORS["dis_fast"] = _dis_fast
    _ESTIMATORS["dis_medium"] = _dis_medium

DEFAULT_METHOD = "dis_medium" if "dis_medium" in _ESTIMATORS else "farneback"


def register_flow_estimator(name: str, fn: FlowEstimator) -> None:
    """Add (or replace) a flow backend usable as ``estimate_flow(..., method=name)``."""
    _ESTIMATORS[name] = fn


def available_flow_methods() -> list[str]:
    return sorted(_ESTIMATORS)


def estimate_flow(
    prev_gray: ImageArray, gray: ImageArray, *, method: str | None = None
) -> FlowArray:
    """Forward flow from ``prev_gray`` to ``gray``; (h, w, 2) float32 on ``prev``'s grid.

    Inputs may be uint8/float gray or RGB (converted to BT.709 luma).
    ``method``: ``"dis_medium"`` (default), ``"dis_fast"``, ``"farneback"`` or a
    registered backend.
    """
    a, b = _as_u8_gray(prev_gray), _as_u8_gray(gray)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    name = method or DEFAULT_METHOD
    if name not in _ESTIMATORS:
        raise ValueError(f"unknown flow method {name!r}; available: {available_flow_methods()}")
    flow = _ESTIMATORS[name](a, b)
    if flow.shape != (*a.shape, 2):
        raise ValueError(f"flow backend {name!r} returned shape {flow.shape}")
    return flow


def warp_columns(
    seam_cols: FloatArray,
    flow: FlowArray,
    *,
    x_offset: float = 0.0,
    clamp: bool = True,
) -> FloatArray:
    """Propagate column positions (one or several per row) forward by the flow.

    ``seam_cols`` is (h,) or (n, h): a column per row (per seam), in the
    coordinates of ``flow`` minus ``x_offset`` (i.e. ``flow`` column =
    ``seam_cols + x_offset``). The horizontal flow is linearly interpolated
    along x at each row; vertical motion is ignored (a seam keeps one pixel
    per row). Returns new positions in the same coordinates as ``seam_cols``.
    """
    cols = np.asarray(seam_cols, dtype=np.float64)
    h, w = flow.shape[:2]
    if cols.shape[-1] != h:
        raise ValueError(f"seam rows {cols.shape[-1]} != flow height {h}")
    x = np.clip(cols + x_offset, 0.0, w - 1.0)
    x0 = np.floor(x).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    t = x - x0
    rows = np.arange(h)
    fx = flow[..., 0].astype(np.float64)
    dx = (1.0 - t) * fx[rows, x0] + t * fx[rows, x1]
    out = cols + dx
    if clamp:
        out = np.clip(out, -x_offset, w - 1.0 - x_offset)
    return np.asarray(out, dtype=np.float64)


def warp_image(
    img: ImageArray, flow: FlowArray, *, border: int = cv2.BORDER_REPLICATE
) -> ImageArray:
    """Backward warp: ``out(y, x) = img(y + flow[y, x, 1], x + flow[y, x, 0])`` (bilinear).

    With ``flow = estimate_flow(a, b)``, ``warp_image(b, flow)`` approximates ``a``.
    """
    arr = np.asarray(img)
    h, w = flow.shape[:2]
    if arr.shape[:2] != (h, w):
        raise ValueError(f"image {arr.shape[:2]} and flow {(h, w)} differ")
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    src = arr if arr.dtype in (np.uint8, np.float32) else arr.astype(np.float32)
    out = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=border)
    return np.asarray(out)
