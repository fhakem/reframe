"""Cross-module contracts for M2-M4. **Frozen**: change only during integration.

Conventions
-----------
Resolutions
    * **source**: the decoded, upright frame (e.g. 1920x1080).
    * **analysis**: the *whole* source frame downscaled by the integer factor
      ``FramePlan.factor`` (~540p). Every per-frame analysis output (boxes,
      saliency, column profiles, flow) is at analysis resolution in
      *source-frame* coordinates (not crop coordinates).
    * **work**: the crop window at analysis resolution; seams are computed here
      and applied at full resolution through the removal-order map.

Energy scale ("forward units")
    Forward-energy step costs are |dI| sums on 0..255 luminance, i.e. roughly
    0..510 per pixel. Any per-pixel term added to the seam DP must be expressed
    in those units: ``weight * map`` with ``map`` in [0, 1] and weights of order
    100..5000. Hard protection uses ``carve.PROTECT_COST`` (1e8) via a bool mask,
    never via the energy map.

Memory (8 GB machine)
    No stage may hold a whole shot of full-resolution frames. The analysis pass
    keeps only :class:`FrameSummary` (boxes + 1-D profiles) per frame; the
    render pass decodes again and recomputes 2-D energy maps per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from reframe.core.types import BoolArray, FloatArray, ImageArray, IntArray

BoxLabel = Literal["face", "person", "text", "salient", "manual"]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box, analysis-resolution pixels, source-frame coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    label: BoxLabel = "salient"

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def scaled(self, s: float) -> Box:
        return Box(self.x0 * s, self.y0 * s, self.x1 * s, self.y1 * s, self.score, self.label)


@dataclass(frozen=True)
class SemanticMaps:
    """Per-frame 2-D semantic output (M2), all shaped (h, w) of the input frame."""

    energy: FloatArray
    """Combined extra energy in forward units (saliency, lines, depth...), >= 0."""
    protect: BoolArray
    """Hard-protect mask (faces, people, text) for the seam DP."""
    saliency: FloatArray
    """Normalised saliency in [0, 1]."""
    boxes: list[Box] = field(default_factory=list)


class SemanticAnalyzer(Protocol):
    """M2. Stateless per frame; ``frame`` is uint8 RGB at analysis resolution."""

    def analyze(self, frame: ImageArray) -> SemanticMaps: ...


@dataclass(frozen=True)
class FrameSummary:
    """What the analysis pass keeps per frame (small, 1-D)."""

    index: int
    boxes: list[Box]
    column_saliency: FloatArray
    """(w_analysis,) saliency mass per column, normalised to sum 1 (uniform if empty)."""


@dataclass(frozen=True)
class Shot:
    """Half-open frame range [start, end)."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


class ShotDetector(Protocol):
    """M3. Streaming: feed analysis-resolution frames in order."""

    def push(self, frame: ImageArray) -> bool:
        """Return True if ``frame`` starts a new shot (the first frame returns False)."""
        ...

    def reset(self) -> None: ...


CropMode = Literal["stationary", "pan", "track"]


@dataclass(frozen=True)
class CropPath:
    """M4. Crop window left edge per frame of a shot (analysis px, source coords)."""

    x: FloatArray
    """(len(shot),) left edge; 0 <= x <= w_analysis - crop_w."""
    crop_w: float
    mode: CropMode


SeamHook = "Callable[[int, IntArray], FloatArray]"
"""M3 carve hook signature (documented as a string to avoid a runtime import):
``hook(k, cols) -> extra`` where ``k`` is the seam index being found, ``cols``
(h, w_cur) maps every pixel of the current (partially carved) work image to
its column in the *input* work image, and ``extra`` (h, w_cur) is added to the
base energy for that seam only (forward units)."""


def uniform_profile(w: int) -> FloatArray:
    return np.full(w, 1.0 / max(w, 1), dtype=np.float64)


__all__ = [
    "BoolArray",
    "Box",
    "BoxLabel",
    "CropMode",
    "CropPath",
    "FrameSummary",
    "IntArray",
    "SeamHook",
    "SemanticAnalyzer",
    "SemanticMaps",
    "Shot",
    "ShotDetector",
    "uniform_profile",
]
