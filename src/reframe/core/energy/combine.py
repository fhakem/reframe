"""Combine classical cues into :class:`~reframe.core.contracts.SemanticMaps`.

``energy`` is in **forward units** (see :mod:`reframe.core.contracts`): each
[0, 1] cue is multiplied by a weight of order 100..5000 so it competes with the
forward-energy step costs (~0..510 per pixel). ``protect`` is a hard mask built
from face / person / text boxes grown by a margin; the seam DP turns it into
``carve.PROTECT_COST``.

Every cue sits behind a small protocol (``SaliencyEstimator``, ``BoxDetector``,
``SegmentDetector``) so learned models can be injected into
:class:`ClassicalAnalyzer` without touching this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from reframe.core.contracts import Box, BoxLabel, FrameSummary, SemanticMaps, uniform_profile
from reframe.core.energy.detect import (
    BoxDetector,
    HaarFaceDetector,
    PersonDetector,
    boxes_to_mask,
    boxes_to_soft,
)
from reframe.core.energy.lines import LineMap
from reframe.core.energy.saliency import ClassicalSaliency, SaliencyEstimator
from reframe.core.energy.text import GlyphDetector, MserTextDetector, text_pixel_mask
from reframe.core.types import BoolArray, FloatArray, ImageArray


class SemanticConfig(BaseModel):
    """Weights (forward units per unit of [0, 1] map) and toggles for the classical analyzer."""

    saliency_weight: float = Field(
        400.0, ge=0, description="Salient pixels cost about as much as a strong edge."
    )
    line_weight: float = Field(
        800.0, ge=0, description="Crossing a long straight line costs more than a strong edge."
    )
    box_weight: float = Field(
        1500.0, ge=0, description="Soft cost around detected boxes (beyond the hard margin)."
    )
    gradient_boost: float = Field(
        0.0, ge=0, description="Extra weight on the normalised gradient magnitude (0 = off)."
    )
    faces: bool = True
    people: bool = True
    text: bool = True
    protect_labels: tuple[BoxLabel, ...] = ("face", "person", "text")
    protect_margin: float = Field(0.1, ge=0, description="Box growth, fraction of its size.")
    protect_margin_px: float = Field(2.0, ge=0)
    text_protect: Literal["pixels", "glyphs", "lines"] = Field(
        "pixels",
        description=(
            "'pixels': protect the ink inside each text line (Otsu), so seams can still remove "
            "letter / word spacing, needed when a title is wider than the output; 'glyphs': "
            "protect each detected character box; 'lines': protect whole line boxes."
        ),
    )
    glyph_margin: float = Field(0.08, ge=0, description="Glyph box growth, fraction of size.")
    glyph_margin_px: float = Field(1.0, ge=0)
    max_protect_frac: float = Field(
        0.5,
        gt=0,
        le=1,
        description=(
            "Labels are added to the hard mask in priority order (text, face, person); a label "
            "that would push the protected area past this fraction of the frame is skipped "
            "(it still gets the soft box energy). Text is always kept."
        ),
    )
    column_box_boost: float = Field(
        0.5, ge=0, description="Profile mass added per box (x score), relative to saliency (1)."
    )


#: Hard-protection priority: earlier labels are kept when the area cap is hit.
_PROTECT_PRIORITY: tuple[BoxLabel, ...] = ("text", "face", "person", "manual", "salient")


class ClassicalAnalyzer:
    """Weight-free :class:`~reframe.core.contracts.SemanticAnalyzer` built on OpenCV."""

    def __init__(
        self,
        cfg: SemanticConfig | None = None,
        *,
        saliency: SaliencyEstimator | None = None,
        lines: Callable[[ImageArray], FloatArray] | None = None,
        face_detector: BoxDetector | None = None,
        person_detector: BoxDetector | None = None,
        text_detector: BoxDetector | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else SemanticConfig()
        c = self.cfg
        self.saliency = saliency if saliency is not None else ClassicalSaliency()
        self.lines = lines if lines is not None else (LineMap() if c.line_weight > 0 else None)
        self.detectors: list[BoxDetector] = []
        if c.faces:
            self.detectors.append(face_detector or HaarFaceDetector())
        if c.people:
            self.detectors.append(person_detector or PersonDetector())
        self.text_detector: BoxDetector | None = None
        if c.text:
            self.text_detector = text_detector or MserTextDetector()
            self.detectors.append(self.text_detector)

    def detect(self, frame: ImageArray) -> tuple[list[Box], list[Box] | None]:
        """Return ``(boxes, glyphs)``; ``glyphs`` is None unless glyph-level text is available."""
        boxes: list[Box] = []
        glyphs: list[Box] | None = None
        for d in self.detectors:
            if (
                d is self.text_detector
                and self.cfg.text_protect == "glyphs"
                and isinstance(d, GlyphDetector)
            ):
                lines, glyphs = d.detect_glyphs(frame)
                boxes.extend(lines)
            else:
                boxes.extend(d.detect(frame))
        return boxes, glyphs

    def text_mask(
        self, frame: ImageArray, boxes: Sequence[Box], glyphs: Sequence[Box] | None
    ) -> BoolArray | None:
        """Text protection per ``cfg.text_protect``; None means "use the line boxes"."""
        c = self.cfg
        lines = [b for b in boxes if b.label == "text"]
        h, w = np.asarray(frame).shape[:2]
        if c.text_protect == "pixels":
            return text_pixel_mask(frame, lines) if lines else np.zeros((h, w), dtype=bool)
        if c.text_protect == "glyphs" and glyphs is not None:
            return boxes_to_mask(glyphs, (h, w), margin=c.glyph_margin, margin_px=c.glyph_margin_px)
        return None

    def protect_mask(
        self, boxes: Sequence[Box], shape: tuple[int, int], text_mask: BoolArray | None = None
    ) -> BoolArray:
        """Hard mask from boxes (+ margin); text uses ``text_mask`` when given."""
        c = self.cfg
        mask = np.zeros(shape, dtype=bool)
        cap = c.max_protect_frac * shape[0] * shape[1]
        for label in _PROTECT_PRIORITY:
            if label not in c.protect_labels:
                continue
            if label == "text" and text_mask is not None:
                if not text_mask.any():
                    continue
                m = text_mask
            else:
                sel = [b for b in boxes if b.label == label]
                if not sel:
                    continue
                m = boxes_to_mask(
                    sel, shape, margin=c.protect_margin, margin_px=c.protect_margin_px
                )
            merged = mask | m
            if label != "text" and merged.sum() > cap:
                continue  # too much to hard-protect: left to the soft box energy
            mask = merged
        return mask

    def analyze(self, frame: ImageArray) -> SemanticMaps:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"expected an (h, w, 3) RGB frame, got {arr.shape}")
        h, w = arr.shape[:2]
        c = self.cfg
        sal = self.saliency(arr)
        energy = c.saliency_weight * sal
        if self.lines is not None and c.line_weight > 0:
            energy += c.line_weight * self.lines(arr)
        if c.gradient_boost > 0:
            energy += c.gradient_boost * _grad_norm(arr)
        boxes, glyphs = self.detect(arr)
        if boxes and c.box_weight > 0:
            energy += c.box_weight * boxes_to_soft(boxes, (h, w))
        tmask = self.text_mask(arr, boxes, glyphs) if "text" in c.protect_labels else None
        protect = self.protect_mask(boxes, (h, w), tmask)
        energy = np.nan_to_num(np.maximum(energy, 0.0), nan=0.0, posinf=0.0)
        return SemanticMaps(energy=energy, protect=protect, saliency=sal, boxes=boxes)

    def summarize(self, index: int, maps: SemanticMaps) -> FrameSummary:
        return summarize(index, maps, self.cfg.column_box_boost)


def _grad_norm(frame: ImageArray) -> FloatArray:
    gray = cv2.cvtColor(np.ascontiguousarray(frame[..., :3], np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(gx) + np.abs(gy)
    return np.clip(mag.astype(np.float64) / (8.0 * 255.0), 0.0, 1.0)


def column_profile(maps: SemanticMaps, box_boost: float = 0.5) -> FloatArray:
    """Saliency mass per column, normalised to sum 1 (uniform if empty).

    Each box adds ``box_boost * score`` of mass (the saliency part totals 1),
    spread uniformly over its columns, so detected subjects pull the crop.
    """
    sal = np.nan_to_num(np.asarray(maps.saliency, dtype=np.float64))
    w = sal.shape[1]
    col = sal.sum(axis=0)
    total = float(col.sum())
    prof = col / total if total > 0 else np.zeros(w, dtype=np.float64)
    for b in maps.boxes:
        x0, x1 = max(0, int(np.floor(b.x0))), min(w, int(np.ceil(b.x1)))
        if x1 > x0:
            prof[x0:x1] += box_boost * max(b.score, 0.0) / (x1 - x0)
    s = float(prof.sum())
    return prof / s if s > 0 else uniform_profile(w)


def summarize(index: int, maps: SemanticMaps, box_boost: float = 0.5) -> FrameSummary:
    return FrameSummary(
        index=index, boxes=list(maps.boxes), column_saliency=column_profile(maps, box_boost)
    )


__all__ = ["ClassicalAnalyzer", "SemanticConfig", "column_profile", "summarize"]
