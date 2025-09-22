"""Weight-free text / title detection.

Pipeline (classical, after Neumann & Matas / Chen et al.):

1. **MSER** on the luminance (both polarities come out of OpenCV's MSER) at a
   reduced width: stable extremal regions are character candidates.
2. **Character filters**: size, aspect ratio, fill ratio, and a stroke-width
   test (distance transform: letters have thin, uniform strokes relative to
   their height; blobs do not).
3. **Line grouping**: candidates of similar height that sit on the same
   baseline band and are close horizontally are chained with union-find. Only
   chains of at least ``min_chars`` survive, which rejects isolated blobs and
   most texture.

4. **Flat-card stage**: logos and heavy display fonts often have touching
   letters (one MSER per word, rejected as a glyph). When the frame is a
   "card" -- a dominant flat background agreeing with the frame border -- every
   non-background component is graphics/text and is reported as a text box.
   This is what catches the Big Buck Bunny logo card.

A learned detector (EAST, DBNet, CRAFT...) can replace it via
:class:`reframe.core.energy.detect.BoxDetector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from reframe.core.contracts import Box
from reframe.core.energy.saliency import to_gray_u8
from reframe.core.types import ImageArray


@dataclass(frozen=True)
class _Cand:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@runtime_checkable
class GlyphDetector(Protocol):
    """A text detector that can also return per-character boxes.

    Protecting glyphs rather than whole lines lets seams remove the spacing
    between letters and words, which matters when a title is wider than the
    output frame.
    """

    def detect(self, frame: ImageArray) -> list[Box]: ...

    def detect_glyphs(self, frame: ImageArray) -> tuple[list[Box], list[Box]]: ...


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


class MserTextDetector:
    """Detect horizontal text lines as ``Box(label="text")``."""

    def __init__(
        self,
        *,
        work_width: int = 640,
        min_char_h: int = 6,
        max_char_h_frac: float = 0.3,
        min_chars: int = 3,
        max_stroke_ratio: float = 0.35,
        min_fill: float = 0.12,
        max_fill: float = 0.95,
        flat_card: bool = True,
        card_width: int = 320,
        card_tol: float = 30.0,
        card_min_bg: float = 0.55,
        card_max_fill: float = 0.72,
    ) -> None:
        self.card_max_fill = card_max_fill
        self.flat_card = flat_card
        self.card_width = card_width
        self.card_tol = card_tol
        self.card_min_bg = card_min_bg
        self.work_width = work_width
        self.min_char_h = min_char_h
        self.max_char_h_frac = max_char_h_frac
        self.min_chars = min_chars
        self.max_stroke_ratio = max_stroke_ratio
        self.min_fill = min_fill
        self.max_fill = max_fill
        self._mser = cv2.MSER.create(delta=5, min_area=12, max_variation=0.25)

    # -- stage 1+2 ---------------------------------------------------------------

    def _candidates(self, gray: np.ndarray) -> list[_Cand]:
        h, w = gray.shape
        regions, bboxes = self._mser.detectRegions(gray)
        max_h = self.max_char_h_frac * h
        out: list[_Cand] = []
        seen: set[tuple[int, int, int, int]] = set()
        for region, (x, y, bw, bh) in zip(regions, bboxes, strict=False):
            pts = np.asarray(region).reshape(-1, 2)
            key = (int(x) // 2, int(y) // 2, int(bw) // 2, int(bh) // 2)
            if key in seen:
                continue  # MSER returns near-identical nested regions
            seen.add(key)
            if bh < self.min_char_h or bh > max_h or bw > 0.5 * w:
                continue
            aspect = bw / bh
            if not 0.08 <= aspect <= 2.5:
                continue
            fill = len(pts) / float(bw * bh)
            if not self.min_fill <= fill <= self.max_fill:
                continue
            # Stroke width: high percentile of the distance transform over the region mask.
            m = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
            m[pts[:, 1] - y + 1, pts[:, 0] - x + 1] = 255
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            vals = dt[m > 0]
            if vals.size == 0:
                continue
            stroke = 2.0 * float(np.percentile(vals, 90))
            if stroke > self.max_stroke_ratio * max(bh, bw) and fill > 0.6:
                continue  # solid blob, not a glyph
            out.append(_Cand(int(x), int(y), int(x + bw), int(y + bh)))
        return self._drop_contained(out)

    @staticmethod
    def _drop_contained(c: list[_Cand]) -> list[_Cand]:
        """Remove candidates nested inside a larger one (letter counters, nested MSERs)."""
        c = sorted(c, key=lambda b: -(b.x1 - b.x0) * b.h)
        keep: list[_Cand] = []
        for b in c:
            if not any(
                k.x0 <= b.x0 and k.y0 <= b.y0 and k.x1 >= b.x1 and k.y1 >= b.y1 for k in keep
            ):
                keep.append(b)
        return keep

    # -- stage 3 -----------------------------------------------------------------

    def _group(self, c: list[_Cand]) -> list[list[_Cand]]:
        n = len(c)
        parent = list(range(n))
        order = sorted(range(n), key=lambda i: c[i].x0)
        hmax = max((b.h for b in c), default=0)
        for ii, i in enumerate(order):
            a = c[i]
            for j in order[ii + 1 :]:
                b = c[j]
                mh = max(a.h, b.h)
                if b.x0 - a.x1 > 1.2 * mh:
                    if b.x0 - a.x1 > 1.2 * hmax:
                        break  # sorted by x0: nothing further right can be close
                    continue
                if max(a.h, b.h) > 2.0 * min(a.h, b.h):
                    continue
                if abs(a.cy - b.cy) > 0.5 * mh:
                    continue
                parent[_find(parent, i)] = _find(parent, j)
        groups: dict[int, list[_Cand]] = {}
        for i in range(n):
            groups.setdefault(_find(parent, i), []).append(c[i])
        return [g for g in groups.values() if len(g) >= self.min_chars and self._is_line(g)]

    @staticmethod
    def _is_line(g: list[_Cand]) -> bool:
        """A text line: one band of similar-height glyphs sharing a baseline or a top line."""
        hs = np.array([b.h for b in g], dtype=np.float64)
        mh = float(np.median(hs))
        span = max(b.y1 for b in g) - min(b.y0 for b in g)
        if span > 1.8 * mh or float(hs.std()) > 0.35 * mh:
            return False
        tops = np.array([b.y0 for b in g], dtype=np.float64)
        bottoms = np.array([b.y1 for b in g], dtype=np.float64)
        if min(float(tops.std()), float(bottoms.std())) > 0.15 * mh:
            return False
        # Glyphs are dense along the line: their widths cover much of its extent.
        extent = max(b.x1 for b in g) - min(b.x0 for b in g)
        return sum(b.x1 - b.x0 for b in g) >= 0.45 * extent

    def card_boxes(self, frame: ImageArray) -> list[Box]:
        """Foreground components of a flat-background card ([] if not a card)."""
        arr = np.asarray(frame)
        if arr.ndim != 3:
            arr = np.repeat(arr[..., None], 3, axis=2)
        h, w = arr.shape[:2]
        sw = min(self.card_width, w)
        sh = max(8, round(h * sw / w))
        small = cv2.resize(
            np.ascontiguousarray(arr[..., :3], np.uint8), (sw, sh), interpolation=cv2.INTER_AREA
        ).astype(np.int16)
        b = 3
        ring = np.concatenate(
            [
                small[:b].reshape(-1, 3),
                small[-b:].reshape(-1, 3),
                small[:, :b].reshape(-1, 3),
                small[:, -b:].reshape(-1, 3),
            ]
        )
        bg = np.median(ring, axis=0)
        if (np.abs(ring - bg).max(axis=1) < self.card_tol).mean() < 0.8:
            return []  # the border is not one flat colour: not a card
        fg = np.abs(small - bg).max(axis=2) >= self.card_tol
        if 1.0 - fg.mean() < self.card_min_bg:
            return []
        fg_u8 = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        k = max(3, round(0.02 * sw))
        joined = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, np.ones((k, 2 * k), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        s = w / sw
        out: list[Box] = []
        for i in range(1, n):
            x, y, cw, ch, area = (int(v) for v in stats[i])
            if ch < 4 or area < 0.0005 * sw * sh or cw * ch > 0.6 * sw * sh:
                continue
            # Lettering leaves holes and gaps; solid shapes (discs, panels) do not.
            if float(fg_u8[y : y + ch, x : x + cw].mean()) > self.card_max_fill:
                continue
            out.append(Box(x * s, y * s, min((x + cw) * s, w), min((y + ch) * s, h), 0.9, "text"))
        return out

    def detect_glyphs(self, frame: ImageArray) -> tuple[list[Box], list[Box]]:
        """Return ``(lines, glyphs)``: one box per text line and one per character."""
        gray = to_gray_u8(frame)
        h, w = gray.shape
        if int(gray.max()) == int(gray.min()):
            return [], []
        s = min(1.0, self.work_width / w)
        small = gray
        if s < 1.0:
            size = (self.work_width, max(1, round(h * s)))
            small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        lines: list[Box] = []
        glyphs: list[Box] = []
        for g in self._group(self._candidates(small)):
            score = float(min(1.0, 0.4 + 0.1 * len(g)))
            x0, y0 = min(b.x0 for b in g), min(b.y0 for b in g)
            x1, y1 = max(b.x1 for b in g), max(b.y1 for b in g)
            lines.append(Box(x0 / s, y0 / s, min(x1 / s, w), min(y1 / s, h), score, "text"))
            glyphs += [
                Box(b.x0 / s, b.y0 / s, min(b.x1 / s, w), min(b.y1 / s, h), score, "text")
                for b in g
            ]
        if self.flat_card:
            covered = np.zeros((h, w), dtype=bool)
            for b in lines:
                covered[int(b.y0) : int(np.ceil(b.y1)), int(b.x0) : int(np.ceil(b.x1))] = True
            for b in self.card_boxes(frame):
                region = covered[int(b.y0) : int(np.ceil(b.y1)), int(b.x0) : int(np.ceil(b.x1))]
                if region.size and region.mean() < 0.7:  # not already found by MSER
                    lines.append(b)
                    glyphs.append(b)
        return lines, glyphs

    def detect(self, frame: ImageArray) -> list[Box]:
        return self.detect_glyphs(frame)[0]


def text_pixel_mask(
    frame: ImageArray,
    lines: list[Box],
    *,
    pad: float = 0.15,
    dilate_frac: float = 0.04,
    max_dilate: int = 3,
) -> np.ndarray:
    """Bool mask of the *ink* pixels inside each text line box.

    Within each line box (grown by ``pad`` x its height), Otsu splits the pixels
    in two; the class that does not dominate the box border is the ink. It is
    dilated by ``dilate_frac`` x line height (1..``max_dilate`` px) to cover
    anti-aliasing. Unlike glyph boxes this does
    not depend on every character being found, and unlike line boxes it leaves
    the letter and word spacing free for seams.
    """
    gray = to_gray_u8(frame)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    for b in lines:
        lh = b.y1 - b.y0
        x0, y0 = max(0, int(b.x0 - pad * lh)), max(0, int(b.y0 - pad * lh))
        x1, y1 = min(w, int(np.ceil(b.x1 + pad * lh))), min(h, int(np.ceil(b.y1 + pad * lh)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = np.ascontiguousarray(gray[y0:y1, x0:x1])
        _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = th > 0
        # Polarity: the background is what dominates the ROI's border ring (a heavy
        # font in a tight box can have more ink than background).
        ring = np.concatenate([ink[0], ink[-1], ink[:, 0], ink[:, -1]])
        if ring.mean() > 0.5:
            ink = ~ink
        r = int(np.clip(round(dilate_frac * lh), 1, max_dilate))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ink_d = cv2.dilate(ink.astype(np.uint8), k) > 0
        mask[y0:y1, x0:x1] |= ink_d
    return mask


__all__ = ["GlyphDetector", "MserTextDetector", "text_pixel_mask"]
