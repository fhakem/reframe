"""M2 semantic energy: saliency, lines, text, detection plumbing, combination."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reframe.core.contracts import Box, FrameSummary, SemanticAnalyzer, SemanticMaps
from reframe.core.energy.combine import (
    ClassicalAnalyzer,
    SemanticConfig,
    column_profile,
    summarize,
)
from reframe.core.energy.detect import (
    HaarFaceDetector,
    PersonDetector,
    boxes_to_mask,
    boxes_to_soft,
    iou,
    nms,
)
from reframe.core.energy.lines import LineMap
from reframe.core.energy.saliency import ClassicalSaliency, robust_normalize
from reframe.core.energy.text import MserTextDetector, text_pixel_mask
from reframe.core.seams import carve_width

BLUE = (40, 90, 210)


def title_card(h: int = 180, w: int = 320, text: str = "BUNNY", scale: float = 2.2) -> np.ndarray:
    """Big blue letters on white, like the Big Buck Bunny title cards."""
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(2, int(scale * 3))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(img, text, org, font, scale, BLUE, thick, cv2.LINE_AA)
    return img


def letters_of(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.int16).sum(axis=2) < 600


class FixedBoxes:
    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes

    def detect(self, frame: np.ndarray) -> list[Box]:
        return list(self.boxes)


@pytest.fixture(scope="module")
def analyzer() -> ClassicalAnalyzer:
    return ClassicalAnalyzer()


# ------------------------------------------------------------------ saliency


def test_robust_normalize_flat_and_nan():
    assert np.all(robust_normalize(np.full((5, 7), 3.0)) == 0)
    m = np.zeros((4, 4))
    m[0, 0] = np.nan
    m[1, 1] = 5.0
    out = robust_normalize(m)
    assert np.isfinite(out).all() and out.max() == 1.0 and out.min() == 0.0


@pytest.mark.parametrize("value", [0, 128, 255])
def test_saliency_flat_frame_is_zero(value):
    sal = ClassicalSaliency()(np.full((90, 160, 3), value, dtype=np.uint8))
    assert sal.shape == (90, 160)
    assert np.isfinite(sal).all() and np.all(sal == 0)


@settings(max_examples=15)
@given(st.integers(8, 120), st.integers(8, 200), st.integers(0, 2**32 - 1))
def test_saliency_range_and_shape(h, w, seed):
    img = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    sal = ClassicalSaliency()(img)
    assert sal.shape == (h, w) and sal.dtype == np.float64
    assert np.isfinite(sal).all() and sal.min() >= 0 and sal.max() <= 1


def test_saliency_peaks_on_blob(rng):
    img = np.full((270, 480, 3), 90, dtype=np.uint8)
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)  # mild sensor noise
    cv2.circle(img, (360, 120), 30, (250, 40, 40), -1)
    sal = ClassicalSaliency()(img)
    yy, xx = np.mgrid[:270, :480]
    inside = (xx - 360) ** 2 + (yy - 120) ** 2 <= 30**2
    far = (xx - 360) ** 2 + (yy - 120) ** 2 >= 120**2
    assert sal[inside].mean() > 3 * sal[far].mean()
    py, px = np.unravel_index(np.argmax(sal), sal.shape)
    assert abs(px - 360) < 45 and abs(py - 120) < 45


# --------------------------------------------------------------------- lines


def test_line_map_lights_up_on_lines():
    img = np.full((270, 480, 3), 120, dtype=np.uint8)
    cv2.line(img, (20, 200), (460, 180), (255, 255, 255), 2)  # long near-horizontal
    cv2.line(img, (100, 20), (100, 150), (0, 0, 0), 2)  # vertical
    lm = LineMap()(img)
    assert lm.shape == (270, 480) and lm.min() >= 0 and lm.max() <= 1
    assert lm[190, 240] > 0.5 or lm[189, 240] > 0.5 or lm[191, 240] > 0.5
    assert lm[85, 98:103].max() > 0.3
    assert lm[40:120, 250:450].max() == 0  # nothing where nothing is drawn


def test_line_map_flat_and_noise_free():
    assert np.all(LineMap()(np.zeros((60, 80, 3), dtype=np.uint8)) == 0)
    lsd = LineMap(method="lsd")(np.zeros((60, 80, 3), dtype=np.uint8))
    assert np.all(lsd == 0)


def test_short_segments_are_ignored():
    img = np.full((270, 480, 3), 120, dtype=np.uint8)
    cv2.line(img, (200, 100), (210, 100), (255, 255, 255), 2)  # 10 px << min length
    assert LineMap()(img).max() == 0


# ---------------------------------------------------------------------- text


@pytest.mark.parametrize(
    ("text", "scale", "size"),
    [
        ("BUNNY", 2.2, (180, 320)),
        ("Big Buck Bunny", 3.0, (540, 960)),
        ("subtitle", 0.8, (270, 480)),
    ],
)
def test_text_detected_on_title_cards(text, scale, size):
    img = title_card(size[0], size[1], text=text, scale=scale)
    boxes = MserTextDetector().detect(img)
    assert boxes and all(b.label == "text" for b in boxes)
    letters = letters_of(img)
    mask = boxes_to_mask(boxes, letters.shape, margin=0.0, margin_px=2)
    # Nearly every letter pixel lies inside a detected text box.
    assert mask[letters].mean() > 0.97


def logo_card(h: int = 270, w: int = 480) -> np.ndarray:
    """Heavy touching letters on two lines (like the Big Buck Bunny logo)."""
    img = np.full((h, w, 3), 252, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Big Buck", (90, 120), font, 2.4, BLUE, 22, cv2.LINE_AA)
    cv2.putText(img, "BUNNY", (100, 210), font, 2.8, BLUE, 26, cv2.LINE_AA)
    return img


def test_touching_logo_letters_detected_via_flat_card():
    img = logo_card()
    letters = letters_of(img)
    det = MserTextDetector()
    boxes = det.detect(img)
    assert boxes
    assert boxes_to_mask(boxes, letters.shape, margin=0.0, margin_px=1)[letters].mean() > 0.99
    # The flat-card stage alone covers the lettering (it does not need separable glyphs).
    cards = det.card_boxes(img)
    assert cards and all(b.label == "text" for b in cards)
    assert boxes_to_mask(cards, letters.shape, margin=0.0, margin_px=1)[letters].mean() > 0.9


def test_flat_card_not_fired_on_textured_frames(rng):
    det = MserTextDetector()
    noise = cv2.GaussianBlur(rng.integers(0, 256, (270, 480, 3), dtype=np.uint8), (0, 0), 3)
    assert det.card_boxes(noise) == []
    grad = np.repeat(np.linspace(0, 255, 480, dtype=np.uint8)[None, :, None], 270, 0)
    assert det.card_boxes(np.repeat(grad, 3, axis=2)) == []


def test_text_pixel_mask_heavy_font_polarity():
    # Ink covers most of a tight box: polarity must come from the border, not the majority.
    img = np.full((120, 240, 3), 250, dtype=np.uint8)
    cv2.rectangle(img, (20, 30), (220, 90), BLUE, -1)
    cv2.line(img, (120, 30), (120, 90), (250, 250, 250), 2)  # a thin gap
    ink = letters_of(img)
    assert ink[30:90, 20:220].mean() > 0.9
    m = text_pixel_mask(img, [Box(20, 30, 221, 91, 1.0, "text")])
    assert m[ink].all()
    assert not m[5, 5] and not m[110, 230]


def test_logo_carve_keeps_letters():
    img = logo_card()
    letters = letters_of(img)
    maps = ClassicalAnalyzer().analyze(img)
    assert maps.protect[letters].all()
    n = 480 - int(maps.protect.any(axis=0).sum()) - 10  # feasible: fewer than free columns
    assert n > 40
    assert int((carve_width(img, n).order[letters] >= 0).sum()) > 0
    sem = carve_width(img, n, protect=maps.protect, extra_energy=maps.energy)
    assert int((sem.order[letters] >= 0).sum()) == 0


def test_text_light_on_dark():
    img = np.full((270, 480, 3), 20, dtype=np.uint8)
    cv2.putText(img, "THE END", (60, 160), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 5)
    boxes = MserTextDetector().detect(img)
    assert boxes
    letters = img.sum(axis=2) > 300
    assert boxes_to_mask(boxes, letters.shape, margin=0.0)[letters].mean() > 0.95


def test_text_not_fired_on_blobs_and_flat(rng):
    det = MserTextDetector()
    assert det.detect(np.full((120, 200, 3), 77, dtype=np.uint8)) == []
    img = np.full((270, 480, 3), 200, dtype=np.uint8)
    cv2.circle(img, (120, 130), 60, (200, 30, 30), -1)
    cv2.rectangle(img, (300, 60), (420, 200), (20, 160, 20), -1)
    assert det.detect(img) == []
    noise = cv2.GaussianBlur(rng.integers(0, 256, (270, 480, 3), dtype=np.uint8), (0, 0), 4)
    assert det.detect(noise) == []


# ----------------------------------------------------------------- detection


def test_iou_and_nms():
    a = Box(0, 0, 10, 10, 0.9, "face")
    b = Box(1, 1, 11, 11, 0.5, "face")
    c = Box(1, 1, 11, 11, 0.5, "person")  # other label: never suppressed by a face
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, Box(20, 20, 30, 30)) == 0.0
    kept = nms([b, a, c], 0.4)
    assert a in kept and c in kept and b not in kept


def test_boxes_to_mask_margin_and_clipping():
    m = boxes_to_mask([Box(10, 10, 20, 30)], (40, 40), margin=0.5, margin_px=0)
    ys, xs = np.nonzero(m)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (5, 24, 0, 39)
    edge = boxes_to_mask([Box(-5, -5, 3, 3)], (10, 10), margin=0, margin_px=0)
    assert edge[:3, :3].all() and edge.sum() == 9
    assert not boxes_to_mask([], (5, 5)).any()


def test_boxes_to_soft_range():
    s = boxes_to_soft([Box(10, 10, 30, 30, 0.7)], (50, 60))
    assert s.shape == (50, 60) and s.min() >= 0 and s.max() <= 0.7 + 1e-6
    assert s[20, 20] > 0.6 and s[45, 55] < 0.05


def test_builtin_detectors_run_and_are_quiet_on_flat():
    flat = np.full((270, 480, 3), 128, dtype=np.uint8)
    assert HaarFaceDetector().detect(flat) == []
    assert PersonDetector().detect(flat) == []


# ------------------------------------------------------------------- combine


def test_analyzer_is_a_semantic_analyzer(analyzer):
    a: SemanticAnalyzer = analyzer
    assert hasattr(a, "analyze")


@pytest.mark.parametrize("value", [0, 255])
def test_analyzer_flat_frames(analyzer, value):
    maps = analyzer.analyze(np.full((135, 240, 3), value, dtype=np.uint8))
    assert maps.energy.shape == maps.protect.shape == maps.saliency.shape == (135, 240)
    assert np.isfinite(maps.energy).all() and np.all(maps.energy == 0)
    assert not maps.protect.any() and maps.boxes == []


def test_analyzer_ranges_on_noise(analyzer, rng):
    img = cv2.GaussianBlur(rng.integers(0, 256, (135, 240, 3), dtype=np.uint8), (0, 0), 2)
    maps = analyzer.analyze(img)
    assert maps.protect.dtype == np.bool_
    assert np.isfinite(maps.energy).all() and maps.energy.min() >= 0
    cfg = analyzer.cfg
    bound = cfg.saliency_weight + cfg.line_weight + cfg.box_weight + cfg.gradient_boost
    assert maps.energy.max() <= bound + 1e-6
    assert maps.saliency.min() >= 0 and maps.saliency.max() <= 1


def test_energy_is_in_forward_units():
    img = np.full((135, 240, 3), 90, dtype=np.uint8)
    cv2.circle(img, (60, 60), 20, (255, 255, 0), -1)
    maps = ClassicalAnalyzer(SemanticConfig(faces=False, people=False, text=False)).analyze(img)
    # A raw [0, 1] map would be useless against forward costs of ~0..510.
    assert maps.energy.max() > 100


def test_injected_face_box_is_protected_with_margin():
    face = Box(100, 40, 140, 90, 0.9, "face")
    a = ClassicalAnalyzer(
        SemanticConfig(protect_margin=0.25, protect_margin_px=0),
        face_detector=FixedBoxes([face]),
        person_detector=FixedBoxes([]),
        text_detector=FixedBoxes([]),
    )
    img = np.full((135, 240, 3), 100, dtype=np.uint8)
    maps = a.analyze(img)
    assert maps.boxes == [face]
    assert maps.protect[40:90, 100:140].all()
    assert maps.protect[30, 92] and maps.protect[100, 148]  # margin of 10 x 12.5 px
    assert not maps.protect[20, 60]
    assert maps.energy[65, 120] > 0.5 * a.cfg.box_weight  # soft cost too


def test_protect_labels_toggle_and_area_cap():
    big = Box(0, 0, 200, 135, 1.0, "person")
    text = Box(10, 10, 60, 30, 1.0, "text")
    img = np.full((135, 240, 3), 100, dtype=np.uint8)

    def make(cfg: SemanticConfig, people: list[Box]) -> ClassicalAnalyzer:
        return ClassicalAnalyzer(
            cfg,
            face_detector=FixedBoxes([]),
            person_detector=FixedBoxes(people),
            text_detector=FixedBoxes([text]),
        )

    capped = make(SemanticConfig(text_protect="lines"), [big])
    m = capped.analyze(img).protect
    assert m[20, 30] and not m[100, 150]  # person dropped: would cover > 50 %
    no_text = make(SemanticConfig(protect_labels=("face",), text_protect="lines"), [])
    assert not no_text.analyze(img).protect.any()


def test_text_pixel_mask_protects_ink_not_spacing():
    img = title_card(180, 320, "BUNNY", 1.6)
    letters = letters_of(img)
    lines = MserTextDetector().detect(img)
    m = text_pixel_mask(img, lines)
    assert m[letters].all()
    line_area = boxes_to_mask(lines, letters.shape, margin=0.0, margin_px=0)
    # Gaps between letters stay free: a large part of the line box is unprotected.
    assert (line_area & ~m).sum() > 0.15 * line_area.sum()
    # Columns between letters are fully free top to bottom (seams can pass).
    assert (~m.any(axis=0)).sum() > 320 - (lines[0].x1 - lines[0].x0)


@pytest.mark.parametrize("mode", ["pixels", "glyphs", "lines"])
def test_text_protect_modes_cover_letters(mode):
    img = title_card(180, 320, "BUNNY", 1.6)
    maps = ClassicalAnalyzer(SemanticConfig(text_protect=mode)).analyze(img)
    assert maps.protect[letters_of(img)].all()


def test_detector_toggles():
    a = ClassicalAnalyzer(SemanticConfig(faces=False, people=False, text=False, line_weight=0))
    assert a.detectors == [] and a.lines is None


def test_column_profile_and_summary():
    sal = np.zeros((10, 20))
    sal[:, 15] = 1.0
    maps = SemanticMaps(
        energy=np.zeros((10, 20)),
        protect=np.zeros((10, 20), bool),
        saliency=sal,
        boxes=[Box(0, 0, 4, 10, 1.0, "face")],
    )
    p = column_profile(maps, box_boost=0.5)
    assert p.shape == (20,) and p.sum() == pytest.approx(1.0)
    assert p[15] == pytest.approx(1 / 1.5) and p[0:4].sum() == pytest.approx(0.5 / 1.5)
    empty = SemanticMaps(np.zeros((3, 8)), np.zeros((3, 8), bool), np.zeros((3, 8)))
    np.testing.assert_allclose(column_profile(empty), np.full(8, 1 / 8))
    s = summarize(7, maps)
    assert isinstance(s, FrameSummary) and s.index == 7 and s.boxes == maps.boxes
    np.testing.assert_allclose(s.column_saliency, p)


# --------------------------------------------------------------- integration


def test_carve_keeps_title_intact_with_semantic_energy(analyzer):
    # The protected title (+ margin) leaves ~127 free columns; forward energy alone
    # starts cutting letters once the gaps between them are used up.
    img = title_card(180, 320, "BUNNY", 1.6)
    letters = letters_of(img)
    n = 100
    plain = carve_width(img, n)
    assert (plain.order[letters] >= 0).sum() > 0  # forward energy alone cuts letters

    maps = analyzer.analyze(img)
    assert maps.protect[letters].all()
    sem = carve_width(img, n, protect=maps.protect, extra_energy=maps.energy)
    assert (sem.order[letters] >= 0).sum() == 0  # every letter pixel survives
    assert sem.image.shape == (180, 320 - n, 3)


def test_wide_title_keeps_letters_by_removing_spacing():
    # The line box is wider than the output (320 - 100 = 220 px): protecting the
    # whole line forces seams through letters, protecting the ink does not.
    img = title_card(180, 320, "ONE BIG RABBIT", 1.0)
    letters = letters_of(img)
    n = 100

    def cut(mode: str) -> int:
        maps = ClassicalAnalyzer(SemanticConfig(text_protect=mode)).analyze(img)
        assert maps.boxes and maps.boxes[0].width > 320 - n
        res = carve_width(img, n, protect=maps.protect, extra_energy=maps.energy)
        return int((res.order[letters] >= 0).sum())

    assert int((carve_width(img, n).order[letters] >= 0).sum()) > 0
    assert cut("lines") > 0
    assert cut("pixels") == 0
