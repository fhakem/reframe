"""Smart pipeline (M2-M4 integration + M6 parallel render)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from reframe.config import ReframeConfig
from reframe.core.contracts import Box, CropPath, FrameSummary, Shot
from reframe.core.io.video import probe, read_frames
from reframe.core.smart import (
    ShotPlan,
    _chunks,
    analyze,
    geometry,
    plan_shots,
    reframe_video_smart,
    render_shots,
    text_width,
    with_center_prior,
)
from tests.conftest import needs_ffmpeg


def make_cut_clip(path: Path, size: str = "320x180") -> Path:
    """1 s testsrc2, hard cut, 1 s smptebars, with a sine audio track."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration=1",
         "-f", "lavfi", "-i", f"smptebars=size={size}:rate=25:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "ultrafast", "-c:a", "aac", str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_text_width_unions_columns():
    boxes = [Box(10, 0, 50, 5, label="text"), Box(40, 0, 60, 5, label="text"), Box(0, 0, 99, 9)]
    assert text_width(boxes, 100) == 50


def test_center_prior_moves_mass_to_centre():
    prof = np.zeros(100)
    prof[0] = prof[50] = 0.5
    out = with_center_prior(FrameSummary(0, [], prof))
    assert out.column_saliency.sum() == pytest.approx(1.0)
    assert out.column_saliency[50] > 3 * out.column_saliency[0]


def test_chunks_are_contiguous_and_balanced():
    path = CropPath(np.zeros(1), 1.0, "stationary")
    plans = [ShotPlan(Shot(i * 10, (i + 1) * 10), 0, path) for i in range(9)]
    groups = _chunks(plans, 3)
    assert [p for g in groups for p in g] == plans
    assert len(groups) == 3
    assert all(len(g) == 3 for g in groups)


@needs_ffmpeg
@pytest.mark.slow
def test_analysis_finds_cut_and_plans_each_shot(clip_dir):
    src = make_cut_clip(clip_dir / "cut.mp4")
    info = probe(src)
    cfg = ReframeConfig(seam_ratio=0.3)
    geo = geometry(info, cfg)
    a = analyze(src, info, geo, cfg)
    assert a.n_frames == 50
    assert [(s.start, s.end) for s in a.shots] == [(0, 25), (25, 50)]
    plans = plan_shots(a, geo, cfg)
    assert len(plans) == 2
    for sp in plans:
        assert sp.path.x.shape == (len(sp.shot),)
        crop_w = geo.target_wa + sp.n_seams
        assert np.all(sp.path.x >= 0) and np.all(sp.path.x <= geo.wa - crop_w + 1e-6)


@needs_ffmpeg
@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_smart_end_to_end(clip_dir, tmp_path, workers):
    src = make_cut_clip(clip_dir / f"cut_e2e{workers}.mp4")
    dst = tmp_path / "out.mp4"
    cfg = ReframeConfig(seam_ratio=0.3, workers=workers, x264_preset="ultrafast")
    seen: list[int] = []
    reframe_video_smart(src, dst, cfg, progress=lambda d, t: seen.append(d))
    out = probe(dst)
    assert (out.width, out.height) == (1080, 1920)
    assert out.audio_codec == "aac"
    frames = list(read_frames(dst, out))
    assert len(frames) == 50
    assert all(f.mean() > 5 for f in frames)
    assert seen and max(seen) == 100  # analysis + render


@needs_ffmpeg
@pytest.mark.slow
def test_split_screen_fallback_renders(clip_dir, tmp_path):
    src = make_cut_clip(clip_dir / "cut_split.mp4")
    info = probe(src)
    cfg = ReframeConfig(x264_preset="ultrafast")
    geo = geometry(info, cfg)
    path = CropPath(np.zeros(25), float(geo.target_wa), "stationary")
    pair = (Box(10, 40, 60, 120, label="face"), Box(250, 40, 300, 120, label="face"))
    plans = [ShotPlan(Shot(0, 25), 0, path, "split", pair), ShotPlan(Shot(25, 50), 0, path)]
    dst = tmp_path / "split.mp4"
    assert render_shots(src, info, geo, cfg, plans, dst, audio=True) == 50
    out = probe(dst)
    assert (out.width, out.height) == (1080, 1920)
    assert len(list(read_frames(dst, out))) == 50
