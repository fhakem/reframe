"""End-to-end: CLI -> 1080x1920 H.264/AAC (M1 naive baseline)."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from reframe.cli import app
from reframe.config import ReframeConfig
from reframe.core.io.video import probe, read_frames
from reframe.core.pipeline import plan_frame, reframe_frame
from tests.conftest import make_clip, needs_ffmpeg


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5])
def test_plan_geometry(ratio):
    cfg = ReframeConfig(seam_ratio=ratio)
    p = plan_frame(1920, 1080, cfg)
    assert p.target_w == 608 and p.src_h == 1080
    assert p.crop_w % 2 == 0 and p.crop_w <= 1920
    assert p.crop_w == pytest.approx(608 / (1 - ratio), abs=2)
    assert p.factor == 2
    assert reframe_frame(np.zeros((1080, 1920, 3), np.uint8), p, cfg).shape == (1080, 608, 3)


def test_odd_and_narrow_sources():
    cfg = ReframeConfig()
    p = plan_frame(300, 721, cfg)  # already narrower than 9:16: crop height, no stretch
    assert p.target_w == 300 and p.n_seams == 0
    assert p.src_h == 532 and p.crop_y == (721 - 532) // 2
    assert p.target_w / p.src_h == pytest.approx(9 / 16, rel=0.01)
    frame = np.zeros((721, 300, 3), np.uint8)
    assert reframe_frame(frame, p, cfg).shape == (532, 300, 3)
    p = plan_frame(1920, 1081, cfg)
    assert p.src_h == 1080 and p.crop_y == 0


@needs_ffmpeg
@pytest.mark.slow
def test_cli_end_to_end(clip_dir, tmp_path):
    src = make_clip(clip_dir / "e2e.mp4", size="640x360", seconds=1.0)
    dst = tmp_path / "out.mp4"
    res = CliRunner().invoke(
        app, [str(src), "-o", str(dst), "--preset", "tiktok", "--seam-ratio", "0.3"]
    )
    assert res.exit_code == 0, res.output
    out = probe(dst)
    assert (out.width, out.height) == (1080, 1920)
    assert out.audio_codec == "aac"
    assert out.fps == probe(src).fps
    frames = list(read_frames(dst, out))
    assert len(frames) == pytest.approx(25, abs=1)
    assert all(f.mean() > 5 for f in frames)  # no black frames
