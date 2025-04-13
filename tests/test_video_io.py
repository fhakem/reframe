"""Video I/O: fps, duration, rotation, audio passthrough."""

from __future__ import annotations

import subprocess
from fractions import Fraction

import numpy as np
import pytest

from reframe.core.io.video import VideoWriter, probe, read_frames
from tests.conftest import make_clip, needs_ffmpeg

pytestmark = [needs_ffmpeg, pytest.mark.slow]


def _audio_start(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=start_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    return float(out or 0)


def test_probe_and_read(clip_dir):
    src = make_clip(clip_dir / "basic.mp4", rate="30000/1001", seconds=1.0)
    info = probe(src)
    assert (info.width, info.height) == (640, 360)
    assert info.fps == Fraction(30000, 1001)
    assert info.audio_codec == "aac"
    frames = list(read_frames(src, info))
    assert len(frames) == pytest.approx(30, abs=1)
    assert frames[0].shape == (360, 640, 3) and frames[0].dtype == np.uint8


def test_rotation_is_applied(clip_dir):
    src = make_clip(clip_dir / "rot.mp4", seconds=0.4, audio=False, rotation=90)
    info = probe(src)
    assert info.rotation in (90, 270)
    assert (info.width, info.height) == (360, 640)
    frame = next(iter(read_frames(src, info)))
    assert frame.shape == (640, 360, 3)


def test_early_stop_is_not_an_error(clip_dir):
    src = make_clip(clip_dir / "early.mp4", seconds=2.0, audio=False)
    gen = read_frames(src)
    next(gen)
    gen.close()


def test_roundtrip_keeps_fps_duration_audio(clip_dir, tmp_path):
    src = make_clip(clip_dir / "rt.mp4", rate="25", seconds=2.0)
    info = probe(src)
    dst = tmp_path / "out.mp4"
    with VideoWriter(dst, frame_size=(640, 360), fps=info.fps, out_size=(320, 180),
                     audio_from=info, preset="ultrafast") as w:  # fmt: skip
        for f in read_frames(src, info):
            w.write(f)
    out = probe(dst)
    assert (out.width, out.height) == (320, 180)
    assert out.fps == info.fps
    assert out.audio_codec == "aac"  # stream-copied
    one_frame = 1 / float(info.fps)
    assert abs(out.duration - info.duration) <= one_frame + 1e-6
    assert abs(_audio_start(dst) - _audio_start(src)) < one_frame
    assert out.color_space == "bt709"


def test_writer_rejects_bad_frames(tmp_path):
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "x.mp4", frame_size=(10, 10), fps=Fraction(25), out_size=(11, 10))
    w = VideoWriter(tmp_path / "y.mp4", frame_size=(16, 16), fps=Fraction(25))
    with pytest.raises(ValueError):
        w.write(np.zeros((8, 8, 3), dtype=np.uint8))
    w.write(np.zeros((16, 16, 3), dtype=np.uint8))
    w.close()
