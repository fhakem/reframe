from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    deadline=None,  # first call JIT-compiles Numba kernels
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def make_clip(
    path: Path,
    *,
    size: str = "640x360",
    rate: str = "25",
    seconds: float = 2.0,
    audio: bool = True,
    rotation: int | None = None,
) -> Path:
    """Synthetic clip from lavfi testsrc2 (+ 440 Hz sine), no downloads needed."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}",
    ]  # fmt: skip
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
        cmd += ["-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(path)]
    subprocess.run(cmd, check=True)
    if rotation is not None:
        rotated = path.with_name(path.stem + f"_rot{rotation}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-display_rotation:v:0", str(rotation),
             "-i", str(path), "-c", "copy", str(rotated)],
            check=True,
        )  # fmt: skip
        return rotated
    return path


@pytest.fixture(scope="session")
def clip_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("clips")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
