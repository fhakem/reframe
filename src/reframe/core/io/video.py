"""Video decoding / encoding through the ffmpeg CLI (raw RGB pipes).

Why subprocess pipes rather than PyAV: ffmpeg handles display-matrix rotation
(autorotate), colour-matrix conversion and audio stream-copy natively, which
are exactly the error-prone parts of an I/O layer.

* Frames are decoded upright (rotation applied) as uint8 RGB (H, W, 3).
* Output is H.264 / yuv420p tagged BT.709, constant frame rate equal to the
  source's nominal rate (kept as an exact :class:`~fractions.Fraction`).
* The first audio stream is stream-copied when the codec fits in MP4, else
  re-encoded to AAC. Output is trimmed to the shortest stream.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Generator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

from reframe.core.types import ImageArray

MP4_AUDIO_COPY = {"aac", "mp3", "alac", "ac3", "eac3", "opus", "flac"}
_MATRIX = {
    "bt709": "bt709",
    "bt470bg": "bt601",
    "smpte170m": "bt601",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
}


class FFmpegError(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise FFmpegError(f"`{binary}` not found on PATH")
    return path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    """Display width (after rotation)."""
    height: int
    """Display height (after rotation)."""
    coded_width: int
    coded_height: int
    fps: Fraction
    duration: float
    n_frames: int | None
    rotation: int
    """Display rotation in degrees from the container (0, 90, 180, 270)."""
    color_space: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_range: str | None
    audio_codec: str | None
    pix_fmt: str | None

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def _rotation(stream: dict[str, Any]) -> int:
    rot = 0.0
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = float(sd["rotation"])
    if "rotate" in (stream.get("tags") or {}):
        rot = -float(stream["tags"]["rotate"])
    return round(-rot) % 360


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    cmd = [
        _require("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"no video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    rate = video.get("r_frame_rate") or video.get("avg_frame_rate") or "0/1"
    fps = Fraction(rate)
    if fps <= 0:
        fps = Fraction(video.get("avg_frame_rate") or "25/1")
    rotation = _rotation(video)
    cw, ch = int(video["width"]), int(video["height"])
    w, h = (ch, cw) if rotation in (90, 270) else (cw, ch)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb = video.get("nb_frames")
    return VideoInfo(
        path=path, width=w, height=h, coded_width=cw, coded_height=ch, fps=fps,
        duration=duration, n_frames=int(nb) if nb and nb.isdigit() else None,
        rotation=rotation, color_space=video.get("color_space"),
        color_primaries=video.get("color_primaries"),
        color_transfer=video.get("color_transfer"), color_range=video.get("color_range"),
        audio_codec=audio.get("codec_name") if audio else None,
        pix_fmt=video.get("pix_fmt"),
    )  # fmt: skip


def _input_matrix(info: VideoInfo) -> str:
    if info.color_space in _MATRIX:
        return _MATRIX[info.color_space]
    # Untagged: the usual convention is BT.709 for HD, BT.601 for SD.
    return "bt709" if info.coded_height >= 720 else "bt601"


def read_frames(path: str | Path, info: VideoInfo | None = None) -> Generator[ImageArray]:
    """Yield upright uint8 RGB frames of shape (info.height, info.width, 3)."""
    info = info or probe(path)
    cmd = [
        _require("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:v:0", "-fps_mode", "passthrough",
        "-vf", f"scale=in_color_matrix={_input_matrix(info)}:in_range=auto",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    frame_bytes = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    exhausted = False
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                exhausted = True
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3)
    finally:
        if not exhausted:  # consumer stopped early: not an error
            proc.kill()
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
        if exhausted and code != 0:
            raise FFmpegError(f"ffmpeg decode failed ({code}): {stderr.strip()}")


class VideoWriter:
    """Encode RGB frames to H.264 MP4, muxing audio from ``audio_from``.

    Frames are written at ``frame_size`` (W, H) and scaled to ``out_size``
    (lanczos) by ffmpeg, which avoids an extra resize in Python.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        frame_size: tuple[int, int],
        fps: Fraction,
        out_size: tuple[int, int] | None = None,
        audio_from: VideoInfo | None = None,
        crf: int = 18,
        preset: str = "medium",
    ) -> None:
        self.path = Path(path)
        self.frame_size = frame_size
        out_w, out_h = out_size or frame_size
        if out_w % 2 or out_h % 2:
            raise ValueError(f"yuv420p needs even output dimensions, got {out_w}x{out_h}")
        fw, fh = frame_size
        cmd = [
            _require("ffmpeg"), "-y", "-v", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}",
            "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-",
        ]  # fmt: skip
        if audio_from is not None and audio_from.has_audio:
            cmd += ["-i", str(audio_from.path), "-map", "0:v:0", "-map", "1:a:0"]
            if audio_from.audio_codec in MP4_AUDIO_COPY:
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            cmd += ["-shortest"]
        else:
            cmd += ["-map", "0:v:0"]
        cmd += [
            "-vf", f"scale={out_w}:{out_h}:flags=lanczos:out_color_matrix=bt709:out_range=tv,"
                   "format=yuv420p",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-color_range", "tv", "-movflags", "+faststart", str(self.path),
        ]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames_written = 0

    def write(self, frame: ImageArray) -> None:
        fw, fh = self.frame_size
        if frame.shape != (fh, fw, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame {(fh, fw, 3)}, got {frame.dtype} {frame.shape}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames_written += 1

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
        if self._proc.wait() != 0:
            raise FFmpegError(f"ffmpeg encode failed: {stderr.strip()}")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._proc.kill()
            self._proc.wait()
            return
        self.close()
