"""Command-line entry point: ``reframe input.mp4 -o out.mp4 --seam-ratio 0.25``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from reframe.config import Preset, ReframeConfig
from reframe.core.io.video import probe
from reframe.core.pipeline import plan_frame, reframe_video

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    src: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output .mp4")],
    preset: Annotated[Preset, typer.Option(help="Target platform")] = Preset.tiktok,
    seam_ratio: Annotated[
        float,
        typer.Option(
            min=0.0, max=0.74, help="Fraction of the cropped width removed by seams (0 = crop)"
        ),
    ] = 0.25,
    energy: Annotated[str, typer.Option(help="forward | backward")] = "forward",
    work_height: Annotated[int, typer.Option(help="Seam computation height")] = 540,
    crf: Annotated[int, typer.Option(help="x264 CRF")] = 18,
) -> None:
    """Reframe a landscape video to 9:16 (M1 naive baseline: centre crop + per-frame seams)."""
    cfg = ReframeConfig(
        preset=preset, seam_ratio=seam_ratio, energy=energy, work_height=work_height, crf=crf
    )
    info = probe(src)
    plan = plan_frame(info.width, info.height, cfg)
    typer.echo(
        f"{src.name}: {info.width}x{info.height} @ {float(info.fps):.3f} fps -> "
        f"crop {plan.crop_w}px, {plan.n_seams} seams -> {plan.target_w}x{plan.src_h} -> "
        f"{cfg.out_size[0]}x{cfg.out_size[1]}",
        err=True,
    )

    def progress(done: int, total: int | None) -> None:
        if done % 10 == 0 or done == total:
            sys.stderr.write(f"\r  frame {done}/{total or '?'}")
            sys.stderr.flush()

    reframe_video(src, output, cfg, progress=progress)
    sys.stderr.write("\n")
    typer.echo(str(output))


if __name__ == "__main__":
    app()
