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
    mode: Annotated[str, typer.Option(help="smart | naive (M1 baseline)")] = "smart",
    auto_budget: Annotated[
        bool, typer.Option(help="Smart: reduce seams per shot when they would cut structure")
    ] = True,
    fallback: Annotated[bool, typer.Option(help="Smart: allow blur-pad / split-screen")] = True,
    temporal_lambda: Annotated[float, typer.Option(help="Smart: temporal coherence")] = 0.3,
    workers: Annotated[int, typer.Option(help="Parallel render processes (0 = auto)")] = 0,
    energy: Annotated[str, typer.Option(help="forward | backward")] = "forward",
    work_height: Annotated[int, typer.Option(help="Seam computation height")] = 540,
    crf: Annotated[int, typer.Option(help="x264 CRF")] = 18,
) -> None:
    """Reframe a landscape video to 9:16 (smart: shots + tracked crop + coherent seams)."""
    cfg = ReframeConfig.model_validate(
        dict(
            preset=preset, mode=mode, seam_ratio=seam_ratio, auto_budget=auto_budget,
            fallback=fallback, temporal_lambda=temporal_lambda, workers=workers,
            energy=energy, work_height=work_height, crf=crf,
        )
    )  # fmt: skip
    info = probe(src)
    plan = plan_frame(info.width, info.height, cfg)
    typer.echo(
        f"{src.name}: {info.width}x{info.height} @ {float(info.fps):.3f} fps -> "
        f"[{cfg.mode}] max crop {plan.crop_w}px, max {plan.n_seams} seams -> "
        f"{plan.target_w}x{plan.src_h} -> "
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
