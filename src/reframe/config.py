"""Pipeline configuration (Pydantic) and social-network presets."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from reframe.core.energy.combine import SemanticConfig


class Preset(StrEnum):
    tiktok = "tiktok"
    reels = "reels"
    shorts = "shorts"


#: Output (width, height) per preset. All three platforms currently use 1080x1920.
PRESET_SIZES: dict[Preset, tuple[int, int]] = {
    Preset.tiktok: (1080, 1920),
    Preset.reels: (1080, 1920),
    Preset.shorts: (1080, 1920),
}


class ReframeConfig(BaseModel):
    preset: Preset = Preset.tiktok
    mode: Literal["smart", "naive"] = Field(
        "smart",
        description=(
            "'smart': shots + semantic analysis + tracked crop + temporally coherent seams "
            "+ automatic crop/seam split + fallbacks. 'naive': centre crop + per-frame seams "
            "(M1 baseline, kept for comparison)."
        ),
    )
    seam_ratio: float = Field(
        0.25,
        ge=0.0,
        lt=0.75,
        description=(
            "Fraction of the *cropped* width removed by seams: the frame is cropped to "
            "target_w / (1 - seam_ratio), then carved to target_w. In smart mode with "
            "auto_budget this is the maximum; shots where seams would cut structure get fewer. "
            "0 = pure crop; above ~0.35 seams start cutting through subjects."
        ),
    )
    auto_budget: bool = Field(
        True, description="Smart mode: cut the seam budget per shot when seam costs rise."
    )
    temporal_lambda: float = Field(
        0.3,
        ge=0.0,
        description=(
            "Smart mode: temporal coherence weight (0 = per-frame seams). Measured on Big Buck "
            "Bunny rabbit shots: flow warping error 1.31 at 0, 0.90 at 0.3, 0.88 at 1; from 1 "
            "upward seams stay put while a low-texture subject moves into them and deform it. "
            "Without a subject mask this is a stability / fidelity trade-off."
        ),
    )
    flow_method: Literal["dis_fast", "dis_medium", "farneback", "none"] = Field(
        "dis_fast", description="Optical flow used to propagate seams between frames."
    )
    fallback: bool = Field(
        True, description="Smart mode: allow split-screen / blurred-padding fallbacks per shot."
    )
    energy: str = Field("forward", pattern="^(forward|backward)$")
    work_height: int = Field(
        540, ge=64, description="Analysis and seams run at about this height, then upsampled."
    )
    workers: int = Field(
        0, ge=0, le=16, description="Parallel render processes (0 = auto, 1 = sequential)."
    )
    semantic: SemanticConfig = Field(
        default_factory=lambda: SemanticConfig(saliency_weight=0.0, line_weight=0.0),
        description=(
            "Seam energy terms. Classical saliency and line maps are off by default: on real "
            "footage (Big Buck Bunny) they steered seams *into* large low-texture subjects and "
            "raised jitter. Saliency still drives the crop path; boxes (text/faces/people) are "
            "still protected."
        ),
    )
    crf: int = Field(18, ge=0, le=51)
    x264_preset: str = "medium"

    @property
    def out_size(self) -> tuple[int, int]:
        return PRESET_SIZES[self.preset]
