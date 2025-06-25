"""Pipeline configuration (Pydantic) and social-network presets."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


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
    seam_ratio: float = Field(
        0.25,
        ge=0.0,
        lt=0.75,
        description=(
            "Fraction of the *cropped* width removed by seams: the frame is cropped to "
            "target_w / (1 - seam_ratio), then carved to target_w. 0 = pure centre crop; "
            "above ~0.35 seams start cutting through subjects; 1920->608 by seams only "
            "needs ~0.68 (experiments)."
        ),
    )
    energy: str = Field("forward", pattern="^(forward|backward)$")
    work_height: int = Field(
        540, ge=64, description="Seams are computed at about this height, then upsampled."
    )
    crf: int = Field(18, ge=0, le=51)
    x264_preset: str = "medium"

    @property
    def out_size(self) -> tuple[int, int]:
        return PRESET_SIZES[self.preset]
