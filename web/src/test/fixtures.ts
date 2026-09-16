import type { Job } from "../api";
import type { SchemaResponse } from "../schema";

/** Shape emitted by pydantic for ReframeConfig, plus a few fields it does not have (yet). */
export const schemaResponse: SchemaResponse = {
  schema: {
    $defs: {
      Preset: { enum: ["tiktok", "reels", "shorts"], title: "Preset", type: "string" },
      Box: { type: "object", properties: { x: { type: "integer" } }, title: "Box" },
    },
    properties: {
      energy: { default: "forward", pattern: "^(forward|backward)$", title: "Energy", type: "string" },
      preset: { $ref: "#/$defs/Preset", default: "tiktok" },
      seam_ratio: {
        default: 0.25,
        description: "Fraction of the cropped width removed by seams",
        exclusiveMaximum: 0.75,
        minimum: 0,
        title: "Seam Ratio",
        type: "number",
      },
      work_height: { default: 540, minimum: 64, title: "Work Height", type: "integer" },
      crf: { default: 18, maximum: 51, minimum: 0, title: "Crf", type: "integer" },
      x264_preset: { default: "medium", title: "X264 Preset", type: "string" },
      track_faces: { default: true, title: "Track Faces", type: "boolean" },
      max_shift: {
        anyOf: [{ type: "number" }, { type: "null" }],
        default: null,
        title: "Max Shift",
      },
      roi: { anyOf: [{ $ref: "#/$defs/Box" }, { type: "null" }], default: null },
    },
    title: "ReframeConfig",
    type: "object",
  },
  defaults: {
    preset: "tiktok",
    seam_ratio: 0.25,
    energy: "forward",
    work_height: 540,
    crf: 18,
    x264_preset: "medium",
    track_faces: true,
    max_shift: null,
    roi: null,
  },
  presets: {
    tiktok: { width: 1080, height: 1920 },
    reels: { width: 1080, height: 1920 },
    shorts: { width: 1080, height: 1920 },
  },
  accept: [".mov", ".mp4"],
  max_upload_mb: 500,
};

export function makeJob(over: Partial<Job> = {}): Job {
  return {
    id: "a".repeat(32),
    status: "queued",
    progress: 0,
    error: null,
    config: {},
    original_name: "clip.mp4",
    meta: { width: 1920, height: 1080, duration: 12.5 },
    attempt: 0,
    created_at: 1,
    started_at: null,
    finished_at: null,
    updated_at: 1,
    ...over,
  };
}
