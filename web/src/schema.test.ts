import { describe, expect, it } from "vitest";
import { coerce, fieldsFromSchema, patternOptions } from "./schema";
import { schemaResponse } from "./test/fixtures";

describe("fieldsFromSchema", () => {
  const fields = fieldsFromSchema(schemaResponse.schema);
  const by = Object.fromEntries(fields.map((f) => [f.name, f]));

  it("puts preset and seam_ratio first, then schema order", () => {
    expect(fields.map((f) => f.name)).toEqual([
      "preset",
      "seam_ratio",
      "energy",
      "work_height",
      "crf",
      "x264_preset",
      "track_faces",
      "max_shift",
      "roi",
    ]);
  });

  it("resolves $ref enums and pattern enums to selects", () => {
    expect(by.preset).toMatchObject({ kind: "select", options: ["tiktok", "reels", "shorts"] });
    expect(by.preset.default).toBe("tiktok");
    expect(by.energy).toMatchObject({ kind: "select", options: ["forward", "backward"] });
  });

  it("makes bounded numbers sliders, keeping exclusive bounds out of range", () => {
    expect(by.seam_ratio).toMatchObject({ kind: "range", min: 0, step: 0.01 });
    expect(by.seam_ratio.max).toBeLessThan(0.75);
    expect(by.crf).toMatchObject({ kind: "range", min: 0, max: 51, step: 1, integer: true });
    expect(by.work_height).toMatchObject({ kind: "number", min: 64, integer: true });
  });

  it("handles booleans, nullable unions and unknown shapes", () => {
    expect(by.track_faces.kind).toBe("checkbox");
    expect(by.max_shift).toMatchObject({ kind: "number", nullable: true });
    expect(by.roi).toMatchObject({ kind: "json", nullable: true });
    expect(by.x264_preset.kind).toBe("text");
  });

  it("coerces raw input", () => {
    expect(coerce(by.crf, "20.4")).toBe(20);
    expect(coerce(by.seam_ratio, "0.3")).toBe(0.3);
    expect(coerce(by.max_shift, "")).toBeNull();
    expect(coerce(by.roi, '{"x": 1}')).toEqual({ x: 1 });
    expect(coerce(by.roi, "{oops")).toBeUndefined();
  });

  it("parses only simple alternation patterns", () => {
    expect(patternOptions("^(a|b)$")).toEqual(["a", "b"]);
    expect(patternOptions("^[a-z]+$")).toBeUndefined();
    expect(patternOptions(undefined)).toBeUndefined();
  });
});
