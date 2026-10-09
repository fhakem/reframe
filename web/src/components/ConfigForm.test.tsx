import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { schemaResponse } from "../test/fixtures";
import { ConfigForm } from "./ConfigForm";

describe("ConfigForm", () => {
  const setup = () => {
    const onChange = vi.fn();
    render(
      <ConfigForm schema={schemaResponse} value={{ ...schemaResponse.defaults }} onChange={onChange} />,
    );
    return onChange;
  };

  it("renders a preset select with output sizes", () => {
    setup();
    const select = screen.getByLabelText("Preset") as HTMLSelectElement;
    expect(select.value).toBe("tiktok");
    expect([...select.options].map((o) => o.textContent)).toEqual([
      "tiktok (1080×1920)",
      "reels (1080×1920)",
      "shorts (1080×1920)",
    ]);
  });

  it("renders the crop ↔ seams slider and emits numbers", () => {
    const onChange = setup();
    const slider = screen.getByLabelText("Crop ↔ seams") as HTMLInputElement;
    expect(slider.type).toBe("range");
    expect(Number(slider.max)).toBeLessThan(0.75);
    expect(screen.getByText("crop")).toBeInTheDocument();
    expect(screen.getByText("seams")).toBeInTheDocument();
    expect(screen.getByText("25%")).toBeInTheDocument();
    fireEvent.change(slider, { target: { value: "0.4" } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ seam_ratio: 0.4 }));
  });

  it("generates generic inputs for every other field", () => {
    const onChange = setup();
    expect((screen.getByLabelText("Energy") as HTMLSelectElement).value).toBe("forward");
    expect((screen.getByLabelText("Crf") as HTMLInputElement).type).toBe("range");
    expect((screen.getByLabelText("Work Height") as HTMLInputElement).type).toBe("number");
    expect((screen.getByLabelText("X264 Preset") as HTMLInputElement).value).toBe("medium");
    expect((screen.getByLabelText("Track Faces") as HTMLInputElement).checked).toBe(true);
    expect(screen.getByLabelText("Roi").tagName).toBe("TEXTAREA");

    fireEvent.change(screen.getByLabelText("Work Height"), { target: { value: "720" } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ work_height: 720 }));
    fireEvent.click(screen.getByLabelText("Track Faces"));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ track_faces: false }));
    fireEvent.change(screen.getByLabelText("Preset"), { target: { value: "reels" } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ preset: "reels" }));
  });

  it("does not emit invalid JSON values", () => {
    const onChange = setup();
    fireEvent.change(screen.getByLabelText("Roi"), { target: { value: "{bad" } });
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Roi")).toHaveAttribute("aria-invalid", "true");
  });

  it("lets number fields be cleared and retyped without emitting invalid values", () => {
    const onChange = setup();
    const input = screen.getByLabelText("Work Height") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "" } });
    expect(input.value).toBe("");
    expect(input).toHaveAttribute("aria-invalid", "true");
    fireEvent.change(input, { target: { value: "7" } }); // below minimum 64: draft only
    expect(input.value).toBe("7");
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: "720" } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ work_height: 720 }));
    expect(input).toHaveAttribute("aria-invalid", "false");
  });
});
