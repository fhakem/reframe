import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeJob } from "../test/fixtures";
import { JobList } from "./JobList";

describe("JobList", () => {
  it("shows an empty state", () => {
    render(<JobList jobs={[]} />);
    expect(screen.getByText(/no jobs yet/i)).toBeInTheDocument();
  });

  it("renders status, progress, cancel, download and errors", () => {
    const onCancel = vi.fn();
    const running = makeJob({ id: "1".repeat(32), status: "running", progress: 0.42, original_name: "run.mp4" });
    const done = makeJob({ id: "2".repeat(32), status: "done", progress: 1, original_name: "ok.mov" });
    const failed = makeJob({ id: "3".repeat(32), status: "failed", error: "boom", original_name: "bad.mp4" });
    render(<JobList jobs={[running, done, failed]} onCancel={onCancel} />);

    const r = within(screen.getByTestId(`job-${running.id}`));
    expect(r.getByText("Processing")).toBeInTheDocument();
    expect(r.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    expect(r.getByText("42%")).toBeInTheDocument();
    expect(r.getByText("1920×1080 0:13", { exact: false })).toBeInTheDocument();
    fireEvent.click(r.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledWith(running);

    const d = within(screen.getByTestId(`job-${done.id}`));
    expect(d.queryByRole("progressbar")).toBeNull();
    expect(d.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(d.getByRole("link", { name: "Download" })).toHaveAttribute(
      "href",
      `/api/jobs/${done.id}/result?download=1`,
    );

    const f = within(screen.getByTestId(`job-${failed.id}`));
    expect(f.getByText("Failed")).toBeInTheDocument();
    expect(f.getByText("boom")).toBeInTheDocument();
  });

  it("calls onSelect when a job is clicked", () => {
    const onSelect = vi.fn();
    const job = makeJob();
    render(<JobList jobs={[job]} onSelect={onSelect} selectedId={job.id} />);
    expect(screen.getByTestId(`job-${job.id}`)).toHaveClass("selected");
    fireEvent.click(screen.getByText("clip.mp4"));
    expect(onSelect).toHaveBeenCalledWith(job);
  });
});
