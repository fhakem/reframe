import { act, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App, POLL_ACTIVE_MS } from "./App";
import { makeJob, schemaResponse } from "./test/fixtures";

function mockApi(jobsSequence: ReturnType<typeof makeJob>[][]) {
  let call = 0;
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/config/schema") return Response.json(schemaResponse);
    if (url === "/api/jobs") {
      const jobs = jobsSequence[Math.min(call++, jobsSequence.length - 1)];
      return Response.json({ jobs });
    }
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("App", () => {
  it("loads the schema form and polls job progress until done", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const id = "b".repeat(32);
      const fetchMock = mockApi([
        [makeJob({ id, status: "running", progress: 0.1 })],
        [makeJob({ id, status: "running", progress: 0.6 })],
        [makeJob({ id, status: "done", progress: 1 })],
      ]);
      render(<App />);

      expect(await screen.findByLabelText("Preset")).toBeInTheDocument();
      expect(await screen.findByText("10%")).toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_ACTIVE_MS);
      });
      expect(await screen.findByText("60%")).toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_ACTIVE_MS);
      });
      await waitFor(() => expect(screen.getByRole("link", { name: "Download" })).toBeInTheDocument());
      expect(fetchMock.mock.calls.filter(([u]) => String(u) === "/api/jobs").length).toBe(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows an error when the API is down", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "down" }), { status: 503 })),
    );
    render(<App />);
    expect(await screen.findByText(/could not load settings: down/i)).toBeInTheDocument();
    expect(await screen.findByText(/api unreachable: down/i)).toBeInTheDocument();
  });
});
