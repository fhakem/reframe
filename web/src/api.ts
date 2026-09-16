import type { SchemaResponse } from "./schema";

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  status: JobStatus;
  progress: number;
  error: string | null;
  config: Record<string, unknown>;
  original_name: string;
  meta: {
    width?: number;
    height?: number;
    duration?: number;
    fps?: number;
    output_size?: [number, number];
    [k: string]: unknown;
  };
  attempt: number;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  updated_at: number;
}

export const ACTIVE: ReadonlySet<JobStatus> = new Set(["queued", "running"]);

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function detailMessage(body: unknown, fallback: string): string {
  const d = (body as { detail?: unknown } | null)?.detail;
  if (typeof d === "string") return d;
  if (d && typeof d === "object" && "message" in d) {
    const errs = (d as { errors?: { loc?: unknown[]; msg?: string }[] }).errors ?? [];
    const parts = errs.map((e) => `${(e.loc ?? []).join(".")}: ${e.msg ?? ""}`);
    return [String((d as { message: unknown }).message), ...parts].join(" — ");
  }
  if (Array.isArray(d)) return d.map((e) => e?.msg ?? String(e)).join("; ");
  return fallback;
}

async function json<T>(res: Response): Promise<T> {
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, detailMessage(body, res.statusText));
  return body as T;
}

export const api = {
  schema: (): Promise<SchemaResponse> => fetch("/api/config/schema").then((r) => json(r)),
  jobs: (): Promise<Job[]> =>
    fetch("/api/jobs")
      .then((r) => json<{ jobs: Job[] }>(r))
      .then((b) => b.jobs),
  cancel: (id: string): Promise<Job> =>
    fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }).then((r) => json(r)),
  sourceUrl: (id: string) => `/api/jobs/${encodeURIComponent(id)}/source`,
  resultUrl: (id: string, download = false) =>
    `/api/jobs/${encodeURIComponent(id)}/result${download ? "?download=1" : ""}`,

  /** Multipart upload with progress (XHR: fetch has no upload progress). */
  upload(
    file: File,
    config: Record<string, unknown>,
    onProgress?: (fraction: number) => void,
  ): Promise<Job> {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("config", JSON.stringify(config));
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/jobs");
      xhr.responseType = "json";
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total);
      };
      xhr.onerror = () => reject(new ApiError(0, "network error"));
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as Job);
        else reject(new ApiError(xhr.status, detailMessage(xhr.response, `HTTP ${xhr.status}`)));
      };
      xhr.send(form);
    });
  },
};
