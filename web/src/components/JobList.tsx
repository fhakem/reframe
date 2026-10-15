import { ACTIVE, api, type Job } from "../api";

interface Props {
  jobs: Job[];
  selectedId?: string | null;
  onSelect?: (job: Job) => void;
  onCancel?: (job: Job) => void;
}

const LABEL: Record<Job["status"], string> = {
  queued: "Queued",
  running: "Processing",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

function fmtDuration(s?: number): string {
  if (!s || !Number.isFinite(s)) return "";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

export function JobList({ jobs, selectedId, onSelect, onCancel }: Props) {
  if (jobs.length === 0) {
    return <p className="empty">No jobs yet — drop a video above.</p>;
  }
  return (
    <ul className="job-list" aria-label="Jobs">
      {jobs.map((job) => {
        const pct = Math.round(job.progress * 100);
        const { width, height, duration } = job.meta;
        return (
          <li
            key={job.id}
            className={`job job-${job.status}${job.id === selectedId ? " selected" : ""}`}
            data-testid={`job-${job.id}`}
          >
            <button type="button" className="job-main" onClick={() => onSelect?.(job)}>
              <span className="job-name" title={job.original_name}>
                {job.original_name}
              </span>
              <span className="job-meta">
                {width && height ? `${width}×${height}` : ""} {fmtDuration(duration)}
              </span>
              <span className={`badge badge-${job.status}`}>{LABEL[job.status]}</span>
            </button>
            {ACTIVE.has(job.status) && (
              <div
                className="progress"
                role="progressbar"
                aria-label={`Progress of ${job.original_name}`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={pct}
              >
                <div className="progress-bar" style={{ width: `${pct}%` }} />
                <span className="progress-text">{job.status === "queued" ? "waiting" : `${pct}%`}</span>
              </div>
            )}
            {job.status === "failed" && job.error && <p className="error">{job.error}</p>}
            <div className="job-actions">
              {ACTIVE.has(job.status) && (
                <button type="button" className="secondary" onClick={() => onCancel?.(job)}>
                  Cancel
                </button>
              )}
              {job.status === "done" && (
                <a className="button" href={api.resultUrl(job.id, true)} download>
                  Download
                </a>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
