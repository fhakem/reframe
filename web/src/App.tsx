import { useCallback, useEffect, useRef, useState } from "react";
import { ACTIVE, api, type Job } from "./api";
import { ComparePlayer } from "./components/ComparePlayer";
import { ConfigForm, type ConfigValue } from "./components/ConfigForm";
import { JobList } from "./components/JobList";
import { Uploader } from "./components/Uploader";
import type { SchemaResponse } from "./schema";

export const POLL_ACTIVE_MS = 1000;
export const POLL_IDLE_MS = 5000;

/** Poll the job list: fast while something is queued/running, slow otherwise. */
export function useJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    clearTimeout(timer.current);
    let next = POLL_IDLE_MS;
    try {
      const list = await api.jobs();
      if (!alive.current) return;
      setJobs(list);
      setError(null);
      if (list.some((j) => ACTIVE.has(j.status))) next = POLL_ACTIVE_MS;
    } catch (e) {
      if (!alive.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
    timer.current = setTimeout(() => void refresh(), next);
  }, []);

  useEffect(() => {
    alive.current = true;
    void refresh();
    return () => {
      alive.current = false;
      clearTimeout(timer.current);
    };
  }, [refresh]);

  return { jobs, error, refresh };
}

export function App() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [config, setConfig] = useState<ConfigValue>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ name: string; progress: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { jobs, error: pollError, refresh } = useJobs();

  useEffect(() => {
    api
      .schema()
      .then((s) => {
        setSchema(s);
        setConfig({ ...s.defaults });
      })
      .catch((e: unknown) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, []);

  const onFile = async (file: File) => {
    setUploadError(null);
    setUpload({ name: file.name, progress: 0 });
    try {
      const job = await api.upload(file, config, (p) =>
        setUpload({ name: file.name, progress: p }),
      );
      setSelectedId(job.id);
      await refresh();
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : String(e));
    } finally {
      setUpload(null);
    }
  };

  const onCancel = async (job: Job) => {
    try {
      await api.cancel(job.id);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : String(e));
    }
    await refresh();
  };

  const selected = jobs.find((j) => j.id === selectedId) ?? null;

  return (
    <div className="app">
      <header>
        <h1>Reframe</h1>
        <p className="tagline">16:9 → 9:16 with smart crop and seam carving</p>
      </header>
      <main>
        <section className="panel new-job" aria-label="New job">
          <Uploader
            accept={schema?.accept}
            maxMb={schema?.max_upload_mb}
            disabled={!schema || upload !== null}
            onFile={(f) => void onFile(f)}
          />
          {upload && (
            <div
              className="progress"
              role="progressbar"
              aria-label={`Uploading ${upload.name}`}
              aria-valuenow={Math.round(upload.progress * 100)}
            >
              <div className="progress-bar" style={{ width: `${upload.progress * 100}%` }} />
              <span className="progress-text">Uploading {Math.round(upload.progress * 100)}%</span>
            </div>
          )}
          {uploadError && <p className="error">{uploadError}</p>}
          {loadError && <p className="error">Could not load settings: {loadError}</p>}
          {schema && (
            <ConfigForm
              schema={schema}
              value={config}
              onChange={setConfig}
              disabled={upload !== null}
            />
          )}
        </section>
        <section className="panel jobs" aria-label="Job list">
          <h2>Jobs</h2>
          {pollError && <p className="error">API unreachable: {pollError}</p>}
          <JobList
            jobs={jobs}
            selectedId={selectedId}
            onSelect={(j) => setSelectedId(j.id)}
            onCancel={(j) => void onCancel(j)}
          />
        </section>
        {selected && (
          <section className="panel preview">
            <h2 title={selected.original_name}>{selected.original_name}</h2>
            <ComparePlayer key={selected.id} job={selected} />
          </section>
        )}
      </main>
    </div>
  );
}
