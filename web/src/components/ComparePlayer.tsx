import { useEffect, useRef } from "react";
import { api, type Job } from "../api";

/**
 * Source and result side by side. The source player is the leader (native
 * controls); the result follows play / pause / seek / rate and is drift-corrected.
 */
export function ComparePlayer({ job }: { job: Job }) {
  const src = useRef<HTMLVideoElement>(null);
  const out = useRef<HTMLVideoElement>(null);
  const done = job.status === "done";

  useEffect(() => {
    const a = src.current;
    const b = out.current;
    if (!a || !b || !done) return;
    const sync = () => {
      if (Math.abs(b.currentTime - a.currentTime) > 0.08) b.currentTime = a.currentTime;
    };
    const onPlay = () => {
      sync();
      void b.play().catch(() => undefined);
    };
    const onPause = () => {
      b.pause();
      sync();
    };
    const onRate = () => {
      b.playbackRate = a.playbackRate;
    };
    const handlers: [string, () => void][] = [
      ["play", onPlay],
      ["pause", onPause],
      ["seeking", sync],
      ["seeked", sync],
      ["ratechange", onRate],
      ["timeupdate", sync],
    ];
    for (const [ev, fn] of handlers) a.addEventListener(ev, fn);
    return () => {
      for (const [ev, fn] of handlers) a.removeEventListener(ev, fn);
    };
  }, [done, job.id]);

  const w = job.meta.width ?? 16;
  const h = job.meta.height ?? 9;
  const [ow, oh] = job.meta.output_size ?? [9, 16];
  // Column widths proportional to aspect ratios -> both players get the same height.
  const columns = `${w / h}fr ${ow / oh}fr`;

  return (
    <section className="compare" aria-label="Preview">
      <div className="compare-grid" style={{ gridTemplateColumns: columns }}>
        <figure>
          <video ref={src} src={api.sourceUrl(job.id)} controls playsInline preload="metadata" />
          <figcaption>Source</figcaption>
        </figure>
        <figure>
          {done ? (
            <video ref={out} src={api.resultUrl(job.id)} muted playsInline preload="auto" />
          ) : (
            <div className="placeholder" style={{ aspectRatio: `${ow} / ${oh}` }}>
              {job.status === "running" ? `${Math.round(job.progress * 100)}%` : job.status}
            </div>
          )}
          <figcaption>9:16 result</figcaption>
        </figure>
      </div>
      {done && (
        <div className="compare-actions">
          <a className="button" href={api.resultUrl(job.id, true)} download>
            Download 9:16 MP4
          </a>
        </div>
      )}
    </section>
  );
}
