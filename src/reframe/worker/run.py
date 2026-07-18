"""Job subprocess: ``python -m reframe.worker.run --db jobs.db --storage files/ --attempt N <id>``.

Loads the job row, runs :func:`reframe.core.pipeline.reframe_video` with
``ReframeConfig.model_validate(job.config)``, reports progress into the row and
marks it done/failed. Output is written to a ``.part`` key and renamed on success.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from reframe.worker.jobs import JobStatus, JobStore
from reframe.worker.store import LocalStorage, part_key


class _Superseded(Exception):
    """This attempt is no longer the live one (cancelled or recovered elsewhere)."""


def run_job(store: JobStore, storage: LocalStorage, job_id: str, attempt: int) -> int:
    job = store.get(job_id)
    if job is None or job.status is not JobStatus.running or job.attempt != attempt:
        print(f"job {job_id} attempt {attempt} is not live; exiting", file=sys.stderr)
        return 3
    assert job.output_key is not None
    tmp_key = part_key(job.output_key)
    try:
        # Heavy imports here so a bad environment is reported as a job error.
        from reframe.config import ReframeConfig
        from reframe.core.pipeline import reframe_video

        cfg = ReframeConfig.model_validate(job.config)
        fps = float(job.meta.get("fps") or 0.0)
        est_total = round(float(job.meta.get("duration") or 0.0) * fps) or None
        last = 0.0

        def progress(done: int, total: int | None) -> None:
            nonlocal last
            total = total or est_total
            now = time.monotonic()
            if now - last < 0.25 and done != total:
                return
            last = now
            frac = min(done / total, 0.99) if total else 0.0
            if not store.set_progress(job_id, attempt, frac):
                raise _Superseded

        tmp_path = storage.path(tmp_key)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        info = reframe_video(storage.path(job.input_key), tmp_path, cfg,
                             progress=progress)  # fmt: skip
        storage.rename(tmp_key, job.output_key)
        meta = dict(job.meta)
        meta["output_size"] = list(cfg.out_size)
        meta.setdefault("duration", info.duration)
        if not store.finish(job_id, attempt, JobStatus.done, meta=meta):
            storage.delete(job.output_key)
            return 3
        return 0
    except _Superseded:
        storage.delete(tmp_key)
        return 3
    except BaseException as e:
        storage.delete(tmp_key)
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}"
        store.finish(job_id, attempt, JobStatus.failed, error=msg[:2000])
        return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reframe.worker.run")
    p.add_argument("--db", required=True)
    p.add_argument("--storage", required=True)
    p.add_argument("--attempt", type=int, required=True)
    p.add_argument("job_id")
    a = p.parse_args(argv)
    return run_job(JobStore(a.db), LocalStorage(a.storage), a.job_id, a.attempt)


if __name__ == "__main__":
    sys.exit(main())
