"""Job model, SQLite persistence and an in-process worker pool.

* State lives in SQLite (stdlib ``sqlite3``, WAL), so it survives an API restart.
  SQLite *is* the queue: worker threads claim the oldest ``queued`` row with an
  atomic ``UPDATE ... RETURNING``.
* Each job runs in its own subprocess (``python -m reframe.worker.run``) started
  in a new session, so cancelling kills the whole process group (the Python
  runner *and* its ffmpeg children). The runner reports progress by writing the
  job row; every write is conditional on ``status='running' AND attempt=?`` so a
  stale or cancelled runner can never overwrite a newer state.

Recovery policy (on :meth:`JobQueue.start`): rows left ``running`` by a crashed
API/worker are **re-queued** if they have been attempted fewer than
``max_attempts`` times (default 3), otherwise marked ``failed``. Their partial
output is deleted. A runner that survived the crash notices on its next progress
write that its attempt is no longer current and exits.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from reframe.worker.store import LocalStorage, new_key, part_key


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL = frozenset({JobStatus.done, JobStatus.failed, JobStatus.cancelled})


@dataclass(frozen=True)
class Job:
    id: str
    status: JobStatus
    progress: float
    error: str | None
    config: dict[str, Any]
    input_key: str
    output_key: str | None
    original_name: str
    meta: dict[str, Any]
    attempt: int
    pid: int | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    updated_at: float

    def public(self) -> dict[str, Any]:
        """JSON-safe view for the API (no storage keys / pids)."""
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "error": self.error,
            "config": self.config,
            "original_name": self.original_name,
            "meta": self.meta,
            "attempt": self.attempt,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    config TEXT NOT NULL,
    input_key TEXT NOT NULL,
    output_key TEXT,
    original_name TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    attempt INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
"""


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        status=JobStatus(row["status"]),
        progress=float(row["progress"]),
        error=row["error"],
        config=json.loads(row["config"]),
        input_key=row["input_key"],
        output_key=row["output_key"],
        original_name=row["original_name"],
        meta=json.loads(row["meta"]),
        attempt=int(row["attempt"]),
        pid=row["pid"],
        created_at=float(row["created_at"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=float(row["updated_at"]),
    )


class JobStore:
    """Thin SQLite DAO. One short-lived connection per call: safe across threads/processes."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    # -- creation / reads ---------------------------------------------------
    def create(
        self,
        *,
        config: dict[str, Any],
        input_key: str,
        original_name: str,
        meta: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> Job:
        now = time.time()
        jid = job_id or uuid.uuid4().hex
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (id, status, config, input_key, original_name, meta,"
                " created_at, updated_at) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)",
                (jid, json.dumps(config), input_key, original_name, json.dumps(meta or {}), now,
                 now),
            )  # fmt: skip
        job = self.get(jid)
        assert job is not None
        return job

    def get(self, job_id: str) -> Job | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(self, *, limit: int = 100) -> list[Job]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    # -- transitions --------------------------------------------------------
    def claim_next(self) -> Job | None:
        """Atomically move the oldest queued job to running (new attempt, new output key)."""
        now = time.time()
        with self._conn() as c:
            row = c.execute(
                "UPDATE jobs SET status='running', attempt=attempt+1, progress=0, error=NULL,"
                " pid=NULL, output_key=?, started_at=?, finished_at=NULL, updated_at=?"
                " WHERE id = (SELECT id FROM jobs WHERE status='queued'"
                "             ORDER BY created_at, rowid LIMIT 1) AND status='queued'"
                " RETURNING *",
                (new_key("results", "mp4"), now, now),
            ).fetchone()
        return _row_to_job(row) if row else None

    def set_pid(self, job_id: str, attempt: int, pid: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET pid=?, updated_at=? WHERE id=? AND attempt=? AND status='running'",
                (pid, time.time(), job_id, attempt),
            )

    def set_progress(self, job_id: str, attempt: int, progress: float) -> bool:
        """Returns False when this attempt is no longer the live one (cancelled, recovered)."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET progress=?, updated_at=? WHERE id=? AND attempt=?"
                " AND status='running'",
                (min(max(progress, 0.0), 1.0), time.time(), job_id, attempt),
            )
        return cur.rowcount == 1

    def finish(
        self,
        job_id: str,
        attempt: int,
        status: JobStatus,
        *,
        error: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """running -> terminal, only if ``attempt`` is still the live one."""
        assert status in TERMINAL
        now = time.time()
        sets = "status=?, error=?, finished_at=?, updated_at=?, pid=NULL"
        args: list[Any] = [status.value, error, now, now]
        if status is JobStatus.done:
            sets += ", progress=1"
        if meta is not None:
            sets += ", meta=?"
            args.append(json.dumps(meta))
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE jobs SET {sets} WHERE id=? AND attempt=? AND status='running'",
                (*args, job_id, attempt),
            )
        return cur.rowcount == 1

    def cancel_queued(self, job_id: str) -> bool:
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status='cancelled', finished_at=?, updated_at=?"
                " WHERE id=? AND status='queued'",
                (now, now, job_id),
            )
        return cur.rowcount == 1

    def requeue(self, job_id: str, attempt: int, *, refund: bool) -> bool:
        """running -> queued (``refund`` gives the attempt back, e.g. on graceful shutdown)."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status='queued', progress=0, pid=NULL, started_at=NULL,"
                " attempt=attempt-?, updated_at=? WHERE id=? AND attempt=? AND status='running'",
                (1 if refund else 0, time.time(), job_id, attempt),
            )
        return cur.rowcount == 1

    def recover(self, *, max_attempts: int) -> tuple[list[str], list[str]]:
        """Handle rows left ``running`` by a dead worker. Returns (requeued, failed) ids."""
        requeued: list[str] = []
        failed: list[str] = []
        with self._conn() as c:
            rows = c.execute("SELECT id, attempt FROM jobs WHERE status='running'").fetchall()
        for row in rows:
            if row["attempt"] < max_attempts:
                if self.requeue(row["id"], row["attempt"], refund=False):
                    requeued.append(row["id"])
            elif self.finish(
                row["id"],
                row["attempt"],
                JobStatus.failed,
                error=f"worker died during processing ({row['attempt']} attempts)",
            ):
                failed.append(row["id"])
        return requeued, failed


JobCommand = Callable[[Job], Sequence[str]]


def _kill_group(proc: subprocess.Popen[bytes], *, grace: float = 5.0) -> None:
    """SIGTERM the job's process group, SIGKILL after ``grace`` s, then reap."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":  # pragma: no cover - no process groups
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":  # pragma: no cover
            proc.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()


@dataclass
class _Running:
    job: Job
    proc: subprocess.Popen[bytes]
    reaped: threading.Event = field(default_factory=threading.Event)
    cancel_requested: bool = False


class JobQueue:
    """Worker pool: ``concurrency`` threads, each supervising one job subprocess at a time."""

    def __init__(
        self,
        store: JobStore,
        storage: LocalStorage,
        *,
        concurrency: int = 1,
        max_attempts: int = 3,
        log_dir: str | Path | None = None,
        job_command: JobCommand | None = None,
        poll_interval: float = 0.5,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.store = store
        self.storage = storage
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.log_dir = Path(log_dir) if log_dir else store.db_path.parent / "logs"
        self.job_command: JobCommand = job_command or self.default_command
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._running: dict[str, _Running] = {}
        self._threads: list[threading.Thread] = []
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self.recovered: tuple[list[str], list[str]] = ([], [])

    def default_command(self, job: Job) -> list[str]:
        # Every element is server-generated (paths, uuid hex, int): no user strings, no shell.
        return [
            sys.executable, "-m", "reframe.worker.run",
            "--db", str(self.store.db_path),
            "--storage", str(self.storage.root),
            "--attempt", str(job.attempt),
            job.id,
        ]  # fmt: skip

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            return
        self._stopping.clear()
        requeued, failed = self.store.recover(max_attempts=self.max_attempts)
        for jid in requeued + failed:
            job = self.store.get(jid)
            if job and job.output_key:
                self.storage.delete(part_key(job.output_key))
        self.recovered = (requeued, failed)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.concurrency):
            t = threading.Thread(target=self._loop, name=f"reframe-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, *, timeout: float = 10.0) -> None:
        """Stop workers; running jobs are killed and put back in the queue (attempt refunded)."""
        self._stopping.set()
        self._wake.set()
        with self._lock:
            running = list(self._running.values())
        for r in running:
            _kill_group(r.proc)
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def notify(self) -> None:
        """Wake an idle worker (call after enqueueing)."""
        self._wake.set()

    def is_alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    # -- cancel -------------------------------------------------------------
    def cancel(self, job_id: str, *, timeout: float = 15.0) -> Job | None:
        """Cancel a queued or running job. Returns the updated job (None if unknown).

        For a job running here, returns only once its process group is dead and reaped.
        """
        with self._lock:
            if self.store.cancel_queued(job_id):
                return self.store.get(job_id)
            r = self._running.get(job_id)
            if r is not None:
                r.cancel_requested = True
        if r is not None:
            _kill_group(r.proc)
            r.reaped.wait(timeout)
            return self.store.get(job_id)
        job = self.store.get(job_id)
        if job is not None and job.status is JobStatus.running:
            # Running but not supervised by this process (e.g. orphan): flip the row; the
            # runner's next conditional progress write fails and it exits on its own.
            self.store.finish(job_id, job.attempt, JobStatus.cancelled)
            job = self.store.get(job_id)
        return job

    # -- worker loop --------------------------------------------------------
    def _loop(self) -> None:
        while not self._stopping.is_set():
            r = self._claim_and_spawn()
            if r is None:
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            self._supervise(r)

    def _claim_and_spawn(self) -> _Running | None:
        with self._lock:
            if self._stopping.is_set():
                return None
            job = self.store.claim_next()
            if job is None:
                return None
            log = open(self.log_dir / f"{job.id}.log", "ab")  # noqa: SIM115 - closed below
            try:
                proc = subprocess.Popen(
                    list(self.job_command(job)),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as e:
                self.store.finish(job.id, job.attempt, JobStatus.failed, error=f"spawn: {e}")
                return None
            finally:
                log.close()
            self.store.set_pid(job.id, job.attempt, proc.pid)
            r = _Running(job, proc)
            self._running[job.id] = r
            return r

    def _supervise(self, r: _Running) -> None:
        job = r.job
        try:
            code = r.proc.wait()
            with self._lock:
                self._running.pop(job.id, None)
            if r.cancel_requested:
                self.store.finish(job.id, job.attempt, JobStatus.cancelled)
            elif self._stopping.is_set():
                self.store.requeue(job.id, job.attempt, refund=True)
            else:
                # The runner marks done/failed itself; this only fires if it died silently.
                self.store.finish(
                    job.id,
                    job.attempt,
                    JobStatus.failed,
                    error=f"worker exited with code {code}{self._log_tail(job.id)}",
                )
            final = self.store.get(job.id)
            if job.output_key and (final is None or final.status is not JobStatus.done):
                self.storage.delete(part_key(job.output_key))
                if final is None or final.status is not JobStatus.running:
                    self.storage.delete(job.output_key)
        finally:
            r.reaped.set()

    def _log_tail(self, job_id: str, n: int = 400) -> str:
        try:
            data = (self.log_dir / f"{job_id}.log").read_bytes()[-n:]
        except OSError:
            return ""
        text = data.decode("utf-8", "replace").strip()
        return f": {text}" if text else ""
