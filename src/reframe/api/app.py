"""FastAPI application: upload -> job queue -> download.

Routes (all JSON unless noted)::

    GET  /api/health
    GET  /api/config/schema         ReframeConfig JSON schema + presets + defaults
    POST /api/jobs                  multipart: file=<video>, config=<JSON object>
    GET  /api/jobs                  newest first
    GET  /api/jobs/{id}
    POST /api/jobs/{id}/cancel
    GET  /api/jobs/{id}/result      video/mp4 (409 until done; ?download=1 for attachment)
    GET  /api/jobs/{id}/source      original upload (for the preview player)

The built front-end (``web/dist``) is served at ``/`` when present.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.types import Message

from reframe.config import PRESET_SIZES, ReframeConfig
from reframe.core.io.video import FFmpegError, probe
from reframe.worker.jobs import Job, JobCommand, JobQueue, JobStatus, JobStore
from reframe.worker.store import LocalStorage, SizeLimitExceeded, new_key, sanitize_filename

#: Accepted upload extensions -> media type used when streaming the source back.
VIDEO_TYPES: dict[str, str] = {
    "mp4": "video/mp4",
    "m4v": "video/mp4",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
}
_FORM_OVERHEAD = 64 * 1024  # multipart boundaries + config field
_MAX_CONFIG_BYTES = 64 * 1024


class _BodyTooLarge(Exception):
    pass


def _default_web_dist() -> Path | None:
    env = os.environ.get("REFRAME_WEB_DIST")
    if env:
        return Path(env)
    candidate = Path(__file__).resolve().parents[3] / "web" / "dist"
    return candidate if candidate.is_dir() else None


def create_app(
    data_dir: str | Path,
    *,
    max_upload_mb: float = 500.0,
    concurrency: int = 1,
    max_attempts: int = 3,
    max_duration_s: float | None = None,
    web_dist: str | Path | None = None,
    job_command: JobCommand | None = None,
) -> FastAPI:
    """Build the app. ``data_dir`` holds ``jobs.db``, ``files/`` (uploads, results) and ``logs/``.

    ``job_command`` overrides the job subprocess argv (tests use it to run a fake job).
    """
    data = Path(data_dir).resolve()
    data.mkdir(parents=True, exist_ok=True)
    store = JobStore(data / "jobs.db")
    storage = LocalStorage(data / "files")
    queue = JobQueue(
        store,
        storage,
        concurrency=concurrency,
        max_attempts=max_attempts,
        log_dir=data / "logs",
        job_command=job_command,
    )
    max_bytes = int(max_upload_mb * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        queue.start()
        try:
            yield
        finally:
            await run_in_threadpool(queue.stop)

    app = FastAPI(title="Reframe", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.storage = storage
    app.state.queue = queue
    app.state.max_upload_bytes = max_bytes

    def get_job(job_id: uuid.UUID) -> Job:
        job = store.get(job_id.hex)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
            "workers": queue.concurrency,
            "workers_alive": queue.is_alive(),
            "max_upload_mb": max_upload_mb,
        }

    @app.get("/api/config/schema")
    def config_schema() -> dict[str, Any]:
        return {
            "schema": ReframeConfig.model_json_schema(),
            "defaults": ReframeConfig().model_dump(mode="json"),
            "presets": {p.value: {"width": w, "height": h} for p, (w, h) in PRESET_SIZES.items()},
            "accept": sorted(f".{e}" for e in VIDEO_TYPES),
            "max_upload_mb": max_upload_mb,
        }

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request) -> dict[str, Any]:
        declared = request.headers.get("content-length")
        limit = max_bytes + _FORM_OVERHEAD
        if declared is not None and declared.isdigit() and int(declared) > limit:
            raise HTTPException(413, f"upload larger than {max_upload_mb:g} MB")

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            msg = await request.receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return msg

        limited = Request(request.scope, limited_receive)
        try:
            form = await limited.form(max_files=1, max_fields=4, max_part_size=_MAX_CONFIG_BYTES)
        except _BodyTooLarge:
            raise HTTPException(413, f"upload larger than {max_upload_mb:g} MB") from None
        try:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(422, "missing 'file' part")
            name = sanitize_filename(upload.filename)
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in VIDEO_TYPES:
                raise HTTPException(
                    415, f"unsupported file type; accepted: {', '.join(sorted(VIDEO_TYPES))}"
                )
            raw_cfg = form.get("config") or "{}"
            if not isinstance(raw_cfg, str):
                raise HTTPException(422, "'config' must be a JSON string field")
            try:
                cfg_obj = json.loads(raw_cfg)
            except json.JSONDecodeError as e:
                raise HTTPException(422, f"config is not valid JSON: {e}") from None
            if not isinstance(cfg_obj, dict):
                raise HTTPException(422, "config must be a JSON object")
            try:
                cfg = ReframeConfig.model_validate(cfg_obj)
            except ValidationError as e:
                raise HTTPException(
                    422, {"message": "invalid config", "errors": e.errors(include_url=False,
                                                                          include_context=False)}
                ) from None  # fmt: skip

            key = new_key("uploads", ext)
            try:
                await run_in_threadpool(storage.put, key, upload.file, max_bytes=max_bytes)
            except SizeLimitExceeded:
                raise HTTPException(413, f"upload larger than {max_upload_mb:g} MB") from None
        finally:
            await form.close()

        try:
            info = await run_in_threadpool(probe, storage.path(key))
            if info.width < 2 or info.height < 2:
                raise FFmpegError("empty video stream")
            if max_duration_s is not None and info.duration > max_duration_s:
                storage.delete(key)
                raise HTTPException(422, f"video longer than {max_duration_s:g} s")
        except FFmpegError:
            storage.delete(key)
            raise HTTPException(422, "file is not a decodable video") from None

        meta = {
            "width": info.width,
            "height": info.height,
            "fps": float(info.fps),
            "duration": info.duration,
            "n_frames": info.n_frames,
            "has_audio": info.has_audio,
            "source_type": VIDEO_TYPES[ext],
        }
        job = store.create(
            config=cfg.model_dump(mode="json"), input_key=key, original_name=name, meta=meta
        )
        queue.notify()
        return job.public()

    @app.get("/api/jobs")
    def list_jobs(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"jobs": [j.public() for j in store.list_jobs(limit=limit)]}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: uuid.UUID) -> dict[str, Any]:
        return get_job(job_id).public()

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: uuid.UUID) -> dict[str, Any]:
        job = get_job(job_id)
        if job.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
            raise HTTPException(409, f"job already {job.status.value}")
        updated = await run_in_threadpool(queue.cancel, job.id)
        return (updated or job).public()

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: uuid.UUID, download: bool = False) -> FileResponse:
        job = get_job(job_id)
        if job.status is not JobStatus.done or not job.output_key:
            raise HTTPException(409, f"job is {job.status.value}, not done")
        path = storage.path(job.output_key)
        if not path.is_file():
            raise HTTPException(410, "result file is missing")
        stem = job.original_name.rsplit(".", 1)[0] or "video"
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{stem}_9x16.mp4",
            content_disposition_type="attachment" if download else "inline",
        )

    @app.get("/api/jobs/{job_id}/source")
    def job_source(job_id: uuid.UUID) -> FileResponse:
        job = get_job(job_id)
        path = storage.path(job.input_key)
        if not path.is_file():
            raise HTTPException(410, "source file is missing")
        return FileResponse(path, media_type=str(job.meta.get("source_type", "video/mp4")))

    dist = Path(web_dist) if web_dist is not None else _default_web_dist()
    if dist is not None and (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app
