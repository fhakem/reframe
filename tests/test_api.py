"""M5: HTTP API + SQLite job queue + subprocess worker."""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")

from fastapi.testclient import TestClient

from reframe.api.app import create_app
from reframe.core.io.video import probe
from reframe.worker.jobs import Job, JobStatus, JobStore
from reframe.worker.store import (
    LocalStorage,
    SizeLimitExceeded,
    StorageError,
    new_key,
    sanitize_filename,
)
from tests.conftest import make_clip, needs_ffmpeg

# A fake job: spawns a grandchild (to check the whole process group dies), records its
# pid, then sleeps. Argv is built by the test, never from user input.
_SLEEPER = (
    "import subprocess, sys, time\n"
    "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "open(sys.argv[1], 'w').write(str(c.pid))\n"
    "time.sleep(120)\n"
)


def sleeper_command(tmp: Path):
    def cmd(job: Job) -> list[str]:
        return [sys.executable, "-c", _SLEEPER, str(tmp / f"{job.id}.child")]

    return cmd


def wait_for(pred, timeout: float = 20.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError("timed out waiting for condition")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover
        return True
    return True


def fake_job(client: TestClient) -> str:
    """Insert a queued job directly (no upload / probe needed)."""
    storage: LocalStorage = client.app.state.storage  # type: ignore[attr-defined]
    store: JobStore = client.app.state.store  # type: ignore[attr-defined]
    key = new_key("uploads", "mp4")
    storage.put(key, io.BytesIO(b"x"))
    job = store.create(config={}, input_key=key, original_name="fake.mp4")
    client.app.state.queue.notify()  # type: ignore[attr-defined]
    return job.id


@pytest.fixture
def sleeper_client(tmp_path: Path):
    app = create_app(tmp_path / "data", job_command=sleeper_command(tmp_path))
    with TestClient(app) as c:
        yield c


# -- basics ------------------------------------------------------------------


def test_health_and_schema(sleeper_client: TestClient):
    r = sleeper_client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["workers_alive"] is True
    s = sleeper_client.get("/api/config/schema").json()
    assert "seam_ratio" in s["schema"]["properties"]
    assert s["defaults"]["seam_ratio"] == pytest.approx(0.25)
    assert s["presets"]["tiktok"] == {"width": 1080, "height": 1920}
    assert ".mp4" in s["accept"]


def test_unknown_job_404(sleeper_client: TestClient):
    jid = "0" * 32
    for method, path in [
        ("GET", f"/api/jobs/{jid}"),
        ("GET", f"/api/jobs/{jid}/result"),
        ("GET", f"/api/jobs/{jid}/source"),
        ("POST", f"/api/jobs/{jid}/cancel"),
    ]:
        assert sleeper_client.request(method, path).status_code == 404
    # not a uuid -> validation error, never touches storage
    assert sleeper_client.get("/api/jobs/..%2f..%2fetc").status_code in (404, 422)


def test_rejects_unknown_extension(sleeper_client: TestClient):
    r = sleeper_client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert sleeper_client.get("/api/jobs").json()["jobs"] == []


@needs_ffmpeg
def test_rejects_non_video_content(sleeper_client: TestClient, tmp_path: Path):
    r = sleeper_client.post(
        "/api/jobs", files={"file": ("fake.mp4", os.urandom(4096), "video/mp4")}
    )
    assert r.status_code == 422
    uploads = tmp_path / "data" / "files" / "uploads"
    assert not uploads.exists() or list(uploads.iterdir()) == []


def test_rejects_invalid_config(sleeper_client: TestClient):
    for cfg in ['{"seam_ratio": 0.9}', "not json", "[1, 2]", '{"preset": "myspace"}']:
        r = sleeper_client.post(
            "/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")}, data={"config": cfg}
        )
        assert r.status_code == 422, cfg


def test_oversize_rejected_413(tmp_path: Path):
    app = create_app(tmp_path / "data", max_upload_mb=0.01, job_command=sleeper_command(tmp_path))
    with TestClient(app) as c:
        r = c.post("/api/jobs", files={"file": ("big.mp4", b"\0" * 200_000, "video/mp4")})
        assert r.status_code == 413
        assert c.get("/api/jobs").json()["jobs"] == []


def test_oversize_chunked_upload_413(tmp_path: Path):
    """No Content-Length (chunked): the limit is enforced while streaming the body."""
    app = create_app(tmp_path / "data", max_upload_mb=0.01, job_command=sleeper_command(tmp_path))
    boundary = "xBOUNDARYx"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode()

    def body():
        yield head
        for _ in range(50):
            yield b"\0" * 8192
        yield f"\r\n--{boundary}--\r\n".encode()

    with TestClient(app) as c:
        r = c.post(
            "/api/jobs",
            content=body(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        assert r.status_code == 413
        assert "content-length" not in r.request.headers
        assert c.get("/api/jobs").json()["jobs"] == []


def test_storage_limits_and_keys(tmp_path: Path):
    st = LocalStorage(tmp_path)
    with pytest.raises(SizeLimitExceeded):
        st.put("uploads/" + "a" * 32 + ".mp4", io.BytesIO(b"x" * 100), max_bytes=10)
    assert not any(tmp_path.rglob("*.mp4"))
    for bad in ["../etc/passwd", "uploads/../../x.mp4", "/abs.mp4", "uploads/evil.mp4"]:
        with pytest.raises(StorageError):
            st.path(bad)
    assert sanitize_filename("../../etc/pa<ss>wd.mp4") == "passwd.mp4"
    assert sanitize_filename("C:\\Users\\x\\clip.MOV") == "clip.MOV"
    assert sanitize_filename("") == "video"


# -- job lifecycle -----------------------------------------------------------


def test_result_before_done_409_and_cancel_running(sleeper_client: TestClient, tmp_path: Path):
    jid = fake_job(sleeper_client)
    job = wait_for(
        lambda: (j := sleeper_client.get(f"/api/jobs/{jid}").json())["status"] == "running" and j
    )
    assert job["progress"] == 0
    assert sleeper_client.get(f"/api/jobs/{jid}/result").status_code == 409

    store: JobStore = sleeper_client.app.state.store  # type: ignore[attr-defined]
    pid = wait_for(lambda: store.get(jid).pid)  # type: ignore[union-attr]
    child_file = tmp_path / f"{jid}.child"
    child = int(wait_for(lambda: child_file.exists() and child_file.read_text()))
    assert pid_alive(pid) and pid_alive(child)

    # a second job waits in the queue (concurrency 1) and can be cancelled there
    queued = fake_job(sleeper_client)
    assert sleeper_client.get(f"/api/jobs/{queued}").json()["status"] == "queued"
    r = sleeper_client.post(f"/api/jobs/{queued}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"

    r = sleeper_client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert not pid_alive(pid)
    wait_for(lambda: not pid_alive(child), timeout=5)  # grandchild is reparented, then dies
    assert sleeper_client.post(f"/api/jobs/{jid}/cancel").status_code == 409
    assert sleeper_client.get(f"/api/jobs/{jid}/result").status_code == 409
    listed = {j["id"]: j["status"] for j in sleeper_client.get("/api/jobs").json()["jobs"]}
    assert listed == {jid: "cancelled", queued: "cancelled"}


def test_restart_recovery(tmp_path: Path):
    data = tmp_path / "data"
    store = JobStore(data / "jobs.db")
    storage = LocalStorage(data / "files")
    ids = []
    for attempts in (1, 3):
        key = f"uploads/{attempts:032d}.mp4"
        storage.put(key, io.BytesIO(b"x"))
        jid = store.create(config={}, input_key=key, original_name="a.mp4").id
        with store._conn() as c:  # simulate a worker that died mid-job
            c.execute(
                "UPDATE jobs SET status='running', attempt=?, pid=999999 WHERE id=?",
                (attempts, jid),
            )
        ids.append(jid)
    retry, exhausted = ids

    app = create_app(data, job_command=sleeper_command(tmp_path))
    with TestClient(app) as c:
        assert app.state.queue.recovered == ([retry], [exhausted])
        j = c.get(f"/api/jobs/{exhausted}").json()
        assert j["status"] == "failed" and "worker died" in j["error"]
        # re-queued, then picked up again by the new worker as attempt 2
        j = wait_for(lambda: (j := c.get(f"/api/jobs/{retry}").json())["attempt"] == 2 and j)
        assert j["status"] == "running"

    # graceful shutdown puts the running job back in the queue with its attempt refunded
    j2 = JobStore(data / "jobs.db").get(retry)
    assert j2 is not None and j2.status is JobStatus.queued and j2.attempt == 1


@needs_ffmpeg
@pytest.mark.slow
def test_upload_to_result_end_to_end(tmp_path: Path):
    clip = make_clip(tmp_path / "tiny clip.mp4", size="160x90", seconds=0.5)
    app = create_app(tmp_path / "data")
    with TestClient(app) as c:
        with open(clip, "rb") as f:
            r = c.post(
                "/api/jobs",
                files={"file": ("../../tiny clip.mp4", f, "video/mp4")},
                data={"config": '{"seam_ratio": 0.25, "x264_preset": "ultrafast", "crf": 30}'},
            )
        assert r.status_code == 201, r.text
        job = r.json()
        assert job["status"] in ("queued", "running")
        assert job["original_name"] == "tiny clip.mp4"
        assert job["meta"]["width"] == 160 and job["meta"]["height"] == 90
        assert job["config"]["x264_preset"] == "ultrafast"

        src = c.get(f"/api/jobs/{job['id']}/source")
        assert src.status_code == 200 and src.content == clip.read_bytes()

        final = wait_for(
            lambda: (
                (j := c.get(f"/api/jobs/{job['id']}").json())["status"] in ("done", "failed") and j
            ),
            timeout=50,
            interval=0.2,
        )
        assert final["status"] == "done", final
        assert final["progress"] == 1

        res = c.get(f"/api/jobs/{job['id']}/result?download=1")
        assert res.status_code == 200
        assert res.headers["content-type"] == "video/mp4"
        assert "attachment" in res.headers["content-disposition"]
        out = tmp_path / "out.mp4"
        out.write_bytes(res.content)
        info = probe(out)
        assert (info.width, info.height) == (1080, 1920)
        assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 409


def test_serves_web_dist(tmp_path: Path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Reframe</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(tmp_path / "data", web_dist=dist, job_command=sleeper_command(tmp_path))
    with TestClient(app) as c:
        assert "Reframe" in c.get("/").text
        assert c.get("/assets/app.js").status_code == 200
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/../jobs.db").status_code == 404


def test_main_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import uvicorn
    from fastapi import FastAPI

    from reframe.api.__main__ import main

    seen: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    main(["--data-dir", str(tmp_path / "d"), "--port", "8123", "--concurrency", "2"])
    assert isinstance(seen["app"], FastAPI)
    assert seen["port"] == 8123 and seen["host"] == "127.0.0.1"
    assert (tmp_path / "d" / "jobs.db").is_file()
