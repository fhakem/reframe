"""``python -m reframe.api --data-dir ./data --port 8000``: run the API with uvicorn."""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from reframe.api.app import create_app

    p = argparse.ArgumentParser(prog="reframe.api", description="Reframe HTTP API + web UI")
    p.add_argument("--host", default=os.environ.get("REFRAME_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("REFRAME_PORT", "8000")))
    p.add_argument("--data-dir", default=os.environ.get("REFRAME_DATA_DIR", "./reframe-data"))
    p.add_argument("--max-upload-mb", type=float, default=500.0)
    p.add_argument("--concurrency", type=int, default=1, help="parallel jobs")
    p.add_argument("--web-dist", default=None, help="built front-end dir (default: web/dist)")
    a = p.parse_args(argv)
    app = create_app(
        a.data_dir,
        max_upload_mb=a.max_upload_mb,
        concurrency=a.concurrency,
        web_dist=a.web_dist,
    )
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
