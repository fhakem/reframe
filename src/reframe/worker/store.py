"""Blob storage for uploaded sources and rendered results.

Keys are generated server-side (``<kind>/<uuid4 hex>.<ext>``); a client
filename is never used to build a path. :class:`LocalStorage` maps keys to
files under a root directory. An S3 / MinIO backend can implement the same
:class:`Storage` protocol later (``path()`` would then download to a cache).
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

_KIND_RE = re.compile(r"^[a-z]{1,16}$")
_EXT_RE = re.compile(r"^[a-z0-9]{1,8}$")
_KEY_RE = re.compile(r"^[a-z]{1,16}/[0-9a-f]{32}(?:\.part)?\.[a-z0-9]{1,8}$")
_CHUNK = 1 << 20


class StorageError(ValueError):
    pass


class SizeLimitExceeded(StorageError):
    pass


def new_key(kind: str, ext: str) -> str:
    """Fresh opaque key, e.g. ``uploads/3f2a...e1.mp4``."""
    ext = ext.lower().lstrip(".")
    if not _KIND_RE.match(kind) or not _EXT_RE.match(ext):
        raise StorageError(f"invalid key parts: {kind!r}, {ext!r}")
    return f"{kind}/{uuid.uuid4().hex}.{ext}"


def part_key(key: str) -> str:
    """Temporary sibling key used while a file is being written."""
    check_key(key)
    stem, ext = key.rsplit(".", 1)
    return f"{stem}.part.{ext}"


def check_key(key: str) -> str:
    if not _KEY_RE.match(key):
        raise StorageError(f"invalid storage key: {key!r}")
    return key


def sanitize_filename(name: str | None, *, max_len: int = 120) -> str:
    """Display-only version of a client filename (never used as a path)."""
    if not name:
        return "video"
    name = unicodedata.normalize("NFKC", name).replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*')
    name = name.strip(" .")
    return name[:max_len] or "video"


@runtime_checkable
class Storage(Protocol):
    def put(self, key: str, data: BinaryIO, *, max_bytes: int | None = None) -> int:
        """Store a stream under ``key``; returns its size. Raises SizeLimitExceeded."""
        ...

    def path(self, key: str) -> Path:
        """Local filesystem path for ``key`` (it may not exist yet)."""
        ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None:
        """Remove ``key``; missing keys are ignored."""
        ...

    def rename(self, src: str, dst: str) -> None: ...


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        check_key(key)
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root):  # defence in depth; the regex already forbids it
            raise StorageError(f"key escapes storage root: {key!r}")
        return p

    def put(self, key: str, data: BinaryIO, *, max_bytes: int | None = None) -> int:
        dst = self.path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path(part_key(key))
        size = 0
        try:
            with open(tmp, "xb") as out:
                while chunk := data.read(_CHUNK):
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise SizeLimitExceeded(f"upload exceeds {max_bytes} bytes")
                    out.write(chunk)
            os.replace(tmp, dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return size

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)

    def rename(self, src: str, dst: str) -> None:
        d = self.path(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.path(src), d)

    def disk_free(self) -> int:
        return shutil.disk_usage(self.root).free
