"""Content identities and atomic metadata publication (single writer per output directory)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".part", delete=False
        ) as f:
            tmp = Path(f.name)
            json.dump(value, f, sort_keys=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def implementation_identity(root: str | Path) -> dict:
    root = Path(root)
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))}


class JsonCache:
    """Content-addressed completed jobs; atomic writes and corruption checks."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, settings, compute):
        path = self.directory / f"{json_sha256(settings)}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["settings"] != settings or saved["sha256"] != json_sha256(saved["result"]):
                raise ValueError(f"Corrupt cache entry: {path}")
            return saved["result"]
        result = compute()
        atomic_json(path, {"settings": settings, "result": result, "sha256": json_sha256(result)})
        return result
