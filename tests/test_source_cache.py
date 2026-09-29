import hashlib
import io
import json
from dataclasses import asdict

import pytest

from ouroboros.data import build, pool_cache, sources


def test_filter_restart_skips_committed_chemistry_and_preserves_dedup(tmp_path, monkeypatch):
    smiles = ["CCCCC", "CCCCCC", "CCCCC", "invalid", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"]
    path = tmp_path / "pool.tsv.gz"
    original = pool_cache._process_batch
    calls = []

    def interrupted(processor, batch, pool):
        calls.append(batch)
        if len(calls) == 2:
            raise RuntimeError("disconnect")
        return original(processor, batch, pool)

    monkeypatch.setattr(pool_cache, "_process_batch", interrupted)
    with pytest.raises(RuntimeError, match="disconnect"):
        build.prepare_pool([("fixture", smiles)], path, workers=1, batch_size=2)
    calls.clear()

    def resumed(processor, batch, pool):
        calls.append(batch)
        return original(processor, batch, pool)

    monkeypatch.setattr(pool_cache, "_process_batch", resumed)
    report = build.prepare_pool([("fixture", smiles)], path, workers=1, batch_size=2)
    assert len(calls) == 2 and calls[0][0][1] == smiles[2]
    assert report["n_in"] == 6 and report["n_out"] == 3
    assert report["rejects"]["duplicate"] == 2
    assert build.prepare_pool([("fixture", smiles)], path, workers=1) == report
    assert len(calls) == 2
    clean = tmp_path / "clean.tsv.gz"
    build.prepare_pool([("fixture", smiles)], clean, workers=2, batch_size=3)
    assert path.read_bytes() == clean.read_bytes()
    with pytest.raises(ValueError, match="identity changed"):
        build.prepare_pool([("fixture", smiles[::-1])], path, workers=1)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build.prepare_pool([("fixture", smiles)], path, workers=1)


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status, self.headers = status, headers or {}


@pytest.mark.parametrize("honor_range", [True, False])
def test_download_resumes_or_restarts_when_range_is_ignored(tmp_path, monkeypatch, honor_range):
    data = b"abcdefghij"
    src = sources.Source(
        "fixture",
        "https://example.test/source",
        "fixture.csv",
        "smiles",
        len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    monkeypatch.setitem(sources.SOURCES, "fixture", src)

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise OSError("disconnect")
            return super().read(4)

    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda *a, **k: Interrupted(data))
    with pytest.raises(OSError, match="disconnect"):
        sources.fetch("fixture", tmp_path)
    assert (tmp_path / "fixture.csv.part").read_bytes() == data[:4]

    def fetch(request, timeout):
        assert request.headers["Range"] == "bytes=4-"
        return (
            Response(data[4:], 206, {"Content-Range": "bytes 4-9/10"})
            if honor_range
            else Response(data)
        )

    monkeypatch.setattr(sources.urllib.request, "urlopen", fetch)
    path = sources.fetch("fixture", tmp_path)
    assert path.read_bytes() == data
    assert sources.fetch("fixture", tmp_path) == path
    path.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError, match="checksum mismatch"):
        sources.fetch("fixture", tmp_path)


def test_wrong_range_never_publishes_source(tmp_path, monkeypatch):
    src = sources.Source(
        "fixture",
        "https://example.test/source",
        "f.csv",
        "smiles",
        10,
        sha256=hashlib.sha256(b"abcdefghij").hexdigest(),
    )
    monkeypatch.setitem(sources.SOURCES, "fixture", src)
    (tmp_path / "f.csv.part").write_bytes(b"abcd")
    (tmp_path / "f.csv.download.json").write_text(json.dumps(asdict(src)))
    monkeypatch.setattr(
        sources.urllib.request,
        "urlopen",
        lambda *a, **k: Response(b"efghij", 206, {"Content-Range": "bytes 0-5/10"}),
    )
    with pytest.raises(ValueError, match="Content-Range"):
        sources.fetch("fixture", tmp_path)
    assert not (tmp_path / "f.csv").exists()
