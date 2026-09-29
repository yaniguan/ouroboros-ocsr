import json

import pytest

from ouroboros.data import build
from ouroboros.data.loader import index_shard


@pytest.fixture
def rows():
    return [
        {"idx": i, "key14": str(i), "smiles": s, "stereo": 0, "src": "fixture"}
        for i, s in enumerate(["CCCCC", "CCCCCC", "CCCCCCC"])
    ]


def test_shard_resume_verifies_inputs_and_bytes(rows, tmp_path):
    kw = dict(shard_size=2, size=64, workers=1)
    build.render_shards(rows, "train", tmp_path, **kw)
    assert build.render_shards(rows, "train", tmp_path, **kw)["shards_written"] == 0
    for changed, opts in [(rows[::-1], kw), (rows, {**kw, "seed": 1}), (rows, {**kw, "size": 128})]:
        with pytest.raises(ValueError, match="identity changed"):
            build.render_shards(changed, "train", tmp_path, **opts)
    shard = next(tmp_path.glob("*.tar"))
    with shard.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="corrupt shard"):
        build.render_shards(rows, "train", tmp_path, **kw)


def test_render_restart_after_published_shard(rows, tmp_path, monkeypatch):
    original = build._render_shard

    def interrupt(job):
        if job[0][0]["idx"] >= 2:
            raise RuntimeError("disconnect")
        return original(job)

    monkeypatch.setattr(build, "_render_shard", interrupt)
    with pytest.raises(RuntimeError, match="disconnect"):
        build.render_shards(rows, "train", tmp_path, shard_size=2, size=64, workers=1)
    first = (tmp_path / "train-000000.tar").read_bytes()
    monkeypatch.setattr(build, "_render_shard", original)
    result = build.render_shards(rows, "train", tmp_path, shard_size=2, size=64, workers=1)
    assert result["shards_written"] == 1
    assert (tmp_path / "train-000000.tar").read_bytes() == first
    idx = tmp_path / "train-000000.idx.json"
    entries = index_shard(tmp_path / "train-000000.tar")
    idx.write_text(json.dumps({"sha256": "old", "entries": []}))
    assert index_shard(tmp_path / "train-000000.tar") == entries


def test_unverified_legacy_shards_are_not_adopted(tmp_path, rows):
    (tmp_path / "train-000000.tar").write_bytes(b"old")
    with pytest.raises(ValueError, match="no generation identity"):
        build.render_shards(rows, "train", tmp_path, workers=1)


def test_composition_identity_includes_exclusion_contents():
    a = build.ComposeConfig(exclude_key14=frozenset({"a"}))
    b = build.ComposeConfig(exclude_key14=frozenset({"b"}))
    assert build.composition_identity([], a) != build.composition_identity([], b)
