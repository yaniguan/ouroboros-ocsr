import hashlib
import io
import json

import pytest
import torch

from ouroboros.weights import export_checkpoint, fetch


def test_export_preserves_weights_without_training_objects(tmp_path):
    original = tmp_path / "train.pt"
    model = {"weight": torch.randn(5, 7), "counter": torch.tensor(3)}
    torch.save(
        {
            "model": model,
            "config": {
                "data": {"image_size": 384, "synthetic_root": "/private/data"},
                "model": {"encoder": {"type": "baseline"}},
            },
            "vocab": ["<pad>"],
            "step": 100,
            "optimizer": {"large": torch.ones(100)},
            "rng": (1, 2, 3),
        },
        original,
    )
    output = tmp_path / "inference.pt"
    receipt = export_checkpoint(original, output, "A")
    restored = torch.load(output, weights_only=True)
    assert "optimizer" not in restored and "rng" not in restored
    assert restored["config"]["data"] == {"image_size": 384}
    assert torch.equal(restored["model"]["weight"], model["weight"])
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_checkpoint(original, output, "A")


def test_fetch_verifies_download_and_never_installs_corruption(tmp_path, monkeypatch):
    data = b"test checkpoint"
    entry = {
        "filename": "test.pt",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "url": "https://github.com/yaniguan/ouroboros-ocsr/releases/download/test/test.pt",
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="checksum"):
        fetch(entry, tmp_path)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(data))
    assert fetch(entry, tmp_path).read_bytes() == data
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **kw: pytest.fail("cached file redownloaded")
    )
    assert fetch(entry, tmp_path).read_bytes() == data
    with pytest.raises(ValueError, match="filename"):
        fetch(entry | {"filename": "../escape.pt"}, tmp_path)


def test_release_manifest_complete():
    from pathlib import Path

    manifest = json.loads((Path(__file__).parents[1] / "ouroboros/weights.json").read_text())
    assert set(manifest["weights"]) == {"A", "B", "C"}
    for arm, entry in manifest["weights"].items():
        assert entry["filename"] == f"ouroboros-{arm}.pt"
        assert len(entry["sha256"]) == 64 and int(entry["sha256"], 16) > 0
        assert entry["bytes"] > 1_000_000
