"""Hugging Face export (scripts/export_hf_release.py), reload (ouroboros.hf) and dry-run upload."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from ouroboros import hf
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.model import build_model

ROOT = Path(__file__).resolve().parents[1]
SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "N#CC(Cl)Br", "C[C@H](N)C(=O)O"]
IMAGE_SIZE = 64
DECODER = {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64, "max_len": 24}
BASELINE = {"type": "baseline", "widths": [8, 16, 16, 32], "blocks": [1, 1, 1, 1],
            "mixer_layers": 1, "mixer_heads": 2, "mixer_ff": 64}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(encoder: dict) -> dict:
    return {
        "seed": 3,
        "data": {"image_size": IMAGE_SIZE, "real_fraction": 0.7, "vocab": "/private/vocab.json"},
        "model": {"encoder": encoder, "decoder": DECODER},
        "train": {"steps": 10, "batch_size": 2, "lr": 3e-5,
                  "init_checkpoint": "/private/checkpoints/ouroboros-C.pt"},
    }


def _checkpoint(tmp_path: Path, encoder: dict = BASELINE):
    """A training-style checkpoint (pickled NumPy RNG state: needs weights_only=False)."""
    torch.manual_seed(0)
    cfg = _cfg(encoder)
    model = build_model(cfg, SmilesTokenizer.build(SMILES))
    torch.nn.init.normal_(model.decoder.tok.weight, std=1.0)
    with torch.no_grad():  # non-trivial BatchNorm statistics
        model.encoder(torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE))
    model.eval()
    with torch.no_grad():  # eval-mode forward: escnn caches its expanded filters as buffers
        model.encoder(torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE))
    path = tmp_path / "C-selected.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "vocab": model.tokenizer.itos,
                "step": 7, "rng": {"numpy": np.random.get_state()}}, path)
    return path, model


def _report(tmp_path: Path) -> Path:
    groups = [{"subset": s, "kind": None, "n": n, "exact": e, "exact_flat": e, "inchikey": e,
               "valid": 0.8, "tanimoto": 0.5, "exact_ci95": [e / 2, min(1.0, e * 1.5)],
               "stereo_elem_acc_given_flat": float("nan")}
              for s, n, e in [("ALL", 298, 0.2), ("ACS", 100, 0.05), ("UOB", 100, 0.3),
                              ("USPTO", 98, 0.25)]]
    report = {"input": "/content/drive/MyDrive/x/C_preds.csv", "n_rows": 298, "n_scored": 298,
              "gold_invalid": {"n": 0, "ids": []}, "groups": groups,
              "footnote": "Scoring parity unverified."}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))  # NaN, as the harness writes it
    return path


def _profiling(tmp_path: Path) -> Path:
    variant = {"samples": 100, "ms_per_image": {"encoder": 3.0, "decode": 20.0, "total": 23.0},
               "images_per_s": 43.5}
    prof = {"name": "C", "device": "cuda", "checkpoints": ["/content/drive/C-selected.pt"],
            "variants": {"beam1": variant, "beam5": variant}}
    path = tmp_path / "infer_C.json"
    path.write_text(json.dumps(prof))
    return path


def _export(tmp_path: Path, encoder: dict = BASELINE, **kw):
    ckpt, model = _checkpoint(tmp_path, encoder)
    out = tmp_path / "hf"
    _load_script("export_hf_release").export(ckpt, out, **kw)
    return out, model


@torch.no_grad()
def _logits(model, images, ids):
    enc = model.encoder(images)
    return model.decoder(ids, enc.tokens, enc.padding_mask)


def _inputs(n_vocab: int):
    g = torch.Generator().manual_seed(1)
    images = torch.rand(3, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    return images, torch.randint(0, n_vocab, (3, 12), generator=g)


def test_export_then_load_pretrained_gives_identical_logits(tmp_path):
    out, model = _export(tmp_path, arm="A", report=_report(tmp_path),
                         profiling=_profiling(tmp_path))
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "vocab.json", "README.md", "metrics.json", "checksums.txt"} <= names
    assert hf.weights_file(out).name in names and hf.weights_file(out, "encoder").name in names

    loaded = hf.load_pretrained(out)
    images, ids = _inputs(len(model.tokenizer))
    assert torch.equal(_logits(model, images, ids), _logits(loaded, images, ids))
    assert loaded.generate(images) == model.generate(images)
    assert loaded.tokenizer.itos == model.tokenizer.itos

    encoder = hf.read_weights(hf.weights_file(out, "encoder"))
    reference = model.encoder.state_dict()
    assert encoder.keys() == reference.keys()
    assert all(torch.equal(encoder[k], reference[k]) for k in reference)

    config = json.loads((out / "config.json").read_text())
    assert config["vocab_size"] == len(model.tokenizer) and config["encoder_type"] == "baseline"
    assert config["data"] == {"image_size": IMAGE_SIZE} and config["step"] == 7
    assert config["symmetry_group"].startswith("none")

    for line in (out / "checksums.txt").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert set(hf.read_checksums(out)) == names - {"checksums.txt"}

    def no_nan(token):
        raise AssertionError(f"non-JSON constant {token}")

    metrics = json.loads((out / "metrics.json").read_text(), parse_constant=no_nan)
    assert metrics["report"]["input"] == "C_preds.csv"  # no machine paths
    assert metrics["profiling"]["checkpoints"] == ["C-selected.pt"]

    card = (out / "README.md").read_text()
    assert "{{" not in card and "license: cc-by-nc-4.0" in card
    assert "| ACS | 100 | 5.00% (2.50%–7.50%) = 5/100 |" in card
    assert card.index("| ACS ") < card.index("| UOB ") < card.index("| USPTO ") < card.index(
        "| **all** "
    )
    assert "beam5" in card and "ouroboros-C.pt" in card  # profiling and training summary
    assert "/private" not in card and "/content" not in card
    assert "/private" not in (out / "config.json").read_text()


def test_load_pretrained_rejects_tampered_files(tmp_path):
    out, _ = _export(tmp_path)
    vocab = json.loads((out / "vocab.json").read_text())
    (out / "vocab.json").write_text(json.dumps(vocab[::-1]))
    with pytest.raises(ValueError, match="checksum"):
        hf.load_pretrained(out)


def test_export_refuses_non_empty_unrelated_folder(tmp_path):
    ckpt, _ = _checkpoint(tmp_path)
    keep = tmp_path / "data"
    keep.mkdir()
    (keep / "precious.txt").write_text("x")
    exporter = _load_script("export_hf_release")
    with pytest.raises(FileExistsError):
        exporter.export(ckpt, keep)
    with pytest.raises(FileExistsError, match="refusing"):
        exporter.export(ckpt, keep, overwrite=True)
    assert (keep / "precious.txt").exists()


def test_export_without_safetensors_falls_back_to_bin(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_safetensors", lambda: None)
    out, model = _export(tmp_path)
    assert (out / "pytorch_model.bin").is_file() and (out / "encoder.bin").is_file()
    assert not list(out.glob("*.safetensors"))
    images, ids = _inputs(len(model.tokenizer))
    assert torch.equal(_logits(model, images, ids), _logits(hf.load_pretrained(out), images, ids))


def test_steerable_export_drops_escnn_caches(tmp_path):
    pytest.importorskip("escnn")
    steerable = {"type": "steerable", "N": 4, "fields": [2, 2, 4, 4], "blocks": [1, 1, 1, 1],
                 "mixer_layers": 1, "mixer_heads": 2, "mixer_ff": 32}
    out, model = _export(tmp_path, encoder=steerable)
    assert any(k.endswith(".filter") for k in model.state_dict())  # the cache existed
    weights = hf.read_weights(hf.weights_file(out))
    assert not [k for k in weights if k.endswith((".filter", ".expanded_bias"))]
    config = json.loads((out / "config.json").read_text())
    assert config["symmetry_group"] == "C4 rotations only, no reflections"
    images, ids = _inputs(len(model.tokenizer))
    expected = _logits(model, images, ids)
    raw = hf.load_pretrained(out, export_encoder=False)
    assert torch.equal(_logits(raw, images, ids), expected)
    exported = hf.load_pretrained(out)  # pure-PyTorch encoder, as predict_real.py uses
    assert torch.allclose(_logits(exported, images, ids), expected, atol=1e-4)


# ------------------------------------------------------------------------------- upload


@pytest.fixture
def offline(monkeypatch):
    """Any huggingface_hub import or socket use fails the test."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import -> ImportError

    def no_network(*args, **kwargs):
        raise AssertionError("network access during a dry run")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)


def test_upload_dry_run_lists_files_without_hub_or_network(tmp_path, offline, capsys):
    out, _ = _export(tmp_path)
    (out / "notes-local.txt").write_text("not part of the release")
    assert _load_script("upload_hf").main([str(out), "--private"]) == 0
    text = capsys.readouterr().out
    assert "DRY RUN" in text and "huggingface.co/yaniguan/ouroboros-ocsr-encoder" in text
    assert "private" in text and "config.json" in text and "checksums.txt" in text
    assert "notes-local.txt" not in text
    assert sys.modules["huggingface_hub"] is None  # never (re)imported


def test_upload_refuses_corrupted_export(tmp_path, offline):
    out, _ = _export(tmp_path)
    (out / "README.md").write_text("edited after export")
    with pytest.raises(SystemExit, match="checksum"):
        _load_script("upload_hf").main([str(out)])


def test_upload_with_yes_sends_only_listed_files_and_no_token(tmp_path, monkeypatch):
    out, _ = _export(tmp_path)
    (out / "notes-local.txt").write_text("x")
    calls = []

    class FakeApi:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def create_repo(self, repo_id, **kwargs):
            calls.append(("create_repo", repo_id, kwargs))
            return f"https://huggingface.co/{repo_id}"

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))
            return types.SimpleNamespace(commit_url="https://huggingface.co/x/commit/1")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    assert _load_script("upload_hf").main([str(out), "--repo-id", "me/test", "--yes"]) == 0
    assert calls[0] == ("init", {})  # authentication left to huggingface_hub
    assert calls[1] == ("create_repo", "me/test",
                        {"repo_type": "model", "private": False, "exist_ok": True})
    sent = calls[2][1]
    assert "token" not in sent and sent["repo_id"] == "me/test"
    assert set(sent["allow_patterns"]) == set(hf.read_checksums(out)) | {"checksums.txt"}
