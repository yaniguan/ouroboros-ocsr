"""Validation boundaries preserve training RNG and committed best-model selection."""

import copy
import importlib.util
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.train.trainer import Trainer, load_checkpoint, rng_state, set_rng_state


@pytest.fixture
def config(tmp_path):
    """Small deterministic blank-image fixture, not a research accuracy benchmark."""
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    labels = ["CCCCC", "CCCCCC", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"]
    blob = io.BytesIO()
    Image.new("L", (32, 32), 255).save(blob, format="PNG")
    for split in ("train", "val"):
        with tarfile.open(root / f"{split}-000000.tar", "w") as tar:
            for i, label in enumerate(labels):
                for ext, data in (
                    ("png", blob.getvalue()),
                    ("json", json.dumps({"smiles": label}).encode()),
                ):
                    item = tarfile.TarInfo(f"{split}_{i}.{ext}")
                    item.size = len(data)
                    tar.addfile(item, io.BytesIO(data))
    vocab = tmp_path / "vocab.json"
    SmilesTokenizer.build(labels).save(vocab)
    return {
        "seed": 7,
        "data": {
            "synthetic_root": str(root),
            "synthetic_max_samples": 4,
            "real_roots": [],
            "image_size": 32,
            "vocab": str(vocab),
            "num_workers": 0,
        },
        "model": {
            "encoder": {
                "type": "baseline",
                "widths": [4, 4, 8, 8],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 32,
            },
            "decoder": {
                "d_model": 16,
                "n_layers": 2,
                "n_heads": 2,
                "d_ff": 32,
                "max_len": 32,
                "dropout": 0.1,
            },
        },
        "train": {
            "device": "cpu",
            "amp": "none",
            "steps": 10,
            "batch_size": 2,
            "warmup": 1,
            "lr": 1e-3,
            "ckpt_every": 9,
            "log_every": 1,
        },
        "validation": {
            "enabled": True,
            "root": str(root),
            "samples": 4,
            "batch_size": 2,
            "every": 2,
            "patience": 2,
            "min_delta": 0.1,
        },
    }


def evaluation_module():
    spec = importlib.util.spec_from_file_location("evaluation_cli", Path("scripts/evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_stereo_and_rng(config, tmp_path, monkeypatch):
    trainer = Trainer(config, tmp_path / "run")
    pending = iter(["CCCCC", "CCCCCC", "C[C@@H](N)C(=O)O", "C[C@@H](N)C(=O)O"])

    def generate(images):
        assert not trainer.model.training
        torch.rand(3)
        np.random.rand()
        random.random()
        return [next(pending) for _ in images]

    monkeypatch.setattr(trainer.model, "generate", generate)
    before = rng_state()
    assert trainer.validate() == 0.75  # a swapped tetrahedral center is incorrect
    assert trainer.model.training
    observed = (torch.rand(3), np.random.rand(), random.random())
    set_rng_state(before)
    assert torch.equal(observed[0], torch.rand(3))
    assert observed[1:] == (np.random.rand(), random.random())


def test_resume_preserves_patience_losses_and_true_best(config, tmp_path, monkeypatch):
    scores = {2: 0.5, 4: 0.55, 6: 0.5}
    monkeypatch.setattr(Trainer, "validate", lambda self: scores[self.step])
    full = Trainer(config, tmp_path / "full")
    full.fit()
    cut = Trainer(config, tmp_path / "cut")
    cut.fit(stop_at=5)  # commits validation at4; step5 must replay after disconnect
    resumed = Trainer(config, tmp_path / "cut")
    assert resumed.step == 4 and resumed.early_stop["bad_checks"] == 1
    resumed.fit()
    assert full.step == resumed.step == 6
    assert full.early_stop == resumed.early_stop
    assert resumed.early_stop["best_step"] == 4  # strict best even below min_delta
    assert resumed.early_stop["stopped"]
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    for run in (full, resumed):
        chosen = evaluation_module().select_checkpoint(run.out, config)
        assert load_checkpoint(chosen)["step"] == 4
    stopped = Trainer(config, tmp_path / "cut")
    monkeypatch.setattr(stopped, "loader", lambda: pytest.fail("stopped run must not restart"))
    stopped.fit()


def test_changed_validation_bytes_reject_resume(config, tmp_path):
    trainer = Trainer(config, tmp_path / "run")
    trainer.save()
    val = Path(config["validation"]["root"]) / "val-000000.tar"
    with val.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="dataset content"):
        Trainer(config, trainer.out)


def test_uncommitted_better_checkpoint_never_selected(config, tmp_path, monkeypatch):
    monkeypatch.setattr(Trainer, "validate", lambda self: self.step / 10)
    trainer = Trainer(config, tmp_path / "run")
    trainer.fit(stop_at=2)
    original_save = trainer.save
    monkeypatch.setattr(trainer, "save", lambda: (_ for _ in ()).throw(RuntimeError("disconnect")))
    trainer.step = 4
    with pytest.raises(RuntimeError, match="disconnect"):
        trainer.check_validation()  # best-step4 published; last.pt still commits best-step2
    assert trainer.best_checkpoint_path(4).exists()
    chosen = evaluation_module().select_checkpoint(trainer.out, config)
    assert load_checkpoint(chosen)["step"] == 2
    monkeypatch.setattr(trainer, "save", original_save)
    resumed = Trainer(config, trainer.out)
    assert resumed.step == 2 and resumed.early_stop["best_step"] == 2


@pytest.mark.parametrize("change", ["test", "empty", "short", "delta"])
def test_invalid_validation_protocol_rejected(config, tmp_path, change):
    cfg = copy.deepcopy(config)
    if change == "test":
        cfg["validation"]["split"] = "test"
    elif change == "empty":
        cfg["validation"]["samples"] = 0
    elif change == "short":
        cfg["validation"]["samples"] = 5
    else:
        cfg["validation"]["min_delta"] = float("nan")
    with pytest.raises(ValueError):
        Trainer(cfg, tmp_path / "bad")
