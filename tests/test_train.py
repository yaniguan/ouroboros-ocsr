"""Trainer: exact resume (model, optimizer, scheduler, RNG, data position, augmentation)."""

import json

import pytest
import torch

from ouroboros.data import build
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.train.trainer import Trainer, lr_lambda


def tiny_cfg(root, vocab, **train):
    return {
        "seed": 3,
        "data": {
            "synthetic_root": str(root / "shards"),
            "synthetic_max_samples": None,
            "real_roots": [],
            "real_fraction": 0.0,
            "image_size": 64,
            "vocab": str(vocab),
            "num_workers": 2,
        },
        "model": {
            "encoder": {
                "type": "baseline",
                "widths": [8, 16, 16, 32],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 64,
            },
            "decoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64, "max_len": 96},
        },
        "train": {
            "device": "cpu",
            "batch_size": 8,
            "steps": 40,
            "lr": 1e-3,
            "warmup": 5,
            "amp": "none",
            "grad_clip": 1.0,
            "rotation_aug": True,
            "log_every": 1,
            "ckpt_every": 10,
            **train,
        },
    }


@pytest.fixture(scope="module")
def vocab(tiny_dataset, tmp_path_factory):
    root, _ = tiny_dataset
    rows = build.read_manifest(root / "manifests" / "train.tsv")
    path = tmp_path_factory.mktemp("vocab") / "vocab.json"
    SmilesTokenizer.build(r["smiles"] for r in rows).save(path)
    return path


def _losses(out_dir):
    return {
        r["step"]: r["loss"]
        for r in map(json.loads, (out_dir / "log.jsonl").read_text().splitlines())
        if "loss" in r
    }


def test_resume_is_exact(tiny_dataset, vocab, tmp_path):
    root, _ = tiny_dataset
    torch.set_num_threads(1)  # deterministic CPU kernels for bit-exact comparison
    cfg = tiny_cfg(root, vocab)

    full = tmp_path / "full"
    Trainer(cfg, full).fit()

    cut = tmp_path / "cut"
    Trainer(cfg, cut).fit(stop_at=27)  # checkpoint at 20, "disconnect" at 27
    assert max(_losses(cut)) == 27
    resumed = Trainer(cfg, cut)
    assert resumed.step == 20
    assert max(_losses(cut)) == 20  # log lines after the checkpoint were dropped
    resumed.fit()

    a, b = _losses(full), _losses(cut)
    assert sorted(a) == sorted(b) == list(range(1, 41))
    diffs = [abs(a[s] - b[s]) / abs(a[s]) for s in range(21, 41)]
    assert max(diffs) == 0.0, diffs


def test_rotation_aug_depends_only_on_seed_and_step(tiny_dataset, vocab, tmp_path):
    root, _ = tiny_dataset
    tr = Trainer(tiny_cfg(root, vocab), tmp_path)
    x = torch.rand(4, 1, 64, 64)
    tr.step = 5
    a = tr.augment(x)
    torch.manual_seed(999)  # global RNG must not matter
    b = tr.augment(x)
    tr.step = 6
    c = tr.augment(x)
    assert torch.equal(a, b) and not torch.equal(a, c)
    tr.cfg["train"]["rotation_aug"] = False
    assert torch.equal(tr.augment(x), x)


def test_lr_schedule():
    assert lr_lambda(0, 10, 100) == pytest.approx(0.1)
    assert lr_lambda(9, 10, 100) == pytest.approx(1.0)
    assert lr_lambda(100, 10, 100) == pytest.approx(0.0, abs=1e-9)
    assert lr_lambda(55, 10, 100) == pytest.approx(0.5)


def test_steerable_resume_and_eval_mode_checkpoints(tiny_dataset, vocab, tmp_path):
    """escnn caches filters only in eval mode; checkpoints from either mode must load."""
    from ouroboros.model import build_model, load_model_state

    root, _ = tiny_dataset
    torch.set_num_threads(1)
    enc = {
        "type": "steerable",
        "N": 4,
        "fields": [2, 2, 2, 4],
        "blocks": [1, 1, 1, 1],
        "mixer_layers": 1,
        "mixer_heads": 2,
        "mixer_ff": 64,
    }
    cfg = tiny_cfg(root, vocab, steps=20, ckpt_every=10)
    cfg["model"]["encoder"] = enc
    full = tmp_path / "full"
    Trainer(cfg, full).fit()
    cut = tmp_path / "cut"
    Trainer(cfg, cut).fit(stop_at=14)
    tr = Trainer(cfg, cut)
    assert tr.step == 10
    tr.fit()
    a, b = _losses(full), _losses(cut)
    assert all(a[s] == b[s] for s in range(11, 21))
    # an eval-mode state (with cached filters) also loads, and gives identical outputs
    tr.model.eval()
    sd = {k: v.clone() for k, v in tr.model.state_dict().items()}
    fresh = build_model(cfg, tr.tok)
    load_model_state(fresh, sd)
    fresh.eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(fresh.encoder(x).tokens, tr.model.encoder(x).tokens, atol=1e-6)
