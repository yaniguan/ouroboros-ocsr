"""Opt-in GPU training throughput accelerators in Trainer (channels_last, fused_adamw, compile)."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from ouroboros.data import build
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.train.trainer import Trainer, build_optimizer, load_checkpoint


def _tiny_cfg(root, vocab, **train):
    return {
        "seed": 5,
        "data": {
            "synthetic_root": str(root / "shards"),
            "synthetic_max_samples": None,
            "real_roots": [],
            "real_fraction": 0.0,
            "image_size": 64,
            "vocab": str(vocab),
            "num_workers": 0,
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
            "decoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "max_len": 96},
        },
        "train": {
            "device": "cpu",
            "batch_size": 4,
            "steps": 6,
            "lr": 1e-3,
            "warmup": 2,
            "amp": "none",
            "grad_clip": 1.0,
            "rotation_aug": False,
            "log_every": 1,
            "ckpt_every": 3,
            **train,
        },
    }


@pytest.fixture(scope="module")
def vocab(tiny_dataset, tmp_path_factory):
    root, _ = tiny_dataset
    rows = build.read_manifest(root / "manifests" / "train.tsv")
    path = tmp_path_factory.mktemp("accel_vocab") / "vocab.json"
    SmilesTokenizer.build(r["smiles"] for r in rows).save(path)
    return path


def _losses(out_dir):
    return [
        r["loss"]
        for r in map(json.loads, (out_dir / "log.jsonl").read_text().splitlines())
        if "loss" in r
    ]


def test_channels_last_converts_weights_inputs_and_survives_resume(tiny_dataset, vocab, tmp_path):
    torch.set_num_threads(1)
    root, _ = tiny_dataset
    cfg = _tiny_cfg(root, vocab, channels_last=True)
    cfg["validation"] = {
        "enabled": True,
        "root": str(root / "shards"),
        "split": "val",
        "samples": 4,
        "every": 3,
        "patience": 5,
        "batch_size": 2,
    }
    run_dir = tmp_path / "cl_run"
    trainer = Trainer(cfg, run_dir)

    conv_weights = [
        p for m in trainer.model.modules() if isinstance(m, nn.Conv2d) for p in [m.weight]
    ]
    assert conv_weights
    assert all(w.is_contiguous(memory_format=torch.channels_last) for w in conv_weights)

    seen_formats: list[bool] = []

    def _hook(_mod, args):
        x = args[0]
        seen_formats.append(x.is_contiguous(memory_format=torch.channels_last))

    handle = trainer.model.encoder.register_forward_pre_hook(_hook)
    try:
        trainer.fit(stop_at=3)
    finally:
        handle.remove()
    assert seen_formats and all(seen_formats)

    resumed = Trainer(cfg, run_dir)
    assert resumed.step == 3
    assert all(
        m.weight.is_contiguous(memory_format=torch.channels_last)
        for m in resumed.model.modules()
        if isinstance(m, nn.Conv2d)
    )
    resumed.fit()
    assert resumed.step == 6


def test_fused_adamw_validation_and_cuda_forwarding(monkeypatch):
    lin = nn.Linear(4, 2)
    opt_default = build_optimizer(lin, {"lr": 1e-3}, device="cpu")
    assert not opt_default.defaults.get("fused", False)

    opt_auto_cpu = build_optimizer(lin, {"lr": 1e-3, "fused_adamw": "auto"}, device="cpu")
    assert not opt_auto_cpu.defaults.get("fused", False)

    with pytest.raises(ValueError, match="CUDA"):
        build_optimizer(lin, {"lr": 1e-3, "fused_adamw": True}, device="cpu")

    captured: list[dict] = []
    real_adamw = torch.optim.AdamW

    def _spy_adamw(params, **kwargs):
        captured.append(dict(kwargs))
        kwargs.pop("fused", None)
        return real_adamw(params, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _spy_adamw)
    build_optimizer(lin, {"lr": 1e-3, "fused_adamw": "auto"}, device="cuda")
    build_optimizer(lin, {"lr": 1e-3, "fused_adamw": True}, device="cuda")
    assert captured[0].get("fused") is True
    assert captured[1].get("fused") is True


def test_torch_compile_runs_cleanly_and_saves_unprefixed_state_dict(
    tiny_dataset, vocab, tmp_path
):
    root, _ = tiny_dataset
    torch.set_num_threads(1)

    eager_dir = tmp_path / "eager"
    eager_cfg = _tiny_cfg(root, vocab)
    eager = Trainer(eager_cfg, eager_dir)
    eager.fit()

    comp_dir = tmp_path / "compiled"
    comp_cfg = _tiny_cfg(root, vocab, compile=True, compile_backend="aot_eager")
    compiled = Trainer(comp_cfg, comp_dir)
    assert compiled._compiled_loss is not None
    compiled.fit(stop_at=3)

    ckpt = load_checkpoint(comp_dir / "ckpt" / "last.pt")
    assert not any(k.startswith("_orig_mod.") for k in ckpt["model"])
    assert set(ckpt["model"].keys()) == set(eager.model.state_dict().keys())

    resumed = Trainer(comp_cfg, comp_dir)
    assert resumed.step == 3
    resumed.fit()
    assert _losses(comp_dir) == pytest.approx(_losses(eager_dir), rel=1e-4)


def test_compile_mode_forwarding(tiny_dataset, vocab, tmp_path, monkeypatch):
    root, _ = tiny_dataset
    calls: list[dict] = []

    def _fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    cfg_ro = _tiny_cfg(root, vocab, compile="reduce-overhead", compile_dynamic=False)
    Trainer(cfg_ro, tmp_path / "a")
    Trainer(_tiny_cfg(root, vocab, compile="none"), tmp_path / "b")
    assert calls == [{"mode": "reduce-overhead", "dynamic": False}]


@pytest.mark.parametrize(
    ("bad_kw", "match"),
    [
        ({"channels_last": "yes"}, "channels_last"),
        ({"fused_adamw": "fast"}, "fused_adamw"),
        ({"fused_adamw": 2}, "fused_adamw"),
        ({"compile": "turbo"}, "train.compile"),
        ({"compile": True, "compile_backend": ""}, "compile_backend"),
        ({"compile": True, "compile_dynamic": "auto"}, "compile_dynamic"),
    ],
)
def test_invalid_accelerator_flags_rejected(tiny_dataset, vocab, tmp_path, bad_kw, match):
    root, _ = tiny_dataset
    with pytest.raises(ValueError, match=match):
        Trainer(_tiny_cfg(root, vocab, **bad_kw), tmp_path / "bad")
