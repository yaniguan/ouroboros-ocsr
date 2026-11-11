"""Zero-escnn pure-PyTorch encoder export, HF reload, and standalone checkpoint tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ouroboros import hf
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.encoder.pure_torch import (
    ExportedEquivAttentionEncoder,
    ExportedSteerableEncoder,
    MaxPoolChannels,
)
from ouroboros.model import build_encoder, build_model
from ouroboros.weights import export_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "N#CC(Cl)Br", "C[C@H](N)C(=O)O"]
IMAGE_SIZE = 64
DECODER = {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64, "max_len": 24}
EXPORTED_STEERABLE = {
    "type": "exported_steerable",
    "N": 4,
    "fields": [2, 2, 4, 4],
    "blocks": [1, 1, 1, 1],
    "mixer_layers": 1,
    "mixer_heads": 2,
    "mixer_ff": 32,
}
EXPORTED_EQUIV_ATTN = {
    "type": "exported_equiv_attention",
    "N": 4,
    "fields": [2, 2, 4, 4],
    "blocks": [1, 1, 1, 1],
    "attn_dim": 16,
    "attn_layers": 1,
    "attn_heads": 2,
    "attn_ff": 32,
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(encoder: dict) -> dict:
    return {
        "seed": 3,
        "data": {"image_size": IMAGE_SIZE},
        "model": {"encoder": encoder, "decoder": DECODER},
    }


def _grid_perm(h: int, k: int) -> torch.Tensor:
    idx = torch.arange(h * h).view(1, h, h)
    return torch.rot90(idx, k, dims=(-2, -1)).reshape(-1)


def _impose_c4_equivariance_(trunk: nn.Module, generator: torch.Generator) -> None:
    """Populate a pure-PyTorch N=4 trunk with exact C4 regular-representation kernels & BN stats."""
    with torch.no_grad():
        for m in trunk.modules():
            if isinstance(m, nn.Conv2d):
                c_out, c_in, kh, kw = m.weight.shape
                f_out = c_out // 4
                if c_in == 1:
                    base = torch.randn(f_out, 1, kh, kw, generator=generator) * 0.2
                    for h in range(4):
                        m.weight[h::4] = torch.rot90(base, h, dims=(-2, -1))
                else:
                    f_in = c_in // 4
                    base = torch.randn(f_out, f_in, 4, kh, kw, generator=generator) * 0.2
                    w = m.weight.view(f_out, 4, f_in, 4, kh, kw)
                    for h_out in range(4):
                        for h_in in range(4):
                            src = base[:, :, (h_in - h_out) % 4]
                            w[:, h_out, :, h_in] = torch.rot90(src, h_out, dims=(-2, -1))
            elif isinstance(m, nn.BatchNorm2d):
                f = m.num_features // 4
                scale = torch.rand(f, 1, generator=generator).add(0.5).expand(f, 4).reshape(-1)
                shift = torch.randn(f, 1, generator=generator).mul(0.1).expand(f, 4).reshape(-1)
                mean = torch.randn(f, 1, generator=generator).mul(0.1).expand(f, 4).reshape(-1)
                var = torch.rand(f, 1, generator=generator).add(0.5).expand(f, 4).reshape(-1)
                m.weight.copy_(scale)
                m.bias.copy_(shift)
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)


@torch.no_grad()
def _logits(model, images: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    enc = model.encoder(images)
    return model.decoder(ids, enc.tokens, enc.padding_mask)


def _inputs(n_vocab: int):
    g = torch.Generator().manual_seed(11)
    images = torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    ids = torch.randint(0, n_vocab, (2, 10), generator=g)
    return images, ids


@pytest.mark.parametrize(
    ("enc_cfg", "cls"),
    [
        (EXPORTED_STEERABLE, ExportedSteerableEncoder),
        (EXPORTED_EQUIV_ATTN, ExportedEquivAttentionEncoder),
    ],
)
def test_pure_torch_encoders_build_and_preserve_c4_invariance_without_escnn(
    monkeypatch, enc_cfg, cls
):
    monkeypatch.setitem(sys.modules, "escnn", None)
    tok = SmilesTokenizer.build(SMILES)
    model = build_model(_cfg(enc_cfg), tok).eval()
    assert isinstance(model.encoder, cls)
    assert not any(type(m).__module__.startswith("escnn") for m in model.modules())

    g = torch.Generator().manual_seed(7)
    _impose_c4_equivariance_(model.encoder.trunk, g)
    if hasattr(model.encoder.head, "layers"):
        with torch.no_grad():
            for layer in model.encoder.head.layers:
                layer.bias.rot_emb.normal_(generator=g)

    x = torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    with torch.no_grad():
        tokens = model.encoder(x).tokens
        assert tokens.shape == (2, (IMAGE_SIZE // 32) ** 2, DECODER["d_model"])
        for k in (1, 2, 3):
            rotated = model.encoder(torch.rot90(x, k, dims=(-2, -1))).tokens
            perm = _grid_perm(IMAGE_SIZE // 32, k)
            assert torch.allclose(rotated, tokens[:, perm], atol=1e-5)
        mirrored = model.encoder(torch.flip(x, dims=(-1,))).tokens
        assert not torch.allclose(mirrored.mean(dim=1), tokens.mean(dim=1), atol=1e-4)


def test_max_pool_channels_validation_and_values():
    with pytest.raises(ValueError, match="kernel_size"):
        MaxPoolChannels(0)
    pool = MaxPoolChannels(4)
    x = torch.arange(24, dtype=torch.float32).view(1, 8, 1, 3)
    out = pool(x)
    assert out.shape == (1, 2, 1, 3)
    assert torch.equal(out[0, 0], x[0, :4].amax(dim=0))
    assert torch.equal(out[0, 1], x[0, 4:].amax(dim=0))


@pytest.mark.parametrize(
    ("orig_type", "pure_cfg"),
    [
        ("steerable", EXPORTED_STEERABLE),
        ("equiv_attention", EXPORTED_EQUIV_ATTN),
    ],
)
def test_hf_load_pretrained_uses_exported_weights_without_escnn(
    tmp_path, monkeypatch, orig_type, pure_cfg
):
    monkeypatch.setitem(sys.modules, "escnn", None)
    tok = SmilesTokenizer.build(SMILES)
    ref = build_model(_cfg(pure_cfg), tok).eval()
    images, ids = _inputs(len(tok))
    expected = _logits(ref, images, ids)

    folder = tmp_path / "hf"
    folder.mkdir()
    exp_state = hf.inference_state(ref.state_dict())
    exp_model_file = hf.save_weights(exp_state, folder, "exported_model")
    exp_enc_file = hf.save_weights(hf.encoder_state(exp_state), folder, "exported_encoder")
    # Simulate un-exported raw weights file also present in the HF release folder
    raw_model_file = hf.save_weights({"placeholder": torch.zeros(1)}, folder, "model")

    recorded_enc_cfg = {k: v for k, v in pure_cfg.items() if k != "type"} | {"type": orig_type}
    config = {
        "format": hf.FORMAT,
        "architecture": "OCSRModel",
        "arm": "C",
        "model": {"encoder": recorded_enc_cfg, "decoder": DECODER},
        "data": {"image_size": IMAGE_SIZE},
        "vocab_size": len(tok),
        "encoder_type": orig_type,
        "exported_encoder_type": pure_cfg["type"],
        "weights_file": raw_model_file.name,
        "exported_weights_file": exp_model_file.name,
        "exported_encoder_weights_file": exp_enc_file.name,
    }
    (folder / hf.CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
    (folder / hf.VOCAB_FILE).write_text(json.dumps(tok.itos) + "\n")
    sums = {
        p.name: hf.sha256(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != hf.CHECKSUMS_FILE
    }
    (folder / hf.CHECKSUMS_FILE).write_text("".join(f"{d}  {n}\n" for n, d in sums.items()))

    loaded = hf.load_pretrained(folder)
    assert not any(type(m).__module__.startswith("escnn") for m in loaded.modules())
    assert torch.equal(_logits(loaded, images, ids), expected)
    assert loaded.generate(images) == ref.generate(images)

    # Fine-tuning mode (export_encoder=False) requires escnn and gives a clear ImportError
    with pytest.raises(ImportError, match="escnn"):
        hf.load_pretrained(folder, export_encoder=False)

    # Tampering exported_model weights is caught by checksum verification
    bad = dict(exp_state)
    first_key = next(iter(bad))
    bad[first_key] = bad[first_key] + 1.0
    hf.save_weights(bad, folder, "exported_model")
    with pytest.raises(ValueError, match="checksum"):
        hf.load_pretrained(folder)


def test_standalone_pure_torch_checkpoint_loads_in_predict_real_without_escnn(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "escnn", None)
    tok = SmilesTokenizer.build(SMILES)
    cfg = _cfg(EXPORTED_STEERABLE) | {"data": {"image_size": IMAGE_SIZE, "synthetic_root": "/priv"}}
    ref = build_model(cfg, tok).eval()
    train_ckpt = tmp_path / "train.pt"
    torch.save(
        {
            "model": ref.state_dict(),
            "config": cfg,
            "vocab": tok.itos,
            "step": 12,
            "rng": {"numpy": np.random.get_state()},
        },
        train_ckpt,
    )
    out_ckpt = tmp_path / "ouroboros-C.pt"
    receipt = export_checkpoint(train_ckpt, out_ckpt, "C", pure_torch=True)
    assert receipt["step"] == 12

    restored = torch.load(out_ckpt, map_location="cpu", weights_only=True)
    assert restored["pure_torch"] is True
    assert restored["config"]["data"] == {"image_size": IMAGE_SIZE}

    predict_real = _load_script("predict_real")
    loaded, _ = predict_real.load_checkpoint(out_ckpt, torch.device("cpu"))
    images, ids = _inputs(len(tok))
    assert torch.equal(_logits(loaded, images, ids), _logits(ref, images, ids))


@pytest.mark.parametrize(
    ("enc_cfg", "exp_type"),
    [
        (
            {
                "type": "steerable",
                "N": 4,
                "fields": [2, 2, 4, 4],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 32,
            },
            "exported_steerable",
        ),
        (
            {
                "type": "equiv_attention",
                "N": 4,
                "fields": [2, 2, 4, 4],
                "blocks": [1, 1, 1, 1],
                "attn_dim": 16,
                "attn_layers": 1,
                "attn_heads": 2,
                "attn_ff": 32,
            },
            "exported_equiv_attention",
        ),
    ],
)
def test_escnn_to_pure_torch_hf_and_checkpoint_roundtrip(tmp_path, monkeypatch, enc_cfg, exp_type):
    pytest.importorskip("escnn")
    torch.manual_seed(0)
    tok = SmilesTokenizer.build(SMILES)
    cfg = _cfg(enc_cfg)
    model = build_model(cfg, tok)
    model.train()
    with torch.no_grad():
        model.encoder(torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE))
    model.eval()
    with torch.no_grad():
        model.encoder(torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE))

    # 1. Direct .export() leaves zero escnn modules or buffers and matches pure builder keys
    exp_enc = model.encoder.export()
    assert not any(type(m).__module__.startswith("escnn") for m in exp_enc.modules())
    pure_cfg = {"type": exp_type} | {k: v for k, v in enc_cfg.items() if k != "type"}
    pure_template = build_encoder(pure_cfg, DECODER["d_model"], IMAGE_SIZE)
    assert exp_enc.state_dict().keys() == pure_template.state_dict().keys()

    ckpt = tmp_path / "selected.pt"
    torch.save(
        {"model": model.state_dict(), "config": cfg, "vocab": tok.itos, "step": 5},
        ckpt,
    )
    hf_dir = tmp_path / "hf"
    _load_script("export_hf_release").export(ckpt, hf_dir, arm="C")
    pure_pt = tmp_path / "pure.pt"
    export_checkpoint(ckpt, pure_pt, "C", pure_torch=True)

    images, ids = _inputs(len(tok))
    expected = _logits(model, images, ids)

    # 2. Reload both artifacts with escnn completely blocked
    monkeypatch.setitem(sys.modules, "escnn", None)
    loaded_hf = hf.load_pretrained(hf_dir)
    assert not any(type(m).__module__.startswith("escnn") for m in loaded_hf.modules())
    assert torch.allclose(_logits(loaded_hf, images, ids), expected, atol=1e-4)

    loaded_pt, _ = _load_script("predict_real").load_checkpoint(pure_pt, torch.device("cpu"))
    assert not any(type(m).__module__.startswith("escnn") for m in loaded_pt.modules())
    assert torch.allclose(_logits(loaded_pt, images, ids), expected, atol=1e-4)
