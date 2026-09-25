"""Arm E: learned canonicalization — C4 invariance of the whole encoder (Phase 5, optional)."""

import torch

from ouroboros.encoder.canonicalize import CanonConfig, CanonicalizedEncoder

TOL = 1e-4


def small() -> CanonicalizedEncoder:
    torch.manual_seed(0)
    enc = CanonicalizedEncoder(
        CanonConfig(
            N=4,
            orient_fields=[2, 2, 4],
            downsample=2,
            base={"widths": [8, 8, 16, 16], "blocks": [1, 1, 1, 1], "mixer_heads": 4},
            d_model=64,
            image_size=128,
        )
    )
    enc.train()
    with torch.no_grad():
        enc(torch.rand(4, 1, 128, 128))
    return enc.eval()


def rel_err(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def test_orientation_logits_shift_and_output_invariant():
    enc = small()
    x = torch.rand(3, 1, 128, 128)
    with torch.no_grad():
        logits = enc.orient(x)
        t = enc(x).tokens
        for k in (1, 2, 3):
            xr = torch.rot90(x, k, dims=(-2, -1))
            lr = enc.orient(xr)
            assert rel_err(lr, torch.roll(logits, k, dims=1)) < TOL  # equivariant logits
            assert rel_err(enc(xr).tokens, t) < TOL  # invariant encoder output


def test_prior_loss_targets():
    enc = small()
    x = torch.rand(2, 1, 128, 128)
    enc.train()
    enc(x)
    upright = enc.prior_loss(None)
    rotated = enc.prior_loss(torch.tensor([90.0, 181.0]))
    assert torch.isfinite(upright) and torch.isfinite(rotated)
    upright.backward()
    assert any(p.grad is not None for p in enc.orient.parameters())
