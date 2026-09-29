"""Equivariance / invariance of the steerable encoder (Phase 3 DoD) and sanity checks."""

import pytest
import torch
from escnn import nn as enn

from ouroboros.encoder.steerable import SteerableConfig, SteerableEncoder

TOL = 1e-4


def small(N: int, image_size: int = 128) -> SteerableEncoder:
    torch.manual_seed(0)
    cfg = SteerableConfig(
        N=N,
        fields=[4, 4, 8, 8],
        blocks=[1, 1, 1, 1],
        d_model=64,
        mixer_heads=4,
        mixer_ff=128,
        image_size=image_size,
    )
    enc = SteerableEncoder(cfg)
    # non-trivial BN statistics, then eval mode (running stats; deterministic)
    enc.train()
    with torch.no_grad():
        for _ in range(3):
            enc(torch.rand(4, 1, image_size, image_size))
    return enc.eval()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def rot_element(enc: SteerableEncoder, k: int):
    """escnn group element for a rotation by k * 90 deg (same direction as torch.rot90)."""
    N = enc.c.N
    return enc.gspace.fibergroup.element(k * N // 4)


@pytest.mark.parametrize("N", [4, 8, 16])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_feature_maps_equivariant_90(N, k):
    enc = small(N)
    x = torch.rand(2, 1, 128, 128)
    xr = torch.rot90(x, k, dims=(-2, -1))
    el = rot_element(enc, k)
    # our pixel rotation == escnn's action on the (trivial) input field
    assert torch.equal(enn.GeometricTensor(x, enc.in_type).transform(el).tensor, xr)
    with torch.no_grad():
        f, fr = enc.features(x), enc.features(xr)
    assert rel_err(f.transform(el).tensor, fr.tensor) < TOL


def _grid_perm(h: int, k: int) -> torch.Tensor:
    """perm such that tokens(rot90^k x)[i] corresponds to tokens(x)[perm[i]] (row-major)."""
    idx = torch.arange(h * h).view(1, h, h)
    return torch.rot90(idx, k, dims=(-2, -1)).reshape(-1)


@pytest.mark.parametrize("N", [4, 8])
def test_token_set_invariant_up_to_permutation(N):
    enc = small(N)
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        t = enc(x).tokens
        for k in (1, 2, 3):
            tr = enc(torch.rot90(x, k, dims=(-2, -1))).tokens
            perm = _grid_perm(4, k)  # 128 px / 32 = 4x4 tokens
            assert rel_err(tr, t[:, perm]) < TOL
            assert rel_err(tr, t) > 1e-2  # it IS a non-trivial permutation


def test_full_model_logits_invariant_c4():
    from ouroboros.decode.decoder import DecoderConfig, SmilesDecoder

    enc = small(4)
    torch.manual_seed(1)
    dec = SmilesDecoder(DecoderConfig(vocab_size=20, d_model=64, n_layers=2, n_heads=4, d_ff=128))
    dec.eval()
    x = torch.rand(2, 1, 128, 128)
    ids = torch.randint(0, 20, (2, 9))
    with torch.no_grad():
        a = dec(ids, enc(x).tokens)
        b = dec(ids, enc(torch.rot90(x, 1, dims=(-2, -1))).tokens)
    assert rel_err(b, a) < TOL


def test_not_reflection_invariant():
    """C_N only: a mirrored drawing must be able to produce different tokens (stereo!)."""
    enc = small(4)
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        t, tm = enc(x).tokens, enc(torch.flip(x, dims=(-1,))).tokens
        tr = enc(torch.rot90(x, 1, dims=(-2, -1))).tokens
    # the token mean is permutation invariant: equal for a rotation, different for a mirror
    rot_diff, mirror_diff = rel_err(tr.mean(1), t.mean(1)), rel_err(tm.mean(1), t.mean(1))
    assert rot_diff < TOL
    assert mirror_diff > max(100 * rot_diff, 1e-5)  # far above numerical error
    assert "flip" not in type(enc.gspace).__name__.lower()
    assert enc.gspace.fibergroup.order() == 4


def test_baseline_is_not_rotation_invariant():
    from ouroboros.encoder.baseline import BaselineConfig, BaselineEncoder

    torch.manual_seed(0)
    enc = BaselineEncoder(
        BaselineConfig(
            widths=[8, 8, 16, 16], blocks=[1, 1, 1, 1], d_model=64, mixer_heads=4, image_size=128
        )
    ).eval()
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        t, tr = enc(x).tokens, enc(torch.rot90(x, 1, dims=(-2, -1))).tokens
    assert rel_err(tr, t[:, _grid_perm(4, 1)]) > 1e-2


def test_export_matches(tmp_path):
    enc = small(8)
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        ref = enc(x).tokens
        exp = enc.export()
        out = exp(x).tokens
    assert rel_err(out, ref) < TOL
    # the exported trunk contains no escnn modules
    assert not any(isinstance(m, enn.EquivariantModule) for m in exp.trunk.modules())
