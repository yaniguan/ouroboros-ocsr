"""Arm D: group-equivariant self-attention head — equivariance, invariance, export."""

import pytest
import torch

from ouroboros.encoder.equiv_attention import (
    EquivAttentionEncoder,
    EquivAttnConfig,
    GroupRelPosBias,
    lifted_coords,
)

TOL = 1e-4


def small(N: int) -> EquivAttentionEncoder:
    torch.manual_seed(0)
    enc = EquivAttentionEncoder(
        EquivAttnConfig(
            N=N,
            fields=[2, 2, 4, 4],
            blocks=[1, 1, 1, 1],
            d_model=64,
            attn_dim=32,
            attn_layers=2,
            attn_heads=4,
            attn_ff=64,
            image_size=128,
        )
    )
    enc.train()
    with torch.no_grad():
        enc(torch.rand(2, 1, 128, 128))
    return enc.eval()


def rel_err(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def _grid_perm(h: int, k: int) -> torch.Tensor:
    idx = torch.arange(h * h).view(1, h, h)
    return torch.rot90(idx, k, dims=(-2, -1)).reshape(-1)


@pytest.mark.parametrize("N", [4, 8])
def test_token_set_invariant_90(N):
    enc = small(N)
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        t = enc(x).tokens
        for k in (1, 2, 3):
            tr = enc(torch.rot90(x, k, dims=(-2, -1))).tokens
            assert rel_err(tr, t[:, _grid_perm(4, k)]) < TOL


@pytest.mark.parametrize("N", [4, 8])
def test_lifted_tokens_equivariant_90(N):
    """Inside the head, rotating the input permutes (location, rotation) tokens."""
    enc = small(N)
    x = torch.rand(1, 1, 128, 128)
    head = enc.head
    with torch.no_grad():
        a = head.lift(enc.features(x).tensor)
        b = head.lift(enc.features(torch.rot90(x, 1, dims=(-2, -1))).tensor)
        coords = lifted_coords(4, 4)
        for layer in head.layers:
            a, b = layer(a, coords, 4.0), layer(b, coords, 4.0)
    P = 16
    perm_p = _grid_perm(4, 1)
    shift = N // 4  # rot90 = element N/4 of C_N: h -> h + N/4
    idx = torch.arange(P * N).view(P, N)
    perm = idx[perm_p][:, (torch.arange(N) - shift) % N].reshape(-1)
    assert rel_err(b, a[:, perm]) < TOL


def test_bias_depends_on_relative_pose_only():
    torch.manual_seed(0)
    N, P = 8, 9
    bias = GroupRelPosBias(N, heads=2)
    torch.nn.init.normal_(bias.rot_emb)
    coords = lifted_coords(3, 3)
    b = bias(coords, 3.0).view(2, P, N, P, N)
    # rotate the whole configuration by 90 deg: point p -> R p, rotation h -> h + N/4
    R = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rc = coords @ R.T
    match = torch.cdist(rc, coords).argmin(dim=1)  # p -> index of R p
    s = N // 4
    for p in range(P):
        for q in range(P):
            for h in range(N):
                for h2 in range(N):
                    v1 = b[:, p, h, q, h2]
                    v2 = b[:, match[p], (h + s) % N, match[q], (h2 + s) % N]
                    assert torch.allclose(v1, v2, atol=1e-5)


def test_not_reflection_invariant_and_export():
    enc = small(4)
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        t = enc(x).tokens
        tm = enc(torch.flip(x, dims=(-1,))).tokens
        d = torch.cdist(tm[0], t[0]).min(dim=1).values.max()
        assert float(d) / float(t.abs().max()) > 1e-2
        assert rel_err(enc.export()(x).tokens, t) < TOL
