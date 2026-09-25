"""Arm D: steerable conv stem + group-equivariant self-attention (option (b) of constraint 3).

Tokens live on the group-lifted grid: one token per (location x, rotation h in C_N), taken
directly from the regular-representation fields of the steerable trunk (channel h of every field).
Self-attention over all (x, h) tokens uses a relative positional encoding expressed in the
QUERY's rotated frame (Romero et al., "Group Equivariant Stand-Alone Self-Attention"):

    logit((x, h) -> (y, h')) = <q_{x,h}, k_{y,h'}> / sqrt(d) + b( R_h^{-1} (y - x),  h' - h mod N )

Why this is equivariant: rotating the input by g in C_N moves token (x, h) to (g x, g h) (regular
representation = cyclic shift of h). Then R_{gh}^{-1}(g y - g x) = R_h^{-1}(y - x) and
(g h') - (g h) = h' - h, so every logit is unchanged and the attention output is permuted exactly
like its input: the layer is C_N-EQUIVARIANT (a token permutation). Linear maps shared across all
tokens (q, k, v, MLP) commute with that permutation. Unlike option (a), relative ORIENTATION between
tokens is preserved (not just distances), so global handedness can be represented.

The memory handed to the shared decoder is the mean over h of every location's tokens: a
C_N-INVARIANT token per location, i.e. an invariant token SET up to permutation (exact for 90-deg
rotations of the pixel grid; C8/C16 off-grid angles inherit the trunk's interpolation error).

Coordinates are (x right, y up) centred on the image; in this frame escnn's rotation element k
acts as a counter-clockwise rotation by 2*pi*k/N (verified by the equivariance tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from escnn import nn as enn
from torch import nn

from ouroboros.encoder.base import EncoderOutput, ImageEncoder
from ouroboros.encoder.steerable import SteerableConfig, SteerableEncoder
from ouroboros.model import register_encoder


def lifted_coords(h: int, w: int, device=None) -> torch.Tensor:
    """(x right, y up) grid coordinates centred on the image, row-major [h*w, 2]."""
    ys = (h - 1) / 2 - torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


class GroupRelPosBias(nn.Module):
    """b(R_h^{-1} (y - x), h' - h) -> per-head bias for all (x,h) x (y,h') pairs."""

    def __init__(self, N: int, heads: int, hidden: int = 64):
        super().__init__()
        self.N = N
        self.mlp = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, heads))
        self.rot_emb = nn.Parameter(torch.zeros(N, heads))

    def forward(self, coords: torch.Tensor, scale: float) -> torch.Tensor:
        """coords [P, 2] -> bias [heads, P*N, P*N], token order (p, h) with h fastest."""
        N, P = self.N, coords.shape[0]
        rel = coords[None, :, :] - coords[:, None, :]  # [P(x), P(y), 2] = y - x
        ang = -2 * math.pi * torch.arange(N, device=coords.device, dtype=coords.dtype) / N
        c, s = torch.cos(ang), torch.sin(ang)  # R_h^{-1}
        rx = c[:, None, None] * rel[..., 0] - s[:, None, None] * rel[..., 1]  # [N(h), P, P]
        ry = s[:, None, None] * rel[..., 0] + c[:, None, None] * rel[..., 1]
        pos = self.mlp(torch.stack([rx, ry], dim=-1) / scale)  # [N(h), P(x), P(y), heads]
        dh = (torch.arange(N)[None, :] - torch.arange(N)[:, None]) % N  # [N(h), N(h')]
        rot = self.rot_emb[dh.to(coords.device)]  # [N(h), N(h'), heads]
        # logits index: query (x, h), key (y, h')
        b = pos[:, :, :, None, :] + rot[:, None, None, :, :]  # [h, x, y, h', heads]
        b = b.permute(4, 1, 0, 2, 3).reshape(-1, P * N, P * N)  # [heads, (x,h), (y,h')]
        return b


class GroupAttentionLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float, N: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.bias = GroupRelPosBias(N, heads)
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, coords: torch.Tensor, scale: float) -> torch.Tensor:
        B, T, D = x.shape
        bias = self.bias(coords, scale).to(x.dtype)[None]
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.o(y.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class GroupAttentionHead(nn.Module):
    """Regular fields [B, F*N, H, W] -> lifted tokens [B, H*W*N, d] -> group self-attention ->
    mean over h -> invariant tokens [B, H*W, d]."""

    def __init__(
        self,
        in_type: enn.FieldType,
        N: int,
        d: int,
        layers: int,
        heads: int,
        ff: int,
        dropout: float,
    ):
        super().__init__()
        self.N = N
        self.F = len(in_type)
        self.embed = nn.Linear(self.F, d)  # shared over h: commutes with the cyclic shift
        self.layers = nn.ModuleList(
            GroupAttentionLayer(d, heads, ff, dropout, N) for _ in range(layers)
        )
        self.ln = nn.LayerNorm(d)

    def lift(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        # escnn regular field layout: channel = field * N + h
        f = feat.view(B, self.F, self.N, H, W).permute(0, 3, 4, 2, 1)  # [B, H, W, N, F]
        return self.embed(f.reshape(B, H * W * self.N, self.F))  # token order (p, h)

    def forward(self, feat) -> EncoderOutput:
        t = feat.tensor if isinstance(feat, enn.GeometricTensor) else feat
        B, _, H, W = t.shape
        x = self.lift(t)  # equivariant: rotation = permutation of (p, h)
        coords = lifted_coords(H, W, t.device)
        scale = float(max(H, W))
        for layer in self.layers:
            x = layer(x, coords, scale)  # equivariant (permutation of tokens)
        x = self.ln(x).view(B, H * W, self.N, -1).mean(dim=2)  # INVARIANT per location
        return EncoderOutput(tokens=x)


@dataclass
class EquivAttnConfig:
    N: int = 8
    fields: list[int] = field(default_factory=lambda: [7, 14, 27, 55])
    blocks: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    stem_kernel: int = 7
    d_model: int = 512
    attn_dim: int = 160  # FLOP-matched: encoder 10.34 GFLOPs @384 (baseline 10.27)
    attn_layers: int = 2
    attn_heads: int = 8
    attn_ff: int = 640
    dropout: float = 0.1
    image_size: int = 384


class EquivAttentionEncoder(ImageEncoder):
    def __init__(self, c: EquivAttnConfig):
        super().__init__()
        self.c = c
        self.d_model = c.d_model
        trunk_cfg = SteerableConfig(
            N=c.N,
            fields=c.fields,
            blocks=c.blocks,
            stem_kernel=c.stem_kernel,
            d_model=c.d_model,
            image_size=c.image_size,
            mixer_layers=1,
            mixer_heads=1,
            mixer_ff=4,
        )
        base = SteerableEncoder(trunk_cfg)
        self.gspace, self.in_type, self.trunk = base.gspace, base.in_type, base.trunk
        self.head = GroupAttentionHead(
            base.out_type, c.N, c.attn_dim, c.attn_layers, c.attn_heads, c.attn_ff, c.dropout
        )
        self.proj = nn.Linear(c.attn_dim, c.d_model)  # on invariant tokens

    def features(self, images: torch.Tensor) -> enn.GeometricTensor:
        return self.trunk(enn.GeometricTensor(images, self.in_type))

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        out = self.head(self.features(images))
        return EncoderOutput(tokens=self.proj(out.tokens))

    def export(self) -> nn.Module:
        self.eval()
        return _ExportedEquivAttn(self.trunk.export(), self.head, self.proj).eval()


class _ExportedEquivAttn(ImageEncoder):
    def __init__(self, trunk, head, proj):
        super().__init__()
        self.trunk, self.head, self.proj = trunk, head, proj
        self.d_model = proj.out_features

    def forward(self, images):
        return EncoderOutput(tokens=self.proj(self.head(self.trunk(images)).tokens))


@register_encoder("equiv_attention")
def _equiv_attention(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    kw = {k: v for k, v in enc_cfg.items() if k != "type"}
    return EquivAttentionEncoder(EquivAttnConfig(d_model=d_model, image_size=image_size, **kw))
