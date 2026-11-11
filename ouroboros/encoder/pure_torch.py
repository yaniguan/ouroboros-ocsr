"""Zero-escnn pure-PyTorch counterparts of the C_N steerable and equivariant-attention encoders.

Once an ``escnn`` encoder has expanded its steerable filter bases into ordinary ``nn.Conv2d`` and
``nn.BatchNorm2d`` weights via ``.export()``, inference requires only standard PyTorch ops,
channel-wise group pooling over regular ``C_N`` fields, and the pure-PyTorch token heads in this
module. Keeping these architectures free of ``escnn`` imports lets exported Hugging Face releases
and standalone ``--pure-torch`` checkpoints run on any platform (including Windows and Python
versions without ``gfortran``/``py3nj``).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ouroboros.encoder.base import EncoderOutput, ImageEncoder
from ouroboros.encoder.relpos import RelDistMixer, grid_coords
from ouroboros.model import register_encoder

# ----------------------------------------------------------------------------------- configs


@dataclass
class SteerableConfig:
    N: int = 8  # C_N; rotations only
    fields: list[int] = field(default_factory=lambda: [8, 16, 32, 64])  # regular fields/stage
    blocks: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    stem_fields: int | None = None  # default: fields[0]
    stem_kernel: int = 7
    d_model: int = 512
    head: str = "gpool_relpos"
    mixer_layers: int = 2
    mixer_heads: int = 8
    mixer_ff: int = 2048
    dropout: float = 0.1
    image_size: int = 384


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


# ----------------------------------------------------------------------------------- blocks


def default_blur_kernel() -> torch.Tensor:
    k = torch.tensor([1.0, 2.0, 1.0])
    return (k[:, None] * k[None, :] / 16.0)[None, None]


def _blurpool(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError("BlurPool2x2 needs even spatial sizes for exact 90-deg equivariance")
    c = x.shape[1]
    x = F.conv2d(x, kernel.to(x.dtype).expand(c, 1, 3, 3), padding=1, groups=c)
    return F.avg_pool2d(x, 2, 2)


class BlurPoolTorch(nn.Module):
    """Pure-PyTorch 2x2 anti-aliased downsampling matching ``BlurPool2x2.export()``."""

    def __init__(self, kernel: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("kernel", default_blur_kernel() if kernel is None else kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _blurpool(x, self.kernel)


class ResBlockTorch(nn.Module):
    """Pure-PyTorch residual block matching ``ResBlock.export()`` key-for-key."""

    def __init__(self, body: nn.Module, skip: nn.Module | None):
        super().__init__()
        self.body, self.skip = body, skip

    @classmethod
    def from_channels(cls, c_in: int, c_out: int) -> ResBlockTorch:
        body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        skip = (
            nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            if c_in != c_out
            else None
        )
        return cls(body, skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + (x if self.skip is None else self.skip(x)))


class MaxPoolChannels(nn.Module):
    """Channel-wise max pooling over contiguous regular ``C_N`` fields ``[B, F*N, H, W]``."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        return x.view(b, c // self.kernel_size, self.kernel_size, *x.shape[2:]).max(2)[0]


def build_exported_steerable_trunk(
    N: int,
    fields: list[int],
    blocks: list[int],
    stem_fields: int | None = None,
    stem_kernel: int = 7,
) -> tuple[nn.Sequential, int]:
    """Construct the pure-PyTorch ``nn.Sequential`` trunk matching ``SteerableEncoder``'s
    exported trunk layer-for-layer and key-for-key, returning ``(trunk, out_fields)``."""
    cur_fields = stem_fields or fields[0]
    c_stem = cur_fields * N
    mods: list[nn.Module] = [
        nn.Conv2d(1, c_stem, stem_kernel, padding=stem_kernel // 2, bias=False),
        nn.BatchNorm2d(c_stem),
        nn.ReLU(inplace=True),
        BlurPoolTorch(),
        BlurPoolTorch(),
    ]
    c_cur = c_stem
    for i, (nf, nb) in enumerate(zip(fields, blocks, strict=True)):
        if i > 0:
            mods.append(BlurPoolTorch())
        c_out = nf * N
        for _ in range(nb):
            mods.append(ResBlockTorch.from_channels(c_cur, c_out))
            c_cur = c_out
            cur_fields = nf
    return nn.Sequential(*mods), cur_fields


# ------------------------------------------------------------------------------ token heads


class PureGPoolRelPosHead(nn.Module):
    """Pure-PyTorch counterpart of ``GPoolRelPosHead`` without ``escnn.nn.GroupPooling``."""

    def __init__(self, proj: nn.Conv2d, mixer: RelDistMixer):
        super().__init__()
        self.proj = proj
        self.mixer = mixer

    @classmethod
    def from_dims(
        cls, n_inv: int, d_model: int, layers: int, heads: int, ff: int, dropout: float
    ) -> PureGPoolRelPosHead:
        return cls(
            nn.Conv2d(n_inv, d_model, 1),
            RelDistMixer(d_model, layers, heads, ff, dropout),
        )

    def tokens(self, inv: torch.Tensor) -> EncoderOutput:
        f = self.proj(inv)
        _, _, h, w = f.shape
        tok = f.flatten(2).transpose(1, 2)
        return EncoderOutput(tokens=self.mixer(tok, grid_coords(h, w, tok.device)))

    def forward(self, inv: torch.Tensor) -> EncoderOutput:
        return self.tokens(inv)


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
    mean over h -> invariant tokens [B, H*W, d]. Accepts either a raw ``torch.Tensor`` or an
    ``escnn.nn.GeometricTensor`` without importing ``escnn``."""

    def __init__(
        self,
        in_fields,
        N: int,
        d: int,
        layers: int,
        heads: int,
        ff: int,
        dropout: float,
    ):
        super().__init__()
        self.N = N
        self.F = in_fields if isinstance(in_fields, int) else len(in_fields)
        self.embed = nn.Linear(self.F, d)  # shared over h: commutes with the cyclic shift
        self.layers = nn.ModuleList(
            GroupAttentionLayer(d, heads, ff, dropout, N) for _ in range(layers)
        )
        self.ln = nn.LayerNorm(d)

    def lift(self, feat: torch.Tensor) -> torch.Tensor:
        B, _, H, W = feat.shape
        # escnn regular field layout: channel = field * N + h
        f = feat.view(B, self.F, self.N, H, W).permute(0, 3, 4, 2, 1)  # [B, H, W, N, F]
        return self.embed(f.reshape(B, H * W * self.N, self.F))  # token order (p, h)

    def forward(self, feat) -> EncoderOutput:
        t = feat if isinstance(feat, torch.Tensor) else feat.tensor
        B, _, H, W = t.shape
        x = self.lift(t)  # equivariant: rotation = permutation of (p, h)
        coords = lifted_coords(H, W, t.device)
        scale = float(max(H, W))
        for layer in self.layers:
            x = layer(x, coords, scale)  # equivariant (permutation of tokens)
        x = self.ln(x).view(B, H * W, self.N, -1).mean(dim=2)  # INVARIANT per location
        return EncoderOutput(tokens=x)


# ----------------------------------------------------------------------- exported encoders


class ExportedSteerableEncoder(ImageEncoder):
    """Zero-escnn pure-PyTorch inference encoder for arm C / C+."""

    def __init__(
        self,
        trunk: nn.Module,
        gpool_or_head: nn.Module,
        head: PureGPoolRelPosHead | None = None,
        *,
        config: SteerableConfig | None = None,
    ):
        super().__init__()
        self.c = config
        self.trunk = trunk
        if head is None:
            # Backwards-compatible 2-arg form: ExportedSteerableEncoder(trunk, gpool_relpos_head)
            legacy = gpool_or_head
            n_group = legacy.gpool.in_type.size // legacy.gpool.out_type.size
            self.gpool = MaxPoolChannels(n_group)
            self.head = PureGPoolRelPosHead(
                copy.deepcopy(legacy.proj), copy.deepcopy(legacy.mixer)
            )
        else:
            self.gpool = gpool_or_head
            self.head = head
        self.d_model = self.head.proj.out_channels

    @classmethod
    def from_config(cls, c: SteerableConfig) -> ExportedSteerableEncoder:
        if c.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 (even maps at every pooling)")
        if c.head != "gpool_relpos":
            raise KeyError(f"unknown head {c.head!r} (equivariant attention lives in arm D)")
        trunk, out_fields = build_exported_steerable_trunk(
            c.N, c.fields, c.blocks, c.stem_fields, c.stem_kernel
        )
        head = PureGPoolRelPosHead.from_dims(
            out_fields, c.d_model, c.mixer_layers, c.mixer_heads, c.mixer_ff, c.dropout
        )
        return cls(trunk, MaxPoolChannels(c.N), head, config=c)

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        return self.head.tokens(self.gpool(self.trunk(images)))


class ExportedEquivAttentionEncoder(ImageEncoder):
    """Zero-escnn pure-PyTorch inference encoder for arm D."""

    def __init__(
        self,
        trunk: nn.Module,
        head: GroupAttentionHead,
        proj: nn.Linear,
        *,
        config: EquivAttnConfig | None = None,
    ):
        super().__init__()
        self.c = config
        self.trunk = trunk
        self.head = head
        self.proj = proj
        self.d_model = proj.out_features

    @classmethod
    def from_config(cls, c: EquivAttnConfig) -> ExportedEquivAttentionEncoder:
        if c.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 (even maps at every pooling)")
        trunk, out_fields = build_exported_steerable_trunk(
            c.N, c.fields, c.blocks, stem_fields=None, stem_kernel=c.stem_kernel
        )
        head = GroupAttentionHead(
            out_fields, c.N, c.attn_dim, c.attn_layers, c.attn_heads, c.attn_ff, c.dropout
        )
        proj = nn.Linear(c.attn_dim, c.d_model)
        return cls(trunk, head, proj, config=c)

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        return EncoderOutput(tokens=self.proj(self.head(self.trunk(images)).tokens))


@register_encoder("exported_steerable")
def _exported_steerable(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    kw = {k: v for k, v in enc_cfg.items() if k != "type"}
    return ExportedSteerableEncoder.from_config(
        SteerableConfig(d_model=d_model, image_size=image_size, **kw)
    )


@register_encoder("exported_equiv_attention")
def _exported_equiv_attention(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    kw = {k: v for k, v in enc_cfg.items() if k != "type"}
    return ExportedEquivAttentionEncoder.from_config(
        EquivAttnConfig(d_model=d_model, image_size=image_size, **kw)
    )


__all__ = [
    "BlurPoolTorch",
    "EquivAttnConfig",
    "ExportedEquivAttentionEncoder",
    "ExportedSteerableEncoder",
    "GroupAttentionHead",
    "GroupAttentionLayer",
    "GroupRelPosBias",
    "MaxPoolChannels",
    "PureGPoolRelPosHead",
    "ResBlockTorch",
    "SteerableConfig",
    "build_exported_steerable_trunk",
    "lifted_coords",
]
