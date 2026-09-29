"""Steerable C_N encoder (arms C / C+), built with escnn.

Symmetry group: C_N (N in {4, 8, 16}) acting on the plane by rotations only — ``rot2dOnR2``,
never ``flipRot2dOnR2``: reflections are excluded because mirroring a wedge/hash drawing
inverts its stereochemistry (hard constraint 1). This is SE(2)/C_N equivariance of an IMAGE
encoder; it is not E(3) equivariance (that exists only in the MACE stage).

Where equivariance holds / is invariant / is broken
---------------------------------------------------
* ``trunk`` (R2Conv, InnerBatchNorm, ReLU, BlurPool2x2, residual adds): C_N-EQUIVARIANT feature
  maps with regular-representation fields. On the pixel grid, rotations by multiples of 90 deg
  are exact: stride-1 convolutions with symmetric zero padding, and 2x2/stride-2 pooling on
  even-sized maps (pooling windows map onto pooling windows under a 90-deg rotation). No strided
  convolution is used, because stride-2 sampling of an even-sized map is not rotation symmetric.
  For C8/C16 the extra rotations (45, 22.5 deg, ...) are only approximately realised on the
  pixel grid (filters are sampled from exactly steerable continuous bases; the image itself must
  be interpolated) — measured and reported, not assumed.
* ``GroupPooling`` (max over the N channels of each regular field): per-location C_N-INVARIANT
  features. Spatially, the feature map still rotates with the image.
* Tokens = one per grid location, with centred grid coordinates. The token mixer uses only
  pairwise DISTANCES |p_i - p_j| as relative position encoding (rotation invariant; no absolute
  positional embedding, which would break equivariance). A rotation of the input by 90 deg
  therefore permutes the tokens; the mixer is permutation-equivariant, so the output token SET is
  INVARIANT (up to permutation), and the shared decoder's cross-attention ignores permutations.
* Note: pairwise distances cannot distinguish a point pattern from its mirror image, so in this
  head (option a) chirality must be carried by the per-location invariant features (C_N-invariant
  features are not reflection-invariant, so local handedness is preserved). The equivariant
  attention head (option b, ``TokenHead`` interface, arm D) keeps relative ORIENTATION as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as enn
from torch import nn

from ouroboros.encoder.base import EncoderOutput, ImageEncoder
from ouroboros.model import register_encoder

# ----------------------------------------------------------------------------------- blocks


class BlurPool2x2(enn.EquivariantModule):
    """Anti-aliased 2x downsampling: fixed binomial [1,2,1]^2 blur ('same', zero pad 1) then
    2x2 average pooling with stride 2. Channel-wise, so it commutes with any permutation
    representation; exact under 90-deg rotations for even-sized inputs."""

    def __init__(self, in_type: enn.FieldType):
        super().__init__()
        self.in_type = self.out_type = in_type
        k = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("kernel", (k[:, None] * k[None, :] / 16.0)[None, None])

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        return enn.GeometricTensor(_blurpool(x.tensor, self.kernel), self.out_type)

    def evaluate_output_shape(self, s):
        return s[0], s[1], s[2] // 2, s[3] // 2

    def export(self):
        return _BlurPoolTorch(self.kernel.clone()).eval()


def _blurpool(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError("BlurPool2x2 needs even spatial sizes for exact 90-deg equivariance")
    c = x.shape[1]
    x = F.conv2d(x, kernel.to(x.dtype).expand(c, 1, 3, 3), padding=1, groups=c)
    return F.avg_pool2d(x, 2, 2)


class _BlurPoolTorch(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return _blurpool(x, self.kernel)


class ResBlock(enn.EquivariantModule):
    """Equivariant basic residual block: conv3-BN-ReLU-conv3-BN (+ skip) -ReLU."""

    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType):
        super().__init__()
        self.in_type, self.out_type = in_type, out_type
        self.conv1 = enn.R2Conv(in_type, out_type, 3, padding=1, bias=False)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=True)
        self.conv2 = enn.R2Conv(out_type, out_type, 3, padding=1, bias=False)
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type, inplace=True)
        self.skip = (
            enn.SequentialModule(
                enn.R2Conv(in_type, out_type, 1, bias=False), enn.InnerBatchNorm(out_type)
            )
            if in_type != out_type
            else None
        )

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        y = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
        s = x if self.skip is None else self.skip(x)
        return self.relu2(y + s)

    def evaluate_output_shape(self, s):
        return s[0], self.out_type.size, s[2], s[3]

    def export(self):
        self.eval()
        return _ResBlockTorch(
            nn.Sequential(
                self.conv1.export(),
                self.bn1.export(),
                self.relu1.export(),
                self.conv2.export(),
                self.bn2.export(),
            ),
            None if self.skip is None else self.skip.export(),
        ).eval()


class _ResBlockTorch(nn.Module):
    def __init__(self, body, skip):
        super().__init__()
        self.body, self.skip = body, skip

    def forward(self, x):
        return F.relu(self.body(x) + (x if self.skip is None else self.skip(x)))


# ------------------------------------------------------------------------------ token heads


class RelDistMixer(nn.Module):
    """Transformer layers whose only positional signal is an attention bias b_h(|p_i - p_j|).

    Permutation-equivariant in the tokens and invariant to any isometry of the coordinates, so
    it maps a rotated-and-permuted token set to the correspondingly permuted output.
    """

    def __init__(self, d: int, layers: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.h = heads
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d),
                        "qkv": nn.Linear(d, 3 * d),
                        "o": nn.Linear(d, d),
                        "ln2": nn.LayerNorm(d),
                        "ff": nn.Sequential(
                            nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d)
                        ),
                        "bias": nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, heads)),
                    }
                )
            )
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        dist = torch.cdist(coords, coords)[..., None]  # [T, T, 1], rotation invariant
        dist = dist / dist.max().clamp_min(1e-6)
        for L in self.layers:
            bias = L["bias"](dist.to(x.dtype)).permute(2, 0, 1)[None]  # [1, H, T, T]
            h = L["ln1"](x)
            q, k, v = L["qkv"](h).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
            )
            x = x + self.drop(L["o"](y.transpose(1, 2).reshape(B, T, D)))
            x = x + self.drop(L["ff"](L["ln2"](x)))
        return self.ln(x)


def grid_coords(h: int, w: int, device=None) -> torch.Tensor:
    """Token coordinates centred on the image centre, row-major [h*w, 2] (x, y)."""
    ys = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2
    xs = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


class GPoolRelPosHead(nn.Module):
    """Option (a): group pooling -> invariant per-location features -> 1x1 projection ->
    distance-biased token mixer. Output: an invariant token SET (see module docstring)."""

    def __init__(self, in_type: enn.FieldType, d_model: int, layers, heads, ff, dropout):
        super().__init__()
        self.gpool = enn.GroupPooling(in_type)  # C_N-invariant from here on (per location)
        n_inv = self.gpool.out_type.size
        self.proj = nn.Conv2d(n_inv, d_model, 1)  # acts on invariant scalars: still invariant
        self.mixer = RelDistMixer(d_model, layers, heads, ff, dropout)

    def forward(self, feat: enn.GeometricTensor) -> EncoderOutput:
        return self.tokens(self.gpool(feat).tensor)

    def tokens(self, inv: torch.Tensor) -> EncoderOutput:
        f = self.proj(inv)  # [B, D, h, w]
        B, D, h, w = f.shape
        tok = f.flatten(2).transpose(1, 2)
        return EncoderOutput(tokens=self.mixer(tok, grid_coords(h, w, tok.device)))


# ------------------------------------------------------------------------------- encoder


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


class SteerableEncoder(ImageEncoder):
    """Trunk resolution schedule (384 input): 384 -stem-> pool 192 -> pool 96 [stage 1]
    -> pool 48 [stage 2] -> pool 24 [stage 3] -> pool 12 [stage 4] = 144 tokens (stride 32,
    same token grid as the baseline)."""

    def __init__(self, c: SteerableConfig):
        super().__init__()
        if c.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 (even maps at every pooling)")
        self.c = c
        self.d_model = c.d_model
        self.gspace = gspaces.rot2dOnR2(N=c.N)  # C_N: rotations only, NO reflections
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        reg = self.gspace.regular_repr
        stem_t = enn.FieldType(self.gspace, [reg] * (c.stem_fields or c.fields[0]))
        mods: list[enn.EquivariantModule] = [
            enn.R2Conv(self.in_type, stem_t, c.stem_kernel, padding=c.stem_kernel // 2, bias=False),
            enn.InnerBatchNorm(stem_t),
            enn.ReLU(stem_t, inplace=True),
            BlurPool2x2(stem_t),
            BlurPool2x2(stem_t),
        ]
        cur = stem_t
        for i, (nf, nb) in enumerate(zip(c.fields, c.blocks, strict=True)):
            if i > 0:
                mods.append(BlurPool2x2(cur))
            out_t = enn.FieldType(self.gspace, [reg] * nf)
            for _ in range(nb):
                mods.append(ResBlock(cur, out_t))
                cur = out_t
        self.trunk = enn.SequentialModule(*mods)
        self.out_type = cur
        if c.head != "gpool_relpos":
            raise KeyError(f"unknown head {c.head!r} (equivariant attention lives in arm D)")
        self.head = GPoolRelPosHead(
            cur, c.d_model, c.mixer_layers, c.mixer_heads, c.mixer_ff, c.dropout
        )

    def features(self, images: torch.Tensor) -> enn.GeometricTensor:
        """Equivariant feature maps (before group pooling)."""
        return self.trunk(enn.GeometricTensor(images, self.in_type))

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        return self.head(self.features(images))

    def export(self) -> ExportedSteerableEncoder:
        """Pure-PyTorch copy for inference (escnn filters expanded once)."""
        self.eval()
        return ExportedSteerableEncoder(self.trunk.export(), self.head).eval()


class ExportedSteerableEncoder(ImageEncoder):
    def __init__(self, trunk: nn.Module, head: GPoolRelPosHead):
        super().__init__()
        self.trunk = trunk
        self.gpool = head.gpool.export()  # escnn.nn.MaxPoolChannels
        self.head = head
        self.d_model = head.proj.out_channels

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        return self.head.tokens(self.gpool(self.trunk(images)))


@register_encoder("steerable")
def _steerable(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    kw = {k: v for k, v in enc_cfg.items() if k != "type"}
    return SteerableEncoder(SteerableConfig(d_model=d_model, image_size=image_size, **kw))


__all__ = ["SteerableEncoder", "SteerableConfig", "ExportedSteerableEncoder"]
