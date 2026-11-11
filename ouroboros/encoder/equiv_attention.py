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

import copy

import torch
from escnn import nn as enn
from torch import nn

from ouroboros.encoder.base import EncoderOutput, ImageEncoder
from ouroboros.encoder.pure_torch import (
    EquivAttnConfig,
    ExportedEquivAttentionEncoder,
    GroupAttentionHead,
    GroupAttentionLayer,
    GroupRelPosBias,
    lifted_coords,
)
from ouroboros.encoder.steerable import SteerableConfig, SteerableEncoder
from ouroboros.model import register_encoder

_ExportedEquivAttn = ExportedEquivAttentionEncoder


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

    def export(self) -> ExportedEquivAttentionEncoder:
        self.eval()
        return ExportedEquivAttentionEncoder(
            self.trunk.export(),
            copy.deepcopy(self.head),
            copy.deepcopy(self.proj),
            config=copy.deepcopy(self.c),
        ).eval()


@register_encoder("equiv_attention")
def _equiv_attention(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    kw = {k: v for k, v in enc_cfg.items() if k != "type"}
    return EquivAttentionEncoder(EquivAttnConfig(d_model=d_model, image_size=image_size, **kw))


__all__ = [
    "EquivAttnConfig",
    "EquivAttentionEncoder",
    "ExportedEquivAttentionEncoder",
    "GroupAttentionHead",
    "GroupAttentionLayer",
    "GroupRelPosBias",
    "lifted_coords",
]
