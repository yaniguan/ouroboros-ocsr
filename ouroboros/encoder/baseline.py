"""Baseline (non-equivariant) encoder for arms A and B.

ResNet-style CNN (stride 32) -> 1x1 projection to ``d_model`` -> + learned ABSOLUTE 2D positional
embedding -> small Transformer token mixer -> memory tokens.

Symmetry: none. Convolutions are translation-equivariant, but the absolute positional embedding
intentionally breaks rotation equivariance (and translation invariance): this is the standard
control architecture, and rotation robustness can only come from data augmentation (arm B).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ouroboros.encoder.base import EncoderOutput, ImageEncoder


@dataclass
class BaselineConfig:
    in_channels: int = 1
    widths: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    blocks: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    d_model: int = 512
    mixer_layers: int = 2
    mixer_heads: int = 8
    mixer_ff: int = 2048
    dropout: float = 0.1
    image_size: int = 384  # sets the size of the positional-embedding grid


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.skip = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if stride != 1 or cin != cout
            else nn.Identity()
        )

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(y)) + self.skip(x))


class TokenMixer(nn.Module):
    """Pre-LN Transformer encoder layers over memory tokens (shared building block)."""

    def __init__(self, d: int, layers: int, heads: int, ff: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        return self.ln(self.enc(x, mask=mask))


class BaselineEncoder(ImageEncoder):
    def __init__(self, c: BaselineConfig):
        super().__init__()
        self.c = c
        self.d_model = c.d_model
        self.stem = nn.Sequential(
            nn.Conv2d(c.in_channels, c.widths[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(c.widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        stages, cin = [], c.widths[0]
        for i, (w, n) in enumerate(zip(c.widths, c.blocks, strict=True)):
            for j in range(n):
                stages.append(BasicBlock(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
        self.stages = nn.Sequential(*stages)
        self.proj = nn.Conv2d(cin, c.d_model, 1)
        g = c.image_size // 32
        # Absolute 2D positional embedding: BREAKS rotation equivariance on purpose (baseline).
        self.pos = nn.Parameter(torch.zeros(1, g * g, c.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mixer = TokenMixer(c.d_model, c.mixer_layers, c.mixer_heads, c.mixer_ff, c.dropout)

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        f = self.proj(self.stages(self.stem(images)))  # [B, D, g, g]
        tokens = f.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.pos.shape[1]:
            raise ValueError(
                f"image size gives {tokens.shape[1]} tokens, pos-embedding has {self.pos.shape[1]}"
            )
        return EncoderOutput(tokens=self.mixer(tokens + self.pos))
