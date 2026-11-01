"""Rotation-invariant relative-position token mixer (shared by the steerable and baseline arms).

The only positional signal is an attention bias b_h(|p_i - p_j|) computed from token-grid distances,
so the mixer is permutation-equivariant and invariant to isometries of the token coordinates. It has
no dependency on escnn, so a plain CNN trunk can use exactly the same mixer (arm B_rel).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
