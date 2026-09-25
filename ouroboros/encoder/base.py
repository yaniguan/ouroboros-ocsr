"""Encoder interface shared by all experimental arms.

The decoder is identical across arms (hard constraint 4), so every encoder must emit the
same thing: a set of ``d_model``-dim memory tokens plus a padding mask. The decoder
cross-attends to these tokens WITHOUT adding any positional embedding, so whether the
decoder output is rotation invariant is decided entirely by the encoder:

* baseline encoders add absolute 2D positional embeddings -> NOT rotation invariant
  (by design; that is the control arm);
* equivariant encoders emit group-pooled (invariant) tokens mixed with
  rotation-invariant relative position encodings -> the token SET is invariant under
  on-grid rotations, up to a permutation of tokens, which cross-attention ignores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EncoderOutput:
    tokens: torch.Tensor  # [B, T, d_model]
    padding_mask: torch.Tensor | None = None  # [B, T] bool, True = padded (ignored)


class ImageEncoder(nn.Module):
    """Base class: ``forward(images [B, C, H, W]) -> EncoderOutput``."""

    d_model: int

    def forward(self, images: torch.Tensor) -> EncoderOutput:  # pragma: no cover - abstract
        raise NotImplementedError
