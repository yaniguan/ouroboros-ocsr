"""Arm E (optional): learned canonicalization + standard encoder.

A small C_N-equivariant network (escnn, rotations only — no reflections) predicts orientation
logits over the N rotations; the input is rotated back to the predicted canonical orientation and
fed to the unchanged baseline encoder (Kaba et al., "Equivariance with learned canonicalization").

Where equivariance / invariance holds
-------------------------------------
* ``orient``: C_N-EQUIVARIANT. Global average pooling of regular fields followed by a sum over
  fields gives logits that are cyclically SHIFTED when the input is rotated by an element of C_N.
* canonical image = rotate(x, -argmax(logits)): INVARIANT to C_N rotations of the input (exactly for
  90-deg multiples with N = 4, where rotation is a pixel permutation; C8/C16 need interpolation).
* baseline encoder on the canonical image: not equivariant itself, but it only ever sees the
  canonical image, so the whole encoder is INVARIANT (argmax ties aside, which have measure zero).
* The argmax is not differentiable: the orientation network is trained with a canonicalization
  prior (cross-entropy towards the orientation the training image actually has: 0 for upright
  drawings, or the augmentation angle quantized to C_N), added by the trainer via ``prior_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as enn
from torch import nn

from ouroboros.data.loader import rotate_images
from ouroboros.encoder.base import EncoderOutput, ImageEncoder
from ouroboros.encoder.baseline import BaselineConfig, BaselineEncoder
from ouroboros.encoder.steerable import BlurPool2x2, ResBlock
from ouroboros.model import register_encoder


class OrientationNet(nn.Module):
    """C_N-equivariant orientation logits [B, N] from a downsampled image."""

    def __init__(self, N: int, fields: list[int], downsample: int):
        super().__init__()
        self.N, self.downsample = N, downsample
        self.gspace = gspaces.rot2dOnR2(N=N)  # rotations only
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        reg = self.gspace.regular_repr
        t0 = enn.FieldType(self.gspace, [reg] * fields[0])
        mods: list[enn.EquivariantModule] = [
            enn.R2Conv(self.in_type, t0, 5, padding=2, bias=False),
            enn.InnerBatchNorm(t0),
            enn.ReLU(t0, inplace=True),
        ]
        cur = t0
        for nf in fields[1:]:
            mods.append(BlurPool2x2(cur))
            out = enn.FieldType(self.gspace, [reg] * nf)
            mods.append(ResBlock(cur, out))
            cur = out
        self.net = enn.SequentialModule(*mods)
        self.n_fields = len(cur)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.downsample > 1:  # k x k average pooling on even sizes: exact under 90-deg rotations
            images = F.avg_pool2d(images, self.downsample)
        f = self.net(enn.GeometricTensor(images, self.in_type)).tensor  # [B, F*N, h, w]
        B = f.shape[0]
        # escnn regular layout: channel = field * N + h; mean over space, sum over fields
        return f.mean(dim=(2, 3)).view(B, self.n_fields, self.N).sum(dim=1)


@dataclass
class CanonConfig:
    N: int = 4
    orient_fields: list[int] = field(default_factory=lambda: [4, 8, 8, 16])
    downsample: int = 4
    prior_weight: float = 1.0
    base: dict = field(default_factory=dict)  # BaselineConfig overrides for the main encoder
    d_model: int = 512
    image_size: int = 384


class CanonicalizedEncoder(ImageEncoder):
    def __init__(self, c: CanonConfig):
        super().__init__()
        self.c = c
        self.d_model = c.d_model
        self.orient = OrientationNet(c.N, c.orient_fields, c.downsample)
        self.encoder = BaselineEncoder(
            BaselineConfig(d_model=c.d_model, image_size=c.image_size, **c.base)
        )
        self.last_logits: torch.Tensor | None = None

    def canonicalize(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.orient(images)
        k = logits.argmax(dim=1)  # predicted orientation index in C_N
        angles = -k.to(torch.float32) * (360.0 / self.c.N)  # rotate back to canonical
        return rotate_images(images, angles), logits

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        canon, logits = self.canonicalize(images)
        self.last_logits = logits
        return self.encoder(canon)

    def prior_loss(self, angles_deg: torch.Tensor | None) -> torch.Tensor:
        """Cross-entropy towards the true orientation of the last batch (0 = upright)."""
        logits = self.last_logits
        if angles_deg is None:
            target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        else:
            step = 360.0 / self.c.N
            target = (torch.round(angles_deg.to(logits.device) / step).long()) % self.c.N
        return self.c.prior_weight * F.cross_entropy(logits.float(), target)


@register_encoder("canonicalized")
def _canonicalized(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    kw = {k: v for k, v in enc_cfg.items() if k != "type"}
    return CanonicalizedEncoder(CanonConfig(d_model=d_model, image_size=image_size, **kw))
