"""Spatial 2D stereochemistry auxiliary head over image encoder feature maps.

Predicts a 4-channel heatmap ``[B, 4, H_g, W_g]`` aligned with the encoder's spatial grid:
  0: ``chiral_R``         — CIP (R) tetrahedral stereocenters
  1: ``chiral_S``         — CIP (S) tetrahedral stereocenters
  2: ``wedge_dash_bond``  — solid wedge or dashed hash bond midpoints
  3: ``ez_double_bond``   — stereogenic (E)/(Z) double-bond midpoints

Because symmetric 3x3 and 1x1 convolutions on per-cell ``C_N``-invariant scalar maps commute
with 90-degree spatial grid rotations, ``SpatialStereoHead`` preserves exact ``C_4`` spatial
equivariance when attached to ``SteerableEncoder`` or ``ExportedSteerableEncoder``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

STEREO_CHANNELS = ("chiral_R", "chiral_S", "wedge_dash_bond", "ez_double_bond")


@dataclass
class StereoHeadConfig:
    enabled: bool = False
    hidden: int = 128
    out_channels: int = 4
    pos_weight: float = 8.0
    loss_weight: float = 0.5


def _rot90_symmetric_conv3(in_ch: int, out_ch: int) -> nn.Conv2d:
    """Standard 3x3 Conv2d initialized cleanly for dense spatial keypoint heatmaps."""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    nn.init.zeros_(conv.bias)
    return conv


class SpatialStereoHead(nn.Module):
    """Lightweight 2-layer convolutional head mapping ``[B, D, H_g, W_g] -> [B, 4, H_g, W_g]``."""

    def __init__(
        self,
        d_model: int,
        hidden: int = 128,
        out_channels: int = 4,
        pos_weight: float = 8.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_channels = out_channels
        self.pos_weight = float(pos_weight)
        self.net = nn.Sequential(
            _rot90_symmetric_conv3(d_model, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )
        # Prior bias around -2.2 (~0.10 foreground probability) stabilizes early sparse heatmap loss
        nn.init.constant_(self.net[-1].bias, -2.2)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4:
            raise ValueError(
                f"Expected 4D spatial map [B, D, H, W], got shape {tuple(spatial.shape)}"
            )
        return self.net(spatial)

    def loss(self, spatial: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """Positive-weighted binary cross-entropy against soft target heatmaps in ``[0, 1]``."""
        logits = self.forward(spatial).float()
        targets = target_heatmaps.to(device=logits.device, dtype=torch.float32)
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(
                targets, size=logits.shape[-2:], mode="bilinear", align_corners=False
            )
        targets = targets.clamp(0.0, 1.0)
        weights = 1.0 + (self.pos_weight - 1.0) * targets
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)

    @torch.no_grad()
    def predict_probs(self, spatial: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(spatial))


__all__ = ["STEREO_CHANNELS", "SpatialStereoHead", "StereoHeadConfig"]
