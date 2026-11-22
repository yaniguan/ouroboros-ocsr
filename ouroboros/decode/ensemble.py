"""Batched ensemble + test-time augmentation (TTA) decoding over OCSR models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from ouroboros.decode.rerank import Candidate, choose, ensemble_rank
from ouroboros.model import OCSRModel


def tta_pad(images: torch.Tensor, scale: float = 0.9) -> torch.Tensor:
    """Shrink the whole image to ``scale`` and recentre it on a white (zero-ink) canvas."""
    h_in, w_in = images.shape[-2:]
    h, w = max(1, round(h_in * scale)), max(1, round(w_in * scale))
    small = F.interpolate(
        images.float(), size=(h, w), mode="bilinear", align_corners=False, antialias=True
    )
    out = torch.zeros_like(images)
    top, left = (h_in - h) // 2, (w_in - w) // 2
    out[..., top : top + h, left : left + w] = small.clamp(0.0, 1.0).to(images.dtype)
    return out


def tta_sharpen(images: torch.Tensor) -> torch.Tensor:
    """PIL unsharp mask on the grayscale rendering of each single-channel ink image."""
    out = []
    for img in images.detach().float().cpu():
        gray = ((1.0 - img[0].clamp(0.0, 1.0)) * 255.0).round().to(torch.uint8).numpy()
        sharp = Image.fromarray(gray, "L").filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
        out.append(1.0 - torch.from_numpy(np.array(sharp, dtype=np.float32)) / 255.0)
    return torch.stack(out)[:, None].to(images.device, images.dtype)


TTA = {"pad": tta_pad, "sharpen": tta_sharpen}


def predict_batch(
    models: list[OCSRModel],
    images: torch.Tensor,
    beam: int,
    tta: list[str],
    prefer_valid: bool,
    diagnostics: list | None = None,
) -> list[tuple[str, list[Candidate]]]:
    """Per image: ``(prediction, ranked pooled candidates)`` over all (model, view) sources."""
    views = [images, *(TTA[name](images) for name in tta)]
    sources = [(m, x) for m in models for x in views]
    nbest = [m.generate_nbest(x, beam_size=beam) for m, x in sources]  # [source][image]
    n = images.shape[0]
    rescored: list[dict[str, list[float]]] = [{} for _ in range(n)]
    if len(sources) > 1:  # a single source's beam scores are already its own log probs
        flat = [
            (i, s)
            for i in range(n)
            for s in dict.fromkeys(s for per_src in nbest for s, _ in per_src[i])
        ]
        idx, smiles = [i for i, _ in flat], [s for _, s in flat]
        per_source = [m.score_smiles(x, smiles, image_index=idx).tolist() for m, x in sources]
        for j, (i, s) in enumerate(flat):
            rescored[i][s] = [scores[j] for scores in per_source]
    out = []
    for i in range(n):
        ranked = ensemble_rank([per_src[i] for per_src in nbest], rescored[i])
        pred = choose(ranked)[0] if prefer_valid else ranked[0][0]
        if diagnostics is not None:
            predictions = []
            for per_src in nbest:
                if prefer_valid:
                    predictions.append(choose(per_src[i])[0])
                else:
                    predictions.append(max(per_src[i], key=lambda c: c[1])[0])
            diagnostics.append(predictions)
        out.append((pred, ranked))
    return out
