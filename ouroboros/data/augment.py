"""Deterministic CPU real-document augmentation on PIL grayscale images."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def scan_augment(image: Image.Image, rng: np.random.Generator, *,
                 affine: float = 0.025, scan_noise: float = 3.0,
                 morphology: int = 0, threshold: int = 0) -> Image.Image:
    """Small affine/skew, uneven scan background, morphology, optional binarization.

    Positive morphology thickens dark ink; negative thins it. The caller owns
    the RNG; global random state is never consumed. White fills outside the page.
    """
    if not 0 <= affine <= 0.15 or scan_noise < 0 or morphology not in (-1, 0, 1):
        raise ValueError('Invalid scan augmentation parameters')
    if not 0 <= threshold <= 255:
        raise ValueError('threshold must be in [0, 255]')
    img = image.convert('L')
    if affine:
        w, h = img.size
        sx, sy, shear, skew = rng.uniform(-affine, affine, 4)
        tx, ty = rng.uniform(-affine, affine, 2)
        img = img.transform(img.size, Image.Transform.AFFINE,
                            (1 + sx, shear, tx*w - sx*w/2 - shear*h/2,
                             skew, 1 + sy, ty*h - sy*h/2 - skew*w/2),
                            Image.Resampling.BICUBIC, fillcolor=255)
    if scan_noise:
        a = np.asarray(img, dtype=np.float32)
        h, w = a.shape
        # correlated row noise plus a weak illumination gradient and sensor noise
        noise = rng.normal(0, scan_noise, (h, 1)) + rng.normal(0, scan_noise/2, (h, w))
        gradient = np.linspace(0, rng.uniform(-3*scan_noise, 0), w)[None, :]
        img = Image.fromarray(np.clip(a + noise + gradient, 0, 255).astype('uint8'))
    if morphology:
        filtered = img.filter(ImageFilter.MinFilter(3) if morphology > 0 else ImageFilter.MaxFilter(3))
        # Partial morphology preserves one-pixel bonds and small glyphs under thinning.
        img = Image.blend(img, filtered, .6 if morphology > 0 else .4)
    if threshold:
        img = img.point(lambda x: 255 if x >= threshold else 0)
    return img
