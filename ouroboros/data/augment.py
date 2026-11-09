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


def _smooth_field(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    """Low-frequency random field in [-1, 1]: coarse noise bilinearly upsampled."""
    cells = max(2, cells)
    coarse = rng.uniform(-1, 1, (cells, cells)).astype(np.float32)
    img = Image.fromarray(coarse, mode='F').resize((w, h), Image.Resampling.BICUBIC)
    return np.clip(np.asarray(img, dtype=np.float32), -1.5, 1.5)


def _remap(a: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Bilinear sample a at (y+dy, x+dx); outside the page is white."""
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y, x = yy + dy, xx + dx
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - y0, x - x0
    pad = np.pad(a, 1, constant_values=255.0)

    def at(yi, xi):
        return pad[np.clip(yi + 1, 0, h + 1), np.clip(xi + 1, 0, w + 1)]
    return ((1 - fy) * (1 - fx) * at(y0, x0) + (1 - fy) * fx * at(y0, x0 + 1)
            + fy * (1 - fx) * at(y0 + 1, x0) + fy * fx * at(y0 + 1, x0 + 1))


def handdrawn_augment(image: Image.Image, rng: np.random.Generator, *,
                      wobble: float = 1.5, wobble_scale: int = 6, jitter: float = 0.4,
                      width_var: float = 0.5, ink_var: float = 0.25, paper_noise: float = 6.0,
                      blur: float = 0.4) -> Image.Image:
    """Make a clean rendered depiction look hand drawn (ACS hand-drawn style).

    wobble: amplitude in pixels of a smooth elastic warp (pen wobble / wavy lines);
    wobble_scale: number of coarse control cells per side (higher = wigglier);
    jitter: amplitude in pixels of a finer warp (stroke jitter);
    width_var: fraction of the page where strokes are thickened or thinned (line-width variation);
    ink_var: relative spatial variation of ink darkness (pressure / pencil);
    paper_noise: std of paper grain plus a faint fibre texture (grey levels);
    blur: Gaussian blur radius (pixels) applied last.
    The caller owns the RNG; output size and mode ('L') match the input.
    """
    if wobble < 0 or jitter < 0 or not 0 <= width_var <= 1 or not 0 <= ink_var <= 1 \
            or paper_noise < 0 or blur < 0 or wobble_scale < 2:
        raise ValueError('Invalid hand-drawn augmentation parameters')
    img = image.convert('L')
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape
    dy = wobble * _smooth_field(rng, h, w, wobble_scale) + jitter * _smooth_field(rng, h, w, 4 * wobble_scale)
    dx = wobble * _smooth_field(rng, h, w, wobble_scale) + jitter * _smooth_field(rng, h, w, 4 * wobble_scale)
    a = _remap(a, dy, dx)
    if width_var:
        cur = Image.fromarray(np.clip(a, 0, 255).astype('uint8'))
        thick = np.asarray(cur.filter(ImageFilter.MinFilter(3)), dtype=np.float32)
        thin = np.asarray(cur.filter(ImageFilter.MaxFilter(3)), dtype=np.float32)
        field = _smooth_field(rng, h, w, wobble_scale)
        lo, hi = np.quantile(field, [width_var / 2, 1 - width_var / 2])
        a = np.where(field > hi, np.minimum(a, thick), np.where(field < lo, 0.5 * (a + thin), a))
    if ink_var:
        # lighten ink (dark pixels) by a smooth pressure field; white paper stays white
        pressure = 1 - ink_var * (0.5 + 0.5 * _smooth_field(rng, h, w, wobble_scale))
        a = 255 - (255 - a) * np.clip(pressure, 0, 1)
    if paper_noise:
        fibres = _smooth_field(rng, h, w, max(2, w // 4)) * paper_noise
        a = a - np.abs(rng.normal(0, paper_noise, (h, w))) - np.abs(fibres) * 0.5
    out = Image.fromarray(np.clip(a, 0, 255).astype('uint8'))
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out
