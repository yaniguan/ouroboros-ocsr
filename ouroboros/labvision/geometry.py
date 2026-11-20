"""Small numpy geometry helpers: homography, sampling, polar unwrap, Otsu threshold.

Polar angles are measured clockwise in the image from "up" (-y):
a point at angle t and radius r is (cx + r sin t, cy - r cos t).
"""

from __future__ import annotations

import numpy as np


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 homography to points of shape (..., 2)."""
    p = np.asarray(pts, dtype=float)
    flat = p.reshape(-1, 2)
    q = flat @ H[:, :2].T + H[:, 2]
    return (q[:, :2] / q[:, 2:3]).reshape(p.shape)


def _normaliser(p: np.ndarray) -> np.ndarray:
    m = p.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(p - m, axis=1))
    return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1.0]])


def fit_homography(src, dst) -> np.ndarray:
    """Normalised DLT: H with dst ~ H @ src, from >= 4 point pairs."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ts, td = _normaliser(src), _normaliser(dst)
    a, b = apply_h(ts, src), apply_h(td, dst)
    rows = []
    for (x, y), (u, v) in zip(a, b, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    hn = np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(td) @ hn @ ts
    return H / H[2, 2]


def bilinear(img: np.ndarray, x, y) -> np.ndarray:
    """Bilinear sample at (x, y); coordinates are clamped to the image."""
    h, w = img.shape
    x = np.clip(np.asarray(x, float), 0, w - 1)
    y = np.clip(np.asarray(y, float), 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(int), w - 2)
    y0 = np.minimum(np.floor(y).astype(int), h - 2)
    fx, fy = x - x0, y - y0
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def polar_sample(img: np.ndarray, center, radius: float, rho, theta) -> np.ndarray:
    """Sample at radius fractions ``rho`` (rows) and angles ``theta`` (columns, radians)."""
    r = np.asarray(rho, float)[:, None] * radius
    t = np.asarray(theta, float)[None, :]
    return bilinear(img, center[0] + r * np.sin(t), center[1] - r * np.cos(t))


def polar_unwrap(img, center, radius: float, n_r: int = 64, n_theta: int = 720):
    """(n_r, n_theta) unwrap; row k is at rho = (k + 0.5) / n_r, column j at 2 pi j / n_theta."""
    rho = (np.arange(n_r) + 0.5) / n_r
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    return polar_sample(img, center, radius, rho, theta)


def otsu(img: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold of an image with values in [0, 1]."""
    hist, edges = np.histogram(np.clip(img, 0, 1), bins=bins, range=(0, 1))
    p = hist / hist.sum()
    mid = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * mid)
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m0[-1] * w0 - m0) ** 2 / (w0 * w1)
    return float(mid[np.nanargmax(np.where(w1 > 0, between, np.nan))])
