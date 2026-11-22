"""Stand-in well-state classifier: (mean, std) per well patch, nearest centroid."""

from __future__ import annotations

import numpy as np

from .geometry import bilinear


def patch_features(img: np.ndarray, centres: np.ndarray, radius: float) -> np.ndarray:
    """(N, 2) array of (mean, std) over a disk of ``radius`` px at each centre."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    r = max(int(np.floor(radius)), 1)
    g = np.arange(-r, r + 1)
    off = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    off = off[np.hypot(off[:, 0], off[:, 1]) <= max(radius, 0.0)]
    c = np.asarray(centres, float)
    vals = bilinear(img, c[:, None, 0] + off[None, :, 0], c[:, None, 1] + off[None, :, 1])
    return np.column_stack([vals.mean(axis=1), vals.std(axis=1)])


def fit_centroids(features: np.ndarray, labels: np.ndarray, n_classes: int = 4):
    """Per-class centroids in feature space scaled to unit std. Returns (centroids, scale)."""
    labels = np.asarray(labels)
    missing = [k for k in range(n_classes) if not np.any(labels == k)]
    if missing:
        raise ValueError(f"missing training samples for class(es): {missing}")
    scale = features.std(axis=0) + 1e-9
    z = features / scale
    cents = np.array([z[labels == k].mean(axis=0) for k in range(n_classes)])
    return cents, scale


def classify(features: np.ndarray, model) -> np.ndarray:
    cents, scale = model
    d = (((features / scale)[:, None, :] - cents[None]) ** 2).sum(axis=-1)
    return np.argmin(d, axis=1)
