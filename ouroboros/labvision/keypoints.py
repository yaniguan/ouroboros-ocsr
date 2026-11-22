"""Plate keypoints: 4 corners (rect) or centre + radius (circ).

``kp_from_pixels`` is the headline source; ``kp_from_gt`` (ground truth + noise) is diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import apply_h, otsu
from .spec import RectSpec


@dataclass
class Keypoints:
    """Rect: the A1 corner point is the chamfer midpoint (the outline extreme there)."""

    kind: str
    corners: np.ndarray | None = None  # (4, 2) pixel corners, any order
    center: np.ndarray | None = None  # (2,) pixel centre
    radius: float | None = None  # pixels


def gt_keypoints(scene) -> Keypoints:
    spec, H = scene.spec, scene.H
    if isinstance(spec, RectSpec):
        w, h, a1 = spec.width, spec.height, 0.0 if scene.hide_mark else spec.chamfer / 2
        return Keypoints("rect", corners=apply_h(H, [[a1, a1], [w, 0], [w, h], [0, h]]))
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rim = apply_h(H, spec.radius * np.stack([np.sin(t), -np.cos(t)], axis=1))
    c = apply_h(H, np.zeros(2))
    return Keypoints("circ", center=c, radius=float(np.linalg.norm(rim - c, axis=1).mean()))


def kp_from_gt(scene, sigma: float, rng) -> Keypoints:
    """Ground-truth keypoints plus Gaussian noise of ``sigma`` px."""
    kp = gt_keypoints(scene)
    if kp.kind == "rect":
        kp.corners = kp.corners + rng.normal(0, sigma, kp.corners.shape)
    else:
        kp.center = kp.center + rng.normal(0, sigma, 2)
        kp.radius = kp.radius + float(rng.normal(0, sigma))
    return kp


def _rect_corners(pts: np.ndarray) -> np.ndarray:
    """Extreme mask pixels along the 4 diagonals of the mask's PCA frame."""
    m = pts.mean(axis=0)
    e = np.linalg.eigh(np.cov((pts - m).T))[1]
    dirs = e @ np.array([[1, 1, -1, -1], [1, -1, -1, 1]])
    return pts[np.argmax((pts - m) @ dirs, axis=0)].astype(float)


def _kasa(p: np.ndarray):
    a = np.column_stack([p, np.ones(len(p))])
    d, e, f = np.linalg.lstsq(a, -(p**2).sum(axis=1), rcond=None)[0]
    c = np.array([-d / 2, -e / 2])
    return c, float(np.sqrt(max(c @ c - f, 0.0)))


def _circle(mask: np.ndarray, pts: np.ndarray):
    """Farthest mask pixel on 360 rays, Kasa fit, drop the worst 10 %, refit."""
    m = pts.mean(axis=0)
    t = np.radians(np.arange(360))[:, None]
    steps = np.arange(0, np.hypot(*mask.shape), 0.5)[None, :]
    x = np.rint(m[0] + steps * np.sin(t)).astype(int)
    y = np.rint(m[1] - steps * np.cos(t)).astype(int)
    h, w = mask.shape
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    hit = np.zeros(x.shape, dtype=bool)
    hit[ok] = mask[y[ok], x[ok]]
    has = hit.any(axis=1)
    if int(has.sum()) < 4:
        raise ValueError("plate mask has too few rim intersections to fit a circle")
    last = steps.shape[1] - 1 - np.argmax(hit[:, ::-1], axis=1)
    r = steps[0, last][has]
    edge = np.column_stack([m[0] + r * np.sin(t[has, 0]), m[1] - r * np.cos(t[has, 0])])
    c, rad = _kasa(edge)
    res = np.abs(np.linalg.norm(edge - c, axis=1) - rad)
    keep = res <= np.quantile(res, 0.9)
    return _kasa(edge[keep])


def kp_from_pixels(img: np.ndarray, kind: str) -> Keypoints:
    """Keypoints from an Otsu plate mask."""
    mask = img > otsu(img)
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    if len(pts) < 4:
        raise ValueError("plate mask has fewer than 4 foreground pixels")
    if kind == "rect":
        return Keypoints("rect", corners=_rect_corners(pts))
    c, r = _circle(mask, pts)
    return Keypoints("circ", center=c, radius=r)
