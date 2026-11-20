"""Seed-deterministic synthetic renderer for top-view plate images.

The canonical plate drawing is evaluated at H^-1(pixel) for every pixel, then
glare (on the plate only), an occluder, blur and noise are added.
Settings are frozen; the benchmark must not tune them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import apply_h
from .spec import CircSpec, RectSpec, Spec, default_order, grid_index

BACKGROUND, PLATE, MARK = 0.12, 0.78, 0.08
STATE_LEVELS = (0.60, 0.48, 0.28, 0.52)  # empty, clear liquid, coloured, turbid
STATE_NAMES = ("empty", "clear", "coloured", "turbid")
TURBID_STD = 0.08
BLUR_SIGMA, NOISE_STD = 0.7, 0.02


@dataclass
class Scene:
    """A rendered image plus ground truth. H maps canonical plate coords to pixels."""

    image: np.ndarray
    kind: str
    spec: Spec
    H: np.ndarray
    rot_deg: float
    persp: float
    occluded: bool
    mirrored: bool
    hide_mark: bool
    states: np.ndarray  # (96,) state per well, indexed by default_order(spec) - 1


def _fill_wells(img, well, states, rng):
    img = np.where(well, np.asarray(STATE_LEVELS)[states], img)
    turbid = well & (states == 3)
    img[turbid] += rng.normal(0, TURBID_STD, int(turbid.sum()))
    return img


def _draw_rect(u, v, spec: RectSpec, states, hide_mark, rng):
    plate = (u >= 0) & (u <= spec.width) & (v >= 0) & (v <= spec.height)
    if not hide_mark:
        plate &= u + v >= spec.chamfer
    col = np.clip(np.rint((u - spec.x0) / spec.pitch), 0, spec.cols - 1).astype(int)
    row = np.clip(np.rint((v - spec.y0) / spec.pitch), 0, spec.rows - 1).astype(int)
    d = np.hypot(u - spec.x0 - col * spec.pitch, v - spec.y0 - row * spec.pitch)
    img = np.where(plate, PLATE, BACKGROUND)
    img = _fill_wells(img, plate & (d <= spec.well_r), states[row * spec.cols + col], rng)
    if not hide_mark:
        img[np.hypot(u - spec.dot_xy[0], v - spec.dot_xy[1]) <= spec.dot_r] = MARK
    return img, plate


def _draw_circ(u, v, spec: CircSpec, states, hide_mark, rng):
    R = spec.radius
    plate = np.hypot(u, v) <= R
    if not hide_mark:
        plate &= ~((np.abs(u) < spec.notch_w * R / 2) & (v < -(R - spec.notch_d * R)))
    sgn = 1.0 if spec.clockwise else -1.0
    ang = np.mod(np.arctan2(sgn * u, -v), 2 * np.pi)  # from the notch, in the slot direction
    best_d = np.full(u.shape, np.inf)
    best_i = np.zeros(u.shape, dtype=int)
    order = default_order(spec)
    for ring, (n, rr) in enumerate(zip(spec.slots, spec.radii, strict=True)):
        slot = np.mod(np.rint(ang / (2 * np.pi / n)), n).astype(int)
        a = 2 * np.pi * slot / n
        d = np.hypot(u - sgn * rr * R * np.sin(a), v + rr * R * np.cos(a))
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_i = np.where(closer, grid_index(spec, ring, 0, order=order) - 1 + slot, best_i)
    img = np.where(plate, PLATE, BACKGROUND)
    img = _fill_wells(img, plate & (best_d <= spec.well_r * R), states[best_i], rng)
    if not hide_mark:
        t = np.radians(spec.dot_deg)
        dx, dy = sgn * spec.dot_rho * R * np.sin(t), -spec.dot_rho * R * np.cos(t)
        img[np.hypot(u - dx, v - dy) <= spec.dot_r * R] = MARK
    return img, plate


def _random_plate_point(spec: Spec, rng) -> np.ndarray:
    if isinstance(spec, RectSpec):
        return np.array([rng.uniform(0, spec.width), rng.uniform(0, spec.height)])
    r, t = spec.radius * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    return np.array([r * np.sin(t), -r * np.cos(t)])


def _pose(spec: Spec, rng, size: int, rot_deg: float, persp: float) -> np.ndarray:
    if isinstance(spec, RectSpec):
        centre, extent = (spec.width / 2, spec.height / 2), np.hypot(spec.width, spec.height) / 2
    else:
        centre, extent = (0.0, 0.0), spec.radius
    s = rng.uniform(0.9, 1.0) * 0.4 * size / extent
    c, n = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    t_c = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1.0]])
    rot = np.array([[s * c, -s * n, 0], [s * n, s * c, 0], [0, 0, 1.0]])
    phi = rng.uniform(0, 2 * np.pi)
    k = persp / (size / 2)
    proj = np.array([[1, 0, 0], [0, 1, 0], [k * np.cos(phi), k * np.sin(phi), 1.0]])
    jx, jy = rng.uniform(-5, 5, 2)
    t_out = np.array([[1, 0, size / 2 + jx], [0, 1, size / 2 + jy], [0, 0, 1.0]])
    return t_out @ proj @ rot @ t_c


def _occluder(img, rng, size, end):
    """A dark bar (glove, pipette) entering from a random image border towards ``end``."""
    side, f = rng.integers(4), rng.uniform(0, size - 1)
    start = [(f, 0.0), (size - 1.0, f), (f, size - 1.0), (0.0, f)][side]
    a, b = np.asarray(start), np.asarray(end)
    ys, xs = np.mgrid[0:size, 0:size]
    p = np.stack([xs, ys], axis=-1) - a
    t = np.clip((p @ (b - a)) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
    dist = np.linalg.norm(p - t[..., None] * (b - a), axis=-1)
    img[dist <= rng.uniform(0.04, 0.08) * size / 2] = rng.uniform(0.08, 0.22)
    return img


def _blur(img, sigma):
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        pad = np.pad(img, [(r, r) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        n = img.shape[axis]
        img = sum(w * np.take(pad, np.arange(j, j + n), axis=axis) for j, w in enumerate(k))
    return img


def render(
    kind: str,
    rng,
    *,
    size=256,
    rot_deg=None,
    persp=0.05,
    occlude=True,
    glare=True,
    hide_mark=False,
    mirror=False,
    spec: Spec | None = None,
) -> Scene:
    """Render one plate image. ``mirror=True`` flips the output horizontally."""
    if spec is None:
        spec = RectSpec() if kind == "rect" else CircSpec()
    if spec.kind != kind:
        raise ValueError(f"spec kind {spec.kind} does not match {kind}")
    rot = float(rng.uniform(0, 360)) if rot_deg is None else float(rot_deg) % 360
    states = rng.integers(0, len(STATE_LEVELS), 96)
    H = _pose(spec, rng, size, rot, persp)
    ys, xs = np.mgrid[0:size, 0:size]
    uv = apply_h(np.linalg.inv(H), np.stack([xs, ys], axis=-1).astype(float))
    draw = _draw_rect if kind == "rect" else _draw_circ
    img, plate = draw(uv[..., 0], uv[..., 1], spec, states, hide_mark, rng)
    if glare:
        gx, gy = apply_h(H, _random_plate_point(spec, rng))
        sig = rng.uniform(0.04, 0.10) * size
        blob = np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * sig**2))
        img = img + rng.uniform(0.25, 0.5) * blob * plate
    if occlude:
        img = _occluder(img, rng, size, apply_h(H, _random_plate_point(spec, rng)))
    img = _blur(img, BLUR_SIGMA) + rng.normal(0, NOISE_STD, img.shape)
    img = np.clip(img, 0, 1)
    if mirror:
        img = img[:, ::-1].copy()
        H = np.array([[-1, 0, size - 1.0], [0, 1, 0], [0, 0, 1]]) @ H
    return Scene(img, kind, spec, H, rot, persp, occlude, mirror, hide_mark, states)
