"""Decode plate pose from keypoints + image, and map well index <-> pixel.

Rotation symmetry is broken by dark fiducials: the A1 chamfer + dot (rect), and the
rim notch (circ). The circ chiral dot breaks mirror symmetry. A low score margin
sets ``ambiguous``; callers should treat ambiguous decodes as failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import apply_h, bilinear, fit_homography, polar_sample, polar_unwrap
from .keypoints import Keypoints
from .spec import CircSpec, IndexOrder, RectSpec, Spec, grid_index, node_xy, well_xy

MARGIN_MIN = 3.0  # in robust noise units (sigma from the MAD)
SIGMA_FLOOR = 0.01  # avoids divide-by-zero on noise-free images
SIDE_RATIO_MIN = 1.15  # below this the long edge is not trusted: try all 4 rotations


@dataclass
class Decode:
    kind: str
    H: np.ndarray  # canonical plate coords -> pixels
    theta0: float  # plate rotation in degrees, clockwise in the image, [0, 360)
    ambiguous: bool
    mirrored: bool
    margin: float


def _contrast(mark: np.ndarray, surround: np.ndarray) -> np.ndarray:
    """Per-sample darkness of a mark relative to the mean of its local surround."""
    return surround.mean() - mark.ravel()


def _pick(contrasts: list[np.ndarray]) -> tuple[int, float]:
    """Darkest candidate and its margin.

    margin = min(best - second, best) / sigma_MAD of the other candidates' samples, so a
    clear winner must also be darker than its own surround.
    """
    scores = np.array([c.mean() for c in contrasts])
    rank = np.argsort(scores)[::-1]
    rest = np.concatenate([c for k, c in enumerate(contrasts) if k != rank[0]])
    sigma = max(1.4826 * float(np.median(np.abs(rest - np.median(rest)))), SIGMA_FLOOR)
    gap = min(scores[rank[0]] - scores[rank[1]], scores[rank[0]])
    return int(rank[0]), float(gap / sigma)


def _disk(r: float, n: int = 7) -> np.ndarray:
    g = np.linspace(-r, r, n)
    xy = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    return xy[np.hypot(xy[:, 0], xy[:, 1]) <= r]


def _ring(r0: float, r1: float, n: int = 16) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    return np.concatenate([r * np.stack([np.cos(t), np.sin(t)], axis=1) for r in (r0, r1)])


def _polar_contrast(img, c, r, rho, t, half, gap) -> np.ndarray:
    """Mark at angle t (+/- half) vs. the same radii at t +/- (half + gap)."""
    mark = polar_sample(img, c, r, rho, t + np.linspace(-half, half, 5))
    side = half + gap + np.array([0, half / 2])
    return _contrast(mark, polar_sample(img, c, r, rho, t + np.concatenate([side, -side])))


def decode_rect(img: np.ndarray, kp: Keypoints, spec: RectSpec) -> Decode:
    q = np.asarray(kp.corners, float)
    c = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]  # clockwise in the image
    e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)  # e[k] = |q[k] q[k+1]|
    a, b = e[0] + e[2], e[1] + e[3]  # shift s makes edges (s, s + 2) the long 12-column edges
    all4 = max(a, b) / min(a, b) < SIDE_RATIO_MIN
    shifts = [s for s in range(4) if all4 or (a > b) == (s % 2 == 0)]
    w, h = spec.width, spec.height
    # The A1 corner is chamfered: its mask extreme lies on the chamfer edge, not the corner.
    canon = np.array([[spec.chamfer / 2] * 2, [w, 0], [w, h], [0, h]], float)
    hs = [fit_homography(canon, np.roll(q, -s, axis=0)) for s in shifts]
    mark = np.asarray(spec.dot_xy) + _disk(0.5 * spec.dot_r)
    around = np.asarray(spec.dot_xy) + _ring(1.5 * spec.dot_r, 1.8 * spec.dot_r)
    sets = []
    for H in hs:
        p, s = apply_h(H, mark), apply_h(H, around)
        sets.append(_contrast(bilinear(img, p[:, 0], p[:, 1]), bilinear(img, s[:, 0], s[:, 1])))
    best, margin = _pick(sets)
    H = hs[best]
    p0, p1 = apply_h(H, [[w / 2, h / 2], [w / 2 + 1, h / 2]])
    theta0 = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 360)
    return Decode("rect", H, theta0, margin < MARGIN_MIN, False, margin)


def lattice_phase(polar: np.ndarray, k: int) -> float:
    """Phase (radians, in [0, 2 pi / k)) of a k-fold pattern of dark wells.

    phi_k = arg(sum(-I * exp(i k theta))) / k, summed over the rows of a polar unwrap.
    Rolling the rows by n bins shifts phi_k by exactly n * dtheta (mod 2 pi / k).
    """
    row = np.atleast_2d(polar).sum(axis=0)
    theta = 2 * np.pi * np.arange(row.size) / row.size
    z = np.sum(-row * np.exp(1j * k * theta))
    return float(np.mod(np.angle(z) / k, 2 * np.pi / k))


def decode_circ(img: np.ndarray, kp: Keypoints, spec: CircSpec) -> Decode:
    c, r = np.asarray(kp.center, float), float(kp.radius)
    polar = polar_unwrap(img, c, r, 64, 720)
    rho = (np.arange(64) + 0.5) / 64
    fine = 2 * np.pi / 36
    # only the inner ring has a 6-fold lattice; other radii add only nuisance energy at k=6
    phi6 = lattice_phase(polar[np.abs(rho - spec.radii[0]) < 0.05], 6)
    phi36 = lattice_phase(polar[np.abs(rho - spec.radii[-1]) < 0.05], 36)
    theta = np.mod(phi36 + fine * np.round((phi6 - phi36) / fine), np.pi / 3)  # theta0 mod 60
    # notch: dark rim band at theta + 60 k
    cands = theta + np.pi / 3 * np.arange(6)
    half = 0.6 * spec.notch_w / 2 / (1 - spec.notch_d / 2)
    n_rho = 1 - spec.notch_d * np.array([0.2, 0.4, 0.6])
    gap = np.radians(3.0)
    k, m_notch = _pick([_polar_contrast(img, c, r, n_rho, t, half, gap) for t in cands])
    t_notch = cands[k]
    # chiral dot at notch +/- dot_deg; the side tells clockwise from mirrored
    d_half = 0.6 * spec.dot_r / spec.dot_rho
    d_rho = spec.dot_rho + spec.dot_r * np.array([-0.5, 0, 0.5])
    sides = [t_notch + np.radians(spec.dot_deg), t_notch - np.radians(spec.dot_deg)]
    side, m_dot = _pick([_polar_contrast(img, c, r, d_rho, t, d_half, gap) for t in sides])
    mirrored = m_dot >= MARGIN_MIN and side != (0 if spec.clockwise else 1)
    ambiguous = mirrored or min(m_notch, m_dot) < MARGIN_MIN
    s = r / spec.radius
    cs, sn = np.cos(t_notch), np.sin(t_notch)
    H = np.array([[s * cs, -s * sn, c[0]], [s * sn, s * cs, c[1]], [0, 0, 1.0]])
    theta0 = float(np.degrees(t_notch) % 360)
    return Decode("circ", H, theta0, bool(ambiguous), bool(mirrored), min(m_notch, m_dot))


def decode(img: np.ndarray, kp: Keypoints, spec: Spec) -> Decode:
    if isinstance(spec, RectSpec):
        return decode_rect(img, kp, spec)
    return decode_circ(img, kp, spec)


def locate_well(dec: Decode, i: int, spec: Spec, *, order: IndexOrder) -> np.ndarray:
    """Pixel centre of well ``i``."""
    return apply_h(dec.H, well_xy(spec, i, order=order))


def pixel_to_well(dec: Decode, xy, spec: Spec, *, order: IndexOrder) -> int | None:
    """Well index at pixel ``xy``, or None if farther than 0.5 pitch from any well."""
    u, v = apply_h(np.linalg.inv(dec.H), np.asarray(xy, float))
    if isinstance(spec, RectSpec):
        a = int(np.rint((v - spec.y0) / spec.pitch))
        b = int(np.rint((u - spec.x0) / spec.pitch))
        if not (0 <= a < spec.rows and 0 <= b < spec.cols):
            return None
    else:
        a = int(np.argmin(np.abs(np.hypot(u, v) / spec.radius - np.asarray(spec.radii))))
        sgn = 1.0 if spec.clockwise else -1.0
        n = spec.slots[a]
        b = int(np.rint(np.mod(np.arctan2(sgn * u, -v), 2 * np.pi) / (2 * np.pi / n))) % n
    if np.hypot(*(np.array([u, v]) - node_xy(spec, a, b))) > 0.5 * spec.pitch:
        return None
    return grid_index(spec, a, b, order=order)
