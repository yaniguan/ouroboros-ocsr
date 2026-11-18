"""Plate specs, well index conventions and well labels.

Indices are 1-based. The index order is always explicit: there is no default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np


class IndexOrder(Enum):
    ROW_MAJOR = "row_major"  # rect: 1 = A1, 2 = A2, ... (65 -> F5)
    COL_MAJOR = "col_major"  # rect, Opentrons order: 1 = A1, 2 = B1, ... (65 -> A9)
    RING_MAJOR = "ring_major"  # circ: inner ring first, slots clockwise (65 -> R4S04)


@dataclass(frozen=True)
class RectSpec:
    """SBS 96-well plate in mm. Origin at the A1 corner, x along columns, y along rows."""

    rows: int = 8
    cols: int = 12
    width: float = 127.76
    height: float = 85.48
    x0: float = 14.38  # A1 centre
    y0: float = 11.24
    pitch: float = 9.0
    well_r: float = 3.3
    chamfer: float = 2.5  # cut corner at A1 (u + v < chamfer)
    dot_xy: tuple[float, float] = (6.5, 5.8)  # dark dot near A1
    dot_r: float = 2.2
    kind: str = "rect"


@dataclass(frozen=True)
class CircSpec:
    """Circular 96-well plate, centre at the origin, plate radius 1, notch at the top (-y)."""

    slots: tuple[int, ...] = (6, 12, 18, 24, 36)
    radii: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85)
    radius: float = 1.0
    well_r: float = 0.05
    clockwise: bool = True  # slots run clockwise from the notch, as seen in the image
    notch_w: float = 0.08
    notch_d: float = 0.07
    dot_deg: float = 20.0  # chiral dot at notch + 20 deg in the slot direction
    dot_rho: float = 0.94
    dot_r: float = 0.03
    kind: str = "circ"

    @property
    def pitch(self) -> float:
        return 2 * np.pi * self.radii[-1] * self.radius / self.slots[-1]


Spec = RectSpec | CircSpec
N_WELLS = 96


def _check(spec: Spec, order: IndexOrder) -> None:
    if not isinstance(order, IndexOrder):
        raise TypeError(f"order must be an IndexOrder, got {order!r}")
    ok = (IndexOrder.RING_MAJOR,)
    if isinstance(spec, RectSpec):
        ok = (IndexOrder.ROW_MAJOR, IndexOrder.COL_MAJOR)
    if order not in ok:
        raise ValueError(f"{order} is not valid for a {spec.kind} plate")


def grid_pos(spec: Spec, i: int, *, order: IndexOrder) -> tuple[int, int]:
    """0-based (row, col) for rect or (ring, slot) for circ."""
    _check(spec, order)
    if not 1 <= i <= N_WELLS:
        raise ValueError(f"well index {i} out of range 1..{N_WELLS}")
    if order is IndexOrder.ROW_MAJOR:
        return divmod(i - 1, spec.cols)
    if order is IndexOrder.COL_MAJOR:
        col, row = divmod(i - 1, spec.rows)
        return row, col
    rest = i - 1
    for ring, n in enumerate(spec.slots):
        if rest < n:
            return ring, rest
        rest -= n
    raise ValueError("spec slots do not add up to 96")


def grid_index(spec: Spec, a: int, b: int, *, order: IndexOrder) -> int:
    """Inverse of grid_pos."""
    _check(spec, order)
    if order is IndexOrder.ROW_MAJOR:
        return a * spec.cols + b + 1
    if order is IndexOrder.COL_MAJOR:
        return b * spec.rows + a + 1
    return sum(spec.slots[:a]) + b + 1


def label(spec: Spec, i: int, *, order: IndexOrder) -> str:
    a, b = grid_pos(spec, i, order=order)
    if isinstance(spec, RectSpec):
        return f"{'ABCDEFGH'[a]}{b + 1}"
    return f"R{a}S{b:02d}"


def index(spec: Spec, text: str, *, order: IndexOrder) -> int:
    _check(spec, order)
    pat = r"([A-H])(\d{1,2})" if isinstance(spec, RectSpec) else r"R(\d)S(\d{2})"
    m = re.fullmatch(pat, text)
    if m is None:
        raise ValueError(f"bad well label {text!r}")
    if isinstance(spec, RectSpec):
        a, b = "ABCDEFGH".index(m.group(1)), int(m.group(2)) - 1
        ok = 0 <= b < spec.cols
    else:
        a, b = int(m.group(1)), int(m.group(2))
        ok = a < len(spec.slots) and b < spec.slots[a]
    if not ok:
        raise ValueError(f"bad well label {text!r}")
    return grid_index(spec, a, b, order=order)


def node_xy(spec: Spec, a: int, b: int) -> np.ndarray:
    """Canonical centre of (row, col) or (ring, slot)."""
    if isinstance(spec, RectSpec):
        return np.array([spec.x0 + b * spec.pitch, spec.y0 + a * spec.pitch])
    ang = 2 * np.pi * b / spec.slots[a]
    sgn = 1.0 if spec.clockwise else -1.0
    rho = spec.radii[a] * spec.radius
    return np.array([sgn * rho * np.sin(ang), -rho * np.cos(ang)])


def well_xy(spec: Spec, i: int, *, order: IndexOrder) -> np.ndarray:
    return node_xy(spec, *grid_pos(spec, i, order=order))


def all_wells(spec: Spec, *, order: IndexOrder) -> np.ndarray:
    """(96, 2) canonical centres; row k is well k + 1."""
    return np.array([well_xy(spec, i, order=order) for i in range(1, N_WELLS + 1)])


def default_order(spec: Spec) -> IndexOrder:
    """The order used by the renderer and the benchmark for this plate kind."""
    return IndexOrder.ROW_MAJOR if isinstance(spec, RectSpec) else IndexOrder.RING_MAJOR
