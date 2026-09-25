"""Ouroboros: rotation-equivariant OCSR with stereochemistry -> 3D conformer -> MACE energy.

Symmetry per stage (see README):
  image encoder : C_N (discrete rotations) / SE(2), NO reflections
  decoder       : consumes a rotation-invariant token set (no symmetry of its own)
  3D stage      : E(3) via MACE (the only E(3)-equivariant component)
"""

__version__ = "0.0.1"
