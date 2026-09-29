"""Ouroboros: rotation-equivariant OCSR with stereochemistry -> 3D conformer -> MACE energy.

Symmetry per stage (see README):
  image encoder : C_N (discrete rotations) / SE(2), NO reflections
  decoder       : consumes a rotation-invariant token set (no symmetry of its own)
  3D stage      : E(3) via MACE (the only E(3)-equivariant component)
"""

__version__ = "0.0.1"


def _sanitize_mpl_backend() -> None:
    """Jupyter/Colab kernels export MPLBACKEND=module://matplotlib_inline...; subprocesses running
    in another environment (our Colab venv) inherit it and matplotlib then refuses to import.
    Fall back to the headless Agg backend when the requested backend module is not importable."""
    import importlib.util
    import os

    backend = os.environ.get("MPLBACKEND", "")
    if backend.startswith("module://"):
        module = backend[len("module://") :]
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            os.environ["MPLBACKEND"] = "Agg"


_sanitize_mpl_backend()
