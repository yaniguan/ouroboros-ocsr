"""Verify pure-PyTorch inference dependencies, and optionally restore macOS lie-learn tables.

Pure-PyTorch inference (requirements-inference.txt) needs no Fortran compiler, escnn, or lie-learn.
If escnn/lie-learn are installed (e.g. via requirements-escnn.txt on macOS for training), this
script also checks and restores any omitted upstream SO(3) tables using pinned SHA256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import urllib.request
from pathlib import Path

TABLES = {
    "J_dense_0-150.npy": "8ead49e1e62054dab3a826fd281467782bc7b71785f78a429ceb3a76e02380f5",
    "J_block_0-150.npy": "48f7ac5041e70ca0bf80d0623b538cc978bac0fb49c82ab6c6f2ad33348d9e68",
}
BASE = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/lie_learn/"
    "edf012f5f60af320175d2e6269db78b984b5bfc3/lie_learn/representations/SO3/pinchon_hoggan/"
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--require-escnn",
        action="store_true",
        help="fail if escnn/lie-learn training dependencies are not installed",
    )
    args = ap.parse_args(argv)

    import rdkit  # noqa: F401
    import torch  # noqa: F401

    spec = importlib.util.find_spec("lie_learn")
    escnn_spec = importlib.util.find_spec("escnn")
    if escnn_spec is None:
        if args.require_escnn:
            raise RuntimeError("escnn is not installed; install requirements-escnn.txt first")
        print("Pure-PyTorch inference dependencies ready (zero escnn / zero gfortran needed)")
        return 0

    if spec is not None and spec.origin is not None and sys.platform == "darwin":
        directory = Path(spec.origin).parent / "representations/SO3/pinchon_hoggan"
        for name, expected in TABLES.items():
            destination = directory / name
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Unexpected existing table: {destination}")
                continue
            with urllib.request.urlopen(BASE + name, timeout=120) as response:
                data = response.read(40_000_001)
            if len(data) > 40_000_000 or hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Upstream table checksum mismatch: {name}")
            destination.write_bytes(data)
            print(f"Installed verified upstream table: {name}")

    from ouroboros.encoder.steerable import _ensure_escnn_2d_shim

    _ensure_escnn_2d_shim()
    from escnn import gspaces

    gspaces.rot2dOnR2(N=4)
    print("Inference + escnn training dependencies ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
