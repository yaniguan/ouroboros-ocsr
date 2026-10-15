"""Restore upstream lie-learn tables omitted from its macOS source-built wheel.

Run after requirements-inference.txt, using the same virtualenv interpreter.
Only missing files are installed; downloads use pinned upstream content and SHA256.
"""

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


def main():
    spec = importlib.util.find_spec("lie_learn")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install requirements-inference.txt first")
    if sys.platform == "darwin":
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
    from escnn import gspaces

    gspaces.rot2dOnR2(N=4)
    print("Inference dependencies ready")


if __name__ == "__main__":
    main()
