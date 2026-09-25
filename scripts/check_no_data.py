"""Pre-commit guard: refuse to commit datasets, shards, real images or other large binaries.

Installed as .git/hooks/pre-commit by ``scripts/install_git_hooks.sh`` (or via
``.pre-commit-config.yaml``). Checks the staged files; exits 1 on any violation.
Rules (see DATA.md):
  * nothing under data/, data_cache/, real_data/, real/, shards/, runs/, checkpoints/
  * no archives / shard indices / pools: .tar .tar.gz .tgz .zip .idx.json .tsv.gz .npz .ckpt .pt
  * raster images only under benchmarks/ (our own plots), and never with "real" in the path
  * no file larger than 5 MB
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath

BLOCKED_DIRS = {"data", "data_cache", "real_data", "real", "shards", "runs", "checkpoints"}
BLOCKED_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".idx.json",
    ".tsv.gz",
    ".npz",
    ".ckpt",
    ".pt",
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp")
MAX_BYTES = 5 * 1024 * 1024


def violations(files: list[tuple[str, int]]) -> list[str]:
    """``files`` = [(repo-relative posix path, size in bytes)]; returns human-readable problems."""
    out = []
    for path, size in files:
        p = PurePosixPath(path)
        low = path.lower()
        if p.parts and p.parts[0] in BLOCKED_DIRS:
            out.append(f"{path}: inside data directory '{p.parts[0]}/'")
        elif low.endswith(BLOCKED_SUFFIXES):
            out.append(f"{path}: dataset/archive/checkpoint file type")
        elif low.endswith(IMAGE_SUFFIXES):
            if p.parts[0] != "benchmarks":
                out.append(f"{path}: image outside benchmarks/")
            elif "real" in low:
                out.append(f"{path}: image path mentions real data")
        if size > MAX_BYTES:
            out.append(f"{path}: {size / 1e6:.1f} MB > 5 MB")
    return out


def staged_files() -> list[tuple[str, int]]:
    names = (
        subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .split("\0")
    )
    out = []
    for n in filter(None, names):
        size = subprocess.run(
            ["git", "cat-file", "-s", f":{n}"], capture_output=True, check=True, text=True
        ).stdout.strip()
        out.append((n, int(size)))
    return out


def main() -> int:
    problems = violations(staged_files())
    for p in problems:
        print(f"check_no_data: {p}", file=sys.stderr)
    if problems:
        print("Commit refused: data must stay out of git (see DATA.md).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
