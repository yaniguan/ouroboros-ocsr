"""Build ChemVeritas-Bench (smoke / medium100 / gold600) locally on disk.

Generates 4 visual regimes per molecule (V0_clean, V1_rotated_offgrid, V2_crowded_si_scan,
V3_condensed_layout), paired ACS-style Supporting Information (HRMS + 13C/1H NMR), 5 typed
hard negatives, tasks T1-T5 JSONL files, and a SHA256 manifest receipt.

Example:
    python scripts/build_phys_bench.py --out results/chemveritas_smoke --preset smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ouroboros.bench.builder import build_chemveritas_benchmark


def _load_exclude_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("{"):
            obj = json.loads(s)
            k = obj.get("inchikey14") or obj.get("connectivity_key") or str(obj.get("inchikey", ""))
            if k:
                keys.add(k[:14])
        else:
            keys.add(s[:14])
    return keys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Output directory for benchmark files")
    p.add_argument(
        "--preset",
        choices=("smoke", "medium100", "gold600"),
        default="smoke",
        help="Catalog preset size (15, 100, or 600 scaffolds)",
    )
    p.add_argument("--image-size", type=int, default=384, help="Rendered square PNG edge in px")
    p.add_argument("--n-conf", type=int, default=4, help="3D conformers per stereoisomer")
    p.add_argument("--seed", type=int, default=2026, help="Deterministic RNG seed")
    p.add_argument("--no-render", action="store_true", help="Skip PNG rendering (metadata only)")
    p.add_argument(
        "--exclude-keys",
        type=Path,
        default=None,
        help="Optional text/JSONL file of 14-char connectivity InChIKeys to exclude",
    )
    args = p.parse_args(argv)

    manifest = build_chemveritas_benchmark(
        args.out,
        preset=args.preset,
        image_size=args.image_size,
        n_conf=args.n_conf,
        seed=args.seed,
        render_images=not args.no_render,
        exclude_key14=_load_exclude_keys(args.exclude_keys),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
