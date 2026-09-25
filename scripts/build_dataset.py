"""Build a dataset: pool -> compose -> render shards.

Examples
--------
Local 50k (train prefixes give the 10k set too):
    python scripts/build_dataset.py --out data/ds --train-size 50000
1k dry run of the 1M recipe:
    python scripts/build_dataset.py --out /tmp/dry --preset 1m --dry-run 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ouroboros.data import build
from ouroboros.data.sources import iter_source

PRESETS = {  # train sizes; val/test are shared by every size so results are comparable
    "10k": 10_000,
    "50k": 50_000,
    "200k": 200_000,
    "1m": 1_000_000,
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, help="output root (pool, manifests/, shards/)")
    ap.add_argument("--cache", default="data_cache", help="where raw source files are downloaded")
    ap.add_argument("--sources", nargs="+", default=["zinc250k", "moses"])
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--train-size", type=int, default=50_000)
    ap.add_argument("--val-size", type=int, default=5_000)
    ap.add_argument("--test-size", type=int, default=10_000)
    ap.add_argument("--stereo-fraction", type=float, default=0.40)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--pool-limit", type=int, default=None, help="max input SMILES for the pool")
    ap.add_argument(
        "--dry-run",
        type=int,
        default=None,
        metavar="N",
        help="render only the first N train samples (val/test N//10) of the recipe",
    )
    ap.add_argument("--stages", nargs="+", default=["pool", "compose", "render"])
    args = ap.parse_args(argv)

    out = Path(args.out)
    train_size = PRESETS[args.preset] if args.preset else args.train_size
    sizes = {"train": train_size, "val": args.val_size, "test": args.test_size}
    pool_path = out / "pool.tsv.gz"
    t_all = time.time()

    if "pool" in args.stages and not pool_path.exists():
        srcs = [(name, iter_source(name, args.cache)) for name in args.sources]
        print("pool:", build.prepare_pool(srcs, pool_path, args.workers, args.pool_limit))

    if "compose" in args.stages:
        cfg = build.ComposeConfig(stereo_fraction=args.stereo_fraction, sizes=sizes, seed=args.seed)
        print("compose:", build.compose(build.read_pool(pool_path), cfg, out / "manifests"))

    if "render" in args.stages:
        for split in build.SPLITS:
            rows = build.read_manifest(out / "manifests" / f"{split}.tsv")
            if args.dry_run is not None:
                rows = rows[: args.dry_run if split == "train" else max(1, args.dry_run // 10)]
            st = build.render_shards(
                rows,
                split,
                out / "shards",
                shard_size=args.shard_size,
                size=args.image_size,
                seed=args.seed,
                workers=args.workers,
            )
            print("render:", json.dumps(st))
    print(f"total wall time {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
