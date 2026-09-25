"""Ingest a real-document manifest (CSV/JSONL: image, smiles[, source, split, id]) into shards.

    python scripts/ingest_real.py --manifest /path/uspto/manifest.csv --out real/uspto

Writes ``{split}-*.tar`` shards, ``ingest_stats.json`` and ``ingest_failures.jsonl`` (every row
that could not be ingested, with its reason). Output directories must stay out of git (DATA.md).
"""

from __future__ import annotations

import argparse
import json

from ouroboros.data.real import ingest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--shard-size", type=int, default=1000)
    a = ap.parse_args(argv)
    stats = ingest(a.manifest, a.out, size=a.image_size, shard_size=a.shard_size)
    print(json.dumps(stats, indent=1))
    if stats["n_failures"]:
        print(f"{stats['n_failures']} rows failed; see {a.out}/ingest_failures.jsonl")


if __name__ == "__main__":
    main()
