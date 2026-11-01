"""Download a small, seeded subset of public real OCSR benchmarks -> image files + manifest.

Source: Hugging Face dataset ``Keylab/OCSR-Benchmarks`` (one parquet per benchmark, ~5-20 MB).
Only the parquet for each requested benchmark is fetched; nothing here may be committed (DATA.md).

    python scripts/fetch_real_subset.py --sets USPTO UOB ACS --n 100 --out real_data/subset100
    python scripts/ingest_real.py --manifest real_data/subset100/manifest.csv --out real/subset100

Output: ``<out>/images/<SET>/<id>.png`` and ``<out>/manifest.csv`` with columns
``image,smiles,source,split,id`` (the ``ouroboros.data.real`` manifest interface).
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
from pathlib import Path

REPO = "Keylab/OCSR-Benchmarks"
URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{set}/test-00000-of-00001.parquet"


def fetch_parquet(name: str, cache: Path, revision: str = "main") -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.parquet"
    if not path.exists():
        url = URL.format(repo=REPO, rev=revision, set=name)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def sample_rows(n_total: int, n: int, seed: int) -> list[int]:
    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    return sorted(idx[: min(n, n_total)])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["USPTO", "UOB", "ACS"])
    ap.add_argument("--n", type=int, default=100, help="images per benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default="main")
    a = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from PIL import Image

    out = Path(a.out)
    rows = []
    for name in a.sets:
        table = pq.read_table(fetch_parquet(name, out / "parquet", a.revision))
        cols = table.column_names
        img_col, smi_col = "image", "SMILES" if "SMILES" in cols else "smiles"
        id_col = "image_id" if "image_id" in cols else None
        d = out / "images" / name
        d.mkdir(parents=True, exist_ok=True)
        for i in sample_rows(table.num_rows, a.n, a.seed):
            rec = table.slice(i, 1).to_pylist()[0]
            rid = str(rec[id_col]) if id_col else f"{i:06d}"
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in rid)
            p = d / f"{safe}.png"
            Image.open(io.BytesIO(rec[img_col]["bytes"])).convert("RGB").save(p)
            rows.append({"image": str(p.relative_to(out)), "smiles": rec[smi_col],
                         "source": name, "split": "test", "id": rid})
        print(f"{name}: {table.num_rows} rows, sampled {sum(r['source'] == name for r in rows)}")
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "smiles", "source", "split", "id"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
