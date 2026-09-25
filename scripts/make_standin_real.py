"""Build a STAND-IN for a real-document dataset from synthetic val images (pipeline testing only).

Images are cropped to their ink bounding box with a random margin (non-square, varying size, like
cropped figure regions), written as PNG/JPEG files plus a manifest in the real-data format, and
ingested with the real-data adapter. Nothing here is real data; results on it mean nothing.

    python scripts/make_standin_real.py --src data/full/shards --out data/standin_real
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ouroboros.data.loader import ShardDataset
from ouroboros.data.real import ingest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ds = ShardDataset(a.src, "val")
    rng = random.Random(0)
    rows = []
    for i in range(a.n_train + a.n_test):
        ink = ds[i]["image"][0].numpy()
        ys, xs = np.nonzero(ink > 0.3)
        m = rng.randint(4, 30)
        y0, y1 = max(0, ys.min() - m), min(ink.shape[0], ys.max() + m)
        x0, x1 = max(0, xs.min() - m), min(ink.shape[1], xs.max() + m)
        img = Image.fromarray(((1 - ink[y0:y1, x0:x1]) * 255).astype(np.uint8), "L")
        scale = rng.uniform(0.6, 1.6)
        img = img.resize((max(8, int(img.width * scale)), max(8, int(img.height * scale))))
        ext = rng.choice(["png", "jpg"])
        name = f"images/standin_{i:05d}.{ext}"
        img.save(out / name)
        split = "train" if i < a.n_train else "test"
        rows.append([name, ds.meta(i)["smiles"], "STANDIN", split, f"val{i}"])
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "smiles", "source", "split", "id"])
        w.writerows(rows)
    print(ingest(out / "manifest.csv", out / "shards", size=384))


if __name__ == "__main__":
    main()
