"""Phase 1 verification: splits, label integrity, mirror test, contact sheet, throughput.

Each check writes ``benchmarks/data/<check>.json`` (and the contact sheet PNG).

    python scripts/verify_data.py splits   --root data/ds
    python scripts/verify_data.py labels   --root data/ds --n 10000
    python scripts/verify_data.py mirror   --root data/ds --n 200
    python scripts/verify_data.py sheet    --root data/ds
    python scripts/verify_data.py render-speed --root data/ds --n 300
    python scripts/verify_data.py loader-speed --root data/ds --workers 8
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

from rdkit import Chem

from ouroboros.data import build
from ouroboros.data.render import render, sample_style

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "data"


def _save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = {
        **obj,
        "date": time.strftime("%Y-%m-%d"),
        "host": platform.node(),
        "cpus": __import__("os").cpu_count(),
    }
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(json.dumps(obj, indent=1))


def _manifest(root: Path, split: str) -> list[dict]:
    return build.read_manifest(root / "manifests" / f"{split}.tsv")


def cmd_splits(a):
    """Zero InChIKey overlap between splits: full keys of the labels AND connectivity blocks."""
    full, conn = {}, {}
    for split in build.SPLITS:
        rows = _manifest(a.root, split)
        full[split] = {Chem.MolToInchiKey(Chem.MolFromSmiles(r["smiles"])) for r in rows}
        conn[split] = {r["key14"] for r in rows}
    res = {"sizes": {s: len(full[s]) for s in full}}
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        res[f"overlap_full_{x}_{y}"] = len(full[x] & full[y])
        res[f"overlap_connectivity_{x}_{y}"] = len(conn[x] & conn[y])
    res["pass"] = all(v == 0 for k, v in res.items() if k.startswith("overlap"))
    _save("splits", res)


def cmd_labels(a):
    """Every stored label parses, and equals the canonical SMILES of its manifest source."""
    from ouroboros.data.loader import ShardDataset

    n_ok = n_parse_fail = n_mismatch = n = 0
    per_split = {}
    for split in build.SPLITS:
        src = {
            build.sample_key(split, int(r["idx"])): r["smiles"] for r in _manifest(a.root, split)
        }
        ds = ShardDataset(a.root / "shards", split)
        idx = list(range(len(ds)))
        random.Random(0).shuffle(idx)
        take = idx[: a.n if split == "train" else a.n // 10]
        for i in take:
            shard, s = ds.entries[i]
            meta = ds.meta(i)
            m = Chem.MolFromSmiles(meta["smiles"])
            n += 1
            if m is None:
                n_parse_fail += 1
                continue
            if Chem.MolToSmiles(m) != Chem.CanonSmiles(src[s["key"]]):
                n_mismatch += 1
                continue
            n_ok += 1
        per_split[split] = len(take)
    _save(
        "labels",
        {
            "checked": n,
            "per_split": per_split,
            "parse_fail": n_parse_fail,
            "mismatch": n_mismatch,
            "ok": n_ok,
            "pass": n_ok == n,
        },
    )


def _enantiomer(smi: str) -> str:
    inv = smi.replace("@@", "\0").replace("@", "@@").replace("\0", "@")
    return Chem.CanonSmiles(inv)


def cmd_mirror(a):
    """Mirrored drawings of chiral molecules are labeled as the enantiomer."""
    rows = [r for r in _manifest(a.root, "test") if "@" in r["smiles"]]
    rows = [
        r for r in rows if _enantiomer(r["smiles"]) != Chem.CanonSmiles(r["smiles"])
    ]  # not meso
    rows = rows[: a.n]
    ok = 0
    fails = []
    for i, r in enumerate(rows):
        style = sample_style(random.Random(i))
        mol = Chem.MolFromSmiles(r["smiles"])
        lab = render(mol, style, mirror=True, seed=i).label
        if lab == _enantiomer(r["smiles"]):
            ok += 1
        else:
            fails.append({"smiles": r["smiles"], "mirror_label": lab})
    _save("mirror", {"n": len(rows), "ok": ok, "fails": fails[:20], "pass": ok == len(rows) == a.n})


def cmd_sheet(a):
    """8x8 contact sheet of random training samples (as stored in the shards)."""
    from PIL import Image, ImageDraw

    from ouroboros.data.loader import ShardDataset

    ds = ShardDataset(a.root / "shards", "train")
    idx = random.Random(a.seed).sample(range(len(ds)), 64)
    tile = 192
    sheet = Image.new("L", (8 * tile, 8 * tile), 255)
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        img = (1 - ds[i]["image"][0]).mul(255).byte().numpy()
        im = Image.fromarray(img, mode="L").resize((tile, tile), Image.BILINEAR)
        sheet.paste(im, ((k % 8) * tile, (k // 8) * tile))
        draw.rectangle(
            [(k % 8) * tile, (k // 8) * tile, (k % 8 + 1) * tile - 1, (k // 8 + 1) * tile - 1],
            outline=160,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "contact_sheet.png")
    styles = [ds.meta(i)["style"] for i in idx]
    summary = {
        "fonts": sorted({s["font"] or "rdkit-builtin" for s in styles}),
        "palettes": sorted({s["palette"] for s in styles}),
        "line_width_range": [
            min(s["bond_line_width"] for s in styles),
            max(s["bond_line_width"] for s in styles),
        ],
        "bond_length_range": [
            min(s["bond_length"] for s in styles),
            max(s["bond_length"] for s in styles),
        ],
        "n_blur": sum(s["blur"] > 0 for s in styles),
        "n_noise": sum(s["noise_std"] > 0 or s["salt_pepper"] > 0 for s in styles),
        "n_jpeg": sum(s["jpeg_quality"] > 0 for s in styles),
        "n_explicit_methyl": sum(s["explicit_methyl"] for s in styles),
        "n_comic": sum(s["comic"] for s in styles),
    }
    _save("contact_sheet", {"path": "benchmarks/data/contact_sheet.png", "styles": summary})


def cmd_render_speed(a):
    """Single-process throughput of style sampling + render + PNG encode at 384 px."""
    from ouroboros.data.render import encode_png

    rows = _manifest(a.root, "train")[: a.n]
    mols = [Chem.MolFromSmiles(r["smiles"]) for r in rows]
    rng = random.Random(0)
    render(mols[0], sample_style(rng), size=384)  # warm-up (font loading)
    t0 = time.perf_counter()
    for i, m in enumerate(mols):
        encode_png(render(m, sample_style(rng), size=384, seed=i).image)
    dt = time.perf_counter() - t0
    _save(
        "render_speed",
        {"n": len(mols), "size": 384, "img_per_s": len(mols) / dt, "pass": len(mols) / dt >= 20},
    )


def cmd_loader_speed(a):
    """DataLoader throughput from local shards (decode + tensor + collate), shuffled order."""
    import torch

    from ouroboros.data.loader import ResumableSampler, ShardDataset, collate

    ds = ShardDataset(a.root / "shards", "train")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=64,
        sampler=ResumableSampler(len(ds), seed=0),
        num_workers=a.workers,
        collate_fn=collate,
        persistent_workers=False,
        prefetch_factor=4,
    )
    it = iter(dl)
    for _ in range(a.workers * 2):  # warm-up: spawn workers, fill prefetch queues
        next(it)
    t0 = time.perf_counter()
    n = 0
    while n < a.n:
        n += next(it)["image"].shape[0]
    dt = time.perf_counter() - t0
    _save(
        "loader_speed",
        {
            "workers": a.workers,
            "n": n,
            "img_per_s": n / dt,
            "dataset_len": len(ds),
            "pass": n / dt >= 500,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd", choices=["splits", "labels", "mirror", "sheet", "render-speed", "loader-speed"]
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    defaults = {"labels": 10_000, "mirror": 200, "render-speed": 300, "loader-speed": 20_000}
    if a.n is None:
        a.n = defaults.get(a.cmd, 0)
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
