"""Paired diagnostic set: invariance to drawing, sensitivity to chemistry.

    python scripts/make_diagnostics.py --labels shards/eval:test --out diag --n 500

For each base molecule M (labels come from a real-document test split, drawn synthetically):
  identity-PRESERVING pairs (the answer must not change)
    redraw   M drawn twice with independent random styles (font, line width, layout, noise)
    rot45    the same drawing rotated by 45 degrees (off-grid)
    rot90    the same drawing rotated by 90 degrees (pixel-exact for C4 / C8)
  identity-CHANGING pairs (the answer must change, exactly as the edit says)
    stereo_flip  invert one tetrahedral stereocentre (enantiomer or diastereomer)
    bond_order   raise one acyclic C-C single bond to a double bond
    atom_swap    replace one acyclic CH2 carbon by O or N
Edited molecules are sanitized and checked with RDKit: the standardized InChIKey must differ from
the original (stereo_flip must keep the connectivity block), otherwise the edit is discarded.
Truth for every image is the standardized label derived from the *drawing* (wedge/hash bonds), the
same convention as the training data. Output: ``manifest.jsonl`` + ``images/*.png``. Not for git.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem, RDLogger

from ouroboros.data.render import render, sample_style
from ouroboros.eval.evaluate import rotate_images
from ouroboros.eval.standardize import standard_inchikey, standardize

RDLogger.DisableLog("rdApp.*")
EDITS = ("stereo_flip", "bond_order", "atom_swap")


def _valid(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        Chem.SanitizeMol(mol)
        return mol if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None else None
    except Exception:  # noqa: BLE001 - any RDKit failure means "discard this edit"
        return None


def stereo_flip(mol: Chem.Mol, rng: random.Random) -> Chem.Mol | None:
    centers = [
        i
        for i, _ in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False
        )
    ]
    if not centers:
        return None
    edited = Chem.RWMol(mol)
    atom = edited.GetAtomWithIdx(rng.choice(centers))
    atom.InvertChirality()
    return _valid(edited)


def bond_order(mol: Chem.Mol, rng: random.Random) -> Chem.Mol | None:
    bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and not b.IsInRing()
        and not b.GetIsAromatic()
        and b.GetBeginAtom().GetSymbol() == "C"
        and b.GetEndAtom().GetSymbol() == "C"
        and b.GetBeginAtom().GetTotalNumHs() >= 1
        and b.GetEndAtom().GetTotalNumHs() >= 1
    ]
    if not bonds:
        return None
    edited = Chem.RWMol(mol)
    edited.GetBondWithIdx(rng.choice(bonds)).SetBondType(Chem.BondType.DOUBLE)
    return _valid(edited)


def atom_swap(mol: Chem.Mol, rng: random.Random) -> Chem.Mol | None:
    atoms = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "C"
        and not a.IsInRing()
        and not a.GetIsAromatic()
        and a.GetDegree() == 2
        and a.GetTotalNumHs() == 2
        and a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if not atoms:
        return None
    edited = Chem.RWMol(mol)
    edited.GetAtomWithIdx(rng.choice(atoms)).SetAtomicNum(rng.choice([7, 8]))
    return _valid(edited)


EDIT_FUNCS = {"stereo_flip": stereo_flip, "bond_order": bond_order, "atom_swap": atom_swap}


def _truth(rendered) -> str | None:
    std = standardize(rendered.label)
    return std.smiles if std.valid else None


def _save(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build(labels: list[str], out: Path, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    rows, counts, skipped = [], {k: 0 for k in ("redraw", "rot45", "rot90", *EDITS)}, 0
    pair_id = 0
    for smiles in labels:
        if pair_id >= n:
            break
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        style, seed_px = sample_style(rng), rng.randrange(2**31)
        base = render(mol, style, seed=seed_px)
        base_truth = _truth(base)
        if base_truth is None:
            skipped += 1
            continue
        pid = f"p{pair_id:05d}"
        stem = out / "images" / pid
        _save(base.image, Path(f"{stem}_base.png"))
        rows.append({"pair": pid, "role": "base", "kind": "base", "truth": base_truth})
        # identity-preserving views
        redraw = render(mol, sample_style(rng), seed=rng.randrange(2**31))
        views = {"redraw": (redraw.image, _truth(redraw))}
        tensor = torch.from_numpy(np.asarray(base.image, dtype=np.float32))[None, None] / 255.0
        for name, angle in (("rot45", 45.0), ("rot90", 90.0)):
            rot = rotate_images(1.0 - tensor, torch.tensor([angle]))
            img = Image.fromarray((255 * (1.0 - rot[0, 0]).clamp(0, 1)).byte().numpy(), "L")
            views[name] = (img, base_truth)
        for kind, (img, truth) in views.items():
            if truth is None:
                continue
            _save(img, Path(f"{stem}_{kind}.png"))
            rows.append({"pair": pid, "role": "view", "kind": kind, "truth": truth})
            counts[kind] += 1
        # identity-changing edits
        key0 = standard_inchikey(standardize(base_truth))
        for kind in EDITS:
            edited = EDIT_FUNCS[kind](Chem.Mol(mol), rng)
            if edited is None:
                continue
            drawn = render(edited, style, seed=seed_px)
            truth = _truth(drawn)
            key1 = standard_inchikey(standardize(truth)) if truth else None
            if truth is None or key1 is None or key1 == key0:
                continue
            if kind == "stereo_flip" and key1[:14] != key0[:14]:
                continue
            _save(drawn.image, Path(f"{stem}_{kind}.png"))
            rows.append({"pair": pid, "role": "edit", "kind": kind, "truth": truth})
            counts[kind] += 1
        pair_id += 1
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            r["image"] = f"images/{r['pair']}_{r['kind']}.png"
            f.write(json.dumps(r) + "\n")
    stats = {
        "pairs": pair_id,
        "skipped_unlabelled": skipped,
        "images": len(rows),
        "per_kind": counts,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def read_labels(spec: str) -> list[str]:
    from ouroboros.data.loader import ShardDataset

    root, split = spec.rsplit(":", 1)
    ds = ShardDataset(root, split)
    seen, out = set(), []
    for i in range(len(ds)):
        s = ds.meta(i)["smiles"]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--labels", required=True, help="SHARD_DIR:SPLIT supplying base molecules")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    labels = read_labels(a.labels)
    random.Random(a.seed).shuffle(labels)
    # Alternate stereo-bearing and other molecules so the stereo-flip subset is not starved.
    stereo = [x for x in labels if "@" in x]
    other = [x for x in labels if "@" not in x]
    labels = [x for pair in zip(stereo, other, strict=False) for x in pair]
    labels += stereo[len(other) :] + other[len(stereo) :]
    a.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(labels, a.out, a.n, a.seed), indent=1))


if __name__ == "__main__":
    main()
