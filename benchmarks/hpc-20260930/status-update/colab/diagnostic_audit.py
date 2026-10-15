"""Audit every transferred PNG/label pair and freeze the 200k/1k train/val selections."""

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

from PIL import Image
from rdkit import Chem

from ouroboros.data.loader import ShardDataset
from ouroboros.provenance import atomic_json, file_sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    root = a.root / "data/full"
    if not (a.root / "reports/data-transfer.ok").is_file():
        raise RuntimeError("Transfer is not committed")
    receipt = json.loads((a.root / "reports/data-transfer-receipts.json").read_text())
    for item in receipt["files"]:
        p = root / item["path"]
        if p.stat().st_size != item["bytes"] or file_sha256(p) != item["sha256"]:
            raise ValueError(f"Transfer checksum mismatch: {p}")
    report = {"scope": "201 train shards + all val/test; not a full 1M audit", "splits": {}, "stereo_mismatches": []}
    keysets = {}
    for split, expected_shards in (("train", 201), ("val", 5), ("test", 10)):
        with (root / "manifests" / f"{split}.tsv").open() as stream:
            source = {
                f"{split}_{int(r['idx']):08d}": r
                for r in csv.DictReader(stream, delimiter="\t")
                if int(r["idx"]) < expected_shards * 1000
            }
        shards = sorted((root / "shards").glob(f"{split}-*.tar"))
        if len(shards) != expected_shards:
            raise ValueError(f"Unexpected shard count for {split}")
        seen, connectivity = set(), set()
        for path in shards:
            with tarfile.open(path) as tar:
                pngs, metadata = {}, {}
                for member in tar:
                    key, ext = member.name.rsplit(".", 1)
                    if not member.isfile() or ext not in ("png", "json"):
                        raise ValueError(f"Unexpected tar member: {member.name}")
                    target = pngs if ext == "png" else metadata
                    if key in target:
                        raise ValueError(f"Duplicate member: {member.name}")
                    target[key] = tar.extractfile(member).read()
                if pngs.keys() != metadata.keys():
                    raise ValueError(f"Unpaired PNG/JSON in {path}")
                for key, data in metadata.items():
                    if key in seen or key not in source:
                        raise ValueError(f"Duplicate or unknown key {key}")
                    seen.add(key)
                    meta = json.loads(data)
                    assigned = Chem.MolFromSmiles(source[key]["smiles"])
                    stored = Chem.MolFromSmiles(meta["smiles"])
                    if assigned is None or stored is None:
                        raise ValueError(f"Unparsable label {key}")
                    if Chem.MolToSmiles(assigned) != Chem.MolToSmiles(stored):
                        report["stereo_mismatches"].append(dict(key=key, split=split, assigned=source[key]["smiles"], stored=meta["smiles"]))
                    key14 = Chem.MolToInchiKey(stored)[:14]
                    if key14 != source[key]["key14"] or key14 != meta["key14"]:
                        raise ValueError(f"Connectivity key mismatch {key}")
                    connectivity.add(key14)
                    with Image.open(io.BytesIO(pngs[key])) as image:
                        if image.size != (384, 384):
                            raise ValueError(f"Wrong image size {key}")
                        image.verify()
            print(split, path.name, len(seen), flush=True)
        missing = sorted(source.keys() - seen)
        if (split != "train" and missing) or len(missing) > 28:
            raise ValueError(f"Unexpected missing records in {split}: {len(missing)}")
        keysets[split] = connectivity
        report["splits"][split] = {
            "stored": len(seen), "missing_render_records": missing,
            "manifest_sha256": file_sha256(root / "manifests" / f"{split}.tsv"),
        }
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        if keysets[x] & keysets[y]:
            raise ValueError(f"Connectivity overlap: {x}/{y}")
    report["selections"] = {}
    for split, size in (("train", 200000), ("val", 1000), ("test", 10000)):
        ds = ShardDataset(root / "shards", split, max_samples=size, image_size=384)
        if len(ds) != size:
            raise ValueError(f"Insufficient {split} samples")
        report["selections"][split] = ds.identity()
    report["passed"] = not report["stereo_mismatches"]
    atomic_json(a.root / "reports/data-audit-diagnostic.json", report)
    print("STEREO_MISMATCHES", len(report["stereo_mismatches"]), flush=True)


if __name__ == "__main__":
    main()
