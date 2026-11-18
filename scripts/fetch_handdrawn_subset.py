"""Download a seeded subset of the DECIMER Hand-drawn benchmark -> images + manifest.

Source: DECIMER Hand-drawn molecule images dataset (Brinkhaus et al., J. Cheminform. 2022),
Zenodo record 6456306 (DOI: 10.5281/zenodo.6456306, CC-BY 4.0, 5,088 depictions).
Extracted images and downloaded archives must stay out of git (see DATA.md).

Materialize images from the committed manifest:
    python scripts/fetch_handdrawn_subset.py \
        --from-manifest benchmarks/handdrawn-decimer/manifest.csv \
        --out real_data/handdrawn_decimer
    python scripts/ingest_real.py \
        --manifest real_data/handdrawn_decimer/manifest.csv \
        --out real/handdrawn_decimer

Output: ``<out>/images/DECIMER_HDM/<id>.png`` and ``<out>/manifest.csv`` with columns
``image,smiles,source,split,id`` (the ``ouroboros.data.real`` manifest interface).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

from rdkit import RDLogger

from ouroboros.data import real as real_data
from ouroboros.eval.standardize import leakage_key, standardize
from ouroboros.provenance import file_sha256

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = real_data.DECIMER_HDM_SOURCE
TSV_URL = real_data.DECIMER_HDM_TSV_URL
TSV_MD5 = real_data.DECIMER_HDM_TSV_MD5
TSV_SHA256 = real_data.DECIMER_HDM_TSV_SHA256
IMAGES_URL = real_data.DECIMER_HDM_IMAGES_URL
IMAGES_MD5 = real_data.DECIMER_HDM_IMAGES_MD5
HANDDRAWN_DECIMER_MANIFEST = real_data.HANDDRAWN_DECIMER_MANIFEST
HANDDRAWN_DECIMER_SEED = real_data.HANDDRAWN_DECIMER_SEED
HANDDRAWN_DECIMER_N = real_data.HANDDRAWN_DECIMER_N
EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256 = (
    real_data.EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256
)


def safe_id(rid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in rid)


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verified(
    url: str,
    dst: Path,
    *,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_file():
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
    if expected_md5 and _file_md5(dst) != expected_md5:
        raise ValueError(f"MD5 mismatch for {dst}: {_file_md5(dst)} != {expected_md5}")
    if expected_sha256 and file_sha256(dst) != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {dst}: {file_sha256(dst)} != {expected_sha256}"
        )
    return dst


def read_smiles_tsv(tsv_path: Path) -> list[dict[str, str]]:
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        out = []
        for row in reader:
            rid = (row.get("IDs") or row.get("id") or "").strip()
            smi = (row.get("SMILES") or row.get("smiles") or "").strip()
            if rid and smi:
                out.append({"id": rid, "smiles": smi})
        return out


def load_excluded_key14(manifests: list[str], key_files: list[str]) -> set[str]:
    out: set[str] = set()
    for m in manifests:
        with open(m, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                lk = leakage_key(standardize(row.get("smiles")))
                if lk is not None:
                    out.add(lk[:14])
    for kf in key_files:
        for token in Path(kf).read_text(encoding="utf-8").split():
            if token:
                out.add(token[:14])
    return out


def sample_clean_handdrawn(
    records: list[dict[str, str]],
    n: int,
    seed: int,
    excluded_key14: set[str],
) -> list[dict[str, str]]:
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    seen_key14 = set(excluded_key14)
    picked_indices: list[int] = []
    for idx in order:
        rec = records[idx]
        std = standardize(rec["smiles"])
        lk = leakage_key(std)
        if not std.valid or lk is None or lk[:14] in seen_key14:
            continue
        seen_key14.add(lk[:14])
        picked_indices.append(idx)
        if len(picked_indices) == n:
            break
    if len(picked_indices) < n:
        raise ValueError(
            f"{SOURCE}: requested {n} clean unique molecules, only {len(picked_indices)} found"
        )
    picked_indices.sort()
    return [records[i] for i in picked_indices]


def extract_selected_images(archive_path: Path, wanted_ids: set[str], out_dir: Path) -> None:
    img_dir = out_dir / "images" / SOURCE
    img_dir.mkdir(parents=True, exist_ok=True)
    missing = set(wanted_ids)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            if stem not in missing:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            dst = img_dir / f"{safe_id(stem)}.png"
            dst.write_bytes(src.read())
            missing.remove(stem)
            if not missing:
                break
    if missing:
        sample = sorted(missing)[:5]
        raise KeyError(f"{archive_path}: missing {len(missing)} IDs, e.g. {sample}")


def write_manifest_lf(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "smiles", "source", "split", "id"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def materialize_from_manifest(
    manifest_path: Path,
    out_dir: Path,
    cache_dir: Path,
    expected_sha256: str | None = None,
) -> None:
    default_manifest = (ROOT / HANDDRAWN_DECIMER_MANIFEST).resolve()
    if (
        expected_sha256 is None
        and manifest_path.resolve() == default_manifest
        and EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256
    ):
        expected_sha256 = EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256
    if expected_sha256:
        actual = file_sha256(manifest_path)
        if actual != expected_sha256:
            raise ValueError(
                f"Manifest SHA-256 mismatch for {manifest_path}: {actual} != {expected_sha256}"
            )
    with open(manifest_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    wanted = {row["id"] for row in rows}
    archive = _download_verified(
        IMAGES_URL,
        cache_dir / "DECIMER_HDM_Dataset_Images.tar.gz",
        expected_md5=IMAGES_MD5,
    )
    extract_selected_images(archive, wanted, out_dir)
    dst_manifest = out_dir / "manifest.csv"
    if manifest_path.resolve() != dst_manifest.resolve():
        shutil.copyfile(manifest_path, dst_manifest)
    print(f"Materialized {len(rows)} {SOURCE} images -> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-manifest", type=Path, default=None)
    ap.add_argument("--expected-sha256", default=None)
    ap.add_argument("--n", type=int, default=HANDDRAWN_DECIMER_N)
    ap.add_argument("--seed", type=int, default=HANDDRAWN_DECIMER_SEED)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--exclude-manifest", action="append", default=[])
    ap.add_argument("--exclude-keys", action="append", default=[])
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args(argv)

    cache_dir = args.cache or (args.out / "cache")
    if args.from_manifest is not None:
        materialize_from_manifest(
            args.from_manifest,
            args.out,
            cache_dir,
            expected_sha256=args.expected_sha256,
        )
        return 0

    tsv_path = _download_verified(
        TSV_URL,
        cache_dir / "DECIMER_HDM_Dataset_SMILES.tsv",
        expected_md5=TSV_MD5,
        expected_sha256=TSV_SHA256,
    )
    records = read_smiles_tsv(tsv_path)
    excluded = load_excluded_key14(args.exclude_manifest, args.exclude_keys)
    picked = sample_clean_handdrawn(records, args.n, args.seed, excluded)
    rows = [
        {
            "image": f"images/{SOURCE}/{safe_id(rec['id'])}.png",
            "smiles": rec["smiles"],
            "source": SOURCE,
            "split": "test",
            "id": rec["id"],
        }
        for rec in picked
    ]
    manifest_out = args.out if args.out.suffix == ".csv" else (args.out / "manifest.csv")
    write_manifest_lf(manifest_out, rows)
    if not args.manifest_only:
        archive = _download_verified(
            IMAGES_URL,
            cache_dir / "DECIMER_HDM_Dataset_Images.tar.gz",
            expected_md5=IMAGES_MD5,
        )
        extract_selected_images(
            archive,
            {r["id"] for r in rows},
            manifest_out.parent,
        )
    print(f"Wrote {len(rows)} rows -> {manifest_out} (sha256={file_sha256(manifest_out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
