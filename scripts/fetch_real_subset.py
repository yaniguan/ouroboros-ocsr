"""Download a seeded subset of public real OCSR benchmarks -> image files + manifest.

Source: Hugging Face dataset ``Keylab/OCSR-Benchmarks`` (one parquet per benchmark, ~5-20 MB).
Extracted images and parquet caches must stay out of git (see DATA.md).

Dev-test set (subset100, seed 0):
    python scripts/fetch_real_subset.py --sets USPTO UOB ACS --n 100 --seed 0 \
        --out real_data/subset100
    python scripts/ingest_real.py --manifest real_data/subset100/manifest.csv \
        --out real/subset100

Frozen release test set (frozen-test-v2, materialize images from the committed manifest):
    python scripts/fetch_real_subset.py --from-manifest benchmarks/frozen-test-v2/manifest.csv \
        --out real_data/frozen_test_v2
    python scripts/ingest_real.py --manifest real_data/frozen_test_v2/manifest.csv \
        --out real/frozen_test_v2

Output: ``<out>/images/<SET>/<id>.png`` and ``<out>/manifest.csv`` with columns
``image,smiles,source,split,id`` (the ``ouroboros.data.real`` manifest interface).
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from rdkit import RDLogger

from ouroboros.data import real as real_data
from ouroboros.eval.standardize import leakage_key, standardize
from ouroboros.provenance import file_sha256

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
REPO = real_data.KEYLAB_BENCHMARKS_REPO
KEYLAB_BENCHMARKS_REVISION = real_data.KEYLAB_BENCHMARKS_REVISION
EXPECTED_DEV_TEST_MANIFEST_SHA256 = real_data.EXPECTED_DEV_TEST_MANIFEST_SHA256
EXPECTED_TEST_MANIFEST_SHA256 = real_data.EXPECTED_TEST_MANIFEST_SHA256
FROZEN_TEST_V2_MANIFEST = real_data.FROZEN_TEST_V2_MANIFEST
FROZEN_TEST_V2_SEED = real_data.FROZEN_TEST_V2_SEED
FROZEN_TEST_V2_COUNTS = real_data.FROZEN_TEST_V2_COUNTS
EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256 = real_data.EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256
URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{set}/test-00000-of-00001.parquet"


def safe_id(rid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in rid)


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


def load_exclusions(
    manifests: list[str], key_files: list[str]
) -> tuple[set[tuple[str, str]], set[str]]:
    excluded_ids: set[tuple[str, str]] = set()
    excluded_key14: set[str] = set()
    for m in manifests:
        with open(m, newline="") as f:
            for row in csv.DictReader(f):
                src = (row.get("source") or "").strip()
                rid = str(row.get("id") or "").strip()
                if src and rid:
                    excluded_ids.add((src, rid))
                lk = leakage_key(standardize(row.get("smiles")))
                if lk is not None:
                    excluded_key14.add(lk[:14])
    for kf in key_files:
        for token in Path(kf).read_text().split():
            if token:
                excluded_key14.add(token[:14])
    return excluded_ids, excluded_key14


def sample_clean_rows(
    records: list[dict],
    source: str,
    n: int,
    seed: int,
    excluded_ids: set[tuple[str, str]],
    excluded_key14: set[str],
    id_col: str | None,
    smi_col: str,
) -> list[int]:
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    picked: list[int] = []
    for i in idx:
        rec = records[i]
        rid = str(rec[id_col]) if id_col else f"{i:06d}"
        if (source, rid) in excluded_ids:
            continue
        std = standardize(rec[smi_col])
        lk = leakage_key(std)
        if not std.valid or lk is None or lk[:14] in excluded_key14:
            continue
        picked.append(i)
        if len(picked) == n:
            break
    if len(picked) < n:
        raise ValueError(
            f"{source}: requested {n} clean non-overlapping rows, only {len(picked)} available"
        )
    return sorted(picked)


def materialize_from_manifest(
    manifest_path: Path,
    out: Path,
    cache: Path,
    revision: str,
    expected_sha256: str | None = None,
) -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    frozen_default = (ROOT / real_data.FROZEN_TEST_V2_MANIFEST).resolve()
    if (
        expected_sha256 is None
        and manifest_path.resolve() == frozen_default
        and real_data.EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256
    ):
        expected_sha256 = real_data.EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256
    if expected_sha256 is not None:
        actual = file_sha256(manifest_path)
        if actual != expected_sha256:
            raise ValueError(
                f"Manifest SHA-256 mismatch for {manifest_path}: {actual} != {expected_sha256}"
            )

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    out.mkdir(parents=True, exist_ok=True)
    for source, wanted in by_source.items():
        table = pq.read_table(fetch_parquet(source, cache, revision))
        cols = table.column_names
        img_col = "image"
        id_col = "image_id" if "image_id" in cols else None
        lookup: dict[str, dict] = {}
        for i, rec in enumerate(table.to_pylist()):
            rid = str(rec[id_col]) if id_col else f"{i:06d}"
            lookup[rid] = rec
        for row in wanted:
            rid = str(row["id"])
            if rid not in lookup:
                raise KeyError(f"{source}: id {rid!r} not found in parquet (revision {revision})")
            dst = out / Path(row["image"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(lookup[rid][img_col]["bytes"])).convert("RGB").save(dst)
        print(f"{source}: wrote {len(wanted)} images")

    dst_manifest = out / "manifest.csv"
    if manifest_path.resolve() != dst_manifest.resolve():
        shutil.copyfile(manifest_path, dst_manifest)
    print(f"wrote {len(rows)} rows to {dst_manifest}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--sets", nargs="+", default=["USPTO", "UOB", "ACS"])
    ap.add_argument("--n", type=int, default=100, help="images per benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--cache-dir", default=None, help="parquet cache directory")
    ap.add_argument(
        "--from-manifest",
        default=None,
        help="materialize images for the exact rows listed in an existing manifest CSV",
    )
    ap.add_argument("--expected-sha256", default=None, help="expected SHA-256 of --from-manifest")
    ap.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="exclude (source, id) rows and connectivity blocks present in this manifest CSV",
    )
    ap.add_argument(
        "--exclude-keys",
        action="append",
        default=[],
        help="exclude connectivity blocks (first 14 chars) listed in this key file",
    )
    ap.add_argument(
        "--require-valid-gold",
        action="store_true",
        help="only sample rows whose reference SMILES standardizes cleanly",
    )
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="write only manifest.csv without extracting PNG images",
    )
    ap.add_argument(
        "--lf",
        action="store_true",
        help="write manifest.csv with LF line endings for git-committed manifests",
    )
    a = ap.parse_args(argv)

    out = Path(a.out)
    if a.cache_dir:
        cache = Path(a.cache_dir)
    elif a.manifest_only:
        cache = Path(tempfile.gettempdir()) / "ouroboros_keylab_parquet"
    else:
        cache = out / "parquet"

    if a.from_manifest:
        rev = real_data.KEYLAB_BENCHMARKS_REVISION if a.revision == "main" else a.revision
        materialize_from_manifest(Path(a.from_manifest), out, cache, rev, a.expected_sha256)
        return

    import pyarrow.parquet as pq
    from PIL import Image

    use_clean_filter = bool(a.exclude_manifest or a.exclude_keys or a.require_valid_gold)
    excluded_ids, excluded_key14 = (
        load_exclusions(a.exclude_manifest, a.exclude_keys)
        if use_clean_filter
        else (set(), set())
    )

    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name in a.sets:
        pq_path = fetch_parquet(name, cache, a.revision)
        schema_cols = pq.read_schema(pq_path).names
        smi_col = "SMILES" if "SMILES" in schema_cols else "smiles"
        id_col = "image_id" if "image_id" in schema_cols else None
        read_cols = [c for c in (id_col, smi_col) if c] if a.manifest_only else None
        table = pq.read_table(pq_path, columns=read_cols)
        n_total = table.num_rows
        if use_clean_filter:
            meta_cols = [c for c in (id_col, smi_col) if c]
            meta_rows = table.select(meta_cols).to_pylist()
            chosen = sample_clean_rows(
                meta_rows, name, a.n, a.seed, excluded_ids, excluded_key14, id_col, smi_col
            )
        else:
            chosen = sample_rows(n_total, a.n, a.seed)

        if not a.manifest_only:
            d = out / "images" / name
            d.mkdir(parents=True, exist_ok=True)
        for i in chosen:
            rec = table.slice(i, 1).to_pylist()[0]
            rid = str(rec[id_col]) if id_col else f"{i:06d}"
            rel_img = f"images/{name}/{safe_id(rid)}.png"
            if not a.manifest_only:
                Image.open(io.BytesIO(rec["image"]["bytes"])).convert("RGB").save(out / rel_img)
            rows.append(
                {
                    "image": rel_img,
                    "smiles": rec[smi_col],
                    "source": name,
                    "split": "test",
                    "id": rid,
                }
            )
        print(f"{name}: {n_total} rows, sampled {sum(r['source'] == name for r in rows)}")

    lineterminator = "\n" if a.lf else "\r\n"
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image", "smiles", "source", "split", "id"],
            lineterminator=lineterminator,
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
