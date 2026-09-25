"""Dataset construction: pool -> splits -> stereo-controlled composition -> rendered shards.

Pipeline (each stage writes a file so it can be resumed / inspected):

1. ``prepare_pool``   : sources -> filtered, standardized, stereo-free, deduplicated pool
                         (dedup + split key = InChIKey connectivity block, so all stereoisomers
                         of one constitution land in the same split).
2. ``compose``        : pool -> per-split manifests with a target stereo fraction. The train
                         manifest is ordered so that EVERY prefix has the target fraction, which
                         makes the size ladder 10k < 50k < 200k < 1M nested prefixes.
3. ``render_shards``  : manifest -> WebDataset tar shards (``{key}.png`` + ``{key}.json``).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import random
import tarfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem

from ouroboros.data.filter import FilterConfig, filter_smiles
from ouroboros.data.render import encode_png, render, sample_style
from ouroboros.data.stereo import assign_random_stereo, has_stereo, potential_stereo

SPLITS = ("train", "val", "test")


def _hash01(s: str, salt: str = "") -> float:
    h = hashlib.sha1(f"{salt}|{s}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _hash_int(s: str, salt: str = "") -> int:
    return int.from_bytes(hashlib.sha1(f"{salt}|{s}".encode()).digest()[:8], "big")


# --------------------------------------------------------------------------- 1. pool


def _pool_row(item: tuple[str, str]) -> tuple[str, dict | str]:
    src_name, smi = item
    mol, why = filter_smiles(smi, FilterConfig())
    if mol is None:
        return "reject", why
    Chem.RemoveStereochemistry(mol)
    flat = Chem.MolToSmiles(mol)
    key = Chem.MolToInchiKey(mol)
    if not key:
        return "reject", "inchikey_failed"
    n_tet, n_db = potential_stereo(mol)
    return "ok", {"key14": key[:14], "flat": flat, "n_tet": n_tet, "n_db": n_db, "src": src_name}


def prepare_pool(
    sources: Iterable[tuple[str, Iterable[str]]],
    out_path: str | Path,
    workers: int = os.cpu_count() or 1,
    limit: int | None = None,
) -> dict:
    """Filter + dedupe all ``(source_name, smiles_iter)`` pairs into a gzipped TSV pool."""

    def items() -> Iterator[tuple[str, str]]:
        n = 0
        for name, it in sources:
            for s in it:
                if limit is not None and n >= limit:
                    return
                n += 1
                yield name, s

    seen: set[str] = set()
    reasons: Counter = Counter()
    n_in = n_out = 0
    t0 = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", newline="") as f, Pool(workers) as pool:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key14", "flat", "n_tet", "n_db", "src"])
        for status, res in pool.imap(_pool_row, items(), chunksize=512):
            n_in += 1
            if status != "ok":
                reasons[res] += 1
                continue
            if res["key14"] in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(res["key14"])
            w.writerow([res["key14"], res["flat"], res["n_tet"], res["n_db"], res["src"]])
            n_out += 1
    stats = {
        "n_in": n_in,
        "n_out": n_out,
        "rejects": dict(reasons),
        "seconds": round(time.time() - t0, 1),
    }
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def read_pool(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["n_tet"] = int(r["n_tet"])
        r["n_db"] = int(r["n_db"])
    return rows


# --------------------------------------------------------------------------- 2. compose


@dataclass
class ComposeConfig:
    stereo_fraction: float = 0.40  # target fraction of molecules with >= 1 defined stereo element
    sizes: dict = field(default_factory=lambda: {"train": 50_000, "val": 5_000, "test": 10_000})
    # fractions of the connectivity-key hash space reserved for val / test
    val_frac: float = 0.02
    test_frac: float = 0.02
    seed: int = 0
    # InChIKey connectivity blocks (first 14 chars) that must never enter ANY split, e.g. the
    # molecules of real-document eval sets (leakage control, Am1-A). Stereoisomers are excluded too.
    exclude_key14: frozenset = frozenset()


def split_of(key14: str, cfg: ComposeConfig) -> str:
    u = _hash01(key14, "split")  # independent of cfg.seed: splits never move between runs
    if u < cfg.test_frac:
        return "test"
    if u < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


def _compose_split(rows: list[dict], n: int, cfg: ComposeConfig, split: str) -> list[dict]:
    """Pick ``n`` molecules; every prefix of the output has stereo fraction ~= target.

    Buckets: ``ez`` (has a potential E/Z bond: its drawing always depicts E or Z, so it can
    only be a stereo sample), ``none`` (no potential stereo: non-stereo only) and ``tet``
    (tetrahedral only: usable as either, wedges drawn or not).
    """
    order = sorted(rows, key=lambda r: _hash_int(r["key14"], f"order{cfg.seed}"))
    ez = [r for r in order if r["n_db"] > 0]
    none = [r for r in order if r["n_db"] == 0 and r["n_tet"] == 0]
    tet = [r for r in order if r["n_db"] == 0 and r["n_tet"] > 0]
    for bucket in (ez, none, tet):
        bucket.reverse()  # pop() from the end == take in hash order
    rng = random.Random(f"{cfg.seed}-{split}")
    out: list[dict] = []
    n_stereo = 0
    while len(out) < n:
        want_stereo = n_stereo < cfg.stereo_fraction * (len(out) + 1)
        if want_stereo:
            if not ez and not tet:
                raise RuntimeError(f"{split}: ran out of stereo-capable molecules")
            use_ez = bool(ez) and (not tet or rng.random() < len(ez) / (len(ez) + len(tet)))
            r = (ez if use_ez else tet).pop()
            m = assign_random_stereo(
                Chem.MolFromSmiles(r["flat"]), random.Random(_hash_int(r["key14"], "st")), True
            )
            smi = Chem.MolToSmiles(m)
            if not has_stereo(smi):  # degenerate (e.g. para-stereo collapsed): skip
                continue
        else:
            if not none and not tet:
                raise RuntimeError(f"{split}: ran out of non-stereo molecules")
            use_none = bool(none) and (not tet or rng.random() < len(none) / (len(none) + len(tet)))
            r = (none if use_none else tet).pop()
            smi = r["flat"]
        is_st = has_stereo(smi)
        n_stereo += is_st
        out.append({"key14": r["key14"], "smiles": smi, "stereo": int(is_st), "src": r["src"]})
    return out


def compose(pool: list[dict], cfg: ComposeConfig, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    n_excluded = 0
    for r in pool:
        if r["key14"] in cfg.exclude_key14:
            n_excluded += 1
            continue
        by_split[split_of(r["key14"], cfg)].append(r)
    stats = {}
    for split in SPLITS:
        n = cfg.sizes.get(split, 0)
        if n <= 0:
            continue
        rows = _compose_split(by_split[split], n, cfg, split)
        path = out_dir / f"{split}.tsv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["idx", "key14", "smiles", "stereo", "src"])
            for i, r in enumerate(rows):
                w.writerow([i, r["key14"], r["smiles"], r["stereo"], r["src"]])
        stats[split] = {
            "n": len(rows),
            "available": len(by_split[split]),
            "stereo_fraction": sum(r["stereo"] for r in rows) / max(1, len(rows)),
        }
    stats["excluded_from_pool"] = n_excluded
    conf = {k: v for k, v in cfg.__dict__.items() if k != "exclude_key14"}
    conf["n_exclude_key14"] = len(cfg.exclude_key14)
    (out_dir / "compose.json").write_text(json.dumps({"config": conf, "stats": stats}, indent=1))
    return stats


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# --------------------------------------------------------------------------- 3. shards


def sample_key(split: str, idx: int) -> str:
    return f"{split}_{idx:08d}"


def render_row(row: dict, split: str, size: int, seed: int) -> tuple[str, bytes, dict] | str:
    """Render one manifest row. Returns (key, png, meta) or a drop reason."""
    key = sample_key(split, int(row["idx"]))
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None:
        return "unparsable"
    style = sample_style(random.Random(_hash_int(key, f"style{seed}")))
    try:
        post_seed = _hash_int(key, f"noise{seed}") % 2**32
        r = render(mol, style, size=size, seed=post_seed, apply_postprocess=False)
    except Exception as e:  # noqa: BLE001
        return f"render_error:{type(e).__name__}"
    source = Chem.MolToSmiles(mol)
    if r.label != source:
        return "label_mismatch"
    meta = {"smiles": r.label, "key14": row["key14"], "stereo": int(row["stereo"])}
    meta["style"] = style.to_dict()
    # degradation (blur / noise / JPEG) is deferred to load time, deterministically seeded
    meta["post"] = {"deferred": True, "seed": post_seed}
    return key, encode_png(r.image), meta


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o444
    tar.addfile(info, io.BytesIO(data))


def _render_shard(args) -> dict:
    rows, split, shard_path, size, seed = args
    t0 = time.time()
    drops: Counter = Counter()
    n = 0
    tmp = str(shard_path) + ".part"
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for row in rows:
            res = render_row(row, split, size, seed)
            if isinstance(res, str):
                drops[res] += 1
                continue
            key, png, meta = res
            _add(tar, f"{key}.png", png)
            _add(tar, f"{key}.json", json.dumps(meta).encode())
            n += 1
    os.replace(tmp, shard_path)
    return {"shard": Path(shard_path).name, "n": n, "drops": dict(drops), "sec": time.time() - t0}


def render_shards(
    manifest: list[dict],
    split: str,
    out_dir: str | Path,
    shard_size: int = 1000,
    size: int = 384,
    seed: int = 0,
    workers: int = os.cpu_count() or 1,
    skip_existing: bool = True,
) -> dict:
    """Render ``manifest`` into ``{split}-{i:06d}.tar`` shards. Resumable: finished shards
    (renamed from ``.part`` only on completion) are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(0, len(manifest), shard_size):
        path = out_dir / f"{split}-{i // shard_size:06d}.tar"
        if skip_existing and path.exists():
            continue
        jobs.append((manifest[i : i + shard_size], split, path, size, seed))
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_render_shard, jobs):
            results.append(res)
    drops: Counter = Counter()
    for r in results:
        drops.update(r["drops"])
    stats = {
        "split": split,
        "shards_written": len(results),
        "samples_written": sum(r["n"] for r in results),
        "drops": dict(drops),
        "wall_sec": round(time.time() - t0, 1),
        "workers": workers,
    }
    log = out_dir / f"{split}.render.jsonl"
    with open(log, "a") as f:
        f.write(json.dumps(stats) + "\n")
    return stats
