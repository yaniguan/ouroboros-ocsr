"""Transactional filtering cursor, preserving the existing connectivity deduplication/order."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import itertools
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from ouroboros.provenance import atomic_json, file_sha256


def _process_batch(processor, batch, pool):
    return list(pool.map(processor, batch, chunksize=64) if pool else map(processor, batch))


def prepare(sources, out_path, workers, limit, processor, identity, batch_size=1000):
    if workers < 1 or batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("workers, batch_size and limit must be positive")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Re-read raw input on restart (cheap); completed chemistry batches are never repeated.
    # Spool to disk to bind generator contents without retaining millions of rows in RAM.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as raw:
        digest = hashlib.sha256()
        count = 0
        for name, smiles in sources:
            for smi in smiles:
                if limit is not None and count >= limit:
                    break
                line = json.dumps([name, smi], separators=(",", ":")) + "\n"
                digest.update(line.encode())
                raw.write(line)
                count += 1
            if limit is not None and count >= limit:
                break
        identity = {**identity, "input_sha256": digest.hexdigest(), "input_rows": count}
        cache_path = out_path.with_suffix(out_path.suffix + ".filter.sqlite")
        if out_path.exists() and not cache_path.exists():
            raise ValueError("Legacy pool has no filtering identity; choose a new output path")
        with sqlite3.connect(cache_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS accepted "
                "(position INTEGER PRIMARY KEY, key14 TEXT UNIQUE, row_json TEXT)"
            )
            previous = dict(db.execute("SELECT key, value FROM metadata"))
            if previous:
                if json.loads(previous["identity"]) != identity:
                    raise ValueError("Filtering input/version/code identity changed")
                if previous.get("complete") == "true" and out_path.exists():
                    if file_sha256(out_path) != previous["output_sha256"]:
                        raise ValueError("Completed pool checksum mismatch")
                    return json.loads(previous["report"])
            else:
                db.execute("INSERT INTO metadata VALUES (?, ?)", ("identity", json.dumps(identity)))
                db.commit()
            cursor = int(previous.get("cursor", "0"))
            reasons = Counter(json.loads(previous.get("reasons", "{}")))
            raw.seek(0)
            inputs = (tuple(json.loads(line)) for line in itertools.islice(raw, cursor, None))
            context = ProcessPoolExecutor(workers) if workers > 1 else nullcontext(None)
            with context as pool:
                while batch := list(itertools.islice(inputs, batch_size)):
                    results = _process_batch(processor, batch, pool)
                    with db:
                        for offset, (status, result) in enumerate(results):
                            if status != "ok":
                                reasons[result] += 1
                                continue
                            inserted = db.execute(
                                "INSERT OR IGNORE INTO accepted VALUES (?, ?, ?)",
                                (cursor + offset, result["key14"], json.dumps(result)),
                            ).rowcount
                            if not inserted:
                                reasons["duplicate"] += 1
                        cursor += len(batch)
                        db.executemany(
                            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                            [("cursor", str(cursor)), ("reasons", json.dumps(reasons))],
                        )
            stats = {
                "n_in": cursor,
                "n_out": db.execute("SELECT count(*) FROM accepted").fetchone()[0],
                "rejects": dict(reasons),
                "seconds": round(time.perf_counter() - started, 1),
                "identity": identity,
            }
            with tempfile.NamedTemporaryFile(
                dir=out_path.parent, suffix=".part", delete=False
            ) as f:
                tmp = Path(f.name)
                try:
                    with gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as gz:
                        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                            writer = csv.writer(text, delimiter="\t")
                            fields = ["key14", "flat", "n_tet", "n_db", "src"]
                            writer.writerow(fields)
                            for (record,) in db.execute(
                                "SELECT row_json FROM accepted ORDER BY position"
                            ):
                                row = json.loads(record)
                                writer.writerow([row[k] for k in fields])
                    f.flush()
                    os.fsync(f.fileno())
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
            try:
                output_digest = file_sha256(tmp)
                os.replace(tmp, out_path)
            finally:
                tmp.unlink(missing_ok=True)
            atomic_json(out_path.with_suffix(".stats.json"), stats)
            with db:
                db.executemany(
                    "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                    [
                        ("complete", "true"),
                        ("report", json.dumps(stats)),
                        ("output_sha256", output_digest),
                    ],
                )
            return stats
