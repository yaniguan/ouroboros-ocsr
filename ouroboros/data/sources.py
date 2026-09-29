"""Molecule sources (SMILES lists) with download + local caching."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from ouroboros.provenance import atomic_json, file_sha256


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    smiles_column: str
    size_bytes: int
    sha256: str | None = None
    git_blob_sha1: str | None = None
    delimiter: str = ","


SOURCES = {
    # ZINC250k (Gomez-Bombarelli et al. 2018); some entries carry stereo.
    "zinc250k": Source(
        "zinc250k",
        "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/37b9f96470d4471c0593cffefa448e0a8a184ef6/models/"
        "zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
        "zinc250k.csv",
        "smiles",
        size_bytes=22606589,
        git_blob_sha1="d31726bf929d35cb52fa109db90411c90a000dfe",
    ),
    # MOSES (Polykovskiy et al. 2020), ~1.9M ZINC clean-leads molecules without stereo.
    "moses": Source(
        "moses",
        "https://media.githubusercontent.com/media/molecularsets/moses/"
        "a3866ff959325b60e36f8b15beebdaa3bfaea188/data/dataset_v1.csv",
        "moses.csv",
        "SMILES",
        size_bytes=84482588,
        sha256="bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b",
    ),
}


def _verify(path: Path, src: Source) -> str:
    if path.stat().st_size != src.size_bytes:
        raise ValueError(f"Source size mismatch: {path}")
    digest = file_sha256(path)
    if src.sha256 and digest != src.sha256:
        raise ValueError(f"Source checksum mismatch: {path}")
    if src.git_blob_sha1:
        blob = hashlib.sha1(f"blob {src.size_bytes}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                blob.update(block)
        if blob.hexdigest() != src.git_blob_sha1:
            raise ValueError(f"Source Git blob checksum mismatch: {path}")
    if not src.sha256 and not src.git_blob_sha1:
        raise ValueError("A pinned source checksum is required")
    return digest


def fetch(name: str, cache_dir: str | Path) -> Path:
    """Resume a partial HTTP download and verify pinned bytes before publication/reuse."""
    src = SOURCES[name]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / src.filename
    spec = asdict(src)
    receipt = path.with_suffix(path.suffix + ".source.json")
    if path.exists():
        digest = _verify(path, src)
        if receipt.exists() and json.loads(receipt.read_text()) != {
            "source": spec,
            "sha256": digest,
        }:
            raise ValueError("Source receipt changed; choose a new cache directory")
        atomic_json(receipt, {"source": spec, "sha256": digest})
        return path
    partial = path.with_suffix(path.suffix + ".part")
    identity_path = path.with_suffix(path.suffix + ".download.json")
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != spec:
            raise ValueError("Partial download belongs to a different source")
    elif partial.exists():
        raise ValueError("Unverified partial download; choose a new cache directory")
    else:
        atomic_json(identity_path, spec)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > src.size_bytes:
        raise ValueError("Partial download exceeds pinned source size")
    if offset < src.size_bytes:
        request = urllib.request.Request(src.url, headers={"Accept-Encoding": "identity"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with urllib.request.urlopen(request, timeout=600) as response:
            status = response.status
            if status == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if (
                    not match
                    or int(match[1]) != offset
                    or int(match[2]) != src.size_bytes - 1
                    or int(match[3]) != src.size_bytes
                ):
                    raise ValueError("Unexpected Content-Range")
            elif status == 200:
                offset = 0  # server ignored Range: replace, never append a full response
            else:
                raise ValueError(f"Unexpected download status: {status}")
            with partial.open("ab" if offset else "wb") as stream:
                count = offset
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > src.size_bytes:
                        raise ValueError("Response exceeds pinned source size")
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
    digest = _verify(partial, src)
    atomic_json(receipt, {"source": spec, "sha256": digest})
    os.replace(partial, path)
    return path


def iter_smiles(
    path: str | Path, smiles_column: str | None = None, delimiter: str = ","
) -> Iterator[str]:
    """Yield SMILES from a CSV (with ``smiles_column``) or a .smi file (first token per line)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        if smiles_column is None:
            for line in f:
                tok = line.split()
                if tok:
                    yield tok[0]
            return
        for row in csv.DictReader(f, delimiter=delimiter):
            s = row[smiles_column].strip()
            if s:
                yield s


def iter_source(name: str, cache_dir: str | Path) -> Iterator[str]:
    yield from iter_smiles(
        fetch(name, cache_dir), SOURCES[name].smiles_column, SOURCES[name].delimiter
    )
