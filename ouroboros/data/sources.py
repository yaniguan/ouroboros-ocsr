"""Molecule sources (SMILES lists) with download + local caching."""

from __future__ import annotations

import csv
import gzip
import shutil
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    smiles_column: str


SOURCES = {
    # ZINC250k (Gomez-Bombarelli et al. 2018); some entries carry stereo.
    "zinc250k": Source(
        "zinc250k",
        "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/"
        "zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
        "zinc250k.csv",
        "smiles",
    ),
    # MOSES (Polykovskiy et al. 2020), ~1.9M ZINC clean-leads molecules without stereo.
    "moses": Source(
        "moses",
        "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv",
        "moses.csv",
        "SMILES",
    ),
}


def fetch(name: str, cache_dir: str | Path) -> Path:
    """Download source ``name`` into ``cache_dir`` if not already present; return the path."""
    src = SOURCES[name]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / src.filename
    if not path.exists():
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(src.url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def iter_smiles(path: str | Path, smiles_column: str | None = None) -> Iterator[str]:
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
        for row in csv.DictReader(f):
            s = row[smiles_column].strip()
            if s:
                yield s


def iter_source(name: str, cache_dir: str | Path) -> Iterator[str]:
    yield from iter_smiles(fetch(name, cache_dir), SOURCES[name].smiles_column)
