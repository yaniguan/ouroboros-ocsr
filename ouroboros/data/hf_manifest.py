"""Hugging Face dataset snapshot -> real-data manifests (the ``ouroboros.data.real`` interface).

The layout of a user-made HF dataset is not known in advance, so this module inspects a local
snapshot (``huggingface_hub.snapshot_download(..., repo_type="dataset")``) and detects:

* tables: ``*.parquet``, ``*.jsonl``/``*.json`` (JSON lines), ``*.csv``, ``*.tsv``;
* the image column: an HF ``Image`` struct (``{"bytes", "path"}``), raw bytes, or a string path
  to an image file (resolved against the table's directory, then the snapshot root);
* the SMILES column: a known name (``smiles``, ``SMILES``, ...), else the string column whose
  sampled values parse with RDKit most often;
* the source column (``source``, ``dataset``, ...); without one, the HF config directory name,
  else ``--default-source``;
* the split: a ``split`` column, else the HF file naming (``data/train-0000-of-0001.parquet``,
  ``test/...``); ``validation``/``valid``/``dev`` -> ``val``. Rows with no recognisable split get
  ``default_split`` (``test``: an unknown row is never silently trained on).

Every detection can be overridden explicitly. ``inspect_snapshot`` reports what was found (for a
human to confirm before ingestion); ``write_manifests`` writes one ``manifests/<source>.csv`` per
source (``image, smiles, source, split, id``) plus the extracted image files, ready for
``scripts/ingest_real.py``. Output must stay out of git (DATA.md).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
TABLE_EXTS = {".parquet", ".jsonl", ".json", ".csv", ".tsv"}
SMILES_NAMES = (
    "smiles",
    "canonical_smiles",
    "isomeric_smiles",
    "smi",
    "gt_smiles",
    "label",
    "target",
    "text",
    "ground_truth",
)
IMAGE_NAMES = ("image", "img", "png", "picture", "file_name", "filename", "image_path", "path")
SOURCE_NAMES = ("source", "dataset", "subset", "benchmark", "origin", "domain", "collection")
ID_NAMES = ("id", "image_id", "uid", "key", "name", "file_name", "filename")
SPLIT_NAMES = ("split", "partition", "fold")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "eval": "test",
    "evaluation": "test",
}
_SPLIT_FILE = re.compile(r"^(?P<split>[A-Za-z]+)(?:[-_.].*)?$")
_NON_CONFIG_DIRS = {"data", "default", "images", "image", "imgs", "files"}


def normalize_split(value) -> str | None:
    if value is None:
        return None
    return SPLIT_ALIASES.get(str(value).strip().lower())


def split_from_path(rel: Path) -> str | None:
    """HF naming: ``<config>/<split>-00000-of-00001.parquet``, ``data/<split>/x.parquet``..."""
    m = _SPLIT_FILE.match(rel.stem)
    if m and normalize_split(m.group("split")):
        return normalize_split(m.group("split"))
    for part in reversed(rel.parts[:-1]):
        if normalize_split(part):
            return normalize_split(part)
    return None


def config_from_path(rel: Path) -> str | None:
    """First directory that is neither a split name nor a generic data folder."""
    for part in rel.parts[:-1]:
        if not normalize_split(part) and part.lower() not in _NON_CONFIG_DIRS:
            return part
    return None


def find_tables(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in TABLE_EXTS
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
        and p.name.lower() not in {"dataset_infos.json", "dataset_info.json", "state.json"}
    )


def iter_table(path: Path, batch_size: int = 512) -> Iterator[dict]:
    """Rows as dicts; parquet is streamed in record batches (image bytes stay bounded)."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
    elif suf in (".jsonl", ".json"):
        yield from _iter_json(path)
    else:
        with open(path, newline="") as f:
            yield from csv.DictReader(f, delimiter="\t" if suf == ".tsv" else ",")


def json_rows(obj) -> list[dict]:
    """Records from a parsed JSON document of unknown shape.

    * ``[{...}, ...]``                         -> the list
    * ``{"images": [{...}], ...}``             -> the first list of objects
    * ``{"a.png": {...}, ...}``                -> one row per key (``key`` + the object's fields)
    * ``{"a.png": "CCO", ...}``                -> one row per key (``key``, ``value``)
    Anything else (e.g. a metadata/config object) yields no rows.
    """
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if not isinstance(obj, dict) or not obj:
        return []
    for v in obj.values():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v[:50]):
            return [x for x in v if isinstance(x, dict)]
    vals = list(obj.values())
    if all(isinstance(v, dict) for v in vals):
        return [{"key": k, **v} for k, v in obj.items()]
    if all(isinstance(v, str) for v in vals):
        return [{"key": k, "value": v} for k, v in obj.items()]
    return []


def _first_line_is_object(path: Path) -> bool:
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    return isinstance(json.loads(line), dict)
                except json.JSONDecodeError:
                    return False
    return False


def _iter_json(path: Path) -> Iterator[dict]:
    """JSON lines when the first line is a complete object, else one (possibly pretty-printed)
    JSON document whatever the extension."""
    if _first_line_is_object(path):
        with open(path) as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{path}:{n}: invalid JSON line ({e})") from e
                    if isinstance(r, dict):
                        yield r
        return
    with open(path) as f:
        yield from json_rows(json.load(f))


def _is_image_value(v) -> bool:
    if isinstance(v, dict):
        return bool(v.get("bytes")) or _is_image_value(v.get("path"))
    if isinstance(v, (bytes, bytearray)):
        return True
    return isinstance(v, str) and Path(v).suffix.lower() in IMAGE_EXTS


def _smiles_parse_rate(values: list) -> float:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return 0.0
    ok = 0
    for v in vals:
        if Path(v).suffix.lower() in IMAGE_EXTS or " " in v.strip():
            continue
        if Chem.MolFromSmiles(v.strip()) is not None:
            ok += 1
    return ok / len(vals)


def _pick(names: tuple[str, ...], columns: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


@dataclass
class Columns:
    image: str | None = None
    smiles: str | None = None
    source: str | None = None
    split: str | None = None
    id: str | None = None
    smiles_parse_rate: float = 0.0
    notes: list[str] = field(default_factory=list)


def detect_columns(rows: list[dict], overrides: dict | None = None) -> Columns:
    """Guess the roles of the columns from a sample of rows (explicit overrides win)."""
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    columns = list(rows[0]) if rows else []
    col = Columns()
    image_like = [c for c in columns if any(_is_image_value(r.get(c)) for r in rows)]
    col.image = (
        overrides.get("image") or _pick(IMAGE_NAMES, image_like) or next(iter(image_like), None)
    )
    string_cols = [
        c for c in columns if c != col.image and any(isinstance(r.get(c), str) for r in rows)
    ]
    rates = {c: _smiles_parse_rate([r.get(c) for r in rows]) for c in string_cols}
    named = _pick(SMILES_NAMES, string_cols)
    if overrides.get("smiles"):
        col.smiles = overrides["smiles"]
    elif named and rates.get(named, 0.0) >= 0.5:
        col.smiles = named
    elif rates:
        best = max(rates, key=rates.get)
        col.smiles = best if rates[best] >= 0.5 else None
        if col.smiles and named and named != best:
            col.notes.append(f"column {named!r} looked like a label but {best!r} parses as SMILES")
    col.smiles_parse_rate = rates.get(col.smiles, 0.0) if col.smiles else 0.0
    rest = [c for c in columns if c not in (col.image, col.smiles)]
    col.source = overrides.get("source") or _pick(SOURCE_NAMES, rest)
    col.split = overrides.get("split") or _pick(SPLIT_NAMES, rest)
    # the image column may double as the id when it holds file names (imagefolder layout)
    id_cands = [c for c in columns if c not in (col.smiles, col.source)]
    col.id = overrides.get("id") or _pick(ID_NAMES, id_cands)
    if col.image is None:
        col.notes.append("no image column found")
    if col.smiles is None:
        col.notes.append("no column parses as SMILES (>= 50% of sampled values)")
    return col


def _preview(v, n: int = 80):
    if isinstance(v, dict):
        return {k: _preview(x, n) for k, x in v.items()}
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "..."


def inspect_snapshot(root: str | Path, sample: int = 64, overrides: dict | None = None) -> dict:
    """Describe every table in the snapshot: rows, columns, detected roles, split/source counts."""
    root = Path(root)
    report = {"root": str(root), "layout": snapshot_layout(root), "tables": []}
    for path in find_tables(root):
        rel = path.relative_to(root)
        n = 0
        split_counts: Counter = Counter()
        source_counts: Counter = Counter()
        head = []
        try:
            for r in iter_table(path):
                if len(head) < sample:
                    head.append(r)
                n += 1
        except (ValueError, OSError) as e:  # report the file, keep inspecting the others
            report["tables"].append({"file": str(rel), "error": f"{type(e).__name__}: {e}"})
            continue
        if not head:
            report["tables"].append({"file": str(rel), "rows": 0, **_json_shape(path)})
            continue
        col = detect_columns(head, overrides)
        for r in iter_table(path) if (col.split or col.source) else ():
            if col.split:
                split_counts[str(r.get(col.split))] += 1
            if col.source:
                source_counts[str(r.get(col.source))] += 1
        report["tables"].append(
            {
                "file": str(rel),
                "rows": n,
                "columns": list(head[0]) if head else [],
                "detected": {
                    "image": col.image,
                    "smiles": col.smiles,
                    "smiles_parse_rate_sample": round(col.smiles_parse_rate, 3),
                    "source": col.source,
                    "split": col.split,
                    "id": col.id,
                    "notes": col.notes,
                },
                "split_from_path": split_from_path(rel),
                "config_from_path": config_from_path(rel),
                "split_column_values": dict(split_counts.most_common(20)),
                "source_column_values": dict(source_counts.most_common(50)),
                "example_rows": [{k: _preview(v) for k, v in r.items()} for r in head[:2]],
            }
        )
    return report


def snapshot_layout(root: Path, n_examples: int = 5) -> dict:
    """File counts by extension and by top-level directory, with a few example paths."""
    by_ext: Counter = Counter()
    by_dir: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or any(part.startswith(".") for part in rel.parts):
            continue
        ext = p.suffix.lower() or "<none>"
        by_ext[ext] += 1
        by_dir[rel.parts[0] if len(rel.parts) > 1 else "."] += 1
        if len(examples.setdefault(ext, [])) < n_examples:
            examples[ext].append(f"{rel} ({p.stat().st_size} B)")
    return {"files_by_ext": dict(by_ext), "files_by_top_dir": dict(by_dir), "examples": examples}


def _json_shape(path: Path) -> dict:
    """For a JSON table without rows: its top-level structure, to show what it holds."""
    if path.suffix.lower() not in (".json", ".jsonl"):
        return {}
    try:
        with open(path) as f:
            obj = json.load(f)
    except (ValueError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if isinstance(obj, dict):
        return {"json_top_level_keys": {k: type(v).__name__ for k, v in list(obj.items())[:30]}}
    return {"json_top_level_type": type(obj).__name__}


def _image_ext(data: bytes) -> str:
    sig = {
        b"\x89PNG": ".png",
        b"\xff\xd8\xff": ".jpg",
        b"GIF8": ".gif",
        b"II*\x00": ".tif",
        b"MM\x00*": ".tif",
        b"BM": ".bmp",
        b"RIFF": ".webp",
    }
    for k, ext in sig.items():
        if data.startswith(k):
            return ext
    return ".img"


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "real"


def _resolve_image_path(value: str, table_dir: Path, root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    for base in (table_dir, root):
        if (base / p).exists():
            return base / p
    return table_dir / p  # does not exist: ingestion records image_unreadable


def write_manifests(
    root: str | Path,
    out_dir: str | Path,
    overrides: dict | None = None,
    default_source: str = "real",
    default_split: str = "test",
    split_map: dict | None = None,
) -> dict:
    """Extract images and write ``out_dir/manifests/<source>.csv``. Returns statistics.

    ``split_map`` renames splits after normalisation (e.g. ``{"val": "test"}``); a split mapped
    to ``"drop"`` is excluded. Rows without an image or SMILES value are counted, not dropped
    silently.
    """
    root, out_dir = Path(root), Path(out_dir)
    if normalize_split(default_split) is None:
        raise ValueError(f"default_split {default_split!r} is not a split name")
    split_map = split_map or {}
    img_dir = out_dir / "images"
    man_dir = out_dir / "manifests"
    img_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, tuple] = {}
    counts: Counter = Counter()
    skipped: Counter = Counter()
    tables = []
    for path in find_tables(root):
        rel = path.relative_to(root)
        head = []
        try:
            for r in iter_table(path):
                head.append(r)
                if len(head) >= 64:
                    break
        except (ValueError, OSError) as e:
            tables.append({"file": str(rel), "used": False, "error": f"{type(e).__name__}: {e}"})
            continue
        col = detect_columns(head, overrides)
        if col.image is None or col.smiles is None:
            tables.append({"file": str(rel), "used": False, "notes": col.notes})
            continue
        tables.append({"file": str(rel), "used": True, "columns": col.__dict__ | {}})
        path_split = split_from_path(rel)
        path_source = config_from_path(rel)
        for i, r in enumerate(iter_table(path)):
            smiles = r.get(col.smiles)
            value = r.get(col.image)
            if not isinstance(smiles, str) or not smiles.strip():
                skipped["no_smiles"] += 1
                continue
            split = normalize_split(r.get(col.split)) if col.split else None
            split = split or path_split or default_split
            split = split_map.get(split, split)
            if split == "drop":
                skipped["split_dropped"] += 1
                continue
            source = str(r.get(col.source) or "") if col.source else ""
            source = _safe_name(source or path_source or default_source)
            rid = r.get(col.id) if col.id else None
            rid = str(rid) if isinstance(rid, (str, int)) and str(rid) else f"{rel}#{i}"
            if isinstance(value, dict):
                data, vpath = value.get("bytes"), value.get("path")
            elif isinstance(value, (bytes, bytearray)):
                data, vpath = bytes(value), None
            else:
                data, vpath = None, value
            if data:
                name = f"{source}/{split}/{counts[source, split]:08d}{_image_ext(bytes(data))}"
                (img_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (img_dir / name).write_bytes(bytes(data))
                image = str((img_dir / name).resolve())
            elif isinstance(vpath, str) and vpath:
                image = str(_resolve_image_path(vpath, path.parent, root))
            else:
                skipped["no_image"] += 1
                continue
            if source not in writers:
                fh = open(man_dir / f"{source}.csv", "w", newline="")
                w = csv.DictWriter(fh, fieldnames=["image", "smiles", "source", "split", "id"])
                w.writeheader()
                writers[source] = (fh, w)
            writers[source][1].writerow(
                {
                    "image": image,
                    "smiles": smiles.strip(),
                    "source": source,
                    "split": split,
                    "id": rid,
                }
            )
            counts[source, split] += 1
    for fh, _ in writers.values():
        fh.close()
    stats = {
        "root": str(root),
        "tables": tables,
        "rows": {f"{s}:{sp}": n for (s, sp), n in sorted(counts.items())},
        "sources": sorted(writers),
        "skipped": dict(skipped),
        "manifests": sorted(str(man_dir / f"{s}.csv") for s in writers),
    }
    (out_dir / "hf_manifest_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    return stats
