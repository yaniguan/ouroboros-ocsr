"""Score a predictions file without a model: exact match, InChIKey, validity, Tanimoto, per subset.

    python scripts/eval_predictions.py preds.csv --json report.json --md report.md
    python -m ouroboros.eval.harness runs/X/eval/predictions.jsonl --where angle=0

Input: CSV / TSV with a header row, or JSONL with one object per line. Columns are found by the
first matching alias (override with ``--id-col`` etc.):

  id      id | key | index                 optional; the row number is used if absent
  gold    gold | ref | reference | smiles   required
  pred    pred | prediction                 required
  subset  subset | source | set             optional, e.g. USPTO, UOB, DECIMER
  kind    kind                              optional, rendered | real

The ``predictions.jsonl`` of ``ouroboros.eval.evaluate.run_eval`` is accepted as is; select one
rotation angle with ``--where angle=0`` (values compare as strings, else as numbers).

Rules
* Every metric uses ``metrics.score_pair`` (both sides standardized the same way).
* A missing or blank prediction is an invalid prediction and counts against every metric.
* A gold that does not standardize is excluded from every metric; such rows are counted and
  their ids listed under ``gold_invalid`` (never silently dropped).
* A repeated (subset, kind, id) is an error, so an item can never be counted twice.
* Rendered and real-document rows are never pooled (Am1-D): with several kinds there is one
  overall row per kind and no cross-kind number.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ouroboros.eval.metrics import PairScore, aggregate, bootstrap_ci, score_pair
from ouroboros.eval.standardize import PARITY_FOOTNOTE, PARITY_VERIFIED

ALIASES = {
    "id": ("id", "key", "index"),
    "gold": ("gold", "ref", "reference", "smiles"),
    "pred": ("pred", "prediction"),
    "subset": ("subset", "source", "set"),
    "kind": ("kind",),
}
REQUIRED = ("gold", "pred")
ALL = "ALL"
MAX_LISTED_IDS = 50

METRICS = {
    "exact": "canonical SMILES equal after standardization, full stereochemistry",
    "exact_flat": "canonical SMILES equal with all stereo removed (stereo-agnostic)",
    "inchikey": "standard InChIKeys equal (stereo layers kept; tautomers may match)",
    "valid": "prediction parses and standardizes in RDKit (blank = invalid)",
    "tanimoto": "mean Morgan r=2 2048-bit Tanimoto, chirality ignored, 0 for invalid",
    "tanimoto_1": "fraction with Tanimoto 1.0 (an enantiomer or stereo error still counts)",
}


@dataclass(frozen=True)
class Item:
    id: str
    subset: str | None
    kind: str | None
    gold: str | None
    pred: str | None


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        records = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{n}: expected a JSON object")
            records.append(rec)
        return records
    if suffix in (".csv", ".tsv"):
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ","))
    raise ValueError(f"unsupported predictions file {path.name!r}: use .csv, .tsv or .jsonl")


def resolve_columns(records: Sequence[dict], overrides: dict | None = None) -> dict:
    """Map each role (id, gold, pred, subset, kind) to a column name, or None if absent."""
    present = dict.fromkeys(k for r in records for k in r)
    cols = {}
    for role, aliases in ALIASES.items():
        explicit = (overrides or {}).get(role)
        if explicit:
            if explicit not in present:
                raise ValueError(f"{role} column {explicit!r} not found; have {list(present)}")
            cols[role] = explicit
        else:
            cols[role] = next((a for a in aliases if a in present), None)
    missing = [r for r in REQUIRED if cols[r] is None]
    if missing:
        raise ValueError(
            f"no {' / '.join(missing)} column; have {list(present)} "
            f"(accepted names: {', '.join(f'{r}={ALIASES[r]}' for r in missing)})"
        )
    return cols


def _matches(value, want: str) -> bool:
    if str(value) == want:
        return True
    try:
        return float(value) == float(want)
    except (TypeError, ValueError):
        return False


def parse_where(clauses: Sequence[str]) -> list[tuple[str, str]]:
    out = []
    for c in clauses:
        col, sep, val = c.partition("=")
        if not sep or not col:
            raise ValueError(f"--where expects COLUMN=VALUE, got {c!r}")
        out.append((col, val))
    return out


def _text(v) -> str | None:
    return None if v is None else str(v)


def to_items(records: Sequence[dict], cols: dict, where=()) -> list[Item]:
    for col, _ in where:
        if records and not any(col in r for r in records):
            raise ValueError(f"--where column {col!r} not found")
    items, seen = [], set()
    for n, r in enumerate(records, 1):
        if not all(_matches(r.get(col), val) for col, val in where):
            continue
        item = Item(
            id=str(r[cols["id"]]) if cols["id"] else str(n),
            subset=(_text(r.get(cols["subset"])) or "(none)") if cols["subset"] else None,
            kind=(_text(r.get(cols["kind"])) or "(none)") if cols["kind"] else None,
            gold=_text(r.get(cols["gold"])),
            pred=_text(r.get(cols["pred"])),
        )
        key = (item.subset, item.kind, item.id)
        if key in seen:
            raise ValueError(
                f"duplicate id {item.id!r} (subset={item.subset}, kind={item.kind}); "
                "several rotation angles? select one with --where angle=0"
            )
        seen.add(key)
        items.append(item)
    if not items:
        raise ValueError("no records to score" + (" after --where" if where else ""))
    return items


def _score(pair: tuple[str | None, str | None]) -> PairScore | None:
    pred, gold = pair
    try:
        return score_pair(pred, gold)
    except ValueError:  # raised only when the gold does not standardize
        return None


def score_items(items: Sequence[Item], workers: int = 1) -> list[PairScore | None]:
    pairs = [(it.pred, it.gold) for it in items]
    if workers <= 1 or len(pairs) < 2:
        return [_score(p) for p in pairs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_score, pairs, chunksize=max(1, len(pairs) // (4 * workers))))


def _group_stats(
    members: list[tuple[Item, PairScore | None]], n_boot: int, seed: int
) -> dict:
    scores = [s for _, s in members if s is not None]
    bad_gold = [it.id for it, s in members if s is None]
    stats = aggregate(scores)
    if scores:
        _, lo, hi = bootstrap_ci([s.exact for s in scores], n_resamples=n_boot, seed=seed)
        stats["exact_ci95"] = [lo, hi]
    stats["n_rows"] = len(members)
    stats["n_pred_blank"] = sum(
        1 for it, s in members if s is not None and not (it.pred or "").strip()
    )
    stats["n_gold_invalid"] = len(bad_gold)
    return stats


def build_report(
    items: Sequence[Item],
    scores: Sequence[PairScore | None],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Overall row(s) (one per kind) followed by one row per (kind, subset)."""
    members = list(zip(items, scores, strict=True))
    kinds = sorted({it.kind for it in items}, key=lambda k: (k is None, k or ""))
    has_subsets = any(it.subset is not None for it in items)
    groups = []
    for kind in kinds:
        in_kind = [m for m in members if m[0].kind == kind]
        groups.append({"subset": ALL, "kind": kind, **_group_stats(in_kind, n_boot, seed)})
        if has_subsets:
            for subset in sorted({it.subset for it, _ in in_kind}):
                sub = [m for m in in_kind if m[0].subset == subset]
                groups.append(
                    {"subset": subset, "kind": kind, **_group_stats(sub, n_boot, seed)}
                )
    bad = [it.id for it, s in members if s is None]
    return {
        "n_rows": len(items),
        "n_scored": len(items) - len(bad),
        "gold_invalid": {"n": len(bad), "ids": bad[:MAX_LISTED_IDS]},
        "groups": groups,
        "metrics": METRICS,
        "n_boot": n_boot,
        "pooled_across_kinds": False,
        "parity_verified": PARITY_VERIFIED,
        "footnote": None if PARITY_VERIFIED else PARITY_FOOTNOTE,
    }


def _pct(x) -> str:
    return "–" if x is None or x != x else f"{x:.2%}"


def report_markdown(report: dict, title: str = "OCSR predictions") -> str:
    show_kind = any(g["kind"] is not None for g in report["groups"])
    head = ["subset"] + (["kind"] if show_kind else []) + [
        "n",
        "exact",
        "exact 95% CI",
        "exact (no stereo)",
        "InChIKey",
        "valid",
        "mean Tanimoto",
        "Tanimoto@1.0",
    ]
    lines = [f"# {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in report["groups"]:
        if g["n"]:
            lo, hi = g["exact_ci95"]
            cells = [
                str(g["n"]),
                _pct(g["exact"]),
                f"[{_pct(lo)}, {_pct(hi)}]",
                _pct(g["exact_flat"]),
                _pct(g["inchikey"]),
                _pct(g["valid"]),
                f"{g['tanimoto']:.4f}",
                _pct(g["tanimoto_1"]),
            ]
        else:
            cells = ["0"] + ["–"] * 7
        name = f"**{g['subset']}**" if g["subset"] == ALL else g["subset"]
        lines.append(
            "| " + " | ".join([name] + ([str(g["kind"])] if show_kind else []) + cells) + " |"
        )
    lines.append("")
    gi = report["gold_invalid"]
    ids = ""
    if gi["n"]:
        more = " …" if gi["n"] > len(gi["ids"]) else ""
        ids = f" (ids: {', '.join(gi['ids'])}{more})"
    lines.append(
        f"Rows: {report['n_rows']}; scored: {report['n_scored']}; excluded because the gold "
        f"SMILES does not standardize: {gi['n']}{ids}."
    )
    if show_kind:
        lines.append("Different kinds (rendered / real) are reported separately, never pooled.")
    lines.append(f"CI: percentile bootstrap over items, {report['n_boot']} resamples.")
    lines += ["", "Metrics:", ""]
    lines += [f"- `{k}`: {v}" for k, v in report["metrics"].items()]
    if report["footnote"]:
        lines += ["", f"_{report['footnote']}_"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("predictions", help=".csv, .tsv or .jsonl")
    for role in ALIASES:
        ap.add_argument(f"--{role}-col", default=None, help=f"default: first of {ALIASES[role]}")
    ap.add_argument("--where", action="append", default=[], metavar="COL=VALUE",
                    help="keep only rows where COL equals VALUE (repeatable)")
    ap.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    ap.add_argument("--md", type=Path, default=None, help="write the report as Markdown")
    ap.add_argument("--per-sample", type=Path, default=None,
                    help="write per-row scores as JSONL")
    ap.add_argument("--title", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="do not print the Markdown table")
    a = ap.parse_args(argv)
    if a.n_boot < 1:
        ap.error("--n-boot must be positive")

    records = read_records(a.predictions)
    overrides = {role: getattr(a, f"{role}_col") for role in ALIASES}
    cols = resolve_columns(records, overrides)
    where = parse_where(a.where)
    items = to_items(records, cols, where)
    scores = score_items(items, workers=a.workers)
    report = build_report(items, scores, n_boot=a.n_boot, seed=a.seed)
    report = {
        "input": str(a.predictions),
        "columns": cols,
        "where": [f"{c}={v}" for c, v in where],
        **report,
    }
    md = report_markdown(report, a.title or f"OCSR predictions: {Path(a.predictions).name}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1) + "\n")
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md)
    if a.per_sample:
        a.per_sample.parent.mkdir(parents=True, exist_ok=True)
        with a.per_sample.open("w") as f:
            for it, s in zip(items, scores, strict=True):
                row = {**asdict(it), "gold_invalid": s is None, **(asdict(s) if s else {})}
                f.write(json.dumps(row) + "\n")
    if not a.quiet:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
