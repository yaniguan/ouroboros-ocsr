"""Extract structured HRMS and 13C/1H NMR blocks from Supporting Information text files.

Parses multi-compound experimental characterization text (ACS / RSC / Wiley / ChemRxiv style),
strips residual deuterated solvent 13C peaks, optionally filters by internal HRMS calcd-vs-found
consistency (default <= 5.0 ppm), and emits ChemVeritas-SI JSONL records.

Example:
    python scripts/extract_si_bench.py --si-text paper_si.txt --out results/si_records.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ouroboros.si.parser import parse_si_document


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--si-text",
        type=Path,
        nargs="+",
        required=True,
        help="One or more plain-text/Markdown Supporting Information files",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSONL file path")
    p.add_argument(
        "--max-internal-ppm",
        type=float,
        default=None,
        help="If set, drop records whose internal calcd-vs-found HRMS error exceeds this ppm",
    )
    args = p.parse_args(argv)

    kept = []
    n_total = 0
    for src in args.si_text:
        docs = parse_si_document(Path(src).read_text(encoding="utf-8"))
        for rec in docs:
            n_total += 1
            err = rec.internal_hrms_ppm
            max_ppm = args.max_internal_ppm
            if max_ppm is not None and err is not None and err > max_ppm:
                continue
            d = rec.to_dict()
            d["source_file"] = Path(src).name
            d["internal_hrms_ppm"] = err
            kept.append(d)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in kept), encoding="utf-8"
    )
    summary = {"n_parsed": n_total, "n_kept": len(kept), "out": str(args.out)}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
