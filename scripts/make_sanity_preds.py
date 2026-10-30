"""Write the model-free sanity predictions set (``ouroboros/eval/sanity.py``) as CSV.

    python scripts/make_sanity_preds.py --out benchmarks/eval-harness/sanity_preds.csv
    python scripts/eval_predictions.py benchmarks/eval-harness/sanity_preds.csv \
        --json benchmarks/eval-harness/sanity_report.json \
        --md benchmarks/eval-harness/sanity_report.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ouroboros.eval.sanity import build_records


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows = build_records(size=a.size, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
