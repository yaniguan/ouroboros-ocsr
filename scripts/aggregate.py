"""Aggregate sweep runs: results table (markdown + JSON), data-efficiency, real-fraction and
rotation-sweep plots.

    python scripts/aggregate.py --runs runs/* --out results/

``--average SET [SET ...]`` additionally reports the mean over several eval sets, which is only
allowed for sets of the same kind; mixing rendered and real-document sets raises an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ouroboros.eval import aggregate as agg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--average", nargs="*", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = agg.load_runs([r for r in a.runs if (Path(r) / "eval" / "predictions.jsonl").exists()])
    table = agg.results_table(runs, angle=0.0, n_boot=a.n_boot)
    if a.average:
        sets = agg.pool_sets(table["set_kinds"], a.average)  # raises MixedKindsError
        for r in table["rows"]:
            vals = [r["sets"][s]["exact_mean"] for s in sets if s in r["sets"]]
            r[f"mean_of_{'+'.join(sets)}"] = float(np.mean(vals)) if vals else float("nan")
    (out / "results.json").write_text(json.dumps(table, indent=1))
    (out / "results.md").write_text(agg.table_markdown(table))
    figs = agg.plot_curves(table, "size", "training-set size (synthetic)", out, logx=True)
    figs += agg.plot_curves(table, "real_fraction", "real-data fraction", out, logx=False)
    sweep = agg.rotation_sweep(runs, n_boot=a.n_boot)
    (out / "rotation.json").write_text(json.dumps(sweep, indent=1))
    figs += agg.plot_rotation(sweep, out)
    print(agg.table_markdown(table))
    print("figures:", *[str(f) for f in figs], sep="\n  ")


if __name__ == "__main__":
    main()
