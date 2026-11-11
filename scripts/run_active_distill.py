"""Run active blind-spot selection & replay-regularized conservative shallow-ensemble distillation.

Evaluates Hypotheses H2.1-H2.3 across connectivity-disjoint scaffolds, comparing acquisition
strategies (random, qbc_force, qbc_lcmd_diverse, etc.) and reporting held-out bioisostere
conformational MAE, PES softening slope alpha, control forgetting drift, and UQ correlation.

Example:
    python scripts/run_active_distill.py --preset smoke --budgets 10 24 --out results/distill
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ouroboros.active.committee import SELECTORS
from ouroboros.active.distill import run_active_distillation_benchmark
from ouroboros.active.motifs import build_bioisostere_panel
from ouroboros.active.oracle import QMOracleSpec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 24])
    p.add_argument(
        "--strategies",
        choices=SELECTORS,
        nargs="+",
        default=["random", "qbc_force", "qbc_lcmd_diverse"],
    )
    p.add_argument(
        "--oracle-backend",
        choices=("analytical_cage", "psi4", "pyscf", "tblite"),
        default="analytical_cage",
    )
    p.add_argument("--qm-method", type=str, default="wb97m-d3bj")
    p.add_argument("--qm-basis", type=str, default="def2-tzvppd")
    p.add_argument("--torsion-step-deg", type=float, default=60.0)
    p.add_argument("--n-displaced", type=int, default=1)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    args = p.parse_args(argv)

    records = build_bioisostere_panel(args.preset)
    spec = QMOracleSpec(
        backend=args.oracle_backend,
        method=args.qm_method,
        basis=args.qm_basis,
    )
    report = run_active_distillation_benchmark(
        records,
        budgets=args.budgets,
        strategies=args.strategies,
        torsion_step_deg=args.torsion_step_deg,
        n_displaced=args.n_displaced,
        epochs=args.epochs,
        oracle_spec=spec,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "distillation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
