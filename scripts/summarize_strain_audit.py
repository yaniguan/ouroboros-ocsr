"""Summarize the MACE strain-audit receipt with embedding failures counted apart.

    python scripts/summarize_strain_audit.py

Reads benchmarks/p1_chemveritas/mace_strain_audit.json and writes
benchmarks/p1_chemveritas/mace_strain_audit_summary.json. No new energies are computed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ouroboros.phys.strain import summarize_strain_audit  # noqa: E402

DEFAULT_DIR = ROOT / "benchmarks" / "p1_chemveritas"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_DIR / "mace_strain_audit.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR / "mace_strain_audit_summary.json")
    args = ap.parse_args(argv)
    summary = summarize_strain_audit(json.loads(args.receipt.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
