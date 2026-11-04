"""Report how often the top candidates include the correct molecule, by subset."""
import argparse
import json
from pathlib import Path
from ouroboros.eval.nbest import diagnose


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--k', nargs='+', type=int, default=[1, 3, 5])
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    report = diagnose([json.loads(s) for s in a.input.read_text().splitlines() if s.strip()], a.k)
    output = json.dumps(report, indent=2)
    if a.out:
        a.out.write_text(output + '\n')
    print(output)


if __name__ == '__main__':
    main()
