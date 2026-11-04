"""Measure precision versus coverage on held-out n-best JSONL; tune on validation only."""
import argparse
import json
from pathlib import Path
from ouroboros.decode.confidence import selective_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--margins', type=float, nargs='+', required=True)
    ap.add_argument('--max-disagreement', type=float)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    result = selective_report([json.loads(s) for s in a.input.read_text().splitlines()
                               if s.strip()], a.margins, a.max_disagreement)
    output = json.dumps(result, indent=2)
    if a.out:
        a.out.write_text(output+'\n')
    print(output)


if __name__ == '__main__':
    main()
