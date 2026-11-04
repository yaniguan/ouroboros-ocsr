"""Split extxyz frames and their manifest by molecule, keeping stereoisomers together."""
import argparse
import json
from pathlib import Path
from ase.io import read, write
from ouroboros.geometry.splits import split_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--fractions', type=float, nargs=3, default=[.8, .1, .1])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        ap.error('--out-dir must be empty to avoid overwriting a dataset')
    records = [json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    frames = read(a.frames, index=':', format='extxyz') if a.frames.stat().st_size else []
    result = split_dataset(frames, records, a.fractions, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result.items():
        path = a.out_dir / f'{name}.extxyz'
        if data['frames']:
            write(path, data['frames'], format='extxyz')
        elif name != 'skipped':
            path.write_text('')
        (a.out_dir / f'{name}.manifest.jsonl').write_text(''.join(
            json.dumps(rec)+'\n' for rec in data['manifest']))
    summary = dict(seed=a.seed, fractions=a.fractions,
                   frames={n: len(d['frames']) for n, d in result.items()})
    (a.out_dir / 'split_config.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    main()
