"""Measure CPU augmentation/render throughput and save a synthetic-only contact sheet."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rdkit import Chem

from ouroboros.data.render import RenderStyle, postprocess, render, sample_style
from ouroboros.data.rgroups import substitute

EXAMPLES = [
    ('R1', 'CCCOc1ccc(CC)cc1'), ('X', 'CCCOc1ccc(CC)cc1'),
    ('Ph', 'CCOc1ccccc1'), ('OMe', 'CCc1ccc(OC)cc1'),
    ('CO2Et', 'CCc1ccc(C(=O)OCC)cc1'), ('Boc', 'CCNC(=O)OC(C)(C)C'),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--out', default='benchmarks/augmentation-oct29')
    args = ap.parse_args(argv)
    if args.n < 1:
        ap.error('--n must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mols = [Chem.MolFromSmiles(s) for _, s in EXAMPLES]
    clean = render(mols[0], RenderStyle(), size=args.size, apply_postprocess=False).image
    styles = [sample_style(random.Random(i)) for i in range(args.n)]
    for style in styles:
        style.real_style = True
    t0 = time.perf_counter()
    for i, style in enumerate(styles):
        postprocess(clean, style, np.random.default_rng(i))
    post_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels = []
    for i, style in enumerate(styles):
        name, _ = EXAMPLES[i % len(EXAMPLES)]
        mol = substitute(mols[i % len(mols)], random.Random(i), name)
        result = render(mol, style, size=args.size, seed=i)
        labels.append(result.label)
    render_seconds = time.perf_counter() - t0
    # Each pair: clean drawing and the same labelled graph with real-style degradation.
    sheet = Image.new('RGB', (4*192, 3*220), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, _) in enumerate(EXAMPLES):
        mol = substitute(mols[i], random.Random(i), name)
        style = sample_style(random.Random(i + 10))
        base = render(mol, style, size=args.size, apply_postprocess=False)
        style.real_style = True
        style.noise_std, style.jpeg_quality, style.blur = .025, 50, .4
        aug = postprocess(base.image, style, np.random.default_rng(i + 10))
        x, y = (i % 2)*384, (i // 2)*220
        sheet.paste(base.image.resize((192, 192)), (x, y))
        sheet.paste(aug.resize((192, 192)), (x+192, y))
        draw.text((x+6, y+195), f'{name}: clean / augmented', fill='black')
    sheet.save(out / 'contact_sheet.png')
    stats = {'n': args.n, 'size': args.size, 'seed': 0,
             'postprocess_seconds': post_seconds,
             'postprocess_images_per_second': args.n/post_seconds,
             'substitute_render_augment_seconds': render_seconds,
             'substitute_render_augment_images_per_second': args.n/render_seconds,
             'labelled_outputs': sum('[' in s for s in labels),
             'threads': {key: os.environ.get(key) for key in
                         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
             'scope': 'one CPU process; PIL/numpy augmentation; includes randomized fonts and drawing styles'}
    (out / 'throughput.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
