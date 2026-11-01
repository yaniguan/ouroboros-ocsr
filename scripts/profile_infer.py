"""Profile the prediction path, including ensemble rescoring and candidate ranking."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ouroboros.data.loader import ShardDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts, however invoked
from predict_real import load_checkpoint, predict_batch  # noqa: E402
from profile_train import peak_memory, summarise_phase_times, synchronize  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-samples', type=int, default=100)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    if min(a.beam, a.batch_size, a.max_samples) < 1:
        ap.error('beam, batch-size and max-samples must be positive')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded = [load_checkpoint(p, dev) for p in a.ckpt]
    models = [m for m, _ in loaded]
    size = loaded[0][1]['data']['image_size']
    if any(c['data']['image_size'] != size for _, c in loaded):
        ap.error('ensemble checkpoints must share image_size')
    ds = ShardDataset(a.shards, a.split, image_size=size, max_samples=a.max_samples)
    if not len(ds):
        ap.error('no inference samples')
    results = {}
    with torch.inference_mode():
        for beam in dict.fromkeys((1, a.beam)):
            times = {'encoder': [], 'decode': [], 'total': []}
            counts = []
            encoder_seconds = 0.0
            starts = []
            def before(module, args):
                synchronize(dev)
                starts.append(time.perf_counter())
            def after(module, args, output):
                nonlocal encoder_seconds
                synchronize(dev)
                encoder_seconds += time.perf_counter() - starts.pop()
            hooks = []
            try:
                for model in models:
                    hooks.extend([model.encoder.register_forward_pre_hook(before),
                                  model.encoder.register_forward_hook(after)])
                if dev.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(dev)
                for start in range(0, len(ds), a.batch_size):
                    images = torch.stack([ds[i]['image'] for i in
                                          range(start, min(len(ds), start + a.batch_size))]).to(dev)
                    encoder_seconds = 0.0
                    synchronize(dev)
                    t0 = time.perf_counter()
                    if beam == 1 and len(models) == 1:
                        models[0].generate(images)
                    else:
                        predict_batch(models, images, beam, [], beam > 1)
                    synchronize(dev)
                    total = time.perf_counter() - t0
                    times['encoder'].append(encoder_seconds)
                    times['decode'].append(max(0.0, total - encoder_seconds))
                    times['total'].append(total)
                    counts.append(len(images))
                seconds = sum(times['total'])
                results[f'beam{beam}'] = dict(
                    samples=sum(counts), batches=len(counts), per_batch=summarise_phase_times(times),
                    ms_per_image={k: 1000 * sum(v) / sum(counts) for k, v in times.items()},
                    images_per_s=sum(counts) / seconds, peak_memory=peak_memory(dev))
            finally:
                for hook in hooks:
                    hook.remove()
    result = dict(name=a.name, device=str(dev), checkpoints=a.ckpt, variants=results,
                  decode_includes='candidate generation, ensemble rescoring and ranking; excludes encoder')
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'infer_{a.name}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
