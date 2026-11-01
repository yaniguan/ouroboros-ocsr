"""Profile real training batches without modifying a training run directory."""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import yaml


def summarise_phase_times(times):
    """Summarise seconds as milliseconds; p90 uses linear interpolation."""
    result = {}
    for name, values in times.items():
        values = sorted(v * 1000 for v in values)
        if not values:
            result[name] = None
            continue
        p = (len(values) - 1) * 0.9
        lo = int(p)
        p90 = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (p - lo)
        result[name] = dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values), p90_ms=p90)
    return result


def parse_nvidia_smi_csv(text):
    """Ignore repeated headers, unsupported metrics and partial/invalid samples."""
    gpu, memory = [], []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 3:
            continue
        for values, field in ((gpu, row[1]), (memory, row[2])):
            try:
                values.append(float(field.strip().rstrip('%').strip()))
            except ValueError:
                pass
    def stats(values):
        return dict(mean=statistics.mean(values), max=max(values)) if values else None
    return {'gpu': stats(gpu), 'memory': stats(memory)} if gpu or memory else None


def start_nvidia_smi(path):
    handle = Path(path).open('a')
    try:
        proc = subprocess.Popen([
            'nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
            '--format=csv', '-lms', '200'], stdout=handle, stderr=subprocess.DEVNULL)
    except OSError:
        handle.close()
        return None
    return proc, handle


def stop_nvidia_smi(sampler):
    if sampler is not None:
        proc, handle = sampler
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        finally:
            handle.close()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def peak_memory(device):
    return ({'allocated_bytes': torch.cuda.max_memory_allocated(device),
             'reserved_bytes': torch.cuda.max_memory_reserved(device)}
            if device.type == 'cuda' else None)


def main(argv=None):
    # Imported lazily so the notebook kernel (which lacks the ouroboros venv) can load the
    # nvidia-smi helpers from this file without importing the package.
    from ouroboros.train.trainer import Trainer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', default='train')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--trace-steps', type=int, default=5)
    a = ap.parse_args(argv)
    if a.steps < 1 or a.warmup < 0 or a.trace_steps < 1:
        ap.error('steps and trace-steps must be positive; warmup must be nonnegative')
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    with tempfile.TemporaryDirectory(prefix='ouroboros-profile-') as temporary:
        tr = Trainer(cfg, temporary)
        loader = iter(tr.loader())
        times = {k: [] for k in ('data_loading', 'host_to_device', 'forward', 'backward', 'optimizer')}
        images_count = tokens_count = 0

        def step(record=False):
            nonlocal loader, images_count, tokens_count
            def timed(name, fn):
                synchronize(tr.dev)
                start = time.perf_counter()
                value = fn()
                synchronize(tr.dev)
                if record:
                    times[name].append(time.perf_counter() - start)
                return value
            def fetch():
                nonlocal loader
                try:
                    return next(loader)
                except StopIteration:
                    loader = iter(tr.loader())
                    return next(loader)
            batch = timed('data_loading', fetch)
            images, ids = timed('host_to_device', lambda: (
                batch['image'].to(tr.dev, non_blocking=True), batch['ids'].to(tr.dev, non_blocking=True)))
            def forward():
                augmented, angles = tr.augment_with_angles(images)
                with torch.autocast(tr.dev.type, dtype=torch.bfloat16,
                                    enabled=cfg['train'].get('amp', 'bf16') == 'bf16'):
                    loss = tr.model.loss(augmented, ids, cfg['train'].get('label_smoothing', 0.0))
                    if hasattr(tr.model.encoder, 'prior_loss'):
                        loss = loss + tr.model.encoder.prior_loss(angles)
                return loss
            loss = timed('forward', forward)
            def backward():
                tr.opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg['train'].get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_(tr.model.parameters(), cfg['train']['grad_clip'])
            timed('backward', backward)
            def optimize():
                tr.opt.step()
                tr.sched.step()
                tr.step += 1
            timed('optimizer', optimize)
            if record:
                images_count += len(images)
                tokens_count += int((batch['ids'][:, 1:] != tr.tok.pad_id).sum())
        tr.model.train()
        for _ in range(a.warmup):
            step()
        if tr.dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(tr.dev)
        csv_path = out / f'nvidia_smi_{a.name}.csv'
        sampler = start_nvidia_smi(csv_path) if tr.dev.type == 'cuda' else None
        try:
            for _ in range(a.steps):
                step(True)
        finally:
            stop_nvidia_smi(sampler)
        memory = peak_memory(tr.dev)
        seconds = sum(sum(v) for v in times.values())
        activities = [torch.profiler.ProfilerActivity.CPU]
        if tr.dev.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            for _ in range(a.trace_steps):
                step()
                prof.step()
        prof.export_chrome_trace(str(out / f'trace_{a.name}.json'))
        table = prof.key_averages().table(
            sort_by='cuda_time_total' if tr.dev.type == 'cuda' else 'cpu_time_total', row_limit=40)
        (out / f'keyavg_{a.name}.txt').write_text(table)
        result = dict(name=a.name, device=str(tr.dev), steps=a.steps, warmup=a.warmup,
                      trace_steps=a.trace_steps, phases=summarise_phase_times(times),
                      img_per_s=images_count / seconds, tokens_per_s=tokens_count / seconds,
                      target_tokens=tokens_count, peak_memory=memory,
                      gpu_util=parse_nvidia_smi_csv(csv_path.read_text()) if sampler else None)
        (out / f'train_{a.name}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return result


if __name__ == '__main__':
    main()
