"""Capacity probes and real-shard resume comparisons using the current model/trainer."""

from __future__ import annotations

import copy
import gc
import json
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from ouroboros.data.loader import rotate_batch
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.model import build_model, count_params
from ouroboros.provenance import atomic_json, implementation_identity
from ouroboros.train.trainer import Trainer, build_optimizer, build_scheduler


def validate_device(device, precision):
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda") or precision not in ("fp32", "bf16"):
        raise ValueError("Use cpu/cuda and fp32/bf16")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; CPU smoke checks require --device cpu --precision fp32"
            )
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support bf16")
    elif precision != "fp32":
        raise ValueError("CPU smoke checks require fp32")
    return dev


def environment(device) -> dict:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    return {
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout.strip()),
        "implementation": implementation_identity(root / "ouroboros"),
        "versions": {name: version(name) for name in ("torch", "rdkit", "escnn", "numpy")},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
    }


def capacity(
    cfg, output, steps=20, warmup=3, batch=32, sequence_length=None, device="cuda", precision="bf16"
):
    dev = validate_device(device, precision)
    cfg = copy.deepcopy(cfg)
    length = sequence_length or cfg["model"]["decoder"].get("max_len", 128)
    if (
        batch < 1
        or warmup < 0
        or steps <= warmup
        or not 2 <= length <= (cfg["model"]["decoder"].get("max_len", 128))
    ):
        raise ValueError("Need positive batch, steps > warmup >= 0 and a supported sequence length")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg["train"].update(
        device=str(dev), amp="bf16" if precision == "bf16" else "none", batch_size=batch
    )
    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tok = SmilesTokenizer.load(cfg["data"]["vocab"])
    model = build_model(cfg, tok).to(dev).train()
    opt = build_optimizer(model, cfg["train"])
    sched = build_scheduler(opt, cfg["train"])
    size = cfg["data"]["image_size"]
    images = torch.rand(batch, 1, size, size, device=dev)
    ids = torch.randint(4, len(tok), (batch, length), device=dev)
    ids[:, 0], ids[:, -1] = tok.bos_id, tok.eos_id
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    times, losses = [], []
    for step in range(steps):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        started = time.perf_counter()
        x, angles = images, None
        if cfg["train"].get("rotation_aug", False):
            generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
            angles = torch.rand(batch, generator=generator) * 360
            x = rotate_batch(images, angles)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            loss = model.loss(x, ids, cfg["train"].get("label_smoothing", 0.0))
            if hasattr(model.encoder, "prior_loss"):
                loss = loss + model.encoder.prior_loss(angles)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["train"].get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        sched.step()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
    report = {
        "scope": "synthetic optimizer capacity; excludes data I/O, evaluation and accuracy",
        "cpu_smoke_only": dev.type == "cpu",
        "environment": environment(dev),
        "config": cfg,
        "precision": precision,
        "batch": batch,
        "sequence_length": length,
        "steps": steps,
        "warmup_steps": warmup,
        "parameters": count_params(model),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev)
        if dev.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None,
        "seconds_per_step": times,
        "losses": losses,
        "warm_images_per_second": batch * (steps - warmup) / sum(times[warmup:]),
    }
    atomic_json(output, report)
    return report


def _losses(path):
    return {
        row["step"]: row["loss"]
        for line in (path / "log.jsonl").read_text().splitlines()
        if "loss" in (row := json.loads(line))
    }


def resume_comparison(
    cfg,
    output,
    checkpoint_step=25,
    compared_steps=100,
    kill_after=3,
    tolerance=0.01,
    device="cuda",
    precision="bf16",
):
    dev = validate_device(device, precision)
    if checkpoint_step < 2 or compared_steps <= kill_after or not 0 < kill_after < checkpoint_step:
        raise ValueError("Need 0 < kill_after < checkpoint_step and compared_steps > kill_after")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    cfg = copy.deepcopy(cfg)
    cfg["train"].update(
        device=str(dev),
        amp="bf16" if precision == "bf16" else "none",
        steps=checkpoint_step + compared_steps,
        ckpt_every=checkpoint_step,
        log_every=1,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    full, cut = output / "uninterrupted", output / "resumed"
    started = time.perf_counter()
    original_identity = None
    for directory, stop_at in ((full, None), (cut, checkpoint_step + kill_after), (cut, None)):
        trainer = Trainer(cfg, directory)
        if original_identity is None:
            original_identity = trainer.identity
        elif trainer.identity != original_identity:
            raise ValueError("Dataset changed between benchmark runs")
        if directory == cut and stop_at is None and trainer.step != checkpoint_step:
            raise AssertionError("Did not resume at the intended checkpoint")
        trainer.fit(stop_at=stop_at)
        data_identity = trainer.identity
        del trainer
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    a, b = _losses(full), _losses(cut)
    steps = list(range(checkpoint_step + 1, checkpoint_step + compared_steps + 1))
    differences = [abs(a[s] - b[s]) / max(abs(a[s]), 1e-8) for s in steps]
    report = {
        "scope": "resume comparison on configured dataset shards; no accuracy claim",
        "cpu_smoke_only": dev.type == "cpu",
        "environment": environment(dev),
        "config": cfg,
        "data_identity": data_identity,
        "precision": precision,
        "checkpoint_step": checkpoint_step,
        "compared_steps": compared_steps,
        "mean_relative_loss_difference": float(np.mean(differences)),
        "max_relative_loss_difference": float(np.max(differences)),
        "tolerance": tolerance,
        "passed": bool(np.isfinite(differences).all() and np.mean(differences) <= tolerance),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(output / "report.json", report)
    if not report["passed"]:
        raise AssertionError(f"Resume difference exceeded {tolerance}: {output / 'report.json'}")
    return report
