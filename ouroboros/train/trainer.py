"""Training loop with exact, resumable state (for Colab disconnects).

Everything that influences the next step is restored from the checkpoint:
  * model, optimizer, LR scheduler, AMP-free (bf16 autocast needs no scaler) step counter;
  * data position: samplers are pure functions of (seed, position) -> position = step * batch;
  * augmentation randomness: rotation angles are drawn from a generator seeded by (seed, step);
  * dropout randomness: global torch (CPU + CUDA), numpy and python RNG states;
  * DataLoader worker seeding uses a private generator, so re-creating the loader on resume does
    not consume the global torch RNG.

Run directory layout: ``{out_dir}/config.yaml``, ``ckpt/last.pt``, ``log.jsonl`` (one JSON per
logged step; on resume, lines after the checkpoint step are dropped so logs never duplicate).
"""

from __future__ import annotations

import functools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from ouroboros.data.loader import ShardDataset, collate, rotate_batch
from ouroboros.data.mixture import MixtureSampler
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.model import OCSRModel, build_model, count_params, load_model_state


def _device(cfg: dict) -> torch.device:
    want = cfg["train"].get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    if step < warmup:
        return (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


def rng_state() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(path: Path, state: dict) -> float:
    """Atomic save (write temp file, then rename). Returns seconds taken."""
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return time.perf_counter() - t0


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_datasets(cfg: dict, tok: SmilesTokenizer):
    d = cfg["data"]
    excl = None
    if d.get("exclude_keys"):
        excl = frozenset(k[:14] for k in Path(d["exclude_keys"]).read_text().split())
    synth = ShardDataset(
        d["synthetic_root"],
        "train",
        tokenizer=tok,
        max_samples=d.get("synthetic_max_samples"),
        image_size=d["image_size"],
        exclude_key14=excl,
    )
    reals = [
        ShardDataset(r, "train", tokenizer=tok, image_size=d["image_size"], exclude_key14=excl)
        for r in d.get("real_roots", []) or []
    ]
    real = ConcatDataset(reals) if reals else None
    return synth, real


class Trainer:
    def __init__(self, cfg: dict, out_dir: str | Path):
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.dev = _device(cfg)
        t = cfg["train"]
        self.seed = int(cfg.get("seed", 0))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tok = SmilesTokenizer.load(cfg["data"]["vocab"])
        self.model: OCSRModel = build_model(cfg, self.tok).to(self.dev)
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (no_decay if p.ndim < 2 or n.endswith(".pos") or "norm" in n else decay).append(p)
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": t.get("weight_decay", 0.05)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=t["lr"],
            betas=(0.9, 0.98),
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_lambda(s, t.get("warmup", 1000), t["steps"])
        )
        self.step = 0
        self.synth, self.real = build_datasets(cfg, self.tok)
        f = float(cfg["data"].get("real_fraction", 0.0))
        self.sampler = MixtureSampler(
            len(self.synth), len(self.real) if self.real else 0, f, t["batch_size"], seed=self.seed
        )
        self.dataset = ConcatDataset([self.synth, self.real]) if self.real else self.synth
        self.ckpt_path = self.out / "ckpt" / "last.pt"
        if self.ckpt_path.exists():
            self._resume()

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "step": self.step,
            "sampler_position": self.step * self.cfg["train"]["batch_size"],
            "rng": rng_state(),
            "config": self.cfg,
            "vocab": self.tok.itos,
        }

    def _resume(self) -> None:
        st = load_checkpoint(self.ckpt_path)
        load_model_state(self.model, st["model"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.step = st["step"]
        set_rng_state(st["rng"])
        log = self.out / "log.jsonl"
        if log.exists():  # drop log lines written after the checkpoint by the interrupted run
            keep = [
                ln for ln in log.read_text().splitlines() if json.loads(ln)["step"] <= self.step
            ]
            log.write_text("".join(ln + "\n" for ln in keep))
        print(f"resumed from {self.ckpt_path} at step {self.step}")

    def save(self) -> float:
        return save_checkpoint(self.ckpt_path, self.state())

    # ------------------------------------------------------------------ loop
    def loader(self) -> DataLoader:
        self.sampler.set_position(self.step * self.cfg["train"]["batch_size"])
        nw = int(self.cfg["data"].get("num_workers", 4))
        return DataLoader(
            self.dataset,
            batch_size=self.cfg["train"]["batch_size"],
            sampler=self.sampler,
            num_workers=nw,
            collate_fn=functools.partial(collate, pad_id=self.tok.pad_id),
            pin_memory=self.dev.type == "cuda",
            prefetch_factor=4 if nw > 0 else None,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(12345),  # private: keeps global RNG untouched
        )

    def augment(self, images: torch.Tensor) -> torch.Tensor:
        """Arm B / C+: uniform random rotation in [0, 360) (angles depend only on seed, step)."""
        if not self.cfg["train"].get("rotation_aug", False):
            return images
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.step)
        angles = torch.rand(images.shape[0], generator=g) * 360.0
        return rotate_batch(images, angles)

    def train_step(self, batch: dict) -> float:
        t = self.cfg["train"]
        self.model.train()
        images = self.augment(batch["image"].to(self.dev, non_blocking=True))
        ids = batch["ids"].to(self.dev, non_blocking=True)
        amp = t.get("amp", "bf16") == "bf16"
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=amp):
            loss = self.model.loss(images, ids, t.get("label_smoothing", 0.0))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if t.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["grad_clip"])
        self.opt.step()
        self.sched.step()
        self.step += 1
        return float(loss.detach())

    def log(self, rec: dict) -> None:
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def fit(self, stop_at: int | None = None, callback=None) -> None:
        """Train until ``train.steps`` (or ``stop_at``, used to simulate a disconnect)."""
        t = self.cfg["train"]
        end = min(t["steps"], stop_at) if stop_at is not None else t["steps"]
        if self.step == 0:
            self.log(
                {
                    "step": 0,
                    "event": "start",
                    "params": count_params(self.model),
                    "params_encoder": count_params(self.model.encoder),
                    "params_decoder": count_params(self.model.decoder),
                    "n_synth": len(self.synth),
                    "n_real": len(self.real) if self.real else 0,
                }
            )
        it = iter(self.loader())
        t0, n_img = time.perf_counter(), 0
        while self.step < end:
            batch = next(it)
            loss = self.train_step(batch)
            n_img += batch["image"].shape[0]
            if self.step % t.get("log_every", 50) == 0 or self.step == end:
                dt = time.perf_counter() - t0
                self.log(
                    {
                        "step": self.step,
                        "loss": loss,
                        "lr": self.sched.get_last_lr()[0],
                        "img_per_s": n_img / max(dt, 1e-9),
                    }
                )
                t0, n_img = time.perf_counter(), 0
            if callback is not None:
                callback(self, loss)
            if self.step % t.get("ckpt_every", 1000) == 0 or self.step == t["steps"]:
                self.log({"step": self.step, "event": "ckpt", "save_sec": self.save()})
