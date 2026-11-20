"""Batched OCSR verifier and GRPO reward hook for shared-GPU RL loops.

    from lab.grpo_verifier import make_verifier
    verify = make_verifier(ckpt="checkpoints/ouroboros-A.pt", max_batch_size=16)
    rows = verify(["fig1.png", "fig2.png"])  # one dict per input; never drops rows

Also exposes ``lab.grpo_verifier:make_verifier`` as a ``module:function`` factory for
``scripts/profile_gpu.py`` (wraps preprocess, encode, decode, and score in NVTX ranges).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lab.molecule_context import from_prediction
from ouroboros.data.real import fit_to_canvas
from ouroboros.eval.standardize import standardize
from ouroboros.infer import DEFAULT_CKPT, DEFAULT_THRESHOLD, InputError, load_checkpoint, open_image


@contextlib.contextmanager
def nvtx_range(name: str, *, enabled: bool = True):
    use_nvtx = enabled and torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")
    if use_nvtx:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if use_nvtx:
            torch.cuda.nvtx.range_pop()


class OCSRVerifier:
    """Small, fixed-budget batched OCSR verifier safe alongside a sleeping rollout engine."""

    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        *,
        device: str | torch.device | None = None,
        dtype: str = "auto",
        int8_cpu: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        max_batch_size: int = 16,
        nvtx: bool = True,
        model=None,
        image_size: int | None = None,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if model is None:
            model, cfg = load_checkpoint(ckpt, device=self.device)
            image_size = cfg.get("data", {}).get("image_size", 384)
        else:
            model = model.to(self.device).eval()
            if hasattr(model.encoder, "export"):
                model.encoder = model.encoder.export().to(self.device)

        if dtype == "auto":
            use_fp16 = self.device.type == "cuda"
        elif dtype in ("fp16", "float16"):
            use_fp16 = True
        elif dtype in ("fp32", "float32"):
            use_fp16 = False
        else:
            raise ValueError(f"unknown dtype {dtype!r}; choose auto, fp16, or fp32")

        if int8_cpu and self.device.type == "cpu" and not use_fp16:
            model.decoder = torch.ao.quantization.quantize_dynamic(
                model.decoder, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif use_fp16:
            model = model.half()

        self.model = model
        self.input_dtype = torch.float16 if use_fp16 else torch.float32
        self.image_size = image_size or 384
        self.threshold = threshold
        self.max_batch_size = max_batch_size
        self.nvtx = nvtx

    def _to_tensor(self, img: Image.Image, *, ink_crop: bool = False) -> torch.Tensor:
        canvas = fit_to_canvas(img, self.image_size, ink_crop=ink_crop)
        arr = 1.0 - np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None]

    @torch.inference_mode()
    def verify_batch(
        self,
        images: Sequence[str | bytes | bytearray | Path | Image.Image],
        *,
        ids: Sequence[str] | None = None,
        ink_crop: bool = False,
    ) -> list[dict]:
        """Decode images in bounded mini-batches; never drop bad or low-confidence rows."""
        n = len(images)
        if ids is not None and len(ids) != n:
            raise ValueError("ids and images must have the same length")
        if n == 0:
            return []

        results: list[dict | None] = [None] * n
        valid_indices: list[int] = []
        tensors: list[torch.Tensor] = []

        with nvtx_range("ocsr_verifier/preprocess", enabled=self.nvtx):
            for idx, raw in enumerate(images):
                row_id = str(ids[idx]) if ids is not None else f"img_{idx:04d}"
                try:
                    pil = open_image(raw)
                    tensors.append(self._to_tensor(pil, ink_crop=ink_crop))
                    valid_indices.append(idx)
                except InputError as err:
                    results[idx] = {
                        "id": row_id,
                        "smiles": None,
                        "pred_raw": "",
                        "confidence": 0.0,
                        "status": "invalid",
                        "flagged": True,
                        "usable": False,
                        "reasons": [f"input_error:{err}"],
                    }

        m, tok = self.model, self.model.tokenizer
        for start in range(0, len(valid_indices), self.max_batch_size):
            sub_idx = valid_indices[start : start + self.max_batch_size]
            batch = torch.stack(tensors[start : start + self.max_batch_size]).to(
                device=self.device, dtype=self.input_dtype
            )
            with nvtx_range("ocsr_verifier/encode", enabled=self.nvtx):
                enc = m.encoder(batch)
            with nvtx_range("ocsr_verifier/decode", enabled=self.nvtx):
                out_ids = m.decoder.greedy(
                    enc.tokens, tok.bos_id, tok.eos_id, padding_mask=enc.padding_mask
                )

            trimmed: list[list[int]] = []
            decoded_smiles: list[str] = []
            for row_tensor in out_ids:
                seq = row_tensor.tolist()
                if tok.eos_id in seq:
                    seq = seq[: seq.index(tok.eos_id) + 1]
                trimmed.append(seq)
                decoded_smiles.append(tok.decode(seq))

            max_t = max((len(seq) for seq in trimmed), default=1) + 1
            pad = tok.pad_id
            score_in = torch.tensor(
                [[tok.bos_id, *seq, *([pad] * (max_t - 1 - len(seq)))] for seq in trimmed],
                dtype=torch.long,
                device=self.device,
            )
            with nvtx_range("ocsr_verifier/score", enabled=self.nvtx):
                sum_logp = m.decoder.score(
                    score_in, enc.tokens, enc.padding_mask, pad_id=pad
                ).tolist()

            for local_pos, orig_idx in enumerate(sub_idx):
                row_id = str(ids[orig_idx]) if ids is not None else f"img_{orig_idx:04d}"
                raw_smi = decoded_smiles[local_pos]
                tok_len = max(len(trimmed[local_pos]), 1)
                conf = math.exp(float(sum_logp[local_pos]) / tok_len)
                low_tok = conf < self.threshold
                ctx = from_prediction(
                    {
                        "id": row_id,
                        "pred": raw_smi,
                        "accepted": "false" if low_tok else "true",
                        "rejection_reason": "low_token_probability" if low_tok else "",
                    }
                )
                results[orig_idx] = {
                    "id": row_id,
                    "smiles": ctx.smiles,
                    "pred_raw": raw_smi,
                    "inchikey": ctx.inchikey,
                    "formula": ctx.formula,
                    "confidence": round(conf, 6),
                    "status": ctx.status,
                    "flagged": ctx.status != "ok",
                    "usable": ctx.usable,
                    "reasons": ctx.reasons,
                }

        return [r for r in results if r is not None]

    def reward_batch(
        self,
        images: Sequence[str | bytes | bytearray | Path | Image.Image],
        proposed_smiles: Sequence[str],
    ) -> list[dict]:
        """Code-checked GRPO reward: 1.0 on exact stereo match if ok; flagged rows abstain."""
        if len(images) != len(proposed_smiles):
            raise ValueError("images and proposed_smiles must have the same length")
        verified = self.verify_batch(images)
        out: list[dict] = []
        for row, prop in zip(verified, proposed_smiles, strict=True):
            if row["flagged"] or not row["smiles"]:
                out.append({**row, "reward": 0.0, "abstained": True, "matched": False})
                continue
            prop_std = standardize(prop)
            matched = bool(prop_std.valid and prop_std.smiles == row["smiles"])
            out.append(
                {
                    **row,
                    "reward": 1.0 if matched else 0.0,
                    "abstained": False,
                    "matched": matched,
                }
            )
        return out


def make_verifier(
    ckpt: str | Path = DEFAULT_CKPT,
    *,
    device: str | None = None,
    dtype: str = "auto",
    int8_cpu: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    max_batch_size: int = 16,
    model=None,
    image_size: int | None = None,
) -> Callable[[Sequence], list[dict]]:
    """Factory ``lab.grpo_verifier:make_verifier`` returning a batched callable."""
    verifier = OCSRVerifier(
        ckpt=ckpt,
        device=device,
        dtype=dtype,
        int8_cpu=int8_cpu,
        threshold=threshold,
        max_batch_size=max_batch_size,
        model=model,
        image_size=image_size,
    )
    return verifier.verify_batch


def benchmark_batch_sizes(
    verifier: OCSRVerifier,
    batch_sizes: Sequence[int] = (1, 4, 8, 16, 32),
    *,
    repeats: int = 2,
) -> list[dict]:
    """Measure img/s and peak CUDA memory across batch sizes on synthetic blank canvases."""
    blank = Image.new("L", (verifier.image_size, verifier.image_size), 245)
    # Warm up once
    _ = verifier.verify_batch([blank])
    if verifier.device.type == "cuda":
        torch.cuda.synchronize(verifier.device)

    rows: list[dict] = []
    for bs in batch_sizes:
        verifier.max_batch_size = bs
        inputs = [blank for _ in range(bs)]
        if verifier.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(verifier.device)
            torch.cuda.synchronize(verifier.device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = verifier.verify_batch(inputs)
        if verifier.device.type == "cuda":
            torch.cuda.synchronize(verifier.device)
        elapsed = time.perf_counter() - t0
        total_imgs = bs * repeats
        peak_mb = (
            round(torch.cuda.max_memory_allocated(verifier.device) / (1024 * 1024), 2)
            if verifier.device.type == "cuda"
            else None
        )
        rows.append(
            {
                "batch_size": bs,
                "device": str(verifier.device),
                "img_per_s": round(total_imgs / max(elapsed, 1e-9), 2),
                "ms_per_image": round(1000.0 * elapsed / max(total_imgs, 1), 2),
                "peak_cuda_mb": peak_mb,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*", help="images to verify")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--benchmark-batches", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)

    verifier = OCSRVerifier(
        ckpt=a.ckpt, max_batch_size=a.max_batch_size, threshold=a.threshold
    )
    if a.benchmark_batches is not None:
        sizes = a.benchmark_batches or [1, 4, 8, 16, 32]
        print(json.dumps(benchmark_batch_sizes(verifier, sizes), indent=2))
        return 0
    out = verifier.verify_batch(a.images)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
