"""CPU inference API: molecule image -> SMILES with a confidence score and a low-confidence flag.

    from ouroboros.infer import OCSRPredictor
    p = OCSRPredictor()       # checkpoints/ouroboros-A.pt, int8 decoder, greedy, 1 thread
    p.predict("figure.png")   # path, bytes or PIL image

CLI: python -m ouroboros.infer figure.png [more.png ...]
HTTP (optional, needs fastapi + uvicorn): uvicorn ouroboros.infer:app -> POST /predict (raw bytes)

One model, greedy decoding, no test-time augmentation. The decoder's Linear layers use int8
dynamic quantization by default (int8=False keeps the full fp32 model).
Confidence is the mean per-token probability of the decoded SMILES (exp of the mean token
log-probability) under the same model; ``low_confidence`` is set when it is below
``threshold`` or the SMILES is not valid in RDKit.
"""

import argparse
import copy
import io
import json
import math
import os
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ouroboros.data.real import fit_to_canvas
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.eval.standardize import standardize
from ouroboros.model import build_model, load_model_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "ouroboros-A.pt"
# Picked on even rows of real-subset100 (release arm A); on the odd rows, accepted predictions
# were ~69% exact at ~23% coverage, vs 18% exact overall (scripts/bench_infer.py).
DEFAULT_THRESHOLD = 0.98
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000


class InputError(ValueError):
    """Bad request: the caller should fix the input."""


def load_checkpoint(
    path,
    device="cpu",
    *,
    allow_unsafe_pickle: bool = True,
    apply_int8: bool = True,
):
    """Load an inference/training checkpoint or exported HF folder -> (eval model, config).

    Tries safe ``weights_only=True`` loading first and falls back to ``weights_only=False``
    only when ``allow_unsafe_pickle=True`` (for training checkpoints with RNG state).
    Steerable encoders are swapped for their pure-PyTorch export.
    """
    from ouroboros.weights import resolve_checkpoint_path

    p = resolve_checkpoint_path(path)
    if p.is_dir():
        from ouroboros.hf import CONFIG_FILE, load_pretrained

        cfg = json.loads((p / CONFIG_FILE).read_text(encoding="utf-8"))
        return load_pretrained(p, device=device), cfg
    try:
        state = torch.load(p, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        if not allow_unsafe_pickle:
            raise
        state = torch.load(p, map_location="cpu", weights_only=False)
    model = build_model(state["config"], SmilesTokenizer(state["vocab"]))
    quant = state.get("quantization", {})
    weights = state["model"]
    has_fp16_weights = quant.get("method") == "fp16" or (
        bool(weights)
        and all(
            not v.is_floating_point() or v.dtype == torch.float16 for v in weights.values()
        )
        and any(v.dtype == torch.float16 for v in weights.values())
    )
    if has_fp16_weights:
        model = model.half()
    load_model_state(model, weights)
    model.to(device).eval()
    if hasattr(model.encoder, "export"):
        exported = model.encoder.export()
        if has_fp16_weights:
            exported = exported.half()
        model.encoder = exported.to(device)
    if (
        apply_int8
        and quant.get("apply_on_load")
        and torch.device(device).type == "cpu"
        and not has_fp16_weights
    ):
        model.decoder = torch.ao.quantization.quantize_dynamic(
            model.decoder, {torch.nn.Linear}, dtype=torch.qint8
        )
    return model, state["config"]


def open_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        if image.width * image.height > MAX_PIXELS:
            raise InputError(f"image larger than {MAX_PIXELS} pixels")
        return image
    if isinstance(image, (str, Path)):
        p = Path(image)
        if p.stat().st_size > MAX_BYTES:
            raise InputError(f"image larger than {MAX_BYTES} bytes")
        image = p.read_bytes()
    if not isinstance(image, (bytes, bytearray)) or not image:
        raise InputError("image must be a nonempty path, bytes or PIL image")
    if len(image) > MAX_BYTES:
        raise InputError(f"image larger than {MAX_BYTES} bytes")
    try:
        img = Image.open(io.BytesIO(image))
        if img.width * img.height > MAX_PIXELS:
            raise InputError(f"image larger than {MAX_PIXELS} pixels")
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise InputError(f"could not decode image: {err}") from err
    return img


class OCSRPredictor:
    """Load once, then call predict() per request."""

    def __init__(self, ckpt=DEFAULT_CKPT, *, threads: int = 1, int8: bool = True,
                 threshold: float = DEFAULT_THRESHOLD, model=None, image_size: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        if model is None:
            model, cfg = load_checkpoint(ckpt, apply_int8=int8)
            image_size = cfg.get("data", {}).get("image_size", 384)
        elif int8:
            model = copy.deepcopy(model).eval()
        else:
            model = model.eval()
        self.model = model
        param = next(self.model.parameters())
        is_already_int8 = any(
            isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
            for m in self.model.decoder.modules()
        )
        can_int8 = param.device.type == "cpu" and param.dtype == torch.float32
        if int8 and can_int8 and not is_already_int8:
            self.model.decoder = torch.ao.quantization.quantize_dynamic(
                self.model.decoder, {torch.nn.Linear}, dtype=torch.qint8)
            is_already_int8 = True
        self.image_size = image_size or 384
        self.threshold = threshold
        self.name = Path(ckpt).name + ("+int8" if is_already_int8 else "")
        self.predict(Image.new("L", (64, 64), 255))  # warm up

    def to_tensor(self, img: Image.Image, ink_crop: bool = False) -> torch.Tensor:
        canvas = fit_to_canvas(img, self.image_size, ink_crop=ink_crop)
        return torch.from_numpy(1.0 - np.asarray(canvas, dtype=np.float32) / 255.0)[None, None]

    @torch.inference_mode()
    def predict(self, image, *, ink_crop: bool = False) -> dict:
        t0 = time.perf_counter()
        img = open_image(image)
        param = next(self.model.parameters())
        x = self.to_tensor(img, ink_crop=ink_crop).to(device=param.device, dtype=param.dtype)
        t1 = time.perf_counter()
        m, tok = self.model, self.model.tokenizer
        enc = m.encoder(x)
        ids = m.decoder.greedy(enc.tokens, tok.bos_id, tok.eos_id,
                               padding_mask=enc.padding_mask)[0].tolist()
        if tok.eos_id in ids:
            ids = ids[:ids.index(tok.eos_id) + 1]
        smiles = tok.decode(ids)
        t2 = time.perf_counter()
        seq = torch.tensor([[tok.bos_id, *ids]], dtype=torch.long, device=param.device)
        logp = float(m.decoder.score(seq, enc.tokens, enc.padding_mask)[0]) / max(len(ids), 1)
        confidence = math.exp(logp)
        std = standardize(smiles)
        reasons = []
        if not std.valid:
            reasons.append("invalid_smiles")
        if confidence < self.threshold:
            reasons.append("low_token_probability")
        return {
            "smiles": smiles,
            "canonical_smiles": std.smiles,
            "valid": std.valid,
            "confidence": confidence,
            "low_confidence": bool(reasons),
            "reasons": reasons,
            "model": self.name,
            "decode": "greedy",
            "timing_ms": {"preprocess": (t1 - t0) * 1e3, "decode": (t2 - t1) * 1e3,
                          "total": (time.perf_counter() - t0) * 1e3},
        }


def create_app(predictor: OCSRPredictor | None = None):
    """Small FastAPI app: POST /predict with the raw image bytes as the request body."""
    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Ouroboros OCSR CPU API")
    state = {"predictor": predictor}
    lock = threading.Lock()  # one request at a time: parallel torch calls fight over cores

    def _sync_predict(payload: bytes) -> dict:
        with lock:
            if state["predictor"] is None:
                state["predictor"] = OCSRPredictor(os.environ.get("OUROBOROS_CKPT", DEFAULT_CKPT))
            return state["predictor"].predict(payload)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/predict")
    async def predict(request: Request):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"image larger than {MAX_BYTES} bytes")
        buf = bytearray()
        async for chunk in request.stream():
            buf.extend(chunk)
            if len(buf) > MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"image larger than {MAX_BYTES} bytes")
        try:
            return await run_in_threadpool(_sync_predict, bytes(buf))
        except InputError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

    return app


def __getattr__(name):  # `uvicorn ouroboros.infer:app` builds the app lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fp32", action="store_true", help="keep the decoder in fp32 (no int8)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--ink-crop", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.ckpt).exists():
        raise SystemExit(f"missing {args.ckpt}; run: python -m ouroboros.weights fetch --arm A")
    p = OCSRPredictor(args.ckpt, threads=args.threads, int8=not args.fp32, threshold=args.threshold)
    for path in args.images:
        try:
            res = p.predict(path, ink_crop=args.ink_crop)
        except InputError as err:
            res = {"error": str(err)}
        print(json.dumps({"image": path, **res}))


if __name__ == "__main__":
    main()
