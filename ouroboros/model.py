"""Encoder + shared decoder, built from a config dict.

Only ``cfg["encoder"]`` differs between arms; ``cfg["decoder"]`` must be identical (enforced by
the sweep generator, and the decoder class has no arm-specific options).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ouroboros.decode.decoder import DecoderConfig, SmilesDecoder
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.encoder.base import ImageEncoder

ENCODERS = {}


def register_encoder(name: str):
    def deco(fn):
        ENCODERS[name] = fn
        return fn

    return deco


@register_encoder("baseline")
def _baseline(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    from ouroboros.encoder.baseline import BaselineConfig, BaselineEncoder

    kw = {k: v for k, v in enc_cfg.items() if k != "type"}
    return BaselineEncoder(BaselineConfig(d_model=d_model, image_size=image_size, **kw))


def build_encoder(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    name = enc_cfg["type"]
    if name not in ENCODERS:
        # lazily import optional encoders so that their registration runs
        import importlib

        for mod in ("ouroboros.encoder.steerable", "ouroboros.encoder.equiv_attention"):
            try:
                importlib.import_module(mod)
            except ModuleNotFoundError:
                pass
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder type {name!r}; known: {sorted(ENCODERS)}")
    return ENCODERS[name](enc_cfg, d_model, image_size)


class OCSRModel(nn.Module):
    def __init__(self, encoder: ImageEncoder, decoder: SmilesDecoder, tokenizer: SmilesTokenizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def loss(self, images: torch.Tensor, ids: torch.Tensor, label_smoothing: float = 0.0):
        enc = self.encoder(images)
        logits = self.decoder(ids[:, :-1], enc.tokens, enc.padding_mask)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=label_smoothing,
        )

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int | None = None) -> list[str]:
        enc = self.encoder(images)
        out = self.decoder.greedy(
            enc.tokens,
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
            max_len=max_len,
            padding_mask=enc.padding_mask,
        )
        return [self.tokenizer.decode(row.tolist()) for row in out]


def build_model(cfg: dict, tokenizer: SmilesTokenizer) -> OCSRModel:
    m = cfg["model"]
    dec = DecoderConfig(vocab_size=len(tokenizer), **m.get("decoder", {}))
    encoder = build_encoder(m["encoder"], dec.d_model, cfg["data"]["image_size"])
    return OCSRModel(encoder, SmilesDecoder(dec), tokenizer)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
