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

        for mod in (
            "ouroboros.encoder.steerable",
            "ouroboros.encoder.equiv_attention",
            "ouroboros.encoder.canonicalize",
        ):
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

    @torch.no_grad()
    def generate_nbest(
        self, images: torch.Tensor, beam_size: int = 5, max_len: int | None = None
    ) -> list[list[tuple[str, float]]]:
        """Beam search: per image, up to ``beam_size`` ``(smiles, sum_logprob)``, best first
        (by length-normalized log probability)."""
        enc = self.encoder(images)
        beams = self.decoder.beam(
            enc.tokens,
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
            beam_size=beam_size,
            max_len=max_len,
            padding_mask=enc.padding_mask,
        )
        return [[(self.tokenizer.decode(ids), lp) for ids, lp in hyps] for hyps in beams]

    def _score_ids(self, smiles: str) -> list[int] | None:
        """Token ids (BOS ... EOS) of a SMILES the decoder could emit, else None."""
        try:
            ids = self.tokenizer.encode(smiles)
        except ValueError:
            return None
        if self.tokenizer.unk_id in ids or len(ids) > self.decoder.c.max_len:
            return None
        return ids

    @torch.no_grad()
    def score_smiles(
        self,
        images: torch.Tensor,
        smiles_list: list[str],
        image_index: list[int] | None = None,
    ) -> torch.Tensor:
        """Teacher-forced sum log probability (EOS included) of each SMILES given its image.

        ``smiles_list[i]`` is paired with ``images[i]``, or with ``images[image_index[i]]`` if
        given (encodes each image once when scoring several candidates per image). SMILES that
        cannot be tokenized, contain out-of-vocabulary tokens or exceed ``max_len`` get -inf.
        """
        if image_index is None:
            image_index = list(range(len(smiles_list)))
        if len(image_index) != len(smiles_list):
            raise ValueError("image_index and smiles_list differ in length")
        out = torch.full((len(smiles_list),), float("-inf"))
        ids = [self._score_ids(s) for s in smiles_list]
        keep = [i for i, x in enumerate(ids) if x is not None]
        if not keep:
            return out
        dev = images.device
        enc = self.encoder(images)
        sel = torch.tensor([image_index[i] for i in keep], dtype=torch.long, device=dev)
        memory = enc.tokens.index_select(0, sel)
        pad_mask = None if enc.padding_mask is None else enc.padding_mask.index_select(0, sel)
        T = max(len(ids[i]) for i in keep)
        pad = self.tokenizer.pad_id
        batch = torch.tensor(
            [ids[i] + [pad] * (T - len(ids[i])) for i in keep], dtype=torch.long, device=dev
        )
        lp = self.decoder.score(batch, memory, pad_mask, pad_id=pad)
        out[torch.tensor(keep)] = lp.float().cpu()
        return out


def build_model(cfg: dict, tokenizer: SmilesTokenizer) -> OCSRModel:
    m = cfg["model"]
    dec = DecoderConfig(vocab_size=len(tokenizer), **m.get("decoder", {}))
    encoder = build_encoder(m["encoder"], dec.d_model, cfg["data"]["image_size"])
    return OCSRModel(encoder, SmilesDecoder(dec), tokenizer)


# escnn caches expanded filters as buffers only in eval mode (and a freshly built R2Conv holds a
# zero placeholder). They are derived from the weights, so they are never part of the saved state.
_ESCNN_CACHE_SUFFIXES = (".filter", ".expanded_bias")


def load_model_state(model: nn.Module, state: dict) -> None:
    """Load a checkpoint saved in either train or eval mode (escnn-safe, otherwise strict)."""
    model.train()  # drops escnn's cached filters so that the key sets agree
    state = {k: v for k, v in state.items() if not k.endswith(_ESCNN_CACHE_SUFFIXES)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(_ESCNN_CACHE_SUFFIXES)]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def initialize_finetune(model: OCSRModel, checkpoint: dict) -> None:
    """Load pretrained weights into an explicitly append-only expanded vocabulary.

    Optimizer/scheduler/step are intentionally fresh; exact resume remains the
    Trainer's separate last.pt path. Only tied token/output rows may grow.
    """
    old_vocab = checkpoint['vocab']
    if model.tokenizer.itos[:len(old_vocab)] != old_vocab:
        raise ValueError('Fine-tuning vocabulary must preserve the checkpoint vocabulary prefix')
    state = dict(checkpoint['model'])
    current = model.state_dict()
    for key in ('decoder.tok.weight', 'decoder.head.weight'):
        old, new = state[key], current[key]
        if old.shape[1:] != new.shape[1:] or old.shape[0] != len(old_vocab):
            raise ValueError(f'Incompatible pretrained token matrix: {key}')
        expanded = new.clone()
        expanded[:len(old_vocab)].copy_(old)
        state[key] = expanded
    load_model_state(model, state)
