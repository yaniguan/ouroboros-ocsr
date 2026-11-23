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


_ENCODER_MODULES = {
    "exported_steerable": "ouroboros.encoder.pure_torch",
    "exported_equiv_attention": "ouroboros.encoder.pure_torch",
    "steerable": "ouroboros.encoder.steerable",
    "equiv_attention": "ouroboros.encoder.equiv_attention",
    "canonicalized": "ouroboros.encoder.canonicalize",
}
_ESCNN_ENCODERS = frozenset({"steerable", "equiv_attention", "canonicalized"})


def build_encoder(enc_cfg: dict, d_model: int, image_size: int) -> ImageEncoder:
    import importlib
    import sys

    name = enc_cfg["type"]
    if name in _ESCNN_ENCODERS and "escnn" in sys.modules and sys.modules["escnn"] is None:
        raise ImportError(
            f"encoder {name!r} requires escnn; load an exported pure-PyTorch checkpoint "
            "or install escnn"
        )
    if name not in ENCODERS and name in _ENCODER_MODULES:
        try:
            importlib.import_module(_ENCODER_MODULES[name])
        except ModuleNotFoundError as exc:
            if name in _ESCNN_ENCODERS and (exc.name or "").startswith("escnn"):
                raise ImportError(
                    f"encoder {name!r} requires escnn; load an exported pure-PyTorch checkpoint "
                    "or install escnn"
                ) from exc
            raise
    if name not in ENCODERS:
        known = sorted(ENCODERS | _ENCODER_MODULES.keys())
        raise KeyError(f"unknown encoder type {name!r}; known: {known}")
    return ENCODERS[name](enc_cfg, d_model, image_size)


class OCSRModel(nn.Module):
    _STEREO_TOKENS = frozenset({"@", "@@", "/", "\\"})

    def __init__(
        self,
        encoder: ImageEncoder,
        decoder: SmilesDecoder,
        tokenizer: SmilesTokenizer,
        stereo_head: nn.Module | None = None,
        stereo_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.stereo_head = stereo_head
        self.stereo_loss_weight = float(stereo_loss_weight)

    def _vocab_token_weights(
        self, stereo_token_weight: float, device: torch.device
    ) -> torch.Tensor:
        w = torch.ones(len(self.tokenizer), dtype=torch.float32, device=device)
        for idx, tok in enumerate(self.tokenizer.itos):
            if tok in self._STEREO_TOKENS:
                w[idx] = float(stereo_token_weight)
        return w

    def loss(
        self,
        images: torch.Tensor,
        ids: torch.Tensor,
        label_smoothing: float = 0.0,
        *,
        stereo_targets: torch.Tensor | None = None,
        stereo_loss_weight: float | None = None,
        stereo_token_weight: float = 1.0,
    ):
        enc = self.encoder(images)
        logits = self.decoder(ids[:, :-1], enc.tokens, enc.padding_mask)
        weight = (
            self._vocab_token_weights(stereo_token_weight, logits.device)
            if stereo_token_weight != 1.0
            else None
        )
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1),
            weight=weight,
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=label_smoothing,
        )
        if self.stereo_head is not None and stereo_targets is not None:
            if enc.spatial is None:
                raise ValueError("stereo_head requires an encoder emitting EncoderOutput.spatial")
            aux_w = (
                self.stereo_loss_weight if stereo_loss_weight is None else float(stereo_loss_weight)
            )
            ce = ce + aux_w * self.stereo_head.loss(enc.spatial, stereo_targets)
        return ce

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
    stereo_cfg = m.get("stereo_head") or {}
    stereo_head = None
    loss_w = 0.5
    if stereo_cfg.get("enabled", False):
        from ouroboros.encoder.stereo_head import SpatialStereoHead

        stereo_head = SpatialStereoHead(
            d_model=dec.d_model,
            hidden=int(stereo_cfg.get("hidden", 128)),
            out_channels=int(stereo_cfg.get("out_channels", 4)),
            pos_weight=float(stereo_cfg.get("pos_weight", 8.0)),
        )
        loss_w = float(stereo_cfg.get("loss_weight", 0.5))
    return OCSRModel(encoder, SmilesDecoder(dec), tokenizer, stereo_head, loss_w)


# escnn caches expanded filters as buffers only in eval mode (and a freshly built R2Conv holds a
# zero placeholder). They are derived from the weights, so they are never part of the saved state.
_ESCNN_CACHE_SUFFIXES = (".filter", ".expanded_bias")


def load_model_state(model: nn.Module, state: dict) -> None:
    """Load a checkpoint saved in either train or eval mode (escnn-safe, otherwise strict)."""
    was_training = model.training
    model.train()  # drops escnn's cached filters so that the key sets agree
    state = {k: v for k, v in state.items() if not k.endswith(_ESCNN_CACHE_SUFFIXES)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(_ESCNN_CACHE_SUFFIXES)]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    if not was_training:
        model.eval()


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def initialize_finetune(model: OCSRModel, checkpoint: dict) -> None:
    """Load pretrained weights into an expanded vocabulary and/or longer decoder ``max_len``.

    Optimizer/scheduler/step are intentionally fresh; exact resume remains the
    Trainer's separate last.pt path. Tied token/output rows and 1D decoder
    sequence positions may grow while preserving the pretrained prefix; a newly
    enabled ``stereo_head`` keeps its fresh initialization if absent from ``checkpoint``.
    """
    old_vocab = checkpoint["vocab"]
    if model.tokenizer.itos[: len(old_vocab)] != old_vocab:
        raise ValueError("Fine-tuning vocabulary must preserve the checkpoint vocabulary prefix")
    state = dict(checkpoint["model"])
    current = model.state_dict()
    for key in ("decoder.tok.weight", "decoder.head.weight"):
        old, new = state[key], current[key]
        if old.shape[1:] != new.shape[1:] or old.shape[0] != len(old_vocab):
            raise ValueError(f"Incompatible pretrained token matrix: {key}")
        expanded = new.clone()
        expanded[: len(old_vocab)].copy_(old)
        state[key] = expanded
    if "decoder.pos.weight" in state:
        old_pos, new_pos = state["decoder.pos.weight"], current["decoder.pos.weight"]
        if old_pos.shape[1:] != new_pos.shape[1:] or old_pos.shape[0] > new_pos.shape[0]:
            raise ValueError(
                f"Incompatible decoder.pos.weight shape {tuple(old_pos.shape)} -> "
                f"{tuple(new_pos.shape)}"
            )
        if old_pos.shape[0] < new_pos.shape[0]:
            expanded_pos = new_pos.clone()
            expanded_pos[: old_pos.shape[0]].copy_(old_pos)
            state["decoder.pos.weight"] = expanded_pos
    if model.stereo_head is not None:
        for k, v in current.items():
            if k.startswith("stereo_head.") and k not in state:
                state[k] = v
    load_model_state(model, state)
