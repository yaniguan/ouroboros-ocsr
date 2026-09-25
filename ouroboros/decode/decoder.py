"""Shared autoregressive SMILES decoder (identical for every experimental arm — constraint 4).

Pre-LN Transformer decoder: causal self-attention over SMILES tokens (learned 1D positions of
the *sequence*, unrelated to image geometry) and cross-attention to the encoder's memory tokens.
Cross-attention adds NO positional information to the memory: it treats the memory as a set, so
the decoder's output is invariant to any permutation of the memory tokens. Hence, if an encoder
emits a rotation-invariant token *set*, the whole model is rotation invariant; if it emits
position-dependent tokens (baseline), it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DecoderConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 128


class _Attention(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.h = h
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, mem, mask=None, causal=False, cache=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(B, mem.shape[1], 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:  # incremental self-attention: append new keys/values
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            causal = False  # the single new query may attend to all cached positions
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class DecoderLayer(nn.Module):
    def __init__(self, c: DecoderConfig):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(c.d_model) for _ in range(3))
        self.self_attn = _Attention(c.d_model, c.n_heads, c.dropout)
        self.cross_attn = _Attention(c.d_model, c.n_heads, c.dropout)
        self.ff = nn.Sequential(
            nn.Linear(c.d_model, c.d_ff),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_ff, c.d_model),
        )
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mem, mem_mask=None, cache=None):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, causal=True, cache=cache))
        h = self.ln2(x)
        x = x + self.drop(self.cross_attn(h, mem, mask=mem_mask))
        return x + self.drop(self.ff(self.ln3(x)))


class SmilesDecoder(nn.Module):
    def __init__(self, c: DecoderConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)  # 1D sequence positions only
        self.layers = nn.ModuleList(DecoderLayer(c) for _ in range(c.n_layers))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.drop = nn.Dropout(c.dropout)
        nn.init.normal_(self.tok.weight, std=0.02)  # tied with the output head: keep logits O(1)
        nn.init.normal_(self.pos.weight, std=0.02)

    @staticmethod
    def _mem_mask(padding_mask: torch.Tensor | None, dtype) -> torch.Tensor | None:
        """[B, M] bool (True = pad) -> additive mask broadcastable to [B, H, T, M]."""
        if padding_mask is None:
            return None
        m = torch.zeros(padding_mask.shape, dtype=dtype, device=padding_mask.device)
        return m.masked_fill(padding_mask, float("-inf"))[:, None, None, :]

    def forward(self, ids: torch.Tensor, memory: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """Teacher forcing: ids [B, T] -> logits [B, T, V]."""
        T = ids.shape[1]
        if T > self.c.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.c.max_len}")
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.tok(ids) + self.pos(pos)[None])
        mask = self._mem_mask(padding_mask, x.dtype)
        for layer in self.layers:
            x = layer(x, memory, mask)
        return self.head(self.ln(x))

    @torch.no_grad()
    def greedy(
        self,
        memory: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int | None = None,
        padding_mask=None,
    ) -> torch.Tensor:
        """Greedy decoding with a self-attention KV cache. Returns ids [B, L] (no BOS)."""
        B = memory.shape[0]
        max_len = min(max_len or self.c.max_len, self.c.max_len)
        caches = [{} for _ in self.layers]
        mask = self._mem_mask(padding_mask, memory.dtype)
        cur = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
        out = []
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for t in range(max_len - 1):
            x = self.tok(cur) + self.pos.weight[t][None, None]
            for layer, cache in zip(self.layers, caches, strict=True):
                x = layer(x, memory, mask, cache=cache)
            nxt = self.head(self.ln(x))[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            out.append(nxt)
            done |= nxt == eos_id
            if bool(done.all()):
                break
            cur = nxt[:, None]
        return torch.stack(out, dim=1)
