"""Cross-attention K/V caching and static self-attention KV buffer parity & call-count tests."""

from __future__ import annotations

import pytest
import torch

from ouroboros.decode.decoder import DecoderConfig, SmilesDecoder


def _make_decoder(seed: int = 7, max_len: int = 20) -> SmilesDecoder:
    torch.manual_seed(seed)
    cfg = DecoderConfig(vocab_size=24, d_model=32, n_layers=3, n_heads=4, d_ff=64, max_len=max_len)
    dec = SmilesDecoder(cfg).eval()
    torch.nn.init.normal_(dec.tok.weight, std=1.0)
    return dec


@torch.no_grad()
def _naive_greedy(
    dec: SmilesDecoder,
    memory: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int | None = None,
    padding_mask=None,
) -> torch.Tensor:
    """Full causal-attention greedy reference without any self- or cross-attention cache."""
    B = memory.shape[0]
    limit = min(max_len or dec.c.max_len, dec.c.max_len)
    seq = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
    done = torch.zeros(B, dtype=torch.bool, device=memory.device)
    out = []
    for _ in range(limit - 1):
        logits = dec(seq, memory, padding_mask=padding_mask)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
        out.append(nxt)
        done |= nxt == eos_id
        if bool(done.all()):
            break
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("use_mask", [False, True])
def test_greedy_matches_uncached_causal_forward(use_mask: bool):
    dec = _make_decoder()
    g = torch.Generator().manual_seed(11)
    memory = torch.randn(4, 16, 32, generator=g)
    pad_mask = None
    if use_mask:
        pad_mask = torch.zeros(4, 16, dtype=torch.bool)
        pad_mask[0, -4:] = True
        pad_mask[2, -7:] = True
    got = dec.greedy(memory, bos_id=1, eos_id=2, padding_mask=pad_mask)
    ref = _naive_greedy(dec, memory, bos_id=1, eos_id=2, padding_mask=pad_mask)
    assert torch.equal(got, ref)


@torch.no_grad()
def test_cross_kv_projected_once_per_layer_for_greedy_and_beam():
    dec = _make_decoder(max_len=16)
    memory = torch.randn(3, 12, 32, generator=torch.Generator().manual_seed(3))
    calls: list[tuple[int, tuple[int, ...]]] = []
    handles = [
        layer.cross_attn.kv.register_forward_hook(
            lambda _m, inp, _out, idx=i: calls.append((idx, tuple(inp[0].shape)))
        )
        for i, layer in enumerate(dec.layers)
    ]
    try:
        out = dec.greedy(memory, bos_id=1, eos_id=999)  # never emits EOS 999 -> runs all 15 steps
        assert out.shape == (3, 15)
        assert calls == [(i, (3, 12, 32)) for i in range(len(dec.layers))]

        calls.clear()
        beams = dec.beam(memory, bos_id=1, eos_id=999, beam_size=5)
        assert len(beams) == 3 and all(len(b) == 5 for b in beams)
        # Projected on batch B=3 (not B*K=15), once per layer across all 15 decode steps.
        assert calls == [(i, (3, 12, 32)) for i in range(len(dec.layers))]
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def test_static_self_kv_buffer_reuses_allocation_and_matches_legacy_dict():
    dec = _make_decoder()
    layer = dec.layers[0]
    memory = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(5))
    cross_kv = layer.cross_attn.project_kv(memory)
    static_cache: dict = {"max_len": 8}
    legacy_cache: dict = {}
    ptrs = set()
    for step in range(8):
        x = torch.randn(2, 1, 32, generator=torch.Generator().manual_seed(100 + step))
        y_static = layer(x, None, cache=static_cache, cross_kv=cross_kv)
        y_legacy = layer(x, memory, cache=legacy_cache)
        assert torch.equal(y_static, y_legacy)
        assert static_cache["pos"] == step + 1
        ptrs.add(static_cache["k"].data_ptr())
        ptrs.add(static_cache["v"].data_ptr())
    assert len(ptrs) == 2  # one K buffer + one V buffer allocated once and updated in place


@torch.no_grad()
def test_beam_hypotheses_agree_with_teacher_forced_score():
    dec = _make_decoder(seed=19, max_len=18)
    memory = torch.randn(3, 10, 32, generator=torch.Generator().manual_seed(21))
    pad_mask = torch.zeros(3, 10, dtype=torch.bool)
    pad_mask[1, -3:] = True
    beams = dec.beam(memory, bos_id=1, eos_id=2, beam_size=4, padding_mask=pad_mask)
    for b, hyps in enumerate(beams):
        assert len(hyps) == 4
        for ids, lp in hyps:
            if len(ids) < 17:  # finished with EOS
                seq = torch.tensor([[1, *ids, 2]], dtype=torch.long)
                ref = dec.score(seq, memory[b : b + 1], padding_mask=pad_mask[b : b + 1]).item()
                assert lp == pytest.approx(ref, abs=1e-4)


@torch.no_grad()
def test_greedy_and_beam_under_cpu_bfloat16_autocast():
    dec = _make_decoder()
    memory = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(9))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        g_ids = dec.greedy(memory, bos_id=1, eos_id=2)
        b_hyps = dec.beam(memory, bos_id=1, eos_id=2, beam_size=1)
    for row, hyps in zip(g_ids.tolist(), b_hyps, strict=True):
        trimmed = row[: row.index(2)] if 2 in row else row
        assert hyps[0][0] == trimmed
