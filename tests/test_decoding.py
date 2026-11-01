"""Beam search, teacher-forced scoring, validity reranking, ensembling and TTA at inference."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ouroboros.decode.rerank import choose, ensemble_choose, ensemble_rank, normalized
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.model import OCSRModel, build_model

ROOT = Path(__file__).resolve().parents[1]
MAX_LEN = 24
SMILES = ["CCO", "OCC", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "N#CC(Cl)Br"]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_cfg() -> dict:
    return {
        "data": {"image_size": 64},
        "model": {
            "encoder": {
                "type": "baseline",
                "widths": [8, 16, 16, 32],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 64,
            },
            "decoder": {
                "d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64, "max_len": MAX_LEN,
            },
        },
    }


def tiny_model(seed: int = 0) -> OCSRModel:
    torch.manual_seed(seed)
    model = build_model(tiny_cfg(), SmilesTokenizer.build(SMILES))
    # peaked (non-uniform) next-token distributions: no near-ties, and EOS is actually emitted
    torch.nn.init.normal_(model.decoder.tok.weight, std=1.0)
    return model.eval()


@pytest.fixture(scope="module")
def model() -> OCSRModel:
    return tiny_model()


@pytest.fixture(scope="module")
def images() -> torch.Tensor:
    return torch.rand(4, 1, 64, 64, generator=torch.Generator().manual_seed(1))


def _memory(model, images):
    enc = model.encoder(images)
    return enc.tokens, enc.padding_mask


def _until_eos(row: list[int], eos: int) -> list[int]:
    return row[: row.index(eos) + 1] if eos in row else row


@torch.no_grad()
def test_beam_size_one_equals_greedy(model, images):
    tok = model.tokenizer
    mem, pad = _memory(model, images)
    greedy = model.decoder.greedy(mem, tok.bos_id, tok.eos_id, padding_mask=pad).tolist()
    beams = model.decoder.beam(mem, tok.bos_id, tok.eos_id, beam_size=1, padding_mask=pad)
    for row, hyps in zip(greedy, beams, strict=True):
        assert len(hyps) == 1
        assert hyps[0][0] == [t for t in _until_eos(row, tok.eos_id) if t != tok.eos_id]
    assert [h[0][0] for h in model.generate_nbest(images, beam_size=1)] == model.generate(images)


@torch.no_grad()
def test_beam_returns_k_sorted_hypotheses(model, images):
    tok, K = model.tokenizer, 4
    mem, pad = _memory(model, images)
    for hyps in model.decoder.beam(mem, tok.bos_id, tok.eos_id, beam_size=K, padding_mask=pad):
        assert len(hyps) == K
        assert len({tuple(ids) for ids, _ in hyps}) == K
        # unfinished hypotheses (hit max_len) have MAX_LEN - 1 tokens and no EOS to count
        norm = [lp / (len(ids) + (len(ids) < MAX_LEN - 1)) for ids, lp in hyps]
        assert norm == sorted(norm, reverse=True)
        assert all(tok.eos_id not in ids for ids, _ in hyps)  # BOS may appear from a random model, as in greedy
    nbest = model.generate_nbest(images, beam_size=K)
    assert len(nbest) == len(images) and all(len(c) == K for c in nbest)


@torch.no_grad()
def test_score_matches_greedy_logprob(model, images):
    tok = model.tokenizer
    mem, pad = _memory(model, images)
    greedy = model.decoder.greedy(mem, tok.bos_id, tok.eos_id, padding_mask=pad).tolist()
    beams = model.decoder.beam(mem, tok.bos_id, tok.eos_id, beam_size=1, padding_mask=pad)
    for b, row in enumerate(greedy):
        ids = torch.tensor([[tok.bos_id, *_until_eos(row, tok.eos_id)]])
        lp = model.decoder.score(ids, mem[b : b + 1], None if pad is None else pad[b : b + 1])
        assert lp.item() == pytest.approx(beams[b][0][1], abs=1e-4)
    # string-level scoring (padded batch) agrees wherever the decoded string re-tokenizes to the
    # generated tokens (a random model may emit e.g. <pad> mid-sequence, which decode drops)
    smiles = [tok.decode(h[0][0]) for h in beams]
    scored = model.score_smiles(images, smiles)
    for b, s in enumerate(smiles):
        ids, lp = beams[b][0]
        if len(ids) < MAX_LEN - 1 and model._score_ids(s) == [tok.bos_id, *ids, tok.eos_id]:
            assert scored[b].item() == pytest.approx(lp, abs=1e-4)


@torch.no_grad()
def test_score_smiles_rejects_unscorable(model, images):
    long = "C" * MAX_LEN  # BOS + MAX_LEN atoms + EOS > max_len
    cands = ["CCO", "xyz", "[Pt]", long, "c1ccccc1O"]
    scores = model.score_smiles(images[:2], cands, image_index=[0, 0, 1, 1, 1])
    assert torch.isfinite(scores[[0, 4]]).all() and (scores[[0, 4]] < 0).all()
    assert torch.isneginf(scores[[1, 2, 3]]).all()
    alone = model.score_smiles(images[1:2], ["c1ccccc1O"])
    assert alone[0].item() == pytest.approx(scores[4].item(), rel=1e-5)


def test_choose_prefers_valid():
    cands = [("C1CC", -0.1), ("CCO", -0.5), ("CCN", -0.9)]
    assert choose(cands) == ("CCO", -0.5)
    assert choose([("C1CC", -0.1), ("C(", -0.2)]) == ("C1CC", -0.1)  # none valid: top
    assert choose(cands, valid_fn=lambda s: True) == ("C1CC", -0.1)
    with pytest.raises(ValueError):
        choose([])


def test_ensemble_merges_equivalent_smiles():
    a = [("OCC", -1.0), ("C1CC", -0.1)]
    b = [("C(O)C", -1.2), ("CCN", -3.0)]
    rescored = {
        "OCC": [-1.0, -1.1],
        "C(O)C": [-0.9, -1.2],
        "C1CC": [-0.1, -0.2],
        "CCN": [-4.0, -3.0],
    }
    ranked = ensemble_rank([a, b], rescored)
    assert [s for s, _ in ranked].count("OCC") + [s for s, _ in ranked].count("C(O)C") == 1
    assert len(ranked) == 3
    ethanol = next(c for c in ranked if c[0] in ("OCC", "C(O)C"))
    # represented by the better spelling: mean normalized score over both models
    assert ethanol == ("C(O)C", pytest.approx((-0.9 / 6 + -1.2 / 6) / 2))
    assert ranked[0][0] == "C1CC"  # best score overall, but invalid ...
    assert ensemble_choose([a, b], rescored)[0] == "C(O)C"  # ... so the valid one is chosen
    assert ensemble_choose([a, b], rescored, prefer_valid=False)[0] == "C1CC"
    assert normalized("CCO", -4.0) == pytest.approx(-1.0)
    assert normalized("xyz", -1.0) == float("-inf")


def test_tta_views_keep_ink_convention():
    mod = _load_script("predict_real")
    x = torch.zeros(2, 1, 64, 64)
    x[:, :, 10:54, 30:34] = 1.0  # a vertical bond touching the margins
    padded = mod.tta_pad(x)
    assert padded.shape == x.shape and padded.min() >= 0 and padded.max() <= 1
    assert padded[..., :3, :].abs().max() == 0 and padded[..., -3:, :].abs().max() == 0
    assert padded.sum() < x.sum()  # content shrunk
    sharp = mod.tta_sharpen(x)
    assert sharp.shape == x.shape and sharp.min() >= 0 and sharp.max() <= 1
    assert torch.equal(mod.tta_sharpen(torch.zeros(1, 1, 8, 8)), torch.zeros(1, 1, 8, 8))


def _shards(tmp_path: Path) -> Path:
    from ouroboros.data.real import ingest

    (tmp_path / "img").mkdir()
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "smiles", "source", "split", "id"])
        for i, smi in enumerate(["CCO", "c1ccccc1O", "N#CC(Cl)Br"]):
            arr = np.full((80, 120), 255, np.uint8)
            arr[20 + 10 * i : 30 + 10 * i, 10:-10] = 0
            Image.fromarray(arr, "L").save(tmp_path / "img" / f"{i}.png")
            w.writerow([f"img/{i}.png", smi, "TINY", "test", f"id{i}"])
    ingest(tmp_path / "manifest.csv", tmp_path / "shards", size=64)
    return tmp_path / "shards"


def _ckpt(path: Path, seed: int) -> Path:
    m = tiny_model(seed)
    torch.save({"vocab": m.tokenizer.itos, "config": tiny_cfg(), "model": m.state_dict()}, path)
    return path


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_predict_real_beam_ensemble_tta(tmp_path):
    mod = _load_script("predict_real")
    shards = _shards(tmp_path)
    c0, c1 = _ckpt(tmp_path / "a.pt", 0), _ckpt(tmp_path / "b.pt", 1)
    base = ["--shards", str(shards), "--batch-size", "2"]

    mod.main(["--ckpt", str(c0), *base, "--out", str(tmp_path / "greedy.csv")])
    greedy = _rows(tmp_path / "greedy.csv")
    assert list(greedy[0]) == ["id", "gold", "pred", "subset"] and len(greedy) == 3

    # beam 1 through the n-best path reproduces greedy exactly
    mod.main(["--ckpt", str(c0), *base, "--out", str(tmp_path / "b1.csv"),
              "--nbest-out", str(tmp_path / "b1.jsonl")])
    assert _rows(tmp_path / "b1.csv") == greedy

    out, nb = tmp_path / "ens.csv", tmp_path / "ens.jsonl"
    mod.main(["--ckpt", str(c0), "--ckpt", str(c1), *base, "--beam", "2",
              "--tta", "pad,sharpen", "--out", str(out), "--nbest-out", str(nb)])
    rows = _rows(out)
    assert list(rows[0]) == ["id", "gold", "pred", "subset"]
    assert [r["id"] for r in rows] == [r["id"] for r in greedy]
    recs = [json.loads(line) for line in nb.read_text().splitlines()]
    assert [r["pred"] for r in recs] == [r["pred"] for r in rows]
    for r in recs:
        assert r["pred"] in [c["smiles"] for c in r["candidates"]]
        scores = [c["score"] if c["score"] is not None else -np.inf for c in r["candidates"]]
        assert scores == sorted(scores, reverse=True)

    mod.main(["--ckpt", str(c0), *base, "--beam", "2", "--out", str(tmp_path / "b2.csv")])
    assert len(_rows(tmp_path / "b2.csv")) == 3
