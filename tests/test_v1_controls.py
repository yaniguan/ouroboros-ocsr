"""v1 protocol controls: relative-position baseline, optimizer parity, grid, real splits."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.encoder.relpos import RelDistMixer, grid_coords
from ouroboros.model import build_model
from ouroboros.train.config import load_config
from ouroboros.train.trainer import build_optimizer


def test_relpos_mixer_is_permutation_equivariant_and_isometry_invariant():
    torch.manual_seed(0)
    mixer = RelDistMixer(32, 2, 4, 64, 0.0).eval()
    x, coords = torch.randn(2, 9, 32), grid_coords(3, 3)
    perm = torch.randperm(9)
    with torch.no_grad():
        base = mixer(x, coords)
        assert torch.allclose(mixer(x[:, perm], coords[perm]), base[:, perm], atol=1e-5)
        rot = coords @ torch.tensor([[0.0, -1.0], [1.0, 0.0]])  # 90 degree rotation of the grid
        assert torch.allclose(mixer(x, rot), base, atol=1e-5)


def test_baseline_relative_has_no_position_table():
    tok = SmilesTokenizer.load("configs/vocab.json")
    cfg = load_config("configs/pilot-Brel-200k.yaml")
    assert cfg["model"]["encoder"]["position"] == "relative"
    assert cfg["train"]["rotation_aug"] is True
    model = build_model(cfg, tok)
    assert model.encoder.pos is None
    assert not any(n.endswith(".pos") for n, _ in model.encoder.named_parameters())
    out = model.encoder(torch.zeros(1, 1, 384, 384))
    assert out.tokens.shape == (1, 144, 512)


def test_unknown_position_mode_is_rejected():
    tok = SmilesTokenizer.load("configs/vocab.json")
    cfg = load_config("configs/pilot-B-200k.yaml", ["model.encoder.position=bogus"])
    with pytest.raises(ValueError, match="position"):
        build_model(cfg, tok)


def test_escnn_filter_coefficients_are_weight_decayed():
    pytest.importorskip("escnn")
    tok = SmilesTokenizer.load("configs/vocab.json")
    cfg = load_config("configs/pilot-C-200k.yaml")
    model = build_model(cfg, tok)
    opt = build_optimizer(model, cfg["train"])
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    kernels = [p for n, p in model.named_parameters() if n.endswith(".weights")]
    assert kernels and all(id(p) in decayed for p in kernels)
    undecayed = sum(p.numel() for p in opt.param_groups[1]["params"])
    total = sum(p.numel() for p in model.parameters())
    assert undecayed / total < 0.02  # biases, norms and positions only


def test_grid_is_complete_and_stable():
    from scripts.hpc.v1_grid import ARMS, BUDGETS, SEEDS, grid

    tasks = grid()
    assert len(tasks) == len(ARMS) * len(BUDGETS) * len(SEEDS) == 36
    assert [t["id"] for t in tasks] == list(range(1, 37))
    assert len({t["run_id"] for t in tasks}) == 36
    assert all((t["fraction"] > 0) == (t["budget"] > 0) for t in tasks)
    assert len({t["fraction"] for t in tasks if t["budget"]}) == 1  # same mixing ratio


def _parquet_row(i, pmcid, smiles, key):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, "PNG")
    return {
        "image": {"bytes": buf.getvalue(), "path": f"{pmcid}/f_{i}.png"},
        "smiles": smiles,
        "inchikey": key,
        "n_agree": 3,
        "n_agree_open": 2,
        "engines": "a|b|c",
        "pmcid": pmcid,
        "doi": "x",
        "figure_id": "f",
        "crop_index": i,
        "license": "CC BY",
    }


def test_real_splits_scope_overlap_and_nested_pools(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    import scripts.make_real_splits as splits

    monkeypatch.setattr(splits, "POOLS", (2, 4))
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF", "CCS", "CCCC", "CCCO", "CCCN"]
    train = [_parquet_row(i, f"PMC{i % 5}", s, f"{s}AAAAAAAAAAAAAAA") for i, s in enumerate(smiles)]
    train.append(_parquet_row(20, "PMC9", "CC[O]", "BADBADBADBADBAD"))  # out of vocabulary
    train.append(_parquet_row(21, "PMC9", "C" * 200, "LONGLONGLONGLON"))  # over length
    test = [_parquet_row(0, "PMC100", "CCO", "CCOAAAAAAAAAAAAAAA")]  # same molecule as a train row
    (tmp_path / "hf/data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(train), tmp_path / "hf/data/train-00000-of-00001.parquet")
    pq.write_table(
        pa.Table.from_pylist(test), tmp_path / "hf/data/validation-00000-of-00001.parquet"
    )
    stats = splits.build(tmp_path / "hf", tmp_path / "out", Path("configs/vocab.json"), 128, 2)
    assert stats["dropped_by_scope"]["train_published"] == {
        "out_of_vocabulary": 1,
        "over_length": 1,
    }
    assert len((tmp_path / "out/eval_keys.txt").read_text().split()) == 1  # CCO only
    pool_rows = {
        k: [ln for ln in (tmp_path / f"out/manifest_pool{k}.csv").read_text().splitlines()[1:]]
        for k in (2, 4)
    }
    assert pool_rows[2] == pool_rows[4][:2]  # nested
    manifest = (tmp_path / "out/manifest_eval.csv").read_text()
    assert "PMC100" in manifest
    assert not any(",CCO," in ln for ln in pool_rows[4])  # train copy of a test molecule removed
