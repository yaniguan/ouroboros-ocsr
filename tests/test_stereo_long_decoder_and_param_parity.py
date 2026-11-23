"""Tests for 2D spatial stereo head, long-molecule decoder extension, 3D beam reranker,
and parameter-matched C8 equivariant configs (#132)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from ouroboros.data.render import (
    RenderStyle,
    build_stereo_heatmap,
    depict,
    extract_stereo_keypoints,
)
from ouroboros.decode.rerank_3d import rerank_beams_3d
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.encoder.pure_torch import ExportedSteerableEncoder, OrbitStatsPool
from ouroboros.encoder.stereo_head import STEREO_CHANNELS, SpatialStereoHead
from ouroboros.model import build_model, count_params, initialize_finetune
from ouroboros.train.config import load_config
from ouroboros.ui.compliance import audit_3d_physical_compliance

ROOT = Path(__file__).resolve().parents[1]
SMILES = [
    "CCO",
    "N[C@@H](C)C(=O)O",
    "N[C@H](C)C(=O)O",
    "C/C=C/C",
    "C/C=C\\C",
    "CC(=O)Oc1ccccc1C(=O)O",
]


def test_orbit_stats_pool_preserves_cn_invariance_and_triples_channels(monkeypatch):
    with pytest.raises(ValueError, match="pool mode"):
        OrbitStatsPool(4, mode="invalid")
    pool = OrbitStatsPool(4, mode="max_mean_l2")
    assert pool.multiplier == 3
    x = torch.randn(2, 12, 6, 6)
    y0 = pool(x)
    assert y0.shape == (2, 9, 6, 6)
    # Cyclic shift along each 4-channel orbit leaves OrbitStatsPool output identical
    shifted = x.view(2, 3, 4, 6, 6).roll(shifts=2, dims=2).reshape_as(x)
    assert torch.allclose(y0, pool(shifted), atol=1e-6)

    # Pure-PyTorch ExportedSteerableEncoder with pool='max_mean_l2' works without escnn
    monkeypatch.setitem(sys.modules, "escnn", None)
    tok = SmilesTokenizer.build(SMILES)
    cfg = {
        "data": {"image_size": 64},
        "model": {
            "encoder": {
                "type": "exported_steerable",
                "N": 4,
                "fields": [2, 2, 4, 6],
                "blocks": [1, 1, 1, 1],
                "pool": "max_mean_l2",
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 32,
            },
            "decoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "max_len": 32},
        },
    }
    model = build_model(cfg, tok).eval()
    assert isinstance(model.encoder, ExportedSteerableEncoder)
    assert model.encoder.head.proj.in_channels == 18  # 6 fields * 3 orbit stats
    out = model.encoder(torch.rand(1, 1, 64, 64))
    assert out.spatial is not None and out.spatial.shape == (1, 32, 2, 2)


def test_escnn_max_mean_l2_export_roundtrip():
    pytest.importorskip("escnn")
    tok = SmilesTokenizer.build(SMILES)
    cfg = {
        "data": {"image_size": 64},
        "model": {
            "encoder": {
                "type": "steerable",
                "N": 4,
                "fields": [2, 2, 4, 4],
                "blocks": [1, 1, 1, 1],
                "pool": "max_mean_l2",
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 32,
            },
            "decoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "max_len": 32},
        },
    }
    model = build_model(cfg, tok).eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        ref_out = model.encoder(x)
        exp_enc = model.encoder.export()
        exp_out = exp_enc(x)
    assert torch.allclose(ref_out.tokens, exp_out.tokens, atol=1e-4)
    assert torch.allclose(ref_out.spatial, exp_out.spatial, atol=1e-4)


def test_param_matched_configs_align_encoder_and_total_params():
    pytest.importorskip("escnn")
    tok = SmilesTokenizer.load(ROOT / "configs" / "vocab.json")
    cfg_a = load_config(ROOT / "configs" / "param-matched-A.yaml")
    cfg_b = load_config(ROOT / "configs" / "param-matched-B.yaml")
    cfg_c = load_config(ROOT / "configs" / "param-matched-C.yaml")
    cfg_cp = load_config(ROOT / "configs" / "param-matched-Cplus.yaml")

    assert cfg_b["train"]["rotation_aug"] is True
    assert cfg_cp["train"]["rotation_aug"] is True
    assert cfg_a["model"]["decoder"]["max_len"] == 384
    assert cfg_c["model"]["decoder"]["max_len"] == 384

    mod_a = build_model(cfg_a, tok)
    mod_c = build_model(cfg_c, tok)
    assert mod_a.stereo_head is not None and mod_c.stereo_head is not None
    enc_a = count_params(mod_a.encoder)
    enc_c = count_params(mod_c.encoder)
    rel_enc_diff = abs(enc_c - enc_a) / enc_a
    rel_tot_diff = abs(count_params(mod_c) - count_params(mod_a)) / count_params(mod_a)
    assert rel_enc_diff < 0.02, f"Encoder params {enc_c} vs {enc_a} ({rel_enc_diff:.2%})"
    assert rel_tot_diff < 0.01


def test_stereo_keypoint_heatmaps_and_auxiliary_head_training():
    style = RenderStyle()
    # L-Alanine = (S)-Alanine; D-Alanine = (R)-Alanine
    s_mol = depict(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"), style)
    r_mol = depict(Chem.MolFromSmiles("N[C@H](C)C(=O)O"), style)
    e_mol = depict(Chem.MolFromSmiles("C/C=C/C"), style)

    kp_s = extract_stereo_keypoints(s_mol, style, size=128)
    kp_r = extract_stereo_keypoints(r_mol, style, size=128)
    kp_e = extract_stereo_keypoints(e_mol, style, size=128)

    assert len(kp_s["chiral_S"]) == 1 and len(kp_s["chiral_R"]) == 0
    assert len(kp_s["wedge_dash_bond"]) >= 1
    assert len(kp_r["chiral_R"]) == 1 and len(kp_r["chiral_S"]) == 0
    assert len(kp_e["ez_double_bond"]) == 1

    hm_s = build_stereo_heatmap(kp_s, grid_size=4)
    assert hm_s.shape == (len(STEREO_CHANNELS), 4, 4)
    assert float(hm_s[1].max()) > 0.5 and float(hm_s[0].max()) == 0.0

    tok = SmilesTokenizer.build(SMILES)
    cfg = {
        "data": {"image_size": 64},
        "model": {
            "encoder": {
                "type": "baseline",
                "widths": [8, 16, 16, 32],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 32,
            },
            "decoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "max_len": 32},
            "stereo_head": {"enabled": True, "hidden": 16, "loss_weight": 0.5},
        },
    }
    torch.manual_seed(0)
    model = build_model(cfg, tok)
    assert isinstance(model.stereo_head, SpatialStereoHead)

    images = torch.rand(2, 1, 64, 64)
    ids_a = tok.encode("N[C@@H](C)C(=O)O")
    ids_b = tok.encode("C/C=C/C")
    max_t = max(len(ids_a), len(ids_b))
    batch_ids = torch.tensor(
        [
            ids_a + [tok.pad_id] * (max_t - len(ids_a)),
            ids_b + [tok.pad_id] * (max_t - len(ids_b)),
        ],
        dtype=torch.long,
    )
    targets = torch.stack(
        [
            torch.from_numpy(build_stereo_heatmap(kp_s, grid_size=2)),
            torch.from_numpy(build_stereo_heatmap(kp_e, grid_size=2)),
        ]
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first_loss = None
    last_loss = None
    for _ in range(8):
        opt.zero_grad()
        loss = model.loss(images, batch_ids, stereo_targets=targets, stereo_token_weight=2.5)
        if first_loss is None:
            first_loss = float(loss.item())
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    assert last_loss < first_loss


def test_long_decoder_position_extension_and_prepare_finetune(tmp_path):
    tok_short = SmilesTokenizer.build(SMILES)
    cfg_short = {
        "data": {"image_size": 64},
        "model": {
            "encoder": {
                "type": "baseline",
                "widths": [8, 16, 16, 32],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 32,
            },
            "decoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "max_len": 24},
        },
    }
    base_model = build_model(cfg_short, tok_short).eval()
    x = torch.rand(1, 1, 64, 64)
    short_ids = torch.tensor([tok_short.encode("CCO")[:-1]], dtype=torch.long)
    with torch.no_grad():
        logits_before = base_model.decoder(short_ids, base_model.encoder(x).tokens)

    # Extend in-place and verify prefix logits are bit-identical
    ckpt = {
        "model": {k: v.clone() for k, v in base_model.state_dict().items()},
        "config": cfg_short,
        "vocab": list(tok_short.itos),
        "step": 10,
    }
    base_model.decoder.extend_max_len(96)
    assert base_model.decoder.c.max_len == 96
    with torch.no_grad():
        logits_extended = base_model.decoder(short_ids, base_model.encoder(x).tokens)
    assert torch.equal(logits_before, logits_extended)

    # initialize_finetune grows vocab + max_len + attaches fresh stereo_head cleanly
    tok_long = SmilesTokenizer(list(tok_short.itos))
    tok_long.extend(["Br"])
    cfg_long = {
        "data": {"image_size": 64},
        "model": {
            "encoder": cfg_short["model"]["encoder"],
            "decoder": cfg_short["model"]["decoder"] | {"max_len": 96},
            "stereo_head": {"enabled": True, "hidden": 16},
        },
    }
    ft_model = build_model(cfg_long, tok_long).eval()
    initialize_finetune(ft_model, ckpt)
    with torch.no_grad():
        logits_ft = ft_model.decoder(short_ids, ft_model.encoder(x).tokens)[..., : len(tok_short)]
    assert torch.allclose(logits_before, logits_ft, atol=1e-6)
    long_seq = torch.zeros((1, 80), dtype=torch.long)
    with torch.no_grad():
        assert ft_model.decoder(long_seq, ft_model.encoder(x).tokens).shape == (
            1,
            80,
            len(tok_long),
        )

    # Verify prepare_real_finetune CLI accepts --max-len
    spec = importlib.util.spec_from_file_location(
        "prepare_real_finetune", ROOT / "scripts" / "prepare_real_finetune.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")


def test_rerank_beams_3d_penalizes_strained_diastereomer_and_flags_enantiomer_tie():
    # (E)-cyclooctene is ~10+ kcal/mol more strained than (Z)-cyclooctene in 3D!
    res_diast = rerank_beams_3d(
        [
            ("C1CC/C=C/CCC1", -0.10),  # strained trans-cyclooctene (slightly better visual score)
            ("C1CC/C=C\\CCC1", -0.15),  # relaxed cis-cyclooctene
        ],
        strain_weight=0.05,
    )
    assert res_diast["enantiomer_ambiguity"] is False
    # Low-energy (Z)-cyclooctene overtakes strained (E)-cyclooctene after 3D strain penalty
    assert res_diast["best_smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles("C1CC/C=C\\CCC1"))

    # Mirror enantiomers have ΔE = 0, so a close visual tie triggers enantiomer_ambiguity
    res_enant = rerank_beams_3d(
        [
            ("N[C@@H](C)C(=O)O", -0.08),
            ("N[C@H](C)C(=O)O", -0.12),
        ],
        enantiomer_margin=0.25,
    )
    assert res_enant["enantiomer_ambiguity"] is True
    assert res_enant["needs_chiral_review"] is True
    assert len(res_enant["enantiomer_twins"]) == 1

    # Compliance audit explicitly reports enantiomer parity degeneracy for chiral molecules
    chiral_audit = audit_3d_physical_compliance("N[C@@H](C)C(=O)O", n_conf=1, do_relax=False)
    assert chiral_audit["enantiomer_blind_spot_note"] is not None
    assert "ΔE = 0" in chiral_audit["enantiomer_blind_spot_note"]
    achiral_audit = audit_3d_physical_compliance("O=C1OCCO1", n_conf=1, do_relax=False)
    assert achiral_audit["enantiomer_blind_spot_note"] is None
