"""Regression tests for core ML, chemistry, geometry, data, and labvision audit fixes."""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

from ouroboros.data.hf_manifest import _safe_name, write_manifests
from ouroboros.data.loader import ShardDataset, rotate_images
from ouroboros.decode.confidence import confidence
from ouroboros.decode.decoder import DecoderConfig, SmilesDecoder
from ouroboros.decode.ensemble import predict_batch
from ouroboros.decode.rerank import ensemble_rank
from ouroboros.encoder.canonicalize import CanonConfig, CanonicalizedEncoder
from ouroboros.encoder.equiv_attention import EquivAttentionEncoder, EquivAttnConfig
from ouroboros.eval.standardize import leakage_key, standardize
from ouroboros.geometry.conformers import stereo_matches
from ouroboros.geometry.propagation import categorize, mirror_smiles, same_formula
from ouroboros.hf import CHECKSUMS_FILE, verify_checksums
from ouroboros.labvision.decode import decode
from ouroboros.labvision.geometry import otsu
from ouroboros.labvision.keypoints import kp_from_pixels
from ouroboros.labvision.render import render as render_plate
from ouroboros.labvision.spec import RectSpec
from ouroboros.labvision.wellstate import fit_centroids, patch_features
from ouroboros.provenance import implementation_identity


def test_hf_manifest_rejects_dotdot_and_bad_split_map(tmp_path):
    assert _safe_name("..") == "real"
    assert _safe_name(".") == "real"
    snap = tmp_path / "snap"
    snap.mkdir()
    with pytest.raises(ValueError, match="split_map"):
        write_manifests(snap, tmp_path / "out", split_map={"val": "../escape"})


def test_provenance_posix_paths_and_checksum_traversal_guard(tmp_path):
    pkg = tmp_path / "pkg" / "sub"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("x = 1\n", encoding="utf-8")
    ident = implementation_identity(tmp_path / "pkg")
    assert list(ident) == ["sub/mod.py"]

    secret = tmp_path / "secret.json"
    secret.write_text("{}", encoding="utf-8")
    folder = tmp_path / "model"
    folder.mkdir()
    from ouroboros.provenance import file_sha256

    (folder / CHECKSUMS_FILE).write_text(
        f"{file_sha256(secret)}  ../secret.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(folder)


def test_exported_equiv_attention_supports_fp16():
    cfg = EquivAttnConfig(
        N=4,
        fields=[2, 2],
        blocks=[1, 1],
        d_model=16,
        attn_dim=16,
        attn_layers=1,
        attn_heads=2,
        attn_ff=32,
        image_size=32,
    )
    enc = EquivAttentionEncoder(cfg).eval()
    exported = enc.export().half()
    x = torch.zeros(2, 1, 32, 32, dtype=torch.float16)
    out = exported(x)
    assert out.tokens.dtype == torch.float16
    assert out.tokens.shape[0] == 2


def test_canonicalized_encoder_clears_eval_and_consumed_logits():
    cfg = CanonConfig(N=4, orient_fields=[2, 2], downsample=2, d_model=16, image_size=32)
    enc = CanonicalizedEncoder(cfg)
    x = torch.zeros(2, 1, 32, 32)
    enc.eval()
    enc(x)
    assert enc.last_logits is None
    with pytest.raises(RuntimeError, match="prior_loss"):
        enc.prior_loss(None)

    enc.train()
    enc(x)
    assert enc.last_logits is not None
    loss = enc.prior_loss(None)
    assert torch.isfinite(loss)
    enc.eval()
    assert enc.last_logits is None
    enc.train()
    with torch.no_grad():
        enc(x)
    assert enc.last_logits is None


def test_labelled_pseudo_atom_enantiomer_and_formula():
    assert categorize("C[C@@H](O)[Ph]", "C[C@H](O)[Ph]") == "enantiomer"
    assert categorize("not_a_smiles", "C[C@H](O)[Ph]") == "invalid"
    assert same_formula("CC(=O)[O-].[Na+]", "CC(=O)O")
    with pytest.raises(ValueError, match="unparsable"):
        mirror_smiles("not_a_smiles")


def test_stereo_matches_ring_cis_trans_without_cip_codes():
    trans_smi = "O[C@H]1CC[C@@H](O)CC1"
    cis_smi = "O[C@H]1CC[C@H](O)CC1"

    m_trans = Chem.AddHs(Chem.MolFromSmiles(trans_smi))
    assert AllChem.EmbedMolecule(m_trans, AllChem.ETKDGv3()) == 0
    assert stereo_matches(m_trans, trans_smi)
    assert not stereo_matches(m_trans, cis_smi)
    assert not stereo_matches(m_trans, "not_a_smiles")


def test_leakage_key_labelled_stereoisomers_share_14char_prefix():
    r_key = leakage_key(standardize("[Boc]N[C@H](C)C(=O)O"))
    s_key = leakage_key(standardize("[Boc]N[C@@H](C)C(=O)O"))
    other = leakage_key(standardize("[Boc]NCC(=O)O"))
    assert r_key is not None and s_key is not None and other is not None
    assert r_key.startswith("L") and len(r_key) == 27
    assert r_key != s_key
    assert r_key[:14] == s_key[:14]
    assert r_key[:14] != other[:14]


def test_decoder_max_len_ensemble_tensor_scores_and_confidence_neg_inf():
    dec = SmilesDecoder(DecoderConfig(vocab_size=8, d_model=16, n_heads=2, n_layers=1, d_ff=32))
    mem = torch.zeros(1, 4, 16)
    for bad_len in (0, 1):
        with pytest.raises(ValueError, match="max_len"):
            dec.greedy(mem, bos_id=1, eos_id=2, max_len=bad_len)
        with pytest.raises(ValueError, match="max_len"):
            dec.beam(mem, bos_id=1, eos_id=2, beam_size=2, max_len=bad_len)

    ranked = ensemble_rank(
        [[("CC", -0.4), ("CCC", -0.8)]],
        {"CC": torch.tensor([-0.4, -0.6]), "CCC": np.array([-1.0, -1.2])},
    )
    assert ranked[0][0] == "CC"

    gate = confidence(
        "CC",
        [("CC", -0.1), ("CCC", -0.5), ("unparseable(", float("-inf"))],
        min_margin=0.2,
    )
    assert gate["accepted"] and gate["margin"] == pytest.approx(0.4)
    assert callable(predict_batch)


def test_shard_dataset_nonsquare_resize_close_and_fp16_rotate(tmp_path):
    buf = io.BytesIO()
    Image.new("L", (64, 32), 255).save(buf, format="PNG")
    shard = tmp_path / "test-000000.tar"
    items = (("png", buf.getvalue()), ("json", json.dumps({"smiles": "CCO"}).encode()))
    with tarfile.open(shard, "w") as tar:
        for ext, data in items:
            info = tarfile.TarInfo(f"k0.{ext}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with ShardDataset(tmp_path, "test", image_size=64) as ds:
        item = ds[0]
        assert tuple(item["image"].shape) == (1, 64, 64)
    shard.unlink()  # succeeds on Windows only when cached file handles were closed

    imgs = torch.ones(2, 1, 16, 16, dtype=torch.float16)
    rot = rotate_images(imgs, torch.tensor([30.0, 90.0]))
    assert rot.dtype == torch.float16 and rot.shape == imgs.shape


def test_labvision_uniform_otsu_missing_class_and_mirrored_rect():
    assert 0.0 <= otsu(np.ones((16, 16))) <= 1.0
    assert 0.0 <= otsu(np.zeros((16, 16))) <= 1.0
    with pytest.raises(ValueError, match="fewer than 4"):
        kp_from_pixels(np.zeros((32, 32)), "rect")
    with pytest.raises(ValueError, match="radius"):
        patch_features(np.zeros((16, 16)), [[8.0, 8.0]], radius=-1.0)
    with pytest.raises(ValueError, match="missing training samples"):
        fit_centroids(np.zeros((4, 2)), np.array([0, 0, 1, 1]), n_classes=4)

    spec = RectSpec()
    scene = render_plate(
        "rect",
        np.random.default_rng(11),
        rot_deg=35,
        persp=0.0,
        occlude=False,
        glare=False,
        mirror=True,
        spec=spec,
    )
    dec = decode(scene.image, kp_from_pixels(scene.image, "rect"), spec)
    assert dec.mirrored and dec.ambiguous
