"""String-level SMILES utilities agree with ``score_pair`` and handle invalid input."""

import math

import pytest
from rdkit import Chem

from ouroboros.eval import smiles as S
from ouroboros.eval.metrics import score_pair

ALA = "C[C@H](N)C(=O)O"
D_ALA = "C[C@@H](N)C(=O)O"


def test_canonicalize_and_validity():
    assert S.canonicalize("OCC") == S.canonicalize("CCO") == "CCO"
    assert S.canonicalize("C1=CC=CC=C1") == "c1ccccc1"
    assert S.canonicalize(ALA) != S.canonicalize(D_ALA)
    assert S.canonicalize(ALA, stereo=False) == S.canonicalize(D_ALA, stereo=False)
    assert S.canonicalize("CC(=O)[O-].[Na+]") == "CC(=O)O"  # salt stripped, neutralized
    raw = S.canonicalize("CC(=O)[O-].[Na+]", standardize=False)
    assert raw == Chem.CanonSmiles("[Na+].CC(=O)[O-]") and "." in raw
    for bad in [None, "", "  ", "C1CC(", "Xx"]:
        assert S.canonicalize(bad) is None and not S.is_valid(bad)
        assert not S.is_valid(bad, standardize=False)
    assert S.is_valid("CCO")


def test_exact_and_inchikey_match():
    assert S.exact_match("OCC", "CCO")
    assert not S.exact_match(D_ALA, ALA) and S.exact_match(D_ALA, ALA, stereo=False)
    assert not S.inchikey_match(D_ALA, ALA)
    assert S.inchikey_match(D_ALA, ALA, connectivity_only=True)
    key = Chem.MolToInchiKey(Chem.MolFromSmiles(ALA))
    assert S.inchikey(ALA) == key and len(key) == 27
    assert S.inchikey(ALA, connectivity_only=True) == key[:14] == S.inchikey(D_ALA)[:14]
    assert S.inchikey("C1CC(") is None
    assert S.inchikey_match("Oc1ccccn1", "O=c1cccc[nH]1")  # tautomers share a standard InChIKey
    assert not S.exact_match("Oc1ccccn1", "O=c1cccc[nH]1")
    # an invalid side never matches, including two invalid sides
    assert not S.exact_match("C1CC(", "C1CC(") and not S.inchikey_match("", "")
    assert not S.exact_match("CCO", None)


def test_tanimoto():
    assert S.tanimoto("OCC", "CCO") == 1.0
    assert S.tanimoto(D_ALA, ALA) == 1.0  # chirality ignored by default
    assert S.tanimoto(D_ALA, ALA, chirality=True) < 1.0
    t = S.tanimoto("CCN", "CCO")
    assert 0.0 < t < 1.0
    assert S.tanimoto("C1CC(", "CCO") == 0.0 and S.tanimoto("CCO", "") == 0.0
    assert 0.0 < S.tanimoto("CCCCCCO", "CCCCCCN", radius=3, n_bits=1024) < 1.0


@pytest.mark.parametrize(
    "pred,gold",
    [
        ("OCC", "CCO"),
        (D_ALA, ALA),
        ("CCN", "CCO"),
        ("C1CC(", "CCO"),
        ("", ALA),
        ("CC(=O)[O-].[Na+]", "CC(=O)O"),
        ("C/C=C\\CC", "C/C=C/CC"),
        ("Oc1ccccn1", "O=c1cccc[nH]1"),
    ],
)
def test_agrees_with_score_pair(pred, gold):
    s = score_pair(pred, gold)
    assert S.is_valid(pred) == s.valid
    assert S.exact_match(pred, gold) == s.exact
    assert S.exact_match(pred, gold, stereo=False) == s.exact_flat
    assert S.inchikey_match(pred, gold) == s.inchikey
    assert S.tanimoto(pred, gold) == s.tanimoto


def test_batch_metrics():
    preds = [ALA, D_ALA, "CCN", "", "C1=CC=CC=C1", "CCO"]
    golds = [ALA, ALA, "CCO", "CCO", "c1ccccc1", "C1CC("]  # the last gold is invalid
    m = S.batch_metrics(preds, golds)
    assert m["n"] == 5 and m["n_gold_invalid"] == 1
    assert m["exact"] == pytest.approx(2 / 5)
    assert m["exact_flat"] == pytest.approx(3 / 5)
    assert m["inchikey"] == pytest.approx(2 / 5)
    assert m["valid"] == pytest.approx(4 / 5)
    assert m["tanimoto_1"] == pytest.approx(3 / 5)
    assert m["tanimoto"] == pytest.approx((3 + S.tanimoto("CCN", "CCO")) / 5)
    assert S.score_batch(["CCO"], ["C1CC("]) == [None]
    assert S.batch_metrics([], []) == {"n": 0, "n_gold_invalid": 0}
    with pytest.raises(ValueError):
        S.batch_metrics(["CCO"], [])
    assert math.isnan(S.batch_metrics(["CCO"], ["CCO"])["stereo_elem_acc_strict"])
