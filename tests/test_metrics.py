"""Metric functions against hand-built prediction/reference pairs."""

import math

import pytest
from rdkit import Chem

from ouroboros.eval.metrics import aggregate, bootstrap_ci, score_pair
from ouroboros.eval.standardize import standardize


def _enantiomer(smi: str) -> str:
    return smi.replace("@@", "\0").replace("@", "@@").replace("\0", "@")


def test_identity_and_reordering():
    s = score_pair("OCC", "CCO")
    assert s.valid and s.exact and s.exact_flat and s.inchikey and s.tanimoto == 1.0
    assert (s.stereo_total, s.stereo_correct) == (0, 0)


def test_salt_and_charge_standardization():
    assert score_pair("CC(=O)[O-].[Na+]", "CC(=O)O").exact
    assert score_pair("C[C@H](N)C(=O)O.Cl", "C[C@H](N)C(=O)O").exact
    assert score_pair("O.O.CCCCN", "CCCCN").exact  # solvate
    assert score_pair("CN(=O)=O", "C[N+](=O)[O-]").exact  # nitro normalization
    assert standardize("[Na+].[Cl-]").smiles == "Cl"


def test_enantiomer():
    ref = "C[C@H](N)C(=O)O"
    s = score_pair(_enantiomer(ref), ref)
    assert s.valid and not s.exact and s.exact_flat and not s.inchikey
    assert s.tanimoto == 1.0  # Morgan FP without chirality
    assert (s.stereo_total, s.stereo_correct) == (1, 0)


def test_missing_stereo():
    s = score_pair("CC(N)C(=O)O", "C[C@H](N)C(=O)O")
    assert not s.exact and s.exact_flat and (s.stereo_total, s.stereo_correct) == (1, 0)


def test_one_of_two_centres_wrong():
    ref = "C[C@@H](O)[C@@H](C)Cl"
    s = score_pair("C[C@H](O)[C@@H](C)Cl", ref)
    assert not s.exact and s.exact_flat and (s.stereo_total, s.stereo_correct) == (2, 1)


def test_double_bond_stereo():
    ref = "C/C=C/CC"
    same = score_pair("CC/C=C/C", ref)
    assert same.exact and same.stereo_correct == 1
    s = score_pair("C/C=C\\CC", ref)
    assert not s.exact and s.exact_flat and (s.stereo_total, s.stereo_correct) == (1, 0)


def test_ring_cis_trans_mismatch():
    s = score_pair("C[C@H]1CC[C@H](C)CC1", "C[C@H]1CC[C@@H](C)CC1")
    assert not s.exact and s.exact_flat and s.stereo_total == 2 and s.stereo_correct < 2
    s = score_pair("C[C@@H]1CC[C@H](C)CC1", "C[C@H]1CC[C@@H](C)CC1")  # same molecule
    assert s.exact and s.stereo_correct == 2


@pytest.mark.parametrize(
    "ref",
    [
        "C[C@H](N)C(=O)O",
        "C/C=C/[C@@H](Cl)[C@H](O)c1ccccc1",
        "O[C@H]1C[C@@H](O)C1",
        "C[C@@H]1CCCC[C@H]1N",
        "F/C=C/C=C\\C[C@](C)(Br)Cl",
        "N[C@@H]1C[C@H]2CC[C@@H]1C2",
    ],
)
def test_random_smiles_orderings(ref):
    mol = Chem.MolFromSmiles(ref)
    n_tet = sum(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
    for variant in Chem.MolToRandomSmilesVect(mol, 10, randomSeed=0):
        s = score_pair(variant, ref)
        assert s.exact and s.stereo_correct == s.stereo_total > 0, variant
        # the mirror image keeps every E/Z bond and inverts every tetrahedral centre
        ent = score_pair(_enantiomer(variant), ref)
        if Chem.CanonSmiles(_enantiomer(ref)) == Chem.CanonSmiles(ref):  # achiral (meso/cis)
            assert ent.exact and ent.stereo_correct == ent.stereo_total
        else:
            assert not ent.exact
            assert ent.stereo_correct == ent.stereo_total - n_tet, variant


def test_invalid_and_empty():
    for bad in ["C1CC(", "", "   ", None, "Xx"]:
        s = score_pair(bad, "C[C@H](N)C(=O)O")
        assert not s.valid and not s.exact and not s.inchikey and s.tanimoto == 0.0
        assert (s.stereo_total, s.stereo_correct) == (1, 0)
    with pytest.raises(ValueError):
        score_pair("CCO", "C1CC(")


def test_constitutional_error_and_tautomer_inchikey():
    s = score_pair("CCN", "CCO")
    assert s.valid and not s.exact and not s.exact_flat and s.tanimoto < 1.0
    t = score_pair("Oc1ccccn1", "O=c1cccc[nH]1")  # tautomers: SMILES differ, std InChIKey equal
    assert not t.exact and t.inchikey


def test_aggregate_and_bootstrap():
    ref = "C[C@@H](O)[C@@H](C)Cl"
    scores = [
        score_pair(ref, ref),  # exact, 2/2
        score_pair("C[C@H](O)[C@@H](C)Cl", ref),  # 1/2
        score_pair("CCO", "CCO"),  # exact, no stereo
        score_pair("C1CC(", "CCO"),  # invalid
    ]
    agg = aggregate(scores)
    assert agg["n"] == 4
    assert agg["exact"] == 0.5 and agg["invalid_rate"] == 0.25
    assert agg["stereo_elem_acc_strict"] == 3 / 4
    assert agg["stereo_elem_acc_given_flat"] == 3 / 4
    assert agg["n_with_stereo"] == 2 and agg["exact_on_stereo_subset"] == 0.5
    mean, lo, hi = bootstrap_ci([s.exact for s in scores], n_resamples=2000)
    assert mean == 0.5 and 0.0 <= lo < 0.5 < hi <= 1.0
    assert all(math.isnan(v) for v in bootstrap_ci([]))
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)
