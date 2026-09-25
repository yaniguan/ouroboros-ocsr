"""Phase 6: conformers, stereo from 3D, error categories (hand-constructed pairs)."""

import pytest
from rdkit import Chem

from ouroboros.geometry.conformers import embed, mirror_positions, stereo_from_3d, stereo_preserved
from ouroboros.geometry.propagation import categorize, mirror_smiles, same_formula

CASES = [
    # (prediction, truth, expected category)
    ("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O", "correct"),
    ("N[C@@H](C)C(=O)O", "C[C@H](N)C(=O)O", "correct"),  # same molecule, other atom order
    ("OC(=O)[C@@H](C)N", "C[C@H](N)C(=O)O", "enantiomer"),  # looks alike, is the mirror image
    ("C[C@H](N)C(=O)O.Cl", "C[C@H](N)C(=O)O", "correct"),  # salt stripped
    ("C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O", "enantiomer"),
    ("C[C@H](O)[C@H](C)Cl", "C[C@@H](O)[C@@H](C)Cl", "enantiomer"),  # both centres inverted
    ("C[C@H](O)[C@@H](C)Cl", "C[C@@H](O)[C@@H](C)Cl", "diastereomer"),  # one of two inverted
    ("CC(N)C(=O)O", "C[C@H](N)C(=O)O", "diastereomer"),  # stereo dropped
    ("C/C=C\\C", "C/C=C/C", "diastereomer"),  # E/Z swapped: not a mirror image
    ("C/C=C/[C@@H](Cl)F", "C/C=C/[C@H](Cl)F", "enantiomer"),  # E/Z kept, centre inverted
    ("C/C=C\\[C@@H](Cl)F", "C/C=C/[C@H](Cl)F", "diastereomer"),
    ("C[C@@H]1CC[C@H](C)CC1", "C[C@@H]1CC[C@@H](C)CC1", "diastereomer"),  # cis vs trans
    ("C[C@H]1CC[C@@H](C)CC1", "C[C@@H]1CC[C@H](C)CC1", "correct"),  # achiral: mirror == self
    ("O[C@H]1C[C@@H](O)C1", "O[C@@H]1C[C@H](O)C1", "correct"),  # meso-like, same molecule
    ("CCO", "CCN", "constitutional"),
    ("CC(C)O", "CCCO", "constitutional"),  # isomer, different connectivity
    ("C[C@H](N)C(=O)OC", "C[C@H](N)C(=O)O", "constitutional"),
    ("C1CC(", "CCO", "invalid"),
    ("", "CCO", "invalid"),
]


@pytest.mark.parametrize("pred,truth,cat", CASES)
def test_categorize(pred, truth, cat):
    assert categorize(pred, truth) == cat


def test_mirror_smiles_and_formula():
    assert Chem.CanonSmiles(mirror_smiles("C[C@H](N)C(=O)O")) == Chem.CanonSmiles(
        "C[C@@H](N)C(=O)O"
    )
    assert mirror_smiles("C/C=C/C") == Chem.CanonSmiles("C/C=C/C")
    assert same_formula("CC(C)O", "CCCO") and not same_formula("CCO", "CCN")


@pytest.mark.parametrize("smi", ["C[C@H](N)C(=O)O", "C/C=C/[C@@H](Cl)[C@H](O)c1ccccc1"])
def test_embedding_keeps_stereo_and_mirror_inverts(smi):
    mol, status = embed(smi, n_conf=4, seed=0)
    assert status == "ok" and mol.GetNumConformers() == 4
    assert all(stereo_preserved(mol, smi))
    m = Chem.Mol(mol)
    conf = m.GetConformer(0)
    for i, p in enumerate(mirror_positions(conf.GetPositions())):
        conf.SetAtomPosition(i, p.tolist())
    assert stereo_from_3d(m, 0) == Chem.CanonSmiles(mirror_smiles(smi))


def test_embed_failure_reason():
    assert embed("not_smiles")[1] == "unparsable"
