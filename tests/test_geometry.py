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


def test_stereo_check_ignores_unspecified_centres():
    for smi in ["CC(N)C(=O)O", "C[C@H](N)C(C)CC", "C/C=C/C(O)CC"]:
        mol, _ = embed(smi, n_conf=2, seed=0)
        assert all(stereo_preserved(mol, smi)), smi
    # but a wrong specified centre is caught
    mol, _ = embed("C[C@H](N)C(=O)O", n_conf=1, seed=0)
    assert not stereo_preserved(mol, "C[C@@H](N)C(=O)O")[0]


@pytest.mark.parametrize("timeout_first", [True, False])
def test_embedding_timeout_sentinel(monkeypatch, timeout_first):
    from ouroboros.geometry import conformers as g

    original = g.AllChem.EmbedMultipleConfs
    modes = []

    def timed_out(mol, numConfs, params):
        modes.append(params.useRandomCoords)
        assert params.timeout == 7
        if timeout_first and len(modes) == 2:
            return original(mol, numConfs=numConfs, params=params)
        return [-1]

    monkeypatch.setattr(g.AllChem, "EmbedMultipleConfs", timed_out)
    attempts = []
    mol, status = g.embed("C[C@H](N)C(=O)O", 2, timeout_seconds=7, diagnostics=attempts)
    assert modes == [False, True] and attempts[0]["returned_ids"] == [-1]
    if timeout_first:
        assert status == "ok" and mol.GetNumConformers() == 2
    else:
        assert mol is None and status == "etkdg_failed"


@pytest.mark.parametrize("bad", ["unconverged", "stereo", "nan", "exception"])
def test_only_valid_relaxed_candidates_can_win(monkeypatch, bad):
    from ouroboros.geometry import conformers as g

    calls = []

    def fake_relax(atoms, calc, fmax, steps):
        first = not calls
        calls.append(True)
        if first and bad == "exception":
            raise RuntimeError("optimizer failure")
        pos = atoms.positions.copy()
        if first and bad == "stereo":
            pos[:, 0] *= -1
        energy = float("nan") if first and bad == "nan" else (-10.0 if first else -5.0)
        return g.Relaxed(energy, pos * 0, pos, not (first and bad == "unconverged"), 1, 0.01, 0.0)

    monkeypatch.setattr(g, "relax", fake_relax)
    result = g.lowest_energy_conformer("C[C@H](N)C(=O)O", n_conf=2, calc=object())
    assert result.status == "ok" and result.energy == -5.0
    assert result.converged and result.stereo_ok_final and len(result.failures) == 1


def test_no_converged_candidate_has_no_energy(monkeypatch):
    from ouroboros.geometry import conformers as g

    def failed(atoms, *args):
        return g.Relaxed(-10.0, atoms.positions * 0, atoms.positions, False, 500, 1.0, 0.0)

    monkeypatch.setattr(g, "relax", failed)
    result = g.lowest_energy_conformer("CCCCC", n_conf=2, calc=object())
    assert result.status == "no_valid_conformer" and result.energy is None
    assert not result.converged and len(result.failures) == 2
