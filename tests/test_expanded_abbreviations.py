"""Expanded abbreviation dictionary: isotope stability, SMILES safety and 3D round-trip."""

from __future__ import annotations

import pytest
from rdkit import Chem

from ouroboros.data.rgroups import ABBREVIATIONS, expand_aliases
from ouroboros.geometry.conformers import embed, stereo_preserved
from ouroboros.labels import LABEL_ISOTOPES, LABELS, normalize_labels

LEGACY_PREFIX = (
    'R', *[f'R{i}' for i in range(1, 100)], "R'", 'R’', 'X', 'Ar',
    'Ph', 'OMe', 'OEt', 'Me', 'Et', 'Pr', 'iPr', 'Bu', 'tBu', 'CO2Et',
    'CO2Me', 'Boc', 'Cbz', 'Fmoc', 'Ac', 'Bn', 'Bz', 'Ts', 'Ms',
    'TBS', 'TBDMS', 'TMS', 'PMP', 'TAGa', 'G', 'M', 'L', 'PG', 'Y', 'Z',
)


def _canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_legacy_label_isotope_ids_are_unchanged():
    assert LABELS[: len(LEGACY_PREFIX)] == LEGACY_PREFIX
    for i, label in enumerate(LEGACY_PREFIX):
        assert LABEL_ISOTOPES[label] == 500 + i
    assert set(ABBREVIATIONS) <= set(LABELS)


@pytest.mark.parametrize(
    "smiles",
    [
        "CCN",
        "CN(C)C",
        "C1CC2CCN1NO2",
        "FC(F)(F)c1ccc(C#N)cc1",
        "ClC(Cl)(Cl)C(=O)O",
        "CS(=O)(=O)Nc1ccccc1",
        "C[Si](C)(C)OCC",
        "[Ru]",
        "[Xe]",
        "[Co+2]",
        "[Os]",
        "[Sm+3]",
        "N[C@@H](Cc1ccccc1)C(=O)O",
    ],
)
def test_normalize_labels_never_corrupts_ordinary_smiles(smiles: str):
    assert normalize_labels(smiles) == smiles


@pytest.mark.parametrize(
    "labelled,expected",
    [
        ("CC[TBS]", "CC[Si](C)(C)C(C)(C)C"),
        ("CC[TBDMS]", "CC[Si](C)(C)C(C)(C)C"),
        ("CC[TMS]", "CC[Si](C)(C)C"),
        ("CC[OTBS]", "CCO[Si](C)(C)C(C)(C)C"),
        ("CC[OTMS]", "CCO[Si](C)(C)C"),
        ("N[Ts]", "NS(=O)(=O)c1ccc(C)cc1"),
        ("CC[OTs]", "CCOS(=O)(=O)c1ccc(C)cc1"),
        ("N[Ms]", "NS(=O)(=O)C"),
        ("CC[OMs]", "CCOS(=O)(=O)C"),
        ("N[Cbz]", "NC(=O)OCc1ccccc1"),
        ("N[Fmoc]", "NC(=O)OCC1c2ccccc2-c2ccccc21"),
        ("[CF3]c1ccc([CHO])cc1", "O=Cc1ccc(C(F)(F)F)cc1"),
        ("[CCl3]c1ccc([CO2H])cc1", "O=C(O)c1ccc(C(Cl)(Cl)Cl)cc1"),
        ("CC[COOH]", "CCC(=O)O"),
        ("[OPh]c1ccc([SMe])cc1", "CSc1ccc(Oc2ccccc2)cc1"),
        ("[OBn]c1ccc([OAc])cc1", "CC(=O)Oc1ccc(OCc2ccccc2)cc1"),
        ("[NMe2]c1ccc([SO2Me])cc1", "CN(C)c1ccc(S(C)(=O)=O)cc1"),
        ("[NEt2]c1ccc([SO2Ph])cc1", "CCN(CC)c1ccc(S(=O)(=O)c2ccccc2)cc1"),
        ("[tBu]c1ccc([iPr])cc1", "CC(C)c1ccc(C(C)(C)C)cc1"),
        ("[Ts]N[C@@H](Cc1ccccc1)[CO2H]", "Cc1ccc(S(=O)(=O)N[C@@H](Cc2ccccc2)C(=O)O)cc1"),
        (
            "[OTBS]C[C@H]([NHBoc])[CO2Me]",
            "C(O[Si](C)(C)C(C)(C)C)[C@H](NC(=O)OC(C)(C)C)C(=O)OC",
        ),
    ],
)
def test_multi_alias_expansion_and_chiral_preservation(labelled: str, expected: str):
    expanded = expand_aliases(labelled)
    assert Chem.MolToSmiles(expanded) == _canon(expected)


def test_expanded_chiral_amino_acid_embeds_in_3d_with_stereo_preserved():
    expanded = expand_aliases("[Cbz]N[C@@H]([iPr])[CO2Me]")
    smi = Chem.MolToSmiles(expanded)
    mol, status = embed(smi, n_conf=2, seed=0)
    assert status == "ok" and mol is not None
    assert all(stereo_preserved(mol, smi))
