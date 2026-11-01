"""A small, model-free sanity set for the predictions harness, with known expected outcomes.

Each gold molecule appears in three subsets (``id`` = ``mol<NN>``):

  render   pred = the label RDKit re-perceives from the rendered drawing
           (``ouroboros.data.render``: 2D layout + wedges -> MolBlock -> stereo re-perception).
           Expected: exact match, unless the renderer mis-depicts the stereo (which the dataset
           builder drops as ``label_mismatch``).
  mirror   the same from a mirrored drawing: every tetrahedral centre inverts, E/Z is kept.
           Expected: stereo-agnostic match always; exact match only if the molecule equals its
           mirror image (no tetrahedral stereo, or meso / ring cis-trans only).
  perturb  a deterministic corruption chosen by index (column ``perturbation``):
             random_smiles  another valid SMILES of the same molecule      -> exact
             no_stereo      stereo removed                                 -> exact iff no stereo
             salt           ``.Cl`` counter-ion appended                   -> exact (standardized)
             add_methyl     one extra carbon                               -> wrong constitution
             syntax_error   an unclosed branch appended                    -> invalid
             blank          empty prediction                               -> invalid

These are rule checks for the metric code and the render label path, not model accuracy.
"""

from __future__ import annotations

import random

from rdkit import Chem

from ouroboros.data.render import render, sample_style

GOLD = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O",  # ibuprofen
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "C[C@H](N)C(=O)O",  # alanine
    "N[C@@H](Cc1ccccc1)C(=O)O",  # phenylalanine
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose, anomer unspecified
    "C/C=C/C(=O)O",  # crotonic acid
    "CC(C)=CCC/C(C)=C/CO",  # geraniol
    "O=C(O)/C=C\\C(=O)O",  # maleic acid
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",  # diazepam
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",  # morphine
    "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1",  # menthol
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",  # testosterone
    "OC(=O)[C@@H]1CCCN1",  # proline
    "C[C@@H](O)[C@H](N)C(=O)O",  # threonine
    "O[C@H]1CC[C@@H](O)CC1",  # cyclohexane-1,4-diol (ring cis/trans, achiral)
    "Cl/C=C/Cl",  # trans-1,2-dichloroethene
    "c1ccc2ccccc2c1",  # naphthalene
    "CC[C@H](C)Br",  # 2-bromobutane
    "F[C@](Cl)(Br)I",  # fully halogenated stereocentre, no H
    "CC(C)(C)OC(=O)N[C@@H](C)C(=O)O",  # Boc-alanine
    "CCOC(=O)/C=C/c1ccccc1",  # ethyl cinnamate
]
PERTURBATIONS = ("random_smiles", "no_stereo", "salt", "add_methyl", "syntax_error", "blank")


def perturb(smiles: str, kind: str, seed: int = 0) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if kind == "random_smiles":
        return Chem.MolToRandomSmilesVect(mol, 1, randomSeed=seed)[0]
    if kind == "no_stereo":
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    if kind == "salt":
        return smiles + ".Cl"
    if kind == "add_methyl":
        # bracket atoms such as [C@H] have a fixed H count, so only implicit-H carbons qualify
        site = next(
            (
                a.GetIdx()
                for a in mol.GetAtoms()
                if a.GetSymbol() == "C" and not a.GetNoImplicit() and a.GetNumImplicitHs()
            ),
            None,
        )
        if site is None:
            return "C"  # no C-H to substitute: methane is still a valid, wrong molecule
        rw = Chem.RWMol(mol)
        new = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(site, new, Chem.BondType.SINGLE)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    if kind == "syntax_error":
        return smiles + "C("
    if kind == "blank":
        return ""
    raise ValueError(f"unknown perturbation {kind!r}")


def build_records(gold: list[str] | None = None, size: int = 256, seed: int = 0) -> list[dict]:
    """Rows with id, gold, pred, subset, perturbation (blank outside ``perturb``)."""
    rows = []
    for i, smi in enumerate(gold or GOLD):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"gold {smi!r} does not parse")
        mid = f"mol{i:02d}"
        style = sample_style(random.Random(seed * 1_000_003 + i))
        for subset, mirror in (("render", False), ("mirror", True)):
            r = render(mol, style, size=size, mirror=mirror, seed=i, apply_postprocess=False)
            rows.append(
                {"id": mid, "gold": smi, "pred": r.label, "subset": subset, "perturbation": ""}
            )
        kind = PERTURBATIONS[i % len(PERTURBATIONS)]
        rows.append(
            {
                "id": mid,
                "gold": smi,
                "pred": perturb(smi, kind, seed=seed + i),
                "subset": "perturb",
                "perturbation": kind,
            }
        )
    return rows
