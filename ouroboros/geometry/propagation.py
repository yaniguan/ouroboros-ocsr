"""Error categories of a predicted SMILES vs. the truth, and energy-error propagation.

Categories (both SMILES standardized with the scoring standardization first):
  correct         identical canonical isomeric SMILES
  enantiomer      the prediction is the mirror image of a chiral truth (every tetrahedral centre
                  inverted, E/Z unchanged)
  diastereomer    same constitution, different stereo, not the enantiomer (includes predictions
                  that drop or add stereo)
  constitutional  different constitution (connectivity / atoms / bond orders)
  invalid         the prediction does not parse

Energy propagation: ΔE = E(pred) - E(truth) of the lowest-energy MACE-OFF conformers, only
meaningful between isomers (same molecular formula); otherwise reported as NaN with a flag. By
construction ΔE = 0 for enantiomers — which is exactly why stereo is scored separately.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ouroboros.eval.standardize import flat_smiles, standardize

CATEGORIES = ("correct", "enantiomer", "diastereomer", "constitutional", "invalid")


def mirror_smiles(smiles: str) -> str:
    """Enantiomer by inverting every tetrahedral tag (@ <-> @@); E/Z bonds are unchanged."""
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(mol)


def categorize(pred: str | None, truth: str) -> str:
    p, t = standardize(pred), standardize(truth)
    if not t.valid:
        raise ValueError(f"truth does not standardize: {truth!r}")
    if not p.valid:
        return "invalid"
    if p.smiles == t.smiles:
        return "correct"
    if flat_smiles(p) != flat_smiles(t):
        return "constitutional"
    mirror = Chem.CanonSmiles(mirror_smiles(t.smiles))
    if mirror != t.smiles and p.smiles == mirror:
        return "enantiomer"
    return "diastereomer"


def same_formula(a: str, b: str) -> bool:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    return ma is not None and mb is not None and CalcMolFormula(ma) == CalcMolFormula(mb)
