"""Molecule standardization and filtering.

Rules (Phase 1 DoD): allowed elements H C N O F P S Cl Br I only, 5-60 heavy atoms, neutral.
"Neutral" = net formal charge 0 and no radical electrons, evaluated AFTER standardization
(largest organic fragment + uncharger), so protonated amines / carboxylates from vendor
libraries are neutralized rather than dropped. Charge-separated neutral groups such as nitro
([N+](=O)[O-]) are kept (net charge 0).
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from ouroboros.data.chem import canonical_smiles  # noqa: F401  (re-export convenience)

ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})


@dataclass(frozen=True)
class FilterConfig:
    min_heavy_atoms: int = 5
    max_heavy_atoms: int = 60
    allowed_elements: frozenset[str] = ALLOWED_ELEMENTS
    standardize: bool = True


_FRAG = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest organic fragment and neutralize (acid/base) charges; drop isotopes."""
    mol = _FRAG.choose(mol)
    mol = _UNCHARGER.uncharge(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    Chem.SanitizeMol(mol)
    return mol


def check(mol: Chem.Mol, cfg: FilterConfig = FilterConfig()) -> str | None:
    """Return None if ``mol`` passes all rules, else a short rejection reason."""
    if mol is None:
        return "unparsable"
    if len(Chem.GetMolFrags(mol)) != 1:
        return "multi_fragment"
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in cfg.allowed_elements:
            return "element"
        if atom.GetNumRadicalElectrons():
            return "radical"
        if atom.GetIsotope():
            return "isotope"
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < cfg.min_heavy_atoms:
        return "too_small"
    if n_heavy > cfg.max_heavy_atoms:
        return "too_large"
    if Chem.GetFormalCharge(mol) != 0:
        return "charged"
    return None


def filter_smiles(smiles: str, cfg: FilterConfig = FilterConfig()) -> tuple[Chem.Mol | None, str]:
    """Parse, (optionally) standardize and check. Returns ``(mol, "ok")`` or ``(None, reason)``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparsable"
    if cfg.standardize:
        try:
            mol = standardize(mol)
        except Exception:  # noqa: BLE001 - RDKit raises assorted sanitization errors
            return None, "standardize_failed"
    reason = check(mol, cfg)
    return (None, reason) if reason else (mol, "ok")
