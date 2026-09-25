"""Small RDKit helpers shared across the data pipeline."""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canonical_smiles(smiles: str, isomeric: bool = True) -> str | None:
    """Canonical SMILES (stereo kept if ``isomeric``), or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key or None
