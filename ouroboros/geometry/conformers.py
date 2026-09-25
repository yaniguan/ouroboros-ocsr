"""ETKDG conformer embedding (RDKit). MACE relaxation is added in Phase 6."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem


def embed_conformers(smiles: str, n_conf: int = 10, seed: int = 0) -> Chem.Mol:
    """Add hydrogens and embed ``n_conf`` ETKDGv3 conformers. Raises ValueError on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    if len(cids) == 0:
        raise ValueError(f"ETKDG embedding failed: {smiles!r}")
    return mol
