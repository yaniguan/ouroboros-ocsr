"""Stereo perception and random stereo assignment for dataset composition.

A drawn molecule depicts a definite geometry for every stereogenic double bond (its 2D
layout IS cis or trans), so E/Z is always assigned. Tetrahedral centers are only defined
(and drawn with wedges/hashes) for molecules placed in the stereo bucket; otherwise they are
left unspecified and drawn with plain bonds.
"""

from __future__ import annotations

import random

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def potential_stereo(mol: Chem.Mol) -> tuple[int, int]:
    """(#potential tetrahedral centers, #potential stereo double bonds), ignoring current tags."""
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    n_tet = n_db = 0
    for info in Chem.FindPotentialStereo(m, cleanIt=True, flagPossible=True):
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            n_tet += 1
        elif info.type == Chem.StereoType.Bond_Double:
            n_db += 1
    return n_tet, n_db


def defined_stereo(mol: Chem.Mol) -> tuple[int, int]:
    """(#defined tetrahedral centers, #defined E/Z bonds) on a sanitized mol."""
    n_tet = n_db = 0
    for info in Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True, flagPossible=False):
        if info.specified != Chem.StereoSpecified.Specified:
            continue
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            n_tet += 1
        elif info.type == Chem.StereoType.Bond_Double:
            n_db += 1
    return n_tet, n_db


def has_stereo(smiles_or_mol) -> bool:
    mol = Chem.MolFromSmiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return mol is not None and sum(defined_stereo(mol)) > 0


def _embeddable(mol: Chem.Mol, attempts: int = 5) -> bool:
    """Quick 3D-feasibility check: can ETKDG (chirality enforced) embed it in a few attempts?"""
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.maxIterations = attempts
    params.enforceChirality = True
    return AllChem.EmbedMolecule(m, params) == 0


def assign_random_stereo(mol: Chem.Mol, rng: random.Random, tetrahedral: bool) -> Chem.Mol:
    """Return a copy with all potential E/Z bonds (and, if ``tetrahedral``, all potential
    tetrahedral centers) set to a random configuration. Existing tags are discarded first so
    the configuration is controlled by ``rng`` only.

    Random tags on bridged ring systems can describe geometrically impossible isomers (e.g. an
    inverted norbornane bridgehead). For molecules with bridgehead atoms we draw up to 16 random
    isomers and keep the first that passes a quick ETKDG embedding check (RDKit's own
    ``tryEmbedding`` took ~25 s per bridged molecule — too slow at 1M scale). If none embeds, the
    molecule is returned without stereo (the caller then does not use it as a stereo sample).
    """
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    bridged = tetrahedral and rdMolDescriptors.CalcNumBridgeheadAtoms(m) > 0
    opts = StereoEnumerationOptions(
        tryEmbedding=False,
        onlyUnassigned=True,
        maxIsomers=16 if bridged else 1,
        rand=rng.getrandbits(31),
        unique=True,
    )
    out = None
    for iso in EnumerateStereoisomers(m, options=opts):
        if not bridged or _embeddable(iso):
            out = iso
            break
    if out is None:
        return m
    if not tetrahedral:
        for atom in out.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    # round-trip through SMILES so stereo perception/cleanup is canonical
    return Chem.MolFromSmiles(Chem.MolToSmiles(out))
