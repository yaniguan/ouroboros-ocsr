"""SMILES -> ETKDG conformers -> MACE-OFF relaxation -> lowest-energy conformer.

This is the only E(3)-equivariant stage of the project (MACE: energies invariant, forces
equivariant under rotations, translations and reflections). It is used as an error-propagation
analysis of recognition errors, not as a novel pipeline.

Energies are in eV and forces in eV/Å (MACE-OFF / ASE units). Enantiomers have identical MACE
energies, so energy errors can never reveal a wrong chirality: stereo is evaluated separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

FMAX = 0.05  # eV/Å convergence threshold


def embed_conformers(smiles: str, n_conf: int = 10, seed: int = 0) -> Chem.Mol:
    """Add hydrogens and embed ``n_conf`` ETKDGv3 conformers. Raises ValueError on failure."""
    mol, reason = embed(smiles, n_conf, seed)
    if mol is None:
        raise ValueError(f"{reason}: {smiles!r}")
    return mol


def embed(smiles: str, n_conf: int = 10, seed: int = 0) -> tuple[Chem.Mol | None, str]:
    """ETKDGv3 embedding with a random-coordinate retry. Returns (mol with Hs, "ok") or
    (None, reason)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparsable"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params))
    if not cids:
        params.useRandomCoords = True  # standard fallback for hard ring systems
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params))
        if not cids:
            return None, "etkdg_failed"
    return mol, "ok"


def stereo_from_3d(mol: Chem.Mol, conf_id: int = -1) -> str:
    """Canonical isomeric SMILES with stereo re-perceived from the 3D coordinates only."""
    m = Chem.Mol(mol)
    if conf_id >= 0:
        conf = Chem.Conformer(mol.GetConformer(conf_id))
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
    Chem.RemoveStereochemistry(m)  # forget the input tags: geometry is the only source
    Chem.AssignStereochemistryFrom3D(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def _cip_labels(m: Chem.Mol) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    from rdkit.Chem import rdCIPLabeler

    m = Chem.Mol(m)
    rdCIPLabeler.AssignCIPLabels(m)
    atoms = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    bonds = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetProp("_CIPCode")
        for b in m.GetBonds()
        if b.HasProp("_CIPCode")
    }
    return atoms, bonds


def stereo_matches(mol: Chem.Mol, smiles: str, conf_id: int = -1) -> bool:
    """Does the 3D geometry reproduce every stereo element SPECIFIED in ``smiles``?

    Elements left unspecified in the input are ignored (a 3D structure necessarily picks some
    configuration for them). Compared through CIP labels at the same atom indices (``AddHs``
    appends hydrogens, so heavy-atom indices of the embedded molecule equal the input's).
    """
    ref = Chem.MolFromSmiles(smiles)
    ref_atoms, ref_bonds = _cip_labels(ref)
    if not ref_atoms and not ref_bonds:
        return True
    m = Chem.Mol(mol)
    if conf_id >= 0:
        conf = Chem.Conformer(mol.GetConformer(conf_id))
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
    Chem.RemoveStereochemistry(m)
    Chem.AssignStereochemistryFrom3D(m)
    got_atoms, got_bonds = _cip_labels(Chem.RemoveHs(m))
    return all(got_atoms.get(i) == c for i, c in ref_atoms.items()) and all(
        got_bonds.get(k) == c for k, c in ref_bonds.items()
    )


def stereo_preserved(mol: Chem.Mol, smiles: str) -> list[bool]:
    """Per conformer: does the 3D geometry encode the input's specified stereo elements?"""
    return [stereo_matches(mol, smiles, c.GetId()) for c in mol.GetConformers()]


# ------------------------------------------------------------------------------ MACE


_CALCS: dict = {}


def mace_calculator(model: str = "small", device: str | None = None, dtype: str = "float64"):
    """Cached MACE-OFF23 ASE calculator (weights: Academic Software License, see DATA.md)."""
    import torch
    from mace.calculators import mace_off

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model, device, dtype)
    if key not in _CALCS:
        _CALCS[key] = mace_off(model=model, device=device, default_dtype=dtype)
    return _CALCS[key]


def to_atoms(mol: Chem.Mol, conf_id: int):
    from ase import Atoms

    pos = mol.GetConformer(conf_id).GetPositions()
    return Atoms(symbols=[a.GetSymbol() for a in mol.GetAtoms()], positions=pos)


@dataclass
class Relaxed:
    energy: float  # eV
    forces: np.ndarray  # [n_atoms, 3] eV/Å
    positions: np.ndarray  # [n_atoms, 3] Å
    converged: bool
    steps: int
    fmax: float
    seconds: float


def relax(atoms, calc, fmax: float = FMAX, steps: int = 500, optimizer: str = "LBFGS") -> Relaxed:
    from ase import optimize

    atoms = atoms.copy()
    atoms.calc = calc
    opt = getattr(optimize, optimizer)(atoms, logfile=None)
    t0 = time.perf_counter()
    converged = bool(opt.run(fmax=fmax, steps=steps))
    f = atoms.get_forces()
    return Relaxed(
        energy=float(atoms.get_potential_energy()),
        forces=f,
        positions=atoms.get_positions(),
        converged=converged,
        steps=int(opt.nsteps),
        fmax=float(np.linalg.norm(f, axis=1).max()),
        seconds=time.perf_counter() - t0,
    )


@dataclass
class GeometryResult:
    smiles: str
    status: str  # "ok" | embedding failure reason
    n_conf: int = 0
    energy: float | None = None  # lowest relaxed energy, eV
    forces: np.ndarray | None = None
    positions: np.ndarray | None = None
    symbols: list[str] = field(default_factory=list)
    converged: bool = False
    all_converged: bool = False
    stereo_ok_embed: bool | None = None  # all embedded conformers carry the input stereo
    stereo_ok_final: bool | None = None  # the chosen relaxed conformer carries the input stereo
    seconds: float = 0.0
    conformer_energies: list[float] = field(default_factory=list)


def lowest_energy_conformer(
    smiles: str,
    n_conf: int = 10,
    seed: int = 0,
    calc=None,
    fmax: float = FMAX,
    steps: int = 500,
    mmff_prescreen: int | None = None,
) -> GeometryResult:
    """Embed ``n_conf`` conformers, relax each with MACE-OFF, return the lowest-energy one.

    ``mmff_prescreen=k`` relaxes all conformers with MMFF first and sends only the k lowest to
    MACE (a speed option; off by default).
    """
    t0 = time.perf_counter()
    mol, status = embed(smiles, n_conf, seed)
    if mol is None:
        return GeometryResult(smiles, status, seconds=time.perf_counter() - t0)
    cids = [c.GetId() for c in mol.GetConformers()]
    stereo_embed = all(stereo_preserved(mol, smiles))
    if mmff_prescreen:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        order = np.argsort([e for _, e in res])
        cids = [cids[i] for i in order[:mmff_prescreen]]
    calc = calc or mace_calculator()
    relaxed = [relax(to_atoms(mol, c), calc, fmax, steps) for c in cids]
    best = int(np.argmin([r.energy for r in relaxed]))
    r = relaxed[best]
    final = Chem.Mol(mol)
    final.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for i, p in enumerate(r.positions):
        conf.SetAtomPosition(i, p.tolist())
    final.AddConformer(conf, assignId=True)
    return GeometryResult(
        smiles=smiles,
        status="ok",
        n_conf=len(relaxed),
        energy=r.energy,
        forces=r.forces,
        positions=r.positions,
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        converged=r.converged,
        all_converged=all(x.converged for x in relaxed),
        stereo_ok_embed=stereo_embed,
        stereo_ok_final=stereo_matches(final, smiles, 0),
        seconds=time.perf_counter() - t0,
        conformer_energies=[x.energy for x in relaxed],
    )


def mirror_positions(positions: np.ndarray) -> np.ndarray:
    """Reflect through the yz-plane (x -> -x): the enantiomer's geometry."""
    out = np.array(positions, copy=True)
    out[:, 0] *= -1
    return out
