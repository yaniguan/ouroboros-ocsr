"""Continuous 3D thermodynamic strain, steric clash & CIP relaxation flip evaluation.

Replaces brittle binary topological Bredt rules (McDermott, Houk & Garg, Science 2024)
with continuous 3D conformer energetics and PoseBusters-style nonbonded clash checks, while
guaranteeing exact O(3) parity symmetry (E(M) == E(mirror(M))) via canonical mirror embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ouroboros.eval.standardize import standardize
from ouroboros.geometry.conformers import stereo_matches, to_atoms
from ouroboros.geometry.propagation import mirror_smiles
from ouroboros.phys.epimers import expand_superatoms

EV_TO_KCAL = 23.06054783061903
RT_298K_KCAL = 0.5924849  # R * 298.15 K in kcal/mol
_STRAIN_CACHE: dict[tuple[str, int, int, int, float], StrainResult] = {}


@dataclass
class StrainResult:
    valid: bool
    embed_ok: bool
    canonical_smiles: str | None
    n_conf: int = 0
    min_energy_kcal: float | None = None
    boltzmann_energy_kcal: float | None = None
    per_heavy_energy_kcal: float | None = None
    max_atom_strain_kcal: float = 0.0
    min_nonbonded_dist_a: float | None = None
    has_steric_clash: bool = False
    cip_preserved: bool = True
    conformer_energies_kcal: list[float] = field(default_factory=list)
    boltzmann_weights: list[float] = field(default_factory=list)
    mol_3d: Chem.Mol | None = field(default=None, repr=False)
    intrinsic_penalty_kcal: float = 0.0


def _bounded_embed(smiles: str, n_conf: int = 8, seed: int = 0) -> tuple[Chem.Mol | None, str]:
    """Fast bounded ETKDGv3 embedding that rejects impossible inside-out bridgeheads in <50 ms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "unparsable"
    mol = Chem.AddHs(mol)
    for random_coords, max_it in ((False, 30), (True, 20)):
        probe = AllChem.ETKDGv3()
        probe.randomSeed = seed
        probe.numThreads = 1
        probe.enforceChirality = True
        probe.useRandomCoords = random_coords
        probe.maxIterations = max_it
        if AllChem.EmbedMolecule(mol, params=probe) < 0:
            continue
        if n_conf > 1:
            batch = AllChem.ETKDGv3()
            batch.randomSeed = seed
            batch.numThreads = 1
            batch.enforceChirality = True
            batch.useRandomCoords = random_coords
            batch.maxIterations = max_it
            batch.clearConfs = False
            AllChem.EmbedMultipleConfs(mol, numConfs=n_conf - 1, params=batch)
        if mol.GetNumConformers() > 0:
            return mol, "ok"
    return None, "embed_failed"


def embed_parity_invariant(
    canonical_smiles: str, n_conf: int = 8, seed: int = 0
) -> tuple[Chem.Mol | None, str]:
    """Embed 3D conformers so enantiomers (M, mirror(M)) share identical 3D geometries up to
    spatial reflection (x -> -x), eliminating finite-sample Monte Carlo asymmetry."""
    mol_chk = Chem.MolFromSmiles(canonical_smiles)
    if mol_chk is None:
        return None, "unparsable"
    mirrored = Chem.CanonSmiles(mirror_smiles(canonical_smiles))
    if mirrored >= canonical_smiles:
        return _bounded_embed(canonical_smiles, n_conf=n_conf, seed=seed)

    ref_mol, reason = _bounded_embed(mirrored, n_conf=n_conf, seed=seed)
    if ref_mol is None:
        return None, reason

    # Invert chiral tags and reflect x -> -x on ref_mol, then reorder atoms to match target.
    refl = Chem.Mol(ref_mol)
    for atom in refl.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    for conf in refl.GetConformers():
        pos = conf.GetPositions()
        pos[:, 0] *= -1.0
        for idx, xyz in enumerate(pos):
            conf.SetAtomPosition(idx, xyz.tolist())
    Chem.AssignStereochemistryFrom3D(refl)

    target = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles))
    match = refl.GetSubstructMatch(target, useChirality=True)
    if len(match) != target.GetNumAtoms():
        match = refl.GetSubstructMatch(target, useChirality=False)
    if len(match) != target.GetNumAtoms():
        return _bounded_embed(canonical_smiles, n_conf=n_conf, seed=seed)

    out = Chem.Mol(target)
    out.RemoveAllConformers()
    for conf in refl.GetConformers():
        new_conf = Chem.Conformer(target.GetNumAtoms())
        pos = conf.GetPositions()
        for tgt_idx, src_idx in enumerate(match):
            new_conf.SetAtomPosition(tgt_idx, pos[src_idx].tolist())
        out.AddConformer(new_conf, assignId=True)
    return out, "ok"


def _nonbonded_min_distance(mol: Chem.Mol, conf_id: int) -> tuple[float, bool]:
    """Minimum distance across atom pairs >=3 bonds apart, plus PoseBusters clash flag."""
    n = mol.GetNumAtoms()
    if n < 2:
        return 99.0, False
    pos = mol.GetConformer(conf_id).GetPositions()
    dmat = Chem.GetDistanceMatrix(mol)
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    min_d = 99.0
    clash = False
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i, j] < 3:
                continue
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist < min_d:
                min_d = dist
            cutoff = 1.55 if (atomic_nums[i] == 1 or atomic_nums[j] == 1) else 2.05
            if dist < cutoff:
                clash = True
    return min_d, clash


def _ring_angle_strain_outlier(mol: Chem.Mol, conf_id: int) -> float:
    """Maximum local valence-angle distortion penalty (kcal/mol proxy) on heavy atoms."""
    pos = mol.GetConformer(conf_id).GetPositions()
    ideal_deg = {
        Chem.HybridizationType.SP: 180.0,
        Chem.HybridizationType.SP2: 120.0,
        Chem.HybridizationType.SP3: 109.47,
    }
    max_pen = 0.0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 or atom.GetDegree() < 2:
            continue
        theta0 = ideal_deg.get(atom.GetHybridization(), 109.47)
        nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
        p0 = pos[atom.GetIdx()]
        atom_pen = 0.0
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                v1 = pos[nbrs[a_idx]] - p0
                v2 = pos[nbrs[b_idx]] - p0
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom < 1e-8:
                    continue
                cos_t = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                deg = float(np.degrees(np.arccos(cos_t)))
                delta = max(0.0, abs(deg - theta0) - 12.0)
                atom_pen += 0.012 * (delta**2)
        if atom_pen > max_pen:
            max_pen = atom_pen
    return float(max_pen)


def _eval_mmff_or_uff(mol: Chem.Mol, relax_steps: int) -> list[float]:
    energies: list[float] = []
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    for conf in mol.GetConformers():
        cid = conf.GetId()
        ff = (
            AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if props is not None
            else None
        )
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        if ff is None:
            energies.append(0.0)
            continue
        if relax_steps > 0:
            ff.Minimize(maxIts=relax_steps)
        energies.append(float(ff.CalcEnergy()))
    return energies


def _eval_ase_calc(mol: Chem.Mol, calc, relax_steps: int) -> tuple[list[float], float]:
    from ouroboros.geometry.conformers import relax

    energies: list[float] = []
    node_outlier = 0.0
    for conf in mol.GetConformers():
        cid = conf.GetId()
        atoms = to_atoms(mol, cid)
        if relax_steps > 0:
            res = relax(atoms, calc, steps=relax_steps)
            for idx, xyz in enumerate(res.positions):
                conf.SetAtomPosition(idx, xyz.tolist())
            energies.append(res.energy * EV_TO_KCAL)
        else:
            atoms.calc = calc
            energies.append(float(atoms.get_potential_energy()) * EV_TO_KCAL)
        res_dict = getattr(calc, "results", {}) or {}
        if "node_energy" in res_dict and res_dict["node_energy"] is not None:
            ne = np.asarray(res_dict["node_energy"], dtype=float) * EV_TO_KCAL
            if ne.size:
                med = float(np.median(ne))
                node_outlier = max(node_outlier, float(np.max(ne - med)))
    return energies, node_outlier


def evaluate_3d_strain(
    smiles: str | None,
    *,
    n_conf: int = 8,
    seed: int = 0,
    relax_steps: int = 80,
    calc=None,
    embed_fail_penalty_kcal: float = 45.0,
) -> StrainResult:
    """Embed conformers, compute Boltzmann weights, clash checks, CIP preservation & 3D strain."""
    expanded = expand_superatoms(smiles) if smiles else None
    std = standardize(expanded)
    if not std.valid or std.mol is None or std.smiles is None:
        return StrainResult(
            valid=False,
            embed_ok=False,
            canonical_smiles=None,
            intrinsic_penalty_kcal=embed_fail_penalty_kcal,
        )
    if any(a.GetAtomicNum() == 0 for a in std.mol.GetAtoms()):
        return StrainResult(
            valid=False,
            embed_ok=False,
            canonical_smiles=std.smiles,
            intrinsic_penalty_kcal=embed_fail_penalty_kcal,
        )

    cache_key = (std.smiles, n_conf, seed, relax_steps, embed_fail_penalty_kcal)
    if calc is None and cache_key in _STRAIN_CACHE:
        return _STRAIN_CACHE[cache_key]

    mol_3d, _reason = embed_parity_invariant(std.smiles, n_conf=n_conf, seed=seed)
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        failed = StrainResult(
            valid=True,
            embed_ok=False,
            canonical_smiles=std.smiles,
            cip_preserved=False,
            intrinsic_penalty_kcal=embed_fail_penalty_kcal,
        )
        if calc is None:
            _STRAIN_CACHE[cache_key] = failed
        return failed

    if calc is not None:
        energies, node_outlier = _eval_ase_calc(mol_3d, calc, relax_steps)
    else:
        energies = _eval_mmff_or_uff(mol_3d, relax_steps)
        node_outlier = 0.0

    confs = list(mol_3d.GetConformers())
    e_arr = np.asarray(energies, dtype=float)
    best_i = int(np.argmin(e_arr))
    best_cid = confs[best_i].GetId()
    e_min = float(e_arr[best_i])

    rel = np.clip((e_arr - e_min) / RT_298K_KCAL, 0.0, 80.0)
    raw_w = np.exp(-rel)
    weights = (raw_w / raw_w.sum()).tolist()
    e_boltz = float(np.dot(weights, e_arr))

    min_d, clash = _nonbonded_min_distance(mol_3d, best_cid)
    angle_outlier = _ring_angle_strain_outlier(mol_3d, best_cid)
    max_atom_strain = max(node_outlier, angle_outlier)
    cip_ok = stereo_matches(mol_3d, std.smiles, best_cid)

    n_heavy = max(1, std.mol.GetNumHeavyAtoms())
    per_heavy = e_min / n_heavy
    penalty = (
        max(0.0, per_heavy - 3.5) * 2.0
        + 0.35 * max_atom_strain
        + (18.0 if clash else 0.0)
        + (25.0 if not cip_ok else 0.0)
    )

    result = StrainResult(
        valid=True,
        embed_ok=True,
        canonical_smiles=std.smiles,
        n_conf=len(confs),
        min_energy_kcal=e_min,
        boltzmann_energy_kcal=e_boltz,
        per_heavy_energy_kcal=per_heavy,
        max_atom_strain_kcal=max_atom_strain,
        min_nonbonded_dist_a=min_d,
        has_steric_clash=clash,
        cip_preserved=cip_ok,
        conformer_energies_kcal=[float(x) for x in e_arr],
        boltzmann_weights=[float(w) for w in weights],
        mol_3d=mol_3d,
        intrinsic_penalty_kcal=float(penalty),
    )
    if calc is None:
        if len(_STRAIN_CACHE) >= 4096:
            _STRAIN_CACHE.clear()
        _STRAIN_CACHE[cache_key] = result
    return result
