"""Bridgehead exit-vector torsion scanning, off-equilibrium rattle & PES softening diagnostics.

Extends PES Softening Slope alpha (Deng, Persson & Ceder, npj Comput. Mater. 2025,
10.1038/s41524-024-01500-6) to strained 3D organic bioisosteres, tracking transannular
bridgehead C1...C3 compression (~1.85 A in BCP) alongside 1D dihedral barriers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from ouroboros.active.motifs import BioisostereRecord
from ouroboros.geometry.displacements import displaced_conformers


@dataclass
class ConformerFrame:
    frame_id: str
    mol_id: str
    smiles: str
    connectivity_key: str
    group: str
    family: str
    source: str  # "equilibrium" | "torsion" | "displaced"
    dihedral_deg: float | None
    transannular_c1_c3_a: float | None
    displacement_a: float
    atomic_numbers: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class SofteningDiagnostics:
    mol_id: str
    group: str
    family: str
    n_frames: int
    softening_slope_alpha: float
    barrier_height_error_kcal: float
    rel_energy_mae_kcal: float
    force_rmse_kcal_a: float


def find_bridgehead_transannular_pair(mol: Chem.Mol) -> tuple[int, int] | None:
    """Locate the shortest nonbonded bridgehead C1...C3 pair inside a strained cage/spiro ring."""
    ring_info = mol.GetRingInfo()
    bridgeheads = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and ring_info.NumAtomRings(a.GetIdx()) >= 2
    ]
    if len(bridgeheads) < 2:
        return None
    dmat = Chem.GetDistanceMatrix(mol)
    best_pair: tuple[int, int] | None = None
    best_top = 99
    for i in range(len(bridgeheads)):
        for j in range(i + 1, len(bridgeheads)):
            u, v = bridgeheads[i], bridgeheads[j]
            top_d = int(dmat[u, v])
            if 2 <= top_d < best_top:
                best_top = top_d
                best_pair = (u, v)
    return best_pair


def find_exit_vector_dihedral(mol: Chem.Mol) -> tuple[int, int, int, int] | None:
    """Choose the primary acyclic exit-vector bond (prioritizing bridgehead/S-F5 attachments)."""
    ring_info = mol.GetRingInfo()
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        au, av = mol.GetAtomWithIdx(u), mol.GetAtomWithIdx(v)
        if au.GetAtomicNum() == 1 or av.GetAtomicNum() == 1:
            continue
        nu = [nb.GetIdx() for nb in au.GetNeighbors() if nb.GetIdx() != v]
        nv = [nb.GetIdx() for nb in av.GetNeighbors() if nb.GetIdx() != u]
        if not nu or not nv:
            continue
        # Prefer bonds attached to S(VI)F5, multi-ring bridgeheads, or ring atoms.
        has_sf5 = int(au.GetAtomicNum() == 16 or av.GetAtomicNum() == 16)
        max_rings = max(ring_info.NumAtomRings(u), ring_info.NumAtomRings(v))
        heavy_deg = sum(1 for idx in nu + nv if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)
        nu_heavy = [idx for idx in nu if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1] or nu
        nv_heavy = [idx for idx in nv if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1] or nv
        quad = (nu_heavy[0], u, v, nv_heavy[0])
        candidates.append(((-has_sf5, -max_rings, -heavy_deg), quad))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _embed_relaxed_parent(smiles: str, seed: int = 0) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.enforceChirality = True
    params.maxIterations = 60
    if AllChem.EmbedMolecule(mol, params=params) < 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params=params) < 0:
            raise ValueError(f"failed to embed {smiles!r}")
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is not None:
            ff.Minimize(maxIts=80)
    return mol


def generate_blindspot_frames(
    record: BioisostereRecord,
    *,
    torsion_step_deg: float = 45.0,
    n_displaced: int = 2,
    rattle_amplitude: float = 0.06,
    seed: int = 0,
) -> list[ConformerFrame]:
    """Generate equilibrium + 1D exit-vector torsion scan + rattled off-equilibrium 3D frames."""
    mol = _embed_relaxed_parent(record.smiles, seed=seed)
    z_arr = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    c1_c3 = find_bridgehead_transannular_pair(mol)
    dihedral = find_exit_vector_dihedral(mol)
    eq_pos = mol.GetConformer(0).GetPositions().astype(np.float64)

    def _c13_dist(pos: np.ndarray) -> float | None:
        if c1_c3 is None:
            return None
        return float(np.linalg.norm(pos[c1_c3[0]] - pos[c1_c3[1]]))

    frames: list[ConformerFrame] = []
    top_dmat = Chem.GetDistanceMatrix(mol)
    nonbonded_mask = top_dmat >= 3
    if dihedral is not None and torsion_step_deg > 0:
        n_steps = max(2, int(round(360.0 / torsion_step_deg)))
        a, u, v, b = dihedral
        base_ang = float(rdMolTransforms.GetDihedralDeg(mol.GetConformer(0), a, u, v, b))
        for k in range(n_steps):
            work = Chem.Mol(mol)
            conf = work.GetConformer(0)
            target_deg = (base_ang + k * torsion_step_deg) % 360.0
            rdMolTransforms.SetDihedralDeg(conf, a, u, v, b, target_deg)
            pos = conf.GetPositions().astype(np.float64)
            # Skip unphysical rigid-scan steric overlaps into covalent range (<1.65 A nonbonded)
            dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            if k > 0 and (
                float(dmat.min()) < 0.85
                or (np.any(nonbonded_mask) and float(dmat[nonbonded_mask].min()) < 1.65)
            ):
                continue
            disp = float(np.sqrt(np.mean(np.sum((pos - eq_pos) ** 2, axis=1))))
            frames.append(
                ConformerFrame(
                    frame_id=f"{record.mol_id}_tor_{k:02d}",
                    mol_id=record.mol_id,
                    smiles=record.smiles,
                    connectivity_key=record.connectivity_key,
                    group=record.group,
                    family=record.family,
                    source="equilibrium" if k == 0 else "torsion",
                    dihedral_deg=round(target_deg, 2),
                    transannular_c1_c3_a=_c13_dist(pos),
                    displacement_a=disp,
                    atomic_numbers=z_arr.copy(),
                    positions=pos,
                )
            )
    else:
        frames.append(
            ConformerFrame(
                frame_id=f"{record.mol_id}_eq_00",
                mol_id=record.mol_id,
                smiles=record.smiles,
                connectivity_key=record.connectivity_key,
                group=record.group,
                family=record.family,
                source="equilibrium",
                dihedral_deg=None,
                transannular_c1_c3_a=_c13_dist(eq_pos),
                displacement_a=0.0,
                atomic_numbers=z_arr.copy(),
                positions=eq_pos,
            )
        )

    if n_displaced > 0:
        disp_mol, disp_recs = displaced_conformers(
            mol, record.smiles, n_samples=n_displaced, amplitude=rattle_amplitude, seed=seed + 13
        )
        for rec in disp_recs:
            cid = rec.get("conf_id")
            if cid is None:
                continue
            pos = disp_mol.GetConformer(int(cid)).GetPositions().astype(np.float64)
            frames.append(
                ConformerFrame(
                    frame_id=f"{record.mol_id}_disp_{int(rec['sample_id']):02d}",
                    mol_id=record.mol_id,
                    smiles=record.smiles,
                    connectivity_key=record.connectivity_key,
                    group=record.group,
                    family=record.family,
                    source="displaced",
                    dihedral_deg=None,
                    transannular_c1_c3_a=_c13_dist(pos),
                    displacement_a=float(rec["displacement_max"]),
                    atomic_numbers=z_arr.copy(),
                    positions=pos,
                )
            )
    return frames


def compute_softening_diagnostics(
    frames: Sequence[ConformerFrame],
    pred_energies_kcal: Sequence[float],
    pred_forces_kcal_a: Sequence[np.ndarray],
    ref_energies_kcal: Sequence[float],
    ref_forces_kcal_a: Sequence[np.ndarray],
) -> SofteningDiagnostics:
    """Compute per-molecule conformational relative-energy MAE, barrier error & softening alpha."""
    if not frames:
        raise ValueError("frames must be non-empty")
    e_pred = np.asarray(pred_energies_kcal, dtype=float)
    e_ref = np.asarray(ref_energies_kcal, dtype=float)
    de_pred = e_pred - float(e_pred.min())
    de_ref = e_ref - float(e_ref.min())

    var_ref = float(np.var(de_ref))
    if var_ref > 1e-8 and len(frames) >= 2:
        cov = float(np.mean((de_ref - de_ref.mean()) * (de_pred - de_pred.mean())))
        alpha = cov / var_ref
    else:
        alpha = 1.0

    barrier_err = float(abs(de_pred.max() - de_ref.max()))
    rel_mae = float(np.mean(np.abs(de_pred - de_ref)))
    f_diffs = [
        np.asarray(fp, dtype=float) - np.asarray(fr, dtype=float)
        for fp, fr in zip(pred_forces_kcal_a, ref_forces_kcal_a, strict=True)
    ]
    all_sq = np.concatenate([d.ravel() ** 2 for d in f_diffs])
    f_rmse = float(np.sqrt(np.mean(all_sq))) if all_sq.size else 0.0

    f0 = frames[0]
    return SofteningDiagnostics(
        mol_id=f0.mol_id,
        group=f0.group,
        family=f0.family,
        n_frames=len(frames),
        softening_slope_alpha=float(alpha),
        barrier_height_error_kcal=barrier_err,
        rel_energy_mae_kcal=rel_mae,
        force_rmse_kcal_a=f_rmse,
    )
