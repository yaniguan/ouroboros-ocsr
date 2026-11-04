"""Bounded local coordinate sampling around ETKDG parents; no relaxation or downloads."""
import numpy as np
from rdkit import Chem
from ouroboros.geometry.conformers import stereo_matches


def geometry_rejection(mol, smiles, conf_id, min_distance):
    positions = mol.GetConformer(conf_id).GetPositions()
    if not np.isfinite(positions).all():
        return 'nonfinite_positions'
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    if np.any(distances < min_distance):
        return 'minimum_distance'
    if not stereo_matches(mol, smiles, conf_id):
        return 'stereo_mismatch'
    return None


def displaced_conformers(mol, smiles, n_samples, amplitude=.1, seed=0, min_distance=.6):
    """Return a copy with accepted samples and an audit of every attempted sample.

    Each atom moves uniformly in radius from zero to amplitude along an isotropic
    direction. There are exactly n_samples attempts per original parent, no retries.
    Parent coordinates are unchanged. Rejections never receive a conformer ID.
    """
    if (not isinstance(n_samples, int) or n_samples < 0 or not np.isfinite(amplitude)
            or amplitude < 0 or not np.isfinite(min_distance) or min_distance < 0):
        raise ValueError('count, amplitude and minimum distance must be finite and nonnegative')
    out = Chem.Mol(mol)
    rng = np.random.default_rng(seed)
    records = []
    for parent in mol.GetConformers():
        parent_id = parent.GetId()
        base_reason = geometry_rejection(mol, smiles, parent_id, min_distance)
        positions = parent.GetPositions()
        for sample_id in range(1, n_samples+1):
            directions = rng.normal(size=positions.shape)
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
            offsets = directions * rng.uniform(0, amplitude, size=(len(positions), 1))
            conf = Chem.Conformer(parent)
            for i, pos in enumerate(positions + offsets):
                conf.SetAtomPosition(i, pos.tolist())
            cid = out.AddConformer(conf, assignId=True)
            reason = base_reason or geometry_rejection(out, smiles, cid, min_distance)
            rec = dict(parent_conf_id=parent_id, sample_id=sample_id, seed=seed,
                       displacement_max=float(np.linalg.norm(offsets, axis=1).max()),
                       conf_id=cid if reason is None else None, rejection_reason=reason)
            if reason:
                out.RemoveConformer(cid)
            records.append(rec)
    return out, records
