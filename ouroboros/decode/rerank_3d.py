"""3D diastereomer strain reranking and mirror-enantiomer blind-spot detection over OCSR beams.

Why both parts are needed
-------------------------
1. **Diastereomers & constitutional isomers have distinct 3D energies (`ΔE != 0`):**
   Cis/trans ring flips, bridged cage inversions, and bad valence/ring closures produce measurable
   3D steric clashes or MMFF94s strain penalties that ``rerank_beams_3d`` can penalize or reject.
2. **Pure enantiomers are strictly degenerate on achiral force fields (`ΔE = 0`):**
   Every parity-invariant potential (MMFF94s, UFF, MACE-OFF23, ANI-2x) assigns the exact same
   energy to ``(R)`` and ``(S)`` mirror twins. When two top visual beams are mirror enantiomers
   within ``enantiomer_margin``, 3D physics cannot break the tie; ``rerank_beams_3d`` explicitly
   sets ``enantiomer_ambiguity=True`` so downstream callers route the molecule for 2D wedge/hash
   or chiral/NMR verification instead of giving false certainty.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ouroboros.decode.rerank import Candidate, normalized
from ouroboros.eval.standardize import standardize
from ouroboros.geometry.conformers import embed, stereo_matches
from ouroboros.geometry.propagation import categorize


def _eval_3d_candidate(smiles: str, seed: int = 0) -> dict[str, Any]:
    """Embed standardized SMILES in 3D and evaluate clash, stereo retention, and MMFF94s energy."""
    std = standardize(smiles)
    if not std.valid or std.mol is None or std.smiles is None:
        return {
            "valid_2d": False,
            "canonical_smiles": None,
            "embedded_3d": False,
            "no_steric_clash": False,
            "stereo_3d_preserved": False,
            "energy_kcal": None,
            "energy_kcal_per_heavy": None,
            "min_dist_a": None,
        }
    if any(a.GetAtomicNum() == 0 for a in std.mol.GetAtoms()):
        # Pseudo-atoms / R-groups cannot be embedded in 3D force fields without expansion
        return {
            "valid_2d": True,
            "canonical_smiles": std.smiles,
            "embedded_3d": False,
            "no_steric_clash": True,
            "stereo_3d_preserved": True,
            "energy_kcal": None,
            "energy_kcal_per_heavy": None,
            "min_dist_a": None,
            "has_pseudo_atom": True,
        }
    mol_3d, _ = embed(std.smiles, n_conf=2, seed=seed)
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return {
            "valid_2d": True,
            "canonical_smiles": std.smiles,
            "embedded_3d": False,
            "no_steric_clash": False,
            "stereo_3d_preserved": False,
            "energy_kcal": None,
            "energy_kcal_per_heavy": None,
            "min_dist_a": None,
        }
    cids = [c.GetId() for c in mol_3d.GetConformers()]
    n_heavy = max(std.mol.GetNumHeavyAtoms(), 1)
    best_e: float | None = None
    best_cid = cids[0]
    props = AllChem.MMFFGetMoleculeProperties(mol_3d, mmffVariant="MMFF94s")
    for cid in cids:
        ff = (
            AllChem.MMFFGetMoleculeForceField(mol_3d, props, confId=cid)
            if props is not None
            else AllChem.UFFGetMoleculeForceField(mol_3d, confId=cid)
        )
        if ff is not None:
            ff.Minimize(maxIts=80)
            e_val = float(ff.CalcEnergy())
            if math.isfinite(e_val) and (best_e is None or e_val < best_e):
                best_e = e_val
                best_cid = cid
    conf = mol_3d.GetConformer(best_cid)
    pos = np.asarray(conf.GetPositions(), dtype=np.float64)
    if len(pos) >= 2:
        dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        min_dist = float(dmat.min())
    else:
        min_dist = 99.0
    stereo_ok = bool(stereo_matches(mol_3d, std.smiles, best_cid))
    return {
        "valid_2d": True,
        "canonical_smiles": std.smiles,
        "embedded_3d": True,
        "no_steric_clash": min_dist >= 0.75,
        "stereo_3d_preserved": stereo_ok,
        "energy_kcal": round(best_e, 4) if best_e is not None else None,
        "energy_kcal_per_heavy": round(best_e / n_heavy, 4) if best_e is not None else None,
        "min_dist_a": round(min_dist, 4),
    }


def rerank_beams_3d(
    candidates: Sequence[Candidate],
    *,
    scores_are_normalized: bool = True,
    strain_weight: float = 0.03,
    clash_penalty: float = 2.0,
    embed_fail_penalty: float = 1.5,
    enantiomer_margin: float = 0.25,
    max_candidates: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Rerank ``(smiles, score)`` beams with 3D physical checks and flag enantiomer twin ties."""
    if not candidates:
        raise ValueError("candidates must be non-empty")

    # Merge duplicate canonical molecules, keeping best visual score
    dedup: dict[str, tuple[str, float]] = {}
    for raw_smi, raw_score in candidates:
        vis = float(raw_score) if scores_are_normalized else normalized(raw_smi, float(raw_score))
        std = standardize(raw_smi)
        key = std.smiles if (std.valid and std.smiles) else f"__raw__:{raw_smi}"
        if key not in dedup or vis > dedup[key][1]:
            dedup[key] = (raw_smi, vis)

    ordered = sorted(dedup.values(), key=lambda x: x[1], reverse=True)[:max_candidates]
    evaluated: list[dict[str, Any]] = []
    for raw_smi, vis_score in ordered:
        phys = _eval_3d_candidate(raw_smi, seed=seed)
        evaluated.append({"smiles": raw_smi, "visual_score": vis_score, **phys})

    # Reference strain baseline among isomers sharing the same flat connectivity (diastereomers)
    min_energy_by_flat: dict[str, float] = {}
    for item in evaluated:
        if item["canonical_smiles"] and item["energy_kcal"] is not None:
            mol = Chem.MolFromSmiles(item["canonical_smiles"])
            if mol is not None:
                Chem.RemoveStereochemistry(mol)
                flat_key = Chem.MolToSmiles(mol)
                prev = min_energy_by_flat.get(flat_key)
                if prev is None or item["energy_kcal"] < prev:
                    min_energy_by_flat[flat_key] = item["energy_kcal"]
                item["flat_smiles"] = flat_key

    for item in evaluated:
        penalty = 0.0
        if not item["valid_2d"]:
            penalty += 10.0
        elif not item.get("has_pseudo_atom", False):
            if not item["embedded_3d"] or not item["stereo_3d_preserved"]:
                penalty += embed_fail_penalty
            if not item["no_steric_clash"]:
                penalty += clash_penalty
            flat_key = item.get("flat_smiles")
            if flat_key and item["energy_kcal"] is not None:
                rel_strain = max(0.0, item["energy_kcal"] - min_energy_by_flat[flat_key])
                item["diastereomer_rel_kcal"] = round(rel_strain, 4)
                penalty += strain_weight * rel_strain
        item["combined_score"] = round(item["visual_score"] - penalty, 6)

    ranked = sorted(
        evaluated,
        key=lambda x: (x["valid_2d"], x["combined_score"], x["visual_score"]),
        reverse=True,
    )
    winner = ranked[0]

    # Detect mirror-enantiomer twin collisions against the winning candidate
    enantiomer_twins: list[dict[str, Any]] = []
    if winner["canonical_smiles"]:
        for alt in ranked[1:]:
            if not alt["canonical_smiles"]:
                continue
            rel = categorize(alt["canonical_smiles"], winner["canonical_smiles"])
            gap = winner["visual_score"] - alt["visual_score"]
            if rel == "enantiomer" and gap <= enantiomer_margin:
                enantiomer_twins.append(
                    {
                        "smiles": alt["canonical_smiles"],
                        "visual_gap": round(gap, 6),
                    }
                )

    enantiomer_ambiguity = len(enantiomer_twins) > 0
    return {
        "best_smiles": winner["canonical_smiles"] or winner["smiles"],
        "raw_smiles": winner["smiles"],
        "visual_score": winner["visual_score"],
        "combined_score": winner["combined_score"],
        "enantiomer_ambiguity": enantiomer_ambiguity,
        "enantiomer_twins": enantiomer_twins,
        "needs_chiral_review": enantiomer_ambiguity,
        "note": (
            "Top beams contain mirror enantiomers (ΔE_3D = 0 on achiral force fields); "
            "verify 2D wedge/hash orientation or chiral/NMR evidence."
            if enantiomer_ambiguity
            else None
        ),
        "candidates": ranked,
    }


__all__ = ["rerank_beams_3d"]
