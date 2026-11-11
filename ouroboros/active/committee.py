"""Offset-free MLIP committee force disagreement & LCMD diversity-aware batch selection.

Implements Query-by-Committee atomic force disagreement sigma_F(R), per-molecule offset-removed
relative energy variance sigma_dE(R), E(3)-invariant radial-chemical environment descriptors,
and 5 batch active acquisition strategies (Zaverkin et al., Digital Discovery 2022;
Holzmuller et al., JMLR 2023; Kulichenko et al., J. Phys. Chem. Lett. 2023).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ouroboros.active.torsion import ConformerFrame

SELECTORS: tuple[str, ...] = (
    "random",
    "max_displacement",
    "energy_std",
    "qbc_force",
    "qbc_lcmd_diverse",
)


@dataclass
class CommitteeFrameScore:
    frame_id: str
    mol_id: str
    group: str
    family: str
    mean_energy_kcal: float
    mean_forces_kcal_a: np.ndarray
    rel_energy_std_kcal: float
    sigma_f_max: float
    sigma_f_mean: float
    displacement_a: float
    descriptor: np.ndarray


def compute_frame_descriptor(frame: ConformerFrame, n_rbf: int = 12) -> np.ndarray:
    """E(3)-invariant molecular radial-chemical fingerprint for diversity-aware active selection."""
    pos = frame.positions
    z = frame.atomic_numbers
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(len(z), k=1)
    r = dist[iu]
    zi, zj = z[iu[0]], z[iu[1]]

    centers = np.linspace(1.3, 4.0, n_rbf)
    gamma = 12.0
    rbf = np.exp(-gamma * ((r[:, None] - centers[None, :]) ** 2))  # (P, n_rbf)

    channels = [
        ((zi > 1) & (zj > 1)).astype(float),  # heavy-heavy (captures 1.85-2.20 A cages!)
        (((zi == 6) & (zj > 6)) | ((zj == 6) & (zi > 6))).astype(float),  # C-heteroatom
        ((zi == 9) | (zj == 9)).astype(float),  # fluorinated / SF5 contacts
        ((zi == 1) | (zj == 1)).astype(float),  # H-X steric contacts
    ]
    feats = []
    n_heavy = max(1.0, float(np.sum(z > 1)))
    for ch in channels:
        feats.append((rbf * ch[:, None]).sum(axis=0) / n_heavy)
    vec = np.concatenate(feats, axis=0)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-8)


def evaluate_committee(
    frames: Sequence[ConformerFrame], committee: Sequence
) -> list[CommitteeFrameScore]:
    """Score frames with an M-member MLIP committee, subtracting per-molecule energy offsets."""
    if not frames:
        return []
    if len(committee) < 2:
        raise ValueError("committee must contain at least 2 members")

    m_count = len(committee)
    n_frames = len(frames)
    e_mat = np.zeros((m_count, n_frames), dtype=np.float64)
    f_list: list[list[np.ndarray]] = [[] for _ in range(n_frames)]

    for m_idx, member in enumerate(committee):
        for f_idx, frame in enumerate(frames):
            e_val, f_val = member.predict_frame(frame)
            e_mat[m_idx, f_idx] = e_val
            f_list[f_idx].append(np.asarray(f_val, dtype=np.float64))

    # Subtract each model's mean energy per molecule so constant zero-point offsets cancel!
    by_mol: dict[str, list[int]] = {}
    for idx, frame in enumerate(frames):
        by_mol.setdefault(frame.mol_id, []).append(idx)

    centered_e = np.zeros_like(e_mat)
    for idxs in by_mol.values():
        mol_means = e_mat[:, idxs].mean(axis=1, keepdims=True)
        centered_e[:, idxs] = e_mat[:, idxs] - mol_means

    rel_e_std = centered_e.std(axis=0)
    mean_e = e_mat.mean(axis=0)

    out: list[CommitteeFrameScore] = []
    for idx, frame in enumerate(frames):
        stack_f = np.stack(f_list[idx], axis=0)  # (M, N, 3)
        f_mean = stack_f.mean(axis=0)
        # Per-atom RMS force disagreement across committee: sqrt(1/M sum_m ||F_im - <F_i>||^2)
        f_sq = np.sum((stack_f - f_mean[None, :, :]) ** 2, axis=-1)
        atom_disagree = np.sqrt(np.mean(f_sq, axis=0))
        out.append(
            CommitteeFrameScore(
                frame_id=frame.frame_id,
                mol_id=frame.mol_id,
                group=frame.group,
                family=frame.family,
                mean_energy_kcal=float(mean_e[idx]),
                mean_forces_kcal_a=f_mean,
                rel_energy_std_kcal=float(rel_e_std[idx]),
                sigma_f_max=float(atom_disagree.max()),
                sigma_f_mean=float(atom_disagree.mean()),
                displacement_a=frame.displacement_a,
                descriptor=compute_frame_descriptor(frame),
            )
        )
    return out


def select_active_batch(
    scores: Sequence[CommitteeFrameScore],
    budget: int,
    *,
    strategy: str = "qbc_lcmd_diverse",
    seed: int = 0,
    uncertainty_power: float = 1.25,
) -> list[int]:
    """Select `budget` candidate indices from `scores` using one of the 5 acquisition rules."""
    if strategy not in SELECTORS:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {SELECTORS}")
    n = len(scores)
    if budget <= 0 or n == 0:
        return []
    k = min(budget, n)

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return [int(i) for i in rng.choice(n, size=k, replace=False)]

    if strategy == "max_displacement":
        order = np.argsort([-s.displacement_a for s in scores], kind="mergesort")
        return [int(i) for i in order[:k]]

    if strategy == "energy_std":
        order = np.argsort([-s.rel_energy_std_kcal for s in scores], kind="mergesort")
        return [int(i) for i in order[:k]]

    if strategy == "qbc_force":
        order = np.argsort([-s.sigma_f_max for s in scores], kind="mergesort")
        return [int(i) for i in order[:k]]

    # qbc_lcmd_diverse: Uncertainty-weighted farthest-point / LCMD greedy kernel cover.
    # Balances high atomic force disagreement sigma_F with cross-scaffold coverage in descriptor
    # space so queries never collapse onto adjacent torsion steps of a single molecule.
    desc = np.stack([s.descriptor for s in scores], axis=0)  # (N, D)
    uq = np.array([max(s.sigma_f_max, 1e-6) for s in scores], dtype=np.float64)
    uq_weight = (uq / float(uq.max())) ** uncertainty_power

    first = int(np.argmax(uq_weight))
    chosen = [first]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[first] = True
    min_sq_dist = np.sum((desc - desc[first : first + 1]) ** 2, axis=1)
    mol_counts: dict[str, int] = {scores[first].mol_id: 1}

    while len(chosen) < k:
        # Mild diminishing return on repeated queries from the exact same scaffold
        scaf_factor = np.array(
            [1.0 / (1.0 + 0.6 * mol_counts.get(s.mol_id, 0)) for s in scores], dtype=np.float64
        )
        acq = (min_sq_dist + 0.04) * uq_weight * scaf_factor
        acq[selected_mask] = -1.0
        nxt = int(np.argmax(acq))
        chosen.append(nxt)
        selected_mask[nxt] = True
        mol_counts[scores[nxt].mol_id] = mol_counts.get(scores[nxt].mol_id, 0) + 1
        sq_d = np.sum((desc - desc[nxt : nxt + 1]) ** 2, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, sq_d)

    return chosen
