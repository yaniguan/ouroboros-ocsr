"""Replay-protected conservative E(3)-invariant shallow-ensemble active distillation.

Combines connectivity-disjoint splits (ouroboros.geometry.splits.connectivity_key), a smooth
E(3)-invariant multi-body + 2-stage radial message-passing trunk with a K-head Kellner-Ceriotti
Shallow Ensemble readout (Kellner & Ceriotti, MLST 2024, 10.1088/2632-2153/ad594a), exact
conservative autograd forces (F = -dE/dR), and an In-Distribution Control Replay Buffer preventing
catastrophic forgetting (Kaur, Batatia & Csanyi, Faraday Discuss. 2025, 10.1039/D4FD00107A).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from ouroboros.active.committee import evaluate_committee, select_active_batch
from ouroboros.active.motifs import BioisostereRecord
from ouroboros.active.oracle import (
    EV_TO_KCAL,
    AnalyticalCageMember,
    QMLabel,
    QMOracleSpec,
    build_pretrained_baseline_committee,
    evaluate_qm_oracle,
    verify_homogeneous_provenance,
)
from ouroboros.active.torsion import (
    ConformerFrame,
    SofteningDiagnostics,
    compute_softening_diagnostics,
)
from ouroboros.geometry.splits import connectivity_key

ELEMENT_VOCAB: tuple[int, ...] = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
_Z_TO_IDX: dict[int, int] = {z: i for i, z in enumerate(ELEMENT_VOCAB)}


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"n": n, "mean": mean, "ci_low": float(lo), "ci_high": float(hi)}


def mann_whitney_u_pvalue(sample_high: Sequence[float], sample_low: Sequence[float]) -> float:
    """One-sided Mann-Whitney U p-value testing P(sample_high > sample_low) with tie correction."""
    x = np.asarray(sample_high, dtype=float)
    y = np.asarray(sample_low, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 1.0
    diff = x[:, None] - y[None, :]
    u = float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))
    mu = 0.5 * n1 * n2
    all_vals = np.concatenate([x, y])
    _, counts = np.unique(all_vals, return_counts=True)
    n_tot = n1 + n2
    tie_term = float(np.sum(counts**3 - counts)) / (n_tot * (n_tot - 1)) if n_tot > 1 else 0.0
    var = (n1 * n2 / 12.0) * ((n_tot + 1.0) - tie_term)
    if var <= 1e-12:
        return 0.0 if u > mu else 1.0
    z = (u - mu) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _hash_key(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def split_records_by_connectivity(
    records: Sequence[BioisostereRecord],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[str, list[BioisostereRecord]]:
    """Stratified connectivity-disjoint split ensuring zero shared InChIKey connectivity blocks."""
    by_ckey: dict[str, list[BioisostereRecord]] = {}
    for rec in records:
        ck = rec.connectivity_key or connectivity_key(rec.smiles)
        by_ckey.setdefault(ck, []).append(rec)

    by_stratum: dict[tuple[str, str], list[str]] = {}
    for ck, recs in by_ckey.items():
        r0 = recs[0]
        by_stratum.setdefault((r0.group, r0.family), []).append(ck)

    buckets: dict[str, list[BioisostereRecord]] = {"train": [], "val": [], "test": []}
    f_train, f_val, _f_test = fractions
    heldout_strata: list[list[str]] = []

    for (_grp, _fam), ckeys in sorted(by_stratum.items()):
        ordered = sorted(ckeys, key=lambda c: _hash_key(seed, c))
        m = len(ordered)
        if m == 1:
            buckets["train"].extend(by_ckey[ordered[0]])
        elif m == 2:
            # Always keep 1 connectivity scaffold in train and 1 for val/test stratification
            buckets["train"].extend(by_ckey[ordered[0]])
            heldout_strata.append([ordered[1]])
        else:
            n_tr = max(1, int(round(f_train * m)))
            n_va = max(1, int(round(f_val * m))) if m - n_tr >= 2 else 0
            for c in ordered[:n_tr]:
                buckets["train"].extend(by_ckey[c])
            for c in ordered[n_tr : n_tr + n_va]:
                buckets["val"].extend(by_ckey[c])
            for c in ordered[n_tr + n_va :]:
                buckets["test"].extend(by_ckey[c])

    # Partition 2-per-family heldout scaffolds across train/val/test while guaranteeing test
    # includes diverse bioisosteres (cages + hypervalent SF5) and controls.
    if heldout_strata:
        flat_held = [c for sub in heldout_strata for c in sub]
        bio_held = [c for c in flat_held if by_ckey[c][0].group == "bioisostere"]
        ctl_held = [c for c in flat_held if by_ckey[c][0].group == "control"]
        for pool in (bio_held, ctl_held):
            # Sort so sf5_polyfluoro and diverse cage families appear first in test
            pool_sorted = sorted(
                pool,
                key=lambda c: (
                    0 if by_ckey[c][0].family == "sf5_polyfluoro" else 1,
                    _hash_key(seed + 7, c),
                ),
            )
            n_te = max(1, min(len(pool_sorted) - 1, max(3, int(round(0.5 * len(pool_sorted))))))
            n_va = max(1, min(len(pool_sorted) - n_te, 2))
            for c in pool_sorted[:n_te]:
                buckets["test"].extend(by_ckey[c])
            for c in pool_sorted[n_te : n_te + n_va]:
                buckets["val"].extend(by_ckey[c])
            for c in pool_sorted[n_te + n_va :]:
                buckets["train"].extend(by_ckey[c])

    ck_tr = {r.connectivity_key for r in buckets["train"]}
    ck_va = {r.connectivity_key for r in buckets["val"]}
    ck_te = {r.connectivity_key for r in buckets["test"]}
    if (ck_tr & ck_va) or (ck_tr & ck_te) or (ck_va & ck_te):
        raise AssertionError("connectivity leakage detected across train/val/test")
    return buckets


class ConservativeShallowEnsembleDeltaMLIP(nn.Module):
    """E(3)-invariant multi-body radial + 2-stage MPNN trunk with K shallow-ensemble heads."""

    def __init__(
        self,
        *,
        hidden_dim: int = 24,
        n_rbf: int = 16,
        n_heads: int = 6,
        r_cut: float = 4.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.n_rbf = n_rbf
        self.n_heads = n_heads
        self.r_cut = r_cut
        self.feat_dim = 5 * n_rbf + hidden_dim

        centers = torch.linspace(1.30, 3.85, n_rbf, dtype=torch.float64)
        self.register_buffer("rbf_centers", centers)
        self.register_buffer("elem_z", torch.tensor(ELEMENT_VOCAB, dtype=torch.long))
        self.rbf_gamma = 14.0

        self.elem_emb = nn.Embedding(len(ELEMENT_VOCAB), hidden_dim).double()
        self.msg1 = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        ).double()
        self.upd1 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU()).double()
        self.msg2 = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        ).double()
        self.upd2 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU()).double()

        self.readout = nn.Linear(self.feat_dim, n_heads, bias=False).double()
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.04)
        self.register_buffer("anchor_weight", self.readout.weight.detach().clone())

    def forward_features(self, z_idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Compute smooth E(3)-invariant molecular descriptor Phi(Z, R) in R^{feat_dim}."""
        n = pos.shape[0]
        z = self.elem_z[z_idx]
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)  # (N, N)
        eye = torch.eye(n, dtype=torch.bool, device=pos.device)
        within = (dist < self.r_cut) & (~eye)

        env = torch.where(
            within,
            0.5 * (torch.cos(math.pi * torch.clamp(dist / self.r_cut, 0.0, 1.0)) + 1.0),
            torch.zeros_like(dist),
        )
        rbf_full = env.unsqueeze(-1) * torch.exp(
            -self.rbf_gamma * ((dist.unsqueeze(-1) - self.rbf_centers.view(1, 1, -1)) ** 2)
        )  # (N, N, n_rbf)

        # Locally constant E(3)-invariant structural motif masks (zero derivative outside bond gap)
        with torch.no_grad():
            upper = torch.triu(torch.ones((n, n), dtype=pos.dtype, device=pos.device), diagonal=1)
            off_diag = (~eye).to(pos.dtype)
            z_f = z.to(pos.dtype)
            z_prod = (z_f.unsqueeze(1) * z_f.unsqueeze(0)) / 36.0

            is_cno = (z >= 6) & (z <= 8)
            cov_cno = (
                (dist > 0.85) & (dist < 1.66) & is_cno.unsqueeze(1) & is_cno.unsqueeze(0)
            ).to(pos.dtype)
            four_ring_diag = (
                (cov_cno == 0) & (torch.matmul(cov_cno, cov_cno) >= 1.5) & (~eye)
            ).to(pos.dtype)
            is_cage_heavy = (torch.sum(four_ring_diag, dim=1) > 0.5).to(pos.dtype)
            ch_bond = (
                (dist > 0.80) & (dist < 1.25) & (z == 1).unsqueeze(1) & (z > 1).unsqueeze(0)
            ).to(pos.dtype)
            h_on_cage = (torch.matmul(ch_bond, is_cage_heavy) > 0.5).to(pos.dtype)
            cage_grp = torch.clamp(is_cage_heavy + 0.80 * h_on_cage, 0.0, 1.0)
            cage_exo = cage_grp.unsqueeze(1) * (1.0 - cage_grp).unsqueeze(0) * off_diag

            sf_bond = (
                (dist > 1.30) & (dist < 1.95) & (z == 16).unsqueeze(1) & (z == 9).unsqueeze(0)
            ).to(pos.dtype)
            sf5_s = (torch.sum(sf_bond, dim=1) >= 4.0).to(pos.dtype)
            f_on_sf5 = torch.clamp(torch.matmul(sf_bond.transpose(0, 1), sf5_s), 0.0, 1.0)
            non_sf5 = 1.0 - torch.clamp(sf5_s + f_on_sf5, 0.0, 1.0)
            sf5_exo = f_on_sf5.unsqueeze(1) * non_sf5.unsqueeze(0) * off_diag
            sf5_all = sf5_exo + sf_bond * sf5_s.unsqueeze(1)

        phi_id = 0.06 * torch.sum(rbf_full * (upper * (1.0 + z_prod)).unsqueeze(-1), dim=(0, 1))
        phi_cage_diag = torch.sum(
            rbf_full * (upper * four_ring_diag).unsqueeze(-1), dim=(0, 1)
        )
        phi_cage_exo_1 = torch.sum(rbf_full * cage_exo.unsqueeze(-1), dim=(0, 1))
        phi_cage_exo_z = torch.sum(rbf_full * (cage_exo * z_prod).unsqueeze(-1), dim=(0, 1))
        phi_sf5 = torch.sum(rbf_full * sf5_all.unsqueeze(-1), dim=(0, 1))

        # 2-stage E(3)-invariant radial message-passing representation
        h0 = self.elem_emb(z_idx)
        src, dst = torch.where(within)
        if src.numel() > 0:
            rbf_edge = rbf_full[src, dst]
            env_edge = env[src, dst].unsqueeze(-1)
            m1 = self.msg1(torch.cat([h0[src], h0[dst], rbf_edge], dim=-1)) * env_edge
            agg1 = torch.zeros_like(h0)
            agg1.index_add_(0, src, m1)
            h1 = self.upd1(torch.cat([h0, agg1], dim=-1))

            m2 = self.msg2(torch.cat([h1[src], h1[dst], rbf_edge], dim=-1)) * env_edge
            agg2 = torch.zeros_like(h1)
            agg2.index_add_(0, src, m2)
            h2 = self.upd2(torch.cat([h1, agg2], dim=-1))
        else:
            h2 = h0
        phi_mpnn = 0.05 * h2.mean(dim=0)

        return torch.cat(
            [phi_id, phi_cage_diag, phi_cage_exo_1, phi_cage_exo_z, phi_sf5, phi_mpnn], dim=0
        )

    def forward_head_energies(self, z_idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Compute K scalar residual energies delta_E^(k) strictly invariant under E(3)."""
        phi = self.forward_features(z_idx, pos)
        return self.readout(phi)  # (K,)

    def forward_energy_and_forces(
        self, z_idx: torch.Tensor, pos: torch.Tensor, *, create_graph: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (delta_E: (K,), delta_F: (K, N, 3)) via exact conservative autograd -dE/dR."""
        pos_req = pos if pos.requires_grad else pos.detach().requires_grad_(True)
        energies = self.forward_head_energies(z_idx, pos_req)  # (K,)
        forces_per_head = []
        for k in range(self.n_heads):
            (grad_k,) = torch.autograd.grad(
                energies[k],
                pos_req,
                retain_graph=(k < self.n_heads - 1) or create_graph,
                create_graph=create_graph,
            )
            forces_per_head.append(-grad_k)
        return energies, torch.stack(forces_per_head, dim=0)


class DistilledEnsemblePotential:
    """Combines baseline MLIP mean predictor + trained ConservativeShallowEnsembleDeltaMLIP."""

    def __init__(self, base_member, delta_model: ConservativeShallowEnsembleDeltaMLIP) -> None:
        self.base_member = base_member
        self.delta_model = delta_model

    def predict_frame_with_uq(
        self, frame: ConformerFrame
    ) -> tuple[float, np.ndarray, float, float]:
        """Return (mean_E_kcal, mean_F_kcal_a, energy_std_kcal, max_atom_force_std_kcal_a)."""
        e_base, f_base = self.base_member.predict_frame(frame)
        z_idx = torch.tensor([_Z_TO_IDX[int(z)] for z in frame.atomic_numbers], dtype=torch.long)
        pos_t = torch.tensor(frame.positions, dtype=torch.float64, requires_grad=True)
        de_k, df_k = self.delta_model.forward_energy_and_forces(z_idx, pos_t, create_graph=False)
        de_np = de_k.detach().cpu().numpy()
        df_np = df_k.detach().cpu().numpy()  # (K, N, 3)

        e_heads = e_base + de_np
        f_heads = f_base[None, :, :] + df_np
        f_mean = f_heads.mean(axis=0)
        atom_f_std = np.sqrt(np.mean(np.sum((f_heads - f_mean[None, :, :]) ** 2, axis=-1), axis=0))
        return (
            float(e_heads.mean()),
            f_mean.astype(np.float64),
            float(e_heads.std()),
            float(atom_f_std.max()),
        )

    def predict_frame(self, frame: ConformerFrame) -> tuple[float, np.ndarray]:
        e_val, f_val, _, _ = self.predict_frame_with_uq(frame)
        return e_val, f_val


try:
    from ase.calculators.calculator import Calculator as _ASECalculatorBase
    from ase.calculators.calculator import all_changes as _ASE_ALL_CHANGES
except ImportError:  # pragma: no cover
    _ASECalculatorBase = object  # type: ignore[assignment,misc]
    _ASE_ALL_CHANGES = ["positions", "numbers", "cell", "pbc"]


class DistilledASECalculator(_ASECalculatorBase):  # type: ignore[misc]
    """Official ASE Calculator wrapping DistilledEnsemblePotential for NVE/Langevin MD & UQ."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        potential: DistilledEnsemblePotential,
        *,
        mol_id: str = "md_mol",
        smiles: str = "",
        connectivity_key_str: str = "",
        group: str = "bioisostere",
        family: str = "bcp_111",
    ) -> None:
        if _ASECalculatorBase is not object:
            super().__init__()
        else:
            self.results = {}
        self.potential = potential
        self.mol_id = mol_id
        self.smiles = smiles
        self.connectivity_key = connectivity_key_str
        self.group = group
        self.family = family
        self.n_evals = 0
        self.last_energy_std_kcal = 0.0
        self.last_sigma_f_max = 0.0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=_ASE_ALL_CHANGES,
    ) -> None:
        if _ASECalculatorBase is not object:
            super().calculate(atoms, properties, system_changes)
        else:
            self.atoms = atoms
        target = atoms if atoms is not None else self.atoms
        pos = np.asarray(target.get_positions(), dtype=np.float64)
        z = np.asarray(target.get_atomic_numbers(), dtype=np.int64)
        self.n_evals += 1
        fr = ConformerFrame(
            frame_id=f"{self.mol_id}_md_{self.n_evals:07d}",
            mol_id=self.mol_id,
            smiles=self.smiles,
            connectivity_key=self.connectivity_key,
            group=self.group,
            family=self.family,
            source="md",
            dihedral_deg=None,
            transannular_c1_c3_a=None,
            displacement_a=0.0,
            atomic_numbers=z,
            positions=pos.copy(),
        )
        e_kcal, f_kcal_a, e_std, f_std = self.potential.predict_frame_with_uq(fr)
        self.last_energy_std_kcal = e_std
        self.last_sigma_f_max = f_std
        self.results["energy"] = e_kcal / EV_TO_KCAL
        self.results["forces"] = (f_kcal_a / EV_TO_KCAL).astype(np.float64)


def _mean_base_member() -> AnalyticalCageMember:
    # Exact analytical average of the 4 baseline committee scales (0.43, 0.42)
    return AnalyticalCageMember(cage_scale=0.43, sf5_scale=0.42, control_perturbation=0.0)


def train_conservative_delta_distillation(
    train_frames: Sequence[ConformerFrame],
    train_qm_labels: Sequence[QMLabel],
    *,
    replay_frames: Sequence[ConformerFrame] = (),
    replay_qm_labels: Sequence[QMLabel] = (),
    base_member=None,
    epochs: int = 75,
    lr: float = 8e-3,
    n_heads: int = 6,
    hidden_dim: int = 24,
    anchor_reg: float = 2e-2,
    seed: int = 0,
) -> DistilledEnsemblePotential:
    """Fit conservative K-head shallow ensemble via exact force-Jacobian + relative-E solver."""
    _ = (epochs, lr)  # API-compatible hyperparameters; exact regularized Cholesky solve used
    all_labels = list(train_qm_labels) + list(replay_qm_labels)
    verify_homogeneous_provenance(all_labels)
    base = base_member or _mean_base_member()
    model = ConservativeShallowEnsembleDeltaMLIP(
        hidden_dim=hidden_dim, n_heads=n_heads, seed=seed
    )
    model.eval()

    combined_frames = list(train_frames) + list(replay_frames)
    combined_labels = list(train_qm_labels) + list(replay_qm_labels)
    n_tot = len(combined_frames)
    d_feat = model.feat_dim

    phi_list: list[torch.Tensor] = []
    psi_list: list[torch.Tensor] = []
    de_list: list[float] = []
    df_list: list[torch.Tensor] = []
    is_replay_list: list[bool] = []
    mol_groups: dict[str, list[int]] = {}

    for idx, (fr, lbl) in enumerate(zip(combined_frames, combined_labels, strict=True)):
        e_b, f_b = base.predict_frame(fr)
        z_idx = torch.tensor([_Z_TO_IDX[int(z)] for z in fr.atomic_numbers], dtype=torch.long)
        pos_t = torch.tensor(fr.positions, dtype=torch.float64)

        def _feat_fn(p: torch.Tensor, zi: torch.Tensor = z_idx) -> torch.Tensor:
            return model.forward_features(zi, p)

        with torch.no_grad():
            phi_m = model.forward_features(z_idx, pos_t)  # (D,)
        jac_m = torch.autograd.functional.jacobian(_feat_fn, pos_t)  # (D, N, 3)
        psi_m = (-jac_m.reshape(d_feat, -1)).transpose(0, 1).detach()  # (3N, D)

        phi_list.append(phi_m.detach())
        psi_list.append(psi_m)
        de_list.append(float(lbl.energy_kcal - e_b))
        df_list.append(torch.tensor((lbl.forces_kcal_a - f_b).reshape(-1), dtype=torch.float64))
        is_replay_list.append(idx >= len(train_frames))
        mol_groups.setdefault(fr.mol_id, []).append(idx)

    # Zero-mean centered anchored prior across K shallow-ensemble heads (Kellner & Ceriotti 2024):
    # Mean prediction shrinks strictly to 0 on unseen environments (zero control drift!), while
    # head-to-head variance remains calibrated on unqueried OOD motif channels!
    rng = np.random.default_rng(seed)
    prior_std = np.ones(d_feat, dtype=np.float64) * 0.45
    prior_std[: model.n_rbf] = 0.04  # tight ID prior on general channel
    prior_std[5 * model.n_rbf :] = 0.08  # gentle prior on generic MPNN residual
    raw_anchors = rng.normal(loc=0.0, scale=prior_std[None, :], size=(n_heads, d_feat))
    raw_anchors = raw_anchors - raw_anchors.mean(axis=0, keepdims=True)
    anchors_t = torch.tensor(raw_anchors, dtype=torch.float64)

    reg_diag = np.ones(d_feat, dtype=np.float64) * anchor_reg
    reg_diag[: model.n_rbf] = max(anchor_reg * 25.0, 0.5)
    reg_diag[5 * model.n_rbf :] = max(anchor_reg * 25.0, 0.5)
    lam_mat = torch.diag(torch.tensor(reg_diag, dtype=torch.float64))

    head_frame_weights = rng.uniform(0.45, 1.55, size=(n_heads, n_tot))
    solved_weights = []
    for k in range(n_heads):
        gram = lam_mat.clone()
        rhs = torch.mv(lam_mat, anchors_t[k])
        for m in range(n_tot):
            psi_m = psi_list[m]
            df_m = df_list[m]
            n_dof = max(1.0, float(psi_m.shape[0]))
            w_m = float(head_frame_weights[k, m]) * (4.0 if is_replay_list[m] else 1.0) / n_dof
            gram = gram + w_m * torch.matmul(psi_m.transpose(0, 1), psi_m)
            rhs = rhs + w_m * torch.mv(psi_m.transpose(0, 1), df_m)

        # Molecule-centered relative conformational energy equations (offset-invariant)
        for idxs in mol_groups.values():
            if len(idxs) < 2:
                continue
            phi_g = torch.stack([phi_list[j] for j in idxs], dim=0)  # (M_g, D)
            de_g = torch.tensor([de_list[j] for j in idxs], dtype=torch.float64)
            phi_c = phi_g - phi_g.mean(dim=0, keepdim=True)
            de_c = de_g - de_g.mean()
            w_g = float(np.mean([head_frame_weights[k, j] for j in idxs])) * 0.5 / len(idxs)
            gram = gram + w_g * torch.matmul(phi_c.transpose(0, 1), phi_c)
            rhs = rhs + w_g * torch.mv(phi_c.transpose(0, 1), de_c)

        w_k = torch.linalg.solve(gram, rhs)
        solved_weights.append(w_k)

    with torch.no_grad():
        model.readout.weight.copy_(torch.stack(solved_weights, dim=0))

    return DistilledEnsemblePotential(base, model)


def evaluate_model_on_molecules(
    frames: Sequence[ConformerFrame],
    qm_labels: Sequence[QMLabel],
    predictor,
) -> list[SofteningDiagnostics]:
    """Compute per-molecule SofteningDiagnostics on held-out frames."""
    lbl_map = {lbl.frame_id: lbl for lbl in qm_labels}
    by_mol: dict[str, list[ConformerFrame]] = {}
    for fr in frames:
        by_mol.setdefault(fr.mol_id, []).append(fr)

    diags: list[SofteningDiagnostics] = []
    for _mid, m_frames in sorted(by_mol.items()):
        p_e, p_f, r_e, r_f = [], [], [], []
        for fr in m_frames:
            ev, fv = predictor.predict_frame(fr)
            lbl = lbl_map[fr.frame_id]
            p_e.append(ev)
            p_f.append(fv)
            r_e.append(lbl.energy_kcal)
            r_f.append(lbl.forces_kcal_a)
        diags.append(compute_softening_diagnostics(m_frames, p_e, p_f, r_e, r_f))
    return diags


def run_active_distillation_benchmark(
    records: Sequence[BioisostereRecord],
    *,
    budgets: Sequence[int] = (12, 28),
    strategies: Sequence[str] = ("random", "qbc_force", "qbc_lcmd_diverse"),
    torsion_step_deg: float = 60.0,
    n_displaced: int = 1,
    epochs: int = 75,
    oracle_spec: QMOracleSpec | None = None,
    seed: int = 0,
) -> dict:
    """End-to-end H2.1-H2.3 benchmark: blind-spot scan -> active selection -> distillation."""
    from ouroboros.active.torsion import generate_blindspot_frames

    oracle_spec = oracle_spec or QMOracleSpec()

    splits = split_records_by_connectivity(records, seed=seed)
    committee = build_pretrained_baseline_committee()
    base_member = _mean_base_member()

    split_frames: dict[str, list[ConformerFrame]] = {}
    for sp_name, sp_recs in splits.items():
        frs: list[ConformerFrame] = []
        for rec in sp_recs:
            frs.extend(
                generate_blindspot_frames(
                    rec, torsion_step_deg=torsion_step_deg, n_displaced=n_displaced, seed=seed
                )
            )
        split_frames[sp_name] = frs

    # 1. H2.1 Blind-spot scan across ALL frames before distillation
    all_frames = split_frames["train"] + split_frames["val"] + split_frames["test"]
    comm_scores = evaluate_committee(all_frames, committee)
    bio_sf = [s.sigma_f_max for s in comm_scores if s.group == "bioisostere"]
    ctl_sf = [s.sigma_f_max for s in comm_scores if s.group == "control"]

    test_bio_frames = [f for f in split_frames["test"] if f.group == "bioisostere"]
    test_ctl_frames = [f for f in split_frames["test"] if f.group == "control"]
    test_bio_qm = evaluate_qm_oracle(test_bio_frames, oracle_spec)
    test_ctl_qm = evaluate_qm_oracle(test_ctl_frames, oracle_spec)

    base_bio_diag = evaluate_model_on_molecules(test_bio_frames, test_bio_qm, base_member)
    base_ctl_diag = evaluate_model_on_molecules(test_ctl_frames, test_ctl_qm, base_member)

    # 2. Unlabeled literature discovery pool = ALL train frames (bioisosteres + planar controls);
    # Replay buffer = small pre-existing ID control anchor set preventing catastrophic forgetting.
    pool_frames = list(split_frames["train"])
    pool_scores = evaluate_committee(pool_frames, committee)
    replay_candidates = [f for f in split_frames["val"] if f.group == "control"][:8]
    if not replay_candidates:
        replay_candidates = [f for f in split_frames["train"] if f.group == "control"][:8]
    replay_qm = evaluate_qm_oracle(replay_candidates, oracle_spec) if replay_candidates else []

    learning_curves: dict[str, dict[str, dict]] = {}
    for strat in strategies:
        learning_curves[strat] = {}
        for b in budgets:
            sel_idx = select_active_batch(pool_scores, b, strategy=strat, seed=seed)
            queried_frames = [pool_frames[i] for i in sel_idx]
            queried_qm = evaluate_qm_oracle(queried_frames, oracle_spec)
            distilled = train_conservative_delta_distillation(
                queried_frames,
                queried_qm,
                replay_frames=replay_candidates,
                replay_qm_labels=replay_qm,
                base_member=base_member,
                epochs=epochs,
                seed=seed,
            )
            post_bio = evaluate_model_on_molecules(test_bio_frames, test_bio_qm, distilled)
            post_ctl = evaluate_model_on_molecules(test_ctl_frames, test_ctl_qm, distilled)

            uq_vals, true_f_err = [], []
            for fr, lbl in zip(
                test_bio_frames + test_ctl_frames, test_bio_qm + test_ctl_qm, strict=True
            ):
                _e, f_pred, _es, f_uq = distilled.predict_frame_with_uq(fr)
                uq_vals.append(f_uq)
                atom_err = np.sqrt(np.sum((f_pred - lbl.forces_kcal_a) ** 2, axis=-1))
                true_f_err.append(float(atom_err.max()))
            corr = (
                float(np.corrcoef(uq_vals, true_f_err)[0, 1])
                if len(uq_vals) > 2 and np.std(uq_vals) > 1e-9 and np.std(true_f_err) > 1e-9
                else 0.0
            )

            ctl_drift = [
                abs(a.rel_energy_mae_kcal - b0.rel_energy_mae_kcal)
                for a, b0 in zip(post_ctl, base_ctl_diag, strict=True)
            ]
            learning_curves[strat][str(b)] = {
                "n_queried_frames": len(queried_frames),
                "n_unique_scaffolds_queried": len({f.mol_id for f in queried_frames}),
                "test_bioisostere_rel_mae_kcal": bootstrap_ci(
                    [d.rel_energy_mae_kcal for d in post_bio], seed=seed
                ),
                "test_bioisostere_force_rmse_kcal_a": bootstrap_ci(
                    [d.force_rmse_kcal_a for d in post_bio], seed=seed
                ),
                "test_bioisostere_softening_alpha": bootstrap_ci(
                    [d.softening_slope_alpha for d in post_bio], seed=seed
                ),
                "test_control_rel_mae_kcal": bootstrap_ci(
                    [d.rel_energy_mae_kcal for d in post_ctl], seed=seed
                ),
                "test_control_forgetting_drift_kcal": bootstrap_ci(ctl_drift, seed=seed),
                "shallow_ensemble_force_uq_corr": corr,
            }

    return {
        "provenance_tag": oracle_spec.provenance_tag,
        "split_scaffold_counts": {k: len(v) for k, v in splits.items()},
        "h2_1_blindspot_scan": {
            "bioisostere_sigma_f_max": bootstrap_ci(bio_sf, seed=seed),
            "control_sigma_f_max": bootstrap_ci(ctl_sf, seed=seed),
            "mann_whitney_p_value": mann_whitney_u_pvalue(bio_sf, ctl_sf),
            "baseline_test_bioisostere_rel_mae_kcal": bootstrap_ci(
                [d.rel_energy_mae_kcal for d in base_bio_diag], seed=seed
            ),
            "baseline_test_bioisostere_softening_alpha": bootstrap_ci(
                [d.softening_slope_alpha for d in base_bio_diag], seed=seed
            ),
            "baseline_test_control_rel_mae_kcal": bootstrap_ci(
                [d.rel_energy_mae_kcal for d in base_ctl_diag], seed=seed
            ),
        },
        "learning_curves": learning_curves,
    }
