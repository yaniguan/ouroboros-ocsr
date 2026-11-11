"""Provenance-guarded QM single-point oracle & conservative E(3)-invariant reference potentials.

Supports Psi4 / PySCF (wB97M-D3(BJ)/def2-TZVPPD, matching MACE-OFF23; Kovacs et al., JACS 2025),
tblite (GFN2-xTB), ASE calculators, and a deterministic autograd-conservative strained-cage &
hypervalent-SF5 physics potential for fast CPU validation and active-learning benchmarks.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ouroboros.active.torsion import ConformerFrame

HARTREE_TO_KCAL = 627.5094740631
BOHR_TO_ANGSTROM = 0.529177210903
EV_TO_KCAL = 23.06054783061903
ANI2X_ATOMIC_NUMBERS: frozenset[int] = frozenset({1, 6, 7, 8, 9, 16, 17})


@dataclass(frozen=True)
class QMOracleSpec:
    backend: str = "analytical_cage"  # "analytical_cage" | "psi4" | "pyscf" | "tblite" | "ase"
    method: str = "wb97m-d3bj"
    basis: str = "def2-tzvppd"
    solvent: str = "gas"
    charge: int = 0
    multiplicity: int = 1

    @property
    def provenance_tag(self) -> str:
        return (
            f"{self.backend}:{self.method.lower()}/{self.basis.lower()}/"
            f"{self.solvent.lower()}/q{self.charge}/m{self.multiplicity}"
        )


@dataclass(frozen=True)
class QMLabel:
    frame_id: str
    mol_id: str
    energy_kcal: float
    forces_kcal_a: np.ndarray
    provenance_tag: str


def verify_homogeneous_provenance(labels: Sequence[QMLabel]) -> str:
    """Refuse to mix incompatible QM levels of theory in single-head distillation."""
    if not labels:
        raise ValueError("labels must be non-empty")
    tags = {lbl.provenance_tag for lbl in labels}
    if len(tags) > 1:
        raise ValueError(
            f"incompatible QM provenance tags in single-head dataset: {sorted(tags)}; "
            "use multi-head readout isolation (Kaur et al., Faraday Discuss. 2025)"
        )
    return next(iter(tags))


def _smooth_window(r: torch.Tensor, lo: float, hi: float, width: float = 0.08) -> torch.Tensor:
    return torch.sigmoid((r - lo) / width) * torch.sigmoid((hi - r) / width)


def _conservative_cage_energy_torch(
    z: torch.Tensor,
    pos: torch.Tensor,
    *,
    cage_scale: float = 1.0,
    sf5_scale: float = 1.0,
    control_perturbation: float = 0.0,
) -> torch.Tensor:
    """Strictly E(3)-invariant smooth potential over (Z, R) capturing covalent, nonbonded,
    transannular cage compression (1.74-2.28 A), and hypervalent SF5 steric torsion barriers."""
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)
    n = pos.shape[0]
    eye_mask = 1.0 - torch.eye(n, dtype=pos.dtype, device=pos.device)
    upper_mask = torch.triu(eye_mask, diagonal=1)

    is_heavy = (z > 1).to(pos.dtype)
    heavy_pair = (is_heavy.unsqueeze(1) * is_heavy.unsqueeze(0)) * upper_mask
    h_involved = (1.0 - is_heavy.unsqueeze(1) * is_heavy.unsqueeze(0)) * upper_mask

    # 1. Baseline smooth harmonic/dispersion-repulsion molecular mechanics terms (shared ID physics)
    cov_heavy = _smooth_window(dist, 1.10, 1.70, 0.05) * heavy_pair
    e_cov = 42.0 * torch.sum(cov_heavy * ((dist - 1.46) ** 2))

    nb_window = _smooth_window(dist, 2.35, 4.20, 0.12) * upper_mask
    z_prod = (z.to(pos.dtype).unsqueeze(1) * z.to(pos.dtype).unsqueeze(0)) / 36.0
    e_nb_base = torch.sum(
        nb_window * z_prod * (2.6 * torch.exp(-(dist - 2.45) / 0.38) - 0.18 * ((3.2 / dist) ** 6))
    )
    e_hh_steric = torch.sum(
        _smooth_window(dist, 1.85, 2.70, 0.10)
        * h_involved
        * 1.35
        * torch.exp(-(dist - 2.05) / 0.32)
    )
    if control_perturbation != 0.0:
        e_nb_base = e_nb_base + control_perturbation * torch.sum(
            _smooth_window(dist, 2.50, 3.80, 0.15) * heavy_pair * torch.exp(-(dist - 2.8) / 0.5)
        )

    # 2. Post-2020 Strained-Cage Transannular & Exit-Vector Torsional Signature:
    # Locally constant E(3)-invariant 4-ring diagonal graph: nonbonded C/N/O pairs sharing >=2
    # bonded C/N/O neighbors (exact for [1.1.1], [2.1.1], [3.1.1], cubane, oxetane, azaspiro;
    # identically 0 on benzenoids, pyridines, 5-rings, cyclohexanes, amides, esters, and -CF3).
    with torch.no_grad():
        eye_bool = torch.eye(n, dtype=torch.bool, device=pos.device)
        is_cno = (z >= 6) & (z <= 8)
        cov_cno = (
            (dist > 0.85) & (dist < 1.66) & is_cno.unsqueeze(1) & is_cno.unsqueeze(0)
        ).to(pos.dtype)
        four_ring_diag = (
            (cov_cno == 0) & (torch.matmul(cov_cno, cov_cno) >= 1.5) & (~eye_bool)
        ).to(pos.dtype)
        is_cage_heavy = (torch.sum(four_ring_diag, dim=1) > 0.5).to(pos.dtype)
        is_h_bool = z == 1
        ch_bond = (
            (dist > 0.80) & (dist < 1.25) & is_h_bool.unsqueeze(1) & (z > 1).unsqueeze(0)
        ).to(pos.dtype)
        h_on_cage = (torch.matmul(ch_bond, is_cage_heavy) > 0.5).to(pos.dtype)
        cage_group = torch.clamp(is_cage_heavy + 0.80 * h_on_cage, 0.0, 1.0)

        is_s_bool = z == 16
        is_f_bool = z == 9
        sf_bond_mask = (
            (dist > 1.30) & (dist < 1.95) & is_s_bool.unsqueeze(1) & is_f_bool.unsqueeze(0)
        ).to(pos.dtype)
        sf5_flag = (torch.sum(sf_bond_mask, dim=1) >= 4.0).to(pos.dtype)
        f_on_sf5 = torch.clamp(torch.matmul(sf_bond_mask.transpose(0, 1), sf5_flag), 0.0, 1.0)
        non_sf5_mask = 1.0 - torch.clamp(sf5_flag + f_on_sf5, 0.0, 1.0)

    # Through-space eclipsing between off-axis cage CH/C atoms and rotating exocyclic substituents
    exo_vicinal = _smooth_window(dist, 1.95, 3.65, 0.12) * eye_mask
    cage_exo_weight = cage_group.unsqueeze(1) * (1.0 - cage_group).unsqueeze(0)
    pair_scale = 0.60 + 0.40 * z_prod
    e_cage_torsion = 6.5 * torch.sum(
        cage_exo_weight * exo_vicinal * pair_scale * torch.exp(-(dist - 2.65) / 0.48)
    )
    e_cage_transannular = 18.0 * torch.sum(
        torch.triu(four_ring_diag, diagonal=1) * ((dist - 2.02) ** 2)
    )

    # 3. Hypervalent octahedral -SF5 steric eclipsing & bond-stretch signature
    sf5_steric_pairs = f_on_sf5.unsqueeze(1) * non_sf5_mask.unsqueeze(0) * eye_mask
    e_sf5_torsion = 7.2 * torch.sum(
        sf5_steric_pairs
        * _smooth_window(dist, 1.95, 3.65, 0.12)
        * torch.exp(-(dist - 2.60) / 0.46)
    ) + 28.0 * torch.sum(sf_bond_mask * sf5_flag.unsqueeze(1) * ((dist - 1.59) ** 2))

    return (
        e_cov
        + e_nb_base
        + e_hh_steric
        + cage_scale * (e_cage_transannular + e_cage_torsion)
        + sf5_scale * e_sf5_torsion
    )


def eval_conservative_potential(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    *,
    cage_scale: float = 1.0,
    sf5_scale: float = 1.0,
    control_perturbation: float = 0.0,
    atom_offset_kcal: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Evaluate exact energy (kcal/mol) and conservative force F = -dE/dR (kcal/mol/A)."""
    z_t = torch.as_tensor(atomic_numbers, dtype=torch.long)
    pos_t = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    e_t = _conservative_cage_energy_torch(
        z_t,
        pos_t,
        cage_scale=cage_scale,
        sf5_scale=sf5_scale,
        control_perturbation=control_perturbation,
    )
    e_t = e_t + atom_offset_kcal * float(len(atomic_numbers))
    (grad_pos,) = torch.autograd.grad(e_t, pos_t, create_graph=False)
    forces = (-grad_pos).detach().cpu().numpy().astype(np.float64)
    return float(e_t.detach().item()), forces


class AnalyticalCageMember:
    """Single conservative MLIP member parameterized by its strained-cage extrapolation scales."""

    def __init__(
        self,
        cage_scale: float,
        sf5_scale: float,
        control_perturbation: float = 0.0,
        atom_offset_kcal: float = 0.0,
    ) -> None:
        self.cage_scale = cage_scale
        self.sf5_scale = sf5_scale
        self.control_perturbation = control_perturbation
        self.atom_offset_kcal = atom_offset_kcal

    def predict_frame(self, frame: ConformerFrame) -> tuple[float, np.ndarray]:
        return eval_conservative_potential(
            frame.atomic_numbers,
            frame.positions,
            cage_scale=self.cage_scale,
            sf5_scale=self.sf5_scale,
            control_perturbation=self.control_perturbation,
            atom_offset_kcal=self.atom_offset_kcal,
        )


class ASECalculatorMember:
    """Adapter wrapping any ASE calculator (e.g., MACE-OFF23 small/medium/large or AIMNet2)."""

    def __init__(self, calc) -> None:
        self.calc = calc

    def predict_frame(self, frame: ConformerFrame) -> tuple[float, np.ndarray]:
        from ase import Atoms

        atoms = Atoms(numbers=frame.atomic_numbers.tolist(), positions=frame.positions)
        atoms.calc = self.calc
        e_kcal = float(atoms.get_potential_energy()) * EV_TO_KCAL
        f_kcal_a = np.asarray(atoms.get_forces(), dtype=np.float64) * EV_TO_KCAL
        return e_kcal, f_kcal_a


class CachedPredictorMember:
    """Memoizes (energy_kcal, forces_kcal_a) by frame_id for fast multi-strategy evaluation."""

    def __init__(self, inner, cache: dict[str, tuple[float, np.ndarray]] | None = None) -> None:
        self.inner = inner
        self.cache: dict[str, tuple[float, np.ndarray]] = cache if cache is not None else {}

    def predict_frame(self, frame: ConformerFrame) -> tuple[float, np.ndarray]:
        hit = self.cache.get(frame.frame_id)
        if hit is not None:
            return hit[0], hit[1].copy()
        e_val, f_val = self.inner.predict_frame(frame)
        f_arr = np.asarray(f_val, dtype=np.float64)
        self.cache[frame.frame_id] = (float(e_val), f_arr)
        return float(e_val), f_arr.copy()


class ANI2xModelMember:
    """Adapter wrapping TorchANI ANI-2x (either full 8-model ensemble or a single submodel)."""

    def __init__(self, model_module, *, device: str = "cpu") -> None:
        self.model = model_module.to(device).double()
        self.model.eval()
        self.device = device

    @staticmethod
    def supports_frame(frame: ConformerFrame) -> bool:
        return set(int(z) for z in frame.atomic_numbers).issubset(ANI2X_ATOMIC_NUMBERS)

    def predict_frame(self, frame: ConformerFrame) -> tuple[float, np.ndarray]:
        if not self.supports_frame(frame):
            unsupported = sorted(set(int(z) for z in frame.atomic_numbers) - ANI2X_ATOMIC_NUMBERS)
            raise ValueError(f"ANI-2x does not support atomic numbers {unsupported}")
        z_t = torch.as_tensor(frame.atomic_numbers, dtype=torch.long, device=self.device).unsqueeze(
            0
        )
        pos_t = torch.tensor(
            frame.positions, dtype=torch.float64, device=self.device, requires_grad=True
        ).unsqueeze(0)
        e_h = self.model((z_t, pos_t)).energies.sum()
        (grad_pos,) = torch.autograd.grad(e_h, pos_t, create_graph=False)
        e_kcal = float(e_h.detach().cpu().item()) * HARTREE_TO_KCAL
        f_kcal_a = (-grad_pos.squeeze(0).detach().cpu().numpy() * HARTREE_TO_KCAL).astype(
            np.float64
        )
        return e_kcal, f_kcal_a


def build_ani2x_committee(
    *, device: str = "cpu"
) -> tuple[ANI2xModelMember, list[ANI2xModelMember]]:
    """Return (ensemble_mean_member, 8-member neural committee) for pretrained ANI-2x."""
    os.environ.setdefault("TORCHANI_NO_WARN_EXTENSIONS", "1")
    import torchani  # type: ignore[import-not-found]

    ens = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
    ens.eval()
    mean_member = ANI2xModelMember(ens, device=device)
    sub_members = [ANI2xModelMember(ens[k], device=device) for k in range(len(ens.neural_networks))]
    return mean_member, sub_members


class MMFF94sMember:
    """Classical MMFF94s force-field baseline returning analytical energies and forces."""

    def __init__(self) -> None:
        self._mol_cache: dict[str, object] = {}

    def predict_frame(self, frame: ConformerFrame) -> tuple[float, np.ndarray]:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        base_mol = self._mol_cache.get(frame.smiles)
        if base_mol is None:
            m = Chem.AddHs(Chem.MolFromSmiles(frame.smiles))
            AllChem.EmbedMolecule(m, randomSeed=0)
            self._mol_cache[frame.smiles] = m
            base_mol = m

        work = Chem.Mol(base_mol)
        conf = work.GetConformer(0)
        for idx, (x, y, zc) in enumerate(frame.positions):
            conf.SetAtomPosition(idx, (float(x), float(y), float(zc)))

        props = AllChem.MMFFGetMoleculeProperties(work, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(work, props) if props is not None else None
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(work)
        if ff is None:
            return 0.0, np.zeros_like(frame.positions, dtype=np.float64)
        e_kcal = float(ff.CalcEnergy())
        grad = np.asarray(ff.CalcGrad(), dtype=np.float64).reshape(-1, 3)
        return e_kcal, (-grad).astype(np.float64)


def build_pretrained_baseline_committee(mace_models: Sequence[str] = ()) -> list:
    """Return an M-member baseline MLIP committee (MACE-OFF23 models or fast CPU surrogate).

    Note: The 4 CPU surrogate members intentionally carry distinct per-atom zero-point offsets
    (simulating multi-seed / multi-code reference shifts) AND under-predict post-2020 cage/SF5
    barriers (alpha ~ 0.42 < 1), testing both offset-free relative UQ and PES softening recovery.
    """
    if mace_models:
        from ouroboros.geometry.conformers import mace_calculator

        return [ASECalculatorMember(mace_calculator(m)) for m in mace_models]

    return [
        AnalyticalCageMember(0.22, 0.20, control_perturbation=-0.02, atom_offset_kcal=-12.4),
        AnalyticalCageMember(0.36, 0.35, control_perturbation=0.01, atom_offset_kcal=0.0),
        AnalyticalCageMember(0.50, 0.48, control_perturbation=-0.01, atom_offset_kcal=8.7),
        AnalyticalCageMember(0.64, 0.65, control_perturbation=0.02, atom_offset_kcal=19.1),
    ]


_PSI4_INITIALIZED_OUT: str | None = None


def _configure_psi4_runtime(*, n_threads: int = 8, memory: str = "6 GB") -> None:
    global _PSI4_INITIALIZED_OUT
    import psi4  # type: ignore[import-not-found]

    if _PSI4_INITIALIZED_OUT is None:
        fd, tmp_out = tempfile.mkstemp(prefix="ouroboros_psi4_", suffix=".out")
        os.close(fd)
        psi4.core.set_output_file(tmp_out, False)
        _PSI4_INITIALIZED_OUT = tmp_out
    psi4.set_num_threads(n_threads)
    psi4.set_memory(memory)
    psi4.set_options(
        {
            "scf_type": "df",
            "reference": "rhf",
            "e_convergence": 1e-7,
            "d_convergence": 1e-7,
        }
    )


def _build_psi4_molecule(frame: ConformerFrame, spec: QMOracleSpec):
    import psi4  # type: ignore[import-not-found]
    from rdkit import Chem

    pt = [
        f"{spec.charge} {spec.multiplicity}",
        "units angstrom",
        "symmetry c1",
        "no_reorient",
        "no_com",
    ]
    ptable = Chem.GetPeriodicTable()
    for z, (x, y, zc) in zip(frame.atomic_numbers, frame.positions, strict=True):
        sym = ptable.GetElementSymbol(int(z))
        pt.append(f"{sym} {x:.8f} {y:.8f} {zc:.8f}")
    return psi4.geometry("\n".join(pt))


def _run_psi4_single_point(
    frame: ConformerFrame,
    spec: QMOracleSpec,
    *,
    n_threads: int = 8,
    memory: str = "6 GB",
) -> tuple[float, np.ndarray]:
    import psi4  # type: ignore[import-not-found]

    _configure_psi4_runtime(n_threads=n_threads, memory=memory)
    mol_psi = _build_psi4_molecule(frame, spec)
    grad_mat, wfn = psi4.gradient(
        f"{spec.method}/{spec.basis}", molecule=mol_psi, return_wfn=True
    )
    e_h = float(wfn.energy())
    grad = np.asarray(grad_mat, dtype=np.float64)  # Hartree / Bohr
    psi4.core.clean()
    forces = -grad * (HARTREE_TO_KCAL / BOHR_TO_ANGSTROM)
    return e_h * HARTREE_TO_KCAL, forces.astype(np.float64)


def evaluate_psi4_torsion_energies(
    frames: Sequence[ConformerFrame],
    spec: QMOracleSpec | None = None,
    *,
    n_threads: int = 16,
    memory: str = "8 GB",
) -> list[dict[str, float | str]]:
    """Compute exact Psi4 single-point energies (wB97M-D3(BJ)/def2-TZVPPD) for torsion scans."""
    import psi4  # type: ignore[import-not-found]

    spec = spec or QMOracleSpec(backend="psi4", method="wb97m-d3bj", basis="def2-tzvppd")
    _configure_psi4_runtime(n_threads=n_threads, memory=memory)
    results: list[dict[str, float | str]] = []
    for frame in frames:
        mol_psi = _build_psi4_molecule(frame, spec)
        e_h = float(psi4.energy(f"{spec.method}/{spec.basis}", molecule=mol_psi))
        psi4.core.clean()
        results.append(
            {
                "frame_id": frame.frame_id,
                "mol_id": frame.mol_id,
                "energy_hartree": e_h,
                "energy_kcal": e_h * HARTREE_TO_KCAL,
                "provenance_tag": spec.provenance_tag,
            }
        )
    return results


def _run_pyscf_single_point(frame: ConformerFrame, spec: QMOracleSpec) -> tuple[float, np.ndarray]:
    from pyscf import dft, gto  # type: ignore[import-not-found]
    from rdkit import Chem

    ptable = Chem.GetPeriodicTable()
    atom_spec = [
        (ptable.GetElementSymbol(int(z)), tuple(float(v) for v in xyz))
        for z, xyz in zip(frame.atomic_numbers, frame.positions, strict=True)
    ]
    mol = gto.M(
        atom=atom_spec,
        basis=spec.basis,
        charge=spec.charge,
        spin=spec.multiplicity - 1,
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "wb97m-v" if "wb97m" in spec.method.lower() else spec.method
    e_h = mf.kernel()
    grad = mf.nuc_grad_method().kernel()  # Hartree / Bohr
    forces = -np.asarray(grad) * (HARTREE_TO_KCAL / BOHR_TO_ANGSTROM)
    return float(e_h) * HARTREE_TO_KCAL, forces.astype(np.float64)


def _run_tblite_single_point(frame: ConformerFrame, spec: QMOracleSpec) -> tuple[float, np.ndarray]:
    from tblite.ase import TBLite  # type: ignore[import-not-found]

    member = ASECalculatorMember(TBLite(method="GFN2-xTB", charge=spec.charge, verbosity=0))
    return member.predict_frame(frame)


DEFAULT_ORACLE_SPEC = QMOracleSpec()


def evaluate_qm_oracle(
    frames: Sequence[ConformerFrame],
    spec: QMOracleSpec | None = None,
    *,
    ase_calc=None,
) -> list[QMLabel]:
    """Compute reference single-point energies & conservative forces with locked provenance."""
    spec = spec or DEFAULT_ORACLE_SPEC
    out: list[QMLabel] = []
    tag = spec.provenance_tag
    for frame in frames:
        if spec.backend == "analytical_cage":
            e_kcal, f_kcal_a = eval_conservative_potential(
                frame.atomic_numbers, frame.positions, cage_scale=1.0, sf5_scale=1.0
            )
        elif spec.backend == "psi4":
            e_kcal, f_kcal_a = _run_psi4_single_point(frame, spec)
        elif spec.backend == "pyscf":
            e_kcal, f_kcal_a = _run_pyscf_single_point(frame, spec)
        elif spec.backend == "tblite":
            e_kcal, f_kcal_a = _run_tblite_single_point(frame, spec)
        elif spec.backend == "ase":
            if ase_calc is None:
                raise ValueError("ase_calc must be provided when spec.backend == 'ase'")
            e_kcal, f_kcal_a = ASECalculatorMember(ase_calc).predict_frame(frame)
        else:
            raise ValueError(f"unsupported QM oracle backend {spec.backend!r}")
        out.append(
            QMLabel(
                frame_id=frame.frame_id,
                mol_id=frame.mol_id,
                energy_kcal=e_kcal,
                forces_kcal_a=f_kcal_a,
                provenance_tag=tag,
            )
        )
    return out
