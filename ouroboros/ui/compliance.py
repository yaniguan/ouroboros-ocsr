"""3D Equivariant Molecule Generation & 10-Gate Physical Compliance Engine.

Verifies fundamental chemical, stereochemical, geometric, thermodynamic, and Lie-group
symmetry laws on generated 3D molecular conformers:
  G1. Valence, Octet & Kekulization Topology (single neutral/valid molecular graph)
  G2. Calculator Physical Domain Eligibility (formal charge, spin radicals, element support)
  G3. Stereochemical Specification Completeness (CIP R/S & E/Z vs unassigned stereocenters)
  G4. Distance-Geometry 3D Embedding (ETKDGv3 chirality-enforced conformer generation)
  G5. Pauli Exclusion & Steric Clearance (all-pair d_min >= 0.75 A, nonbonded >= 1.35 A)
  G6. Covalent Bond-Length & Ring/Cage Angle Plausibility (deviation vs covalent radii)
  G7. 3D Chirality & Double-Bond Retention (3D-only CIP re-perception pre/post relaxation)
  G8. Stationary-Point Force Convergence (finite E/F, LBFGS fmax <= threshold, Boltzmann weights)
  G9. Live SE(3) Equivariance & Noether Conservation (SO(3) rotation, R^3 translation,
      zero net force sum_i F_i = 0, zero net torque sum_i r_i x F_i = 0)
  G10. O(3) Reflection Parity & Conservative Potential Field (mirror E invariance, chiral
       enantiomer inversion, and central finite-difference F = -grad_R E verification)
"""

from __future__ import annotations

import base64
import io
import math
import time
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ouroboros.data.render import RenderStyle, encode_png, render
from ouroboros.data.rgroups import expand_aliases
from ouroboros.eval.standardize import standard_inchikey, standardize
from ouroboros.geometry.conformers import (
    FMAX,
    embed,
    mace_calculator,
    mirror_positions,
    relax,
    stereo_from_3d,
    stereo_matches,
    stereo_preserved,
)
from ouroboros.geometry.eligibility import EligibilityPolicy
from ouroboros.geometry.propagation import categorize
from ouroboros.labels import parse_labelled

EV_TO_KCAL = 23.06054783061903
HARTREE_TO_EV = 27.211386245988
KB_EV_PER_K = 8.617333262e-5

MACE_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
ANI2X_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "S", "Cl"})

PRESET_MOLECULES: dict[str, list[dict[str, str]]] = {
    "chiral_drugs": [
        {
            "name": "(S)-Ibuprofen (1 chiral center)",
            "smiles": "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
            "note": "Classic chiral NSAID; tests C_N rotation stability & O(3) mirror inversion.",
        },
        {
            "name": "L-Alanine (zwitterion-free neutral)",
            "smiles": "N[C@@H](C)C(=O)O",
            "note": "Compact chiral amino acid ideal for fast MACE-OFF23 relaxation.",
        },
        {
            "name": "(R)-Thalidomide (glutarimide chiral center)",
            "smiles": "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "note": "Bicyclic chiral drug where enantiomer distinction is critical.",
        },
        {
            "name": "(E)-Resveratrol (trans-alkene stereo)",
            "smiles": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
            "note": "Tests E/Z double-bond stereochemical preservation in 2D and 3D.",
        },
    ],
    "strained_3d_bioisosteres": [
        {
            "name": "Bicyclo[1.1.1]pentane (BCP) Carboxamide",
            "smiles": "CNC(=O)C12CC(C1)C2",
            "note": "Post-2020 phenyl bioisostere with 1.87 A transannular C1...C3 compression.",
        },
        {
            "name": "Cubane Carboxamide (Oh strained cage)",
            "smiles": "NC(=O)C12C3C4C1C5C2C3C45",
            "note": "90-degree cyclobutane cage (~155 kcal/mol ring strain); tests MLIP stiffness.",
        },
        {
            "name": "2-Oxabicyclo[2.1.1]hexane Ester",
            "smiles": "COC(=O)C12COC(C1)C2",
            "note": "Ortho-substituted benzene bioisostere with oxygen-bridged strained cage.",
        },
        {
            "name": "ortho-F-Phenyl-SF5 (Hypervalent S(VI))",
            "smiles": "Fc1ccccc1S(F)(F)(F)(F)F",
            "note": "Octahedral hypervalent sulfur bioisostere; tests MLIP element coverage.",
        },
    ],
    "abbreviations_and_compliance_edges": [
        {
            "name": "Racemic / Unassigned Lactic Acid (Stereo Warn)",
            "smiles": "CC(O)C(=O)O",
            "note": "Triggers Gate 3 warning: unassigned tetrahedral stereocenter.",
        },
        {
            "name": "Abbreviated Boc-Bn-Ph ([Boc] & [Ph] aliases)",
            "smiles": "[Boc]NCc1ccc([Ph])cc1",
            "note": "Tests expand_aliases ([Boc], [Ph]) before 3D generation.",
        },
        {
            "name": "Sodium Acetate Salt (Charge Rejection)",
            "smiles": "CC(=O)[O-].[Na+]",
            "note": "Triggers Gate 1/2 rejection: disconnected salt & unsupported Na+.",
        },
        {
            "name": "Benzyl Radical (Open-Shell Spin Rejection)",
            "smiles": "[CH2]c1ccccc1",
            "note": "Triggers Gate 2 rejection: non-zero radical electron on benzylic carbon.",
        },
        {
            "name": "Arylboronic Pinacol Ester (-Bpin, Element B)",
            "smiles": "CC1(C)OB(c2ccc(F)cc2)OC1(C)C",
            "note": "Triggers Gate 2 domain rejection for MACE-OFF23/ANI-2x (Boron unsupported).",
        },
    ],
}


def png_data_url(smiles_or_mol: str | Chem.Mol, size: int = 300) -> str | None:
    mol = (
        Chem.MolFromSmiles(smiles_or_mol)
        if isinstance(smiles_or_mol, str)
        else Chem.Mol(smiles_or_mol)
    )
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    img = render(mol, RenderStyle(), size=size).image
    return "data:image/png;base64," + base64.b64encode(encode_png(img)).decode("ascii")


class MMFF94sASECalculator(Calculator):
    """Analytical conservative MMFF94s ASE calculator on fixed RDKit molecular topology.

    Because MMFF94s evaluates internal coordinates (bond lengths, bond angles, dihedrals,
    out-of-plane, van der Waals, electrostatics), it is exact SE(3)/O(3) invariant in energy
    and exact SO(3)/O(3) equivariant in Cartesian nuclear forces F = -dE/dR!
    """

    implemented_properties = ["energy", "forces"]
    supported_elements = frozenset({"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"})

    def __init__(self, template_mol_h: Chem.Mol) -> None:
        super().__init__()
        self._mol = Chem.Mol(template_mol_h)
        if self._mol.GetNumConformers() == 0:
            conf = Chem.Conformer(self._mol.GetNumAtoms())
            self._mol.AddConformer(conf, assignId=True)
        self._props = AllChem.MMFFGetMoleculeProperties(self._mol, mmffVariant="MMFF94s")
        self._use_uff = self._props is None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] = ("energy", "forces"),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        target = atoms if atoms is not None else self.atoms
        pos = np.asarray(target.get_positions(), dtype=np.float64)
        mol = Chem.Mol(self._mol)
        conf = mol.GetConformer(0)
        for idx, xyz in enumerate(pos):
            conf.SetAtomPosition(idx, [float(xyz[0]), float(xyz[1]), float(xyz[2])])

        if not self._use_uff:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if ff is None:
            raise RuntimeError("RDKit force field initialization failed for topology")

        e_kcal = float(ff.CalcEnergy())
        grad_kcal_a = np.asarray(ff.CalcGrad(), dtype=np.float64).reshape(-1, 3)
        self.results["energy"] = e_kcal / EV_TO_KCAL
        self.results["forces"] = -grad_kcal_a / EV_TO_KCAL


_ANI_CACHE: dict[str, Any] = {}


class ANI2xASECalculator(Calculator):
    """Exact conservative 8-member TorchANI ANI-2x ASE calculator."""

    implemented_properties = ["energy", "forces"]
    supported_elements = ANI2X_ELEMENTS

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        import torch

        try:
            import torchani
        except ImportError as exc:
            raise RuntimeError("Optional dependency 'torchani' is not installed.") from exc

        if device not in _ANI_CACHE:
            _ANI_CACHE[device] = (
                torchani.models.ANI2x(periodic_table_index=True).to(device).double().eval()
            )
        self._model = _ANI_CACHE[device]
        self._device = device
        self._torch = torch

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] = ("energy", "forces"),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        target = atoms if atoms is not None else self.atoms
        torch = self._torch
        sp = torch.tensor(
            [target.get_atomic_numbers().tolist()], dtype=torch.long, device=self._device
        )
        coords = torch.tensor(
            [target.get_positions().tolist()],
            dtype=torch.float64,
            device=self._device,
            requires_grad=True,
        )
        e_h = self._model((sp, coords)).energies
        grad_h_a = torch.autograd.grad(e_h.sum(), coords)[0][0].detach().cpu().numpy()
        self.results["energy"] = float(e_h.item()) * HARTREE_TO_EV
        self.results["forces"] = (-grad_h_a * HARTREE_TO_EV).astype(np.float64)


def get_3d_calculator(
    potential: str, template_mol_h: Chem.Mol, device: str = "cpu"
) -> tuple[Calculator, str, frozenset[str]]:
    key = potential.strip().lower()
    if key == "mmff94s":
        calc = MMFF94sASECalculator(template_mol_h)
        return calc, "RDKit MMFF94s (Analytical)", calc.supported_elements
    if key in ("mace-small", "small"):
        return (
            mace_calculator("small", device=device, dtype="float64"),
            "MACE-OFF23-small (E(3) MPNN)",
            MACE_ELEMENTS,
        )
    if key in ("mace-medium", "medium"):
        return (
            mace_calculator("medium", device=device, dtype="float64"),
            "MACE-OFF23-medium (E(3) MPNN)",
            MACE_ELEMENTS,
        )
    if key in ("mace-large", "large"):
        return (
            mace_calculator("large", device=device, dtype="float64"),
            "MACE-OFF23-large (E(3) MPNN)",
            MACE_ELEMENTS,
        )
    if key in ("ani2x", "ani-2x"):
        calc = ANI2xASECalculator(device=device)
        return calc, "TorchANI ANI-2x (8-Model Ensemble)", ANI2X_ELEMENTS
    raise ValueError(
        f"Unknown potential {potential!r}; choose mmff94s, mace-small, mace-medium, "
        "mace-large, or ani2x."
    )


def _random_so3_matrix(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _cip_summary(mol_no_h: Chem.Mol) -> dict[str, Any]:
    from rdkit.Chem import rdCIPLabeler

    m = Chem.Mol(mol_no_h)
    rdCIPLabeler.AssignCIPLabels(m)
    chiral_centers = [
        {"atom_idx": a.GetIdx(), "symbol": a.GetSymbol(), "cip": a.GetProp("_CIPCode")}
        for a in m.GetAtoms()
        if a.HasProp("_CIPCode")
    ]
    stereo_bonds = [
        {
            "bond_idx": b.GetIdx(),
            "atoms": [b.GetBeginAtomIdx(), b.GetEndAtomIdx()],
            "cip": b.GetProp("_CIPCode"),
        }
        for b in m.GetBonds()
        if b.HasProp("_CIPCode")
    ]
    unassigned_tet = 0
    unassigned_db = 0
    for el in Chem.FindPotentialStereo(Chem.Mol(m)):
        if el.specified == Chem.StereoSpecified.Unspecified:
            if el.type == Chem.StereoType.Atom_Tetrahedral:
                unassigned_tet += 1
            elif el.type == Chem.StereoType.Bond_Double:
                unassigned_db += 1
    return {
        "chiral_centers": chiral_centers,
        "stereo_bonds": stereo_bonds,
        "n_specified": len(chiral_centers) + len(stereo_bonds),
        "n_unassigned_tetrahedral": unassigned_tet,
        "n_unassigned_double_bonds": unassigned_db,
    }


def _format_xyz(symbols: list[str], positions: np.ndarray, comment: str) -> str:
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, positions, strict=True):
        lines.append(f"{sym:<3s} {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines) + "\n"


def _format_extxyz(
    symbols: list[str], positions: np.ndarray, forces: np.ndarray, energy_ev: float, smiles: str
) -> str:
    from ase.io import write

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info["smiles"] = smiles
    atoms.info["mace_energy"] = float(energy_ev)
    atoms.arrays["mace_forces"] = np.asarray(forces, dtype=np.float64)
    buf = io.StringIO()
    write(buf, atoms, format="extxyz")
    return buf.getvalue()


def audit_3d_physical_compliance(
    smiles: str,
    *,
    potential: str = "mmff94s",
    n_conf: int = 4,
    do_relax: bool = True,
    fmax: float = FMAX,
    steps: int = 120,
    expand_abbrev: bool = True,
    perturb_amplitude_a: float = 0.0,
    inject_steric_clash: bool = False,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run 3D molecule generation + 10-gate physical & equivariance compliance audit."""
    t_start = time.perf_counter()
    raw_input = (smiles or "").strip()
    gates: list[dict[str, Any]] = []

    def add_gate(
        gate_id: str,
        title: str,
        status: str,
        metric: str,
        threshold: str,
        detail: str,
    ) -> None:
        gates.append(
            {
                "id": gate_id,
                "title": title,
                "status": status,  # "pass" | "warn" | "fail"
                "metric": metric,
                "threshold": threshold,
                "detail": detail,
            }
        )

    if not raw_input:
        raise ValueError("Please provide a non-empty SMILES string.")

    # Optional alias expansion ([Boc], [Ph], [OMe], etc.)
    working_smiles = raw_input
    expanded_note = None
    labelled_probe = parse_labelled(raw_input) if expand_abbrev else None
    has_labelled_alias = bool(
        labelled_probe is not None and any(a.GetAtomicNum() == 0 for a in labelled_probe.GetAtoms())
    )
    if expand_abbrev and (has_labelled_alias or "*" in raw_input):
        try:
            exp_mol = expand_aliases(raw_input)
            working_smiles = Chem.MolToSmiles(exp_mol)
            expanded_note = f"Expanded alias SMILES -> {working_smiles}"
        except ValueError as exc:
            add_gate(
                "G1_VALENCE_TOPOLOGY",
                "1. Valence, Octet & Single-Component Topology",
                "fail",
                str(exc),
                "Valid Kekule valence, single molecule",
                f"Alias expansion rejected {raw_input!r}: {exc}",
            )
            return {
                "ok": False,
                "overall_status": "fail",
                "input_smiles": raw_input,
                "standardized_smiles": None,
                "gates": gates,
                "elapsed_seconds": round(time.perf_counter() - t_start, 3),
            }

    # Gate 1: Valence, Kekulization & Single-Component Check
    raw_mol = Chem.MolFromSmiles(working_smiles)
    if raw_mol is None or raw_mol.GetNumAtoms() == 0:
        add_gate(
            "G1_VALENCE_TOPOLOGY",
            "1. Valence, Octet & Single-Component Topology",
            "fail",
            "Unparsable / valence violation",
            "RDKit sanitization & Kekulization",
            f"RDKit rejected {working_smiles!r} due to syntax or atomic valence violation.",
        )
        return {
            "ok": False,
            "overall_status": "fail",
            "input_smiles": raw_input,
            "standardized_smiles": None,
            "gates": gates,
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
        }

    frags = Chem.GetMolFrags(raw_mol)
    has_dummy = any(a.GetAtomicNum() == 0 for a in raw_mol.GetAtoms())
    std = standardize(working_smiles)
    if len(frags) > 1 or has_dummy or not std.valid:
        reason = (
            f"Multi-fragment mixture/salt ({len(frags)} fragments)"
            if len(frags) > 1
            else ("Contains unexpanded R-group dummy (*)" if has_dummy else std.reason)
        )
        add_gate(
            "G1_VALENCE_TOPOLOGY",
            "1. Valence, Octet & Single-Component Topology",
            "fail",
            reason,
            "1 covalently connected component, 0 dummy atoms",
            "3D gas-phase MLIPs require a single well-defined covalent graph without counterions.",
        )
        # Also evaluate Gate 2 on raw_mol so the user sees charge/element diagnostics!
        net_q = sum(a.GetFormalCharge() for a in raw_mol.GetAtoms())
        n_rad = sum(a.GetNumRadicalElectrons() for a in raw_mol.GetAtoms())
        req_els = {a.GetSymbol() for a in raw_mol.GetAtoms() if a.GetAtomicNum() > 0}
        target_els = ANI2X_ELEMENTS if "ani" in potential.lower() else MACE_ELEMENTS
        unsup = sorted(req_els - target_els)
        add_gate(
            "G2_DOMAIN_ELIGIBILITY",
            "2. Calculator Physical Domain Eligibility",
            "fail" if (net_q != 0 or n_rad != 0 or unsup) else "warn",
            f"Q={net_q:+d}, radicals={n_rad}, unsup={unsup or 'none'}",
            "Net charge Q=0, spin singlet (radicals=0), supported elements",
            f"Elements present: {sorted(req_els)}.",
        )
        return {
            "ok": False,
            "overall_status": "fail",
            "input_smiles": raw_input,
            "standardized_smiles": std.smiles if std.valid else None,
            "depiction_2d": png_data_url(raw_mol),
            "gates": gates,
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
        }

    add_gate(
        "G1_VALENCE_TOPOLOGY",
        "1. Valence, Octet & Single-Component Topology",
        "pass",
        f"{std.mol.GetNumAtoms()} heavy atoms, 1 component",
        "Valid Kekule & valence rules",
        expanded_note or f"Canonical isomeric SMILES: {std.smiles}",
    )

    # Gate 2: Calculator Domain Eligibility (Charge, Radicals, Supported Elements)
    target_els = (
        ANI2X_ELEMENTS
        if "ani" in potential.lower()
        else (
            MMFF94sASECalculator.supported_elements
            if potential.lower() == "mmff94s"
            else MACE_ELEMENTS
        )
    )
    policy = EligibilityPolicy(
        supported_elements=target_els,
        require_assigned_stereo=False,
        allow_charge=False,
        allow_radicals=False,
    )
    elig_reasons = policy.reasons(std.mol)
    net_charge = sum(a.GetFormalCharge() for a in std.mol.GetAtoms())
    n_radicals = sum(a.GetNumRadicalElectrons() for a in std.mol.GetAtoms())
    mol_h_probe = Chem.AddHs(Chem.Mol(std.mol))
    formula = CalcMolFormula(mol_h_probe)
    mw = float(Descriptors.MolWt(mol_h_probe))

    if elig_reasons:
        add_gate(
            "G2_DOMAIN_ELIGIBILITY",
            "2. Calculator Physical Domain Eligibility",
            "fail",
            "; ".join(elig_reasons),
            f"Q=0, singlet, Z in {sorted(target_els)}",
            f"Formula {formula} (MW={mw:.2f} Da), Q={net_charge:+d}, radicals={n_radicals}.",
        )
        return {
            "ok": False,
            "overall_status": "fail",
            "input_smiles": raw_input,
            "standardized_smiles": std.smiles,
            "inchikey": standard_inchikey(std),
            "formula": formula,
            "molecular_weight": round(mw, 3),
            "depiction_2d": png_data_url(std.mol),
            "gates": gates,
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
        }

    add_gate(
        "G2_DOMAIN_ELIGIBILITY",
        "2. Calculator Physical Domain Eligibility",
        "pass",
        f"Q=0, radicals=0, {formula} ({mw:.1f} Da)",
        f"Neutral singlet in {potential} domain",
        f"All elements supported by {potential}; InChIKey={standard_inchikey(std)}.",
    )

    # Gate 3: Stereochemical Specification Completeness
    cip_info = _cip_summary(std.mol)
    n_unassigned = cip_info["n_unassigned_tetrahedral"] + cip_info["n_unassigned_double_bonds"]
    if n_unassigned > 0:
        add_gate(
            "G3_STEREO_SPECIFICATION",
            "3. Stereochemical Specification Completeness",
            "warn",
            f"{n_unassigned} unassigned ({cip_info['n_specified']} specified)",
            "0 unassigned stereogenic centers/bonds",
            f"Unspecified tetrahedral={cip_info['n_unassigned_tetrahedral']}, "
            f"double-bond={cip_info['n_unassigned_double_bonds']}; ETKDG picks one isomer.",
        )
    else:
        add_gate(
            "G3_STEREO_SPECIFICATION",
            "3. Stereochemical Specification Completeness",
            "pass",
            f"{cip_info['n_specified']} specified, 0 unassigned",
            "All stereocenters & E/Z bonds explicit",
            "Unambiguous 3D stereoisomer mapping guaranteed.",
        )

    # Gate 4: Distance-Geometry 3D Embedding (ETKDGv3)
    embed_diag: list[dict] = []
    mol_3d, embed_status = embed(
        std.smiles, n_conf=max(1, int(n_conf)), seed=seed, diagnostics=embed_diag
    )
    if mol_3d is None:
        add_gate(
            "G4_ETKDG_3D_EMBEDDING",
            "4. ETKDGv3 3D Conformer Embedding",
            "fail",
            embed_status,
            ">= 1 valid 3D conformer",
            "Distance-geometry triangle bounds failed (anti-Bredt or impossible cage).",
        )
        return {
            "ok": False,
            "overall_status": "fail",
            "input_smiles": raw_input,
            "standardized_smiles": std.smiles,
            "inchikey": standard_inchikey(std),
            "formula": formula,
            "molecular_weight": round(mw, 3),
            "depiction_2d": png_data_url(std.mol),
            "gates": gates,
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
        }

    cids = [c.GetId() for c in mol_3d.GetConformers()]
    embed_stereo_flags = stereo_preserved(mol_3d, std.smiles)
    n_st_ok = sum(embed_stereo_flags)
    add_gate(
        "G4_ETKDG_3D_EMBEDDING",
        "4. ETKDGv3 3D Conformer Embedding",
        "pass",
        f"{len(cids)}/{n_conf} conformers embedded",
        ">= 1 chirality-enforced conformer",
        f"Embedded {len(cids)} conformers ({n_st_ok}/{len(cids)} match 2D CIP tags).",
    )

    # Instantiate 3D potential calculator
    calc, calc_label, _ = get_3d_calculator(potential, mol_3d, device=device)
    symbols = [a.GetSymbol() for a in mol_3d.GetAtoms()]
    atomic_numbers = [int(a.GetAtomicNum()) for a in mol_3d.GetAtoms()]

    # Evaluate / relax conformers
    conf_records: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    rng = np.random.default_rng(seed + 17)

    for cid, st_embed_ok in zip(cids, embed_stereo_flags, strict=True):
        pos_init = np.asarray(mol_3d.GetConformer(cid).GetPositions(), dtype=np.float64)
        if perturb_amplitude_a > 0.0:
            dirs = rng.normal(size=pos_init.shape)
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
            pos_init = pos_init + dirs * rng.uniform(
                0.0, perturb_amplitude_a, size=(len(pos_init), 1)
            )
        if inject_steric_clash and len(pos_init) >= 2:
            # Deliberately place last atom 0.45 A from atom 0 to demonstrate Pauli clash gate!
            pos_init = pos_init.copy()
            pos_init[-1] = pos_init[0] + np.array([0.45, 0.0, 0.0])

        atoms_obj = Atoms(symbols=symbols, positions=pos_init)
        if do_relax and not inject_steric_clash:
            r = relax(atoms_obj, calc, fmax=fmax, steps=steps)
            e_ev = r.energy
            f_ev_a = r.forces
            pos_final = r.positions
            conv = r.converged and (r.fmax <= fmax * 1.02)
            fmax_val = r.fmax
            n_steps = r.steps
        else:
            atoms_obj.calc = calc
            e_ev = float(atoms_obj.get_potential_energy())
            f_ev_a = np.asarray(atoms_obj.get_forces(), dtype=np.float64)
            pos_final = pos_init
            fmax_val = float(np.linalg.norm(f_ev_a, axis=1).max())
            conv = fmax_val <= fmax
            n_steps = 0

        mol_after = Chem.Mol(mol_3d)
        conf_after = mol_after.GetConformer(cid)
        for idx, xyz in enumerate(pos_final):
            conf_after.SetAtomPosition(idx, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
        st_final_ok = bool(stereo_matches(mol_after, std.smiles, cid))
        smi_3d = stereo_from_3d(mol_after, cid)

        entry = {
            "conf_id": int(cid),
            "energy_ev": float(e_ev),
            "energy_kcal": float(e_ev * EV_TO_KCAL),
            "fmax_ev_a": float(fmax_val),
            "rms_force_ev_a": float(np.sqrt(np.mean(np.sum(f_ev_a**2, axis=1)))),
            "converged": bool(conv),
            "steps": int(n_steps),
            "stereo_ok_embed": bool(st_embed_ok),
            "stereo_ok_final": st_final_ok,
            "smiles_from_3d": smi_3d,
            "positions": pos_final,
            "forces": f_ev_a,
        }
        conf_records.append(entry)
        if best_entry is None or entry["energy_ev"] < best_entry["energy_ev"]:
            best_entry = entry

    assert best_entry is not None
    e_min_ev = min(c["energy_ev"] for c in conf_records)
    boltz_raw = [
        math.exp(-max(0.0, c["energy_ev"] - e_min_ev) / (KB_EV_PER_K * 298.15))
        for c in conf_records
    ]
    z_part = sum(boltz_raw) or 1.0
    for c, bw in zip(conf_records, boltz_raw, strict=True):
        c["rel_energy_kcal"] = round((c["energy_ev"] - e_min_ev) * EV_TO_KCAL, 4)
        c["boltzmann_weight"] = round(bw / z_part, 4)

    pos_best = np.asarray(best_entry["positions"], dtype=np.float64)
    forces_best = np.asarray(best_entry["forces"], dtype=np.float64)

    # Gate 5: Pauli Exclusion & Nonbonded Steric Clearance
    diff_mat = pos_best[:, None, :] - pos_best[None, :, :]
    dist_mat = np.linalg.norm(diff_mat, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    d_min_global = float(dist_mat.min())

    bonded_pairs: set[tuple[int, int]] = set()
    neighbors: dict[int, list[int]] = {i: [] for i in range(len(symbols))}
    for b in mol_3d.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded_pairs.add((min(u, v), max(u, v)))
        neighbors[u].append(v)
        neighbors[v].append(u)
    one_three_pairs: set[tuple[int, int]] = set()
    for _center_idx, nbrs in neighbors.items():
        for i_idx in range(len(nbrs)):
            for j_idx in range(i_idx + 1, len(nbrs)):
                a1, a2 = nbrs[i_idx], nbrs[j_idx]
                one_three_pairs.add((min(a1, a2), max(a1, a2)))

    clashes: list[dict[str, Any]] = []
    min_nonbond_a = float("inf")
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pair = (i, j)
            d_ij = float(dist_mat[i, j])
            if pair not in bonded_pairs and pair not in one_three_pairs:
                min_nonbond_a = min(min_nonbond_a, d_ij)
                thr = 1.15 if ("H" in (symbols[i], symbols[j])) else 1.40
                if d_ij < thr:
                    clashes.append(
                        {
                            "i": i,
                            "j": j,
                            "pair": f"{symbols[i]}{i}-{symbols[j]}{j}",
                            "distance_a": round(d_ij, 4),
                        }
                    )
            elif d_ij < 0.75:
                clashes.append(
                    {
                        "i": i,
                        "j": j,
                        "pair": f"{symbols[i]}{i}-{symbols[j]}{j}",
                        "distance_a": round(d_ij, 4),
                    }
                )

    steric_ok = (d_min_global >= 0.75) and (len(clashes) == 0)
    add_gate(
        "G5_PAULI_STERIC_CLEARANCE",
        "5. Pauli Exclusion & Nonbonded Steric Clearance",
        "pass" if steric_ok else "fail",
        f"d_min={d_min_global:.3f} A, nonbond_min={min_nonbond_a:.3f} A ({len(clashes)} clashes)",
        "d_min >= 0.75 A & 0 nonbonded clashes",
        "No atomic overlaps detected." if steric_ok else f"Clashes detected: {clashes[:3]}",
    )

    # Gate 6: Covalent Bond-Length & Bond-Angle Plausibility
    pt = Chem.GetPeriodicTable()
    mol_best = Chem.Mol(mol_3d)
    conf_best = mol_best.GetConformer(best_entry["conf_id"])
    for idx, xyz in enumerate(pos_best):
        conf_best.SetAtomPosition(idx, [float(xyz[0]), float(xyz[1]), float(xyz[2])])

    bonds_payload: list[dict[str, Any]] = []
    max_cov_dev = 0.0
    for b in mol_best.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        r_uv = float(np.linalg.norm(pos_best[u] - pos_best[v]))
        r_cov_sum = float(pt.GetRcovalent(atomic_numbers[u]) + pt.GetRcovalent(atomic_numbers[v]))
        # Multiple bonds are shorter than single-bond covalent sum by ~0.08-0.22 A
        bo = float(b.GetBondTypeAsDouble())
        expected_r = r_cov_sum - (0.11 * max(0.0, bo - 1.0))
        dev = abs(r_uv - expected_r)
        max_cov_dev = max(max_cov_dev, dev)
        bonds_payload.append(
            {
                "u": u,
                "v": v,
                "order": bo,
                "length_a": round(r_uv, 4),
                "expected_a": round(expected_r, 4),
                "deviation_a": round(dev, 4),
            }
        )

    angles_deg: list[float] = []
    for center_idx, nbrs in neighbors.items():
        if len(nbrs) < 2:
            continue
        for i_idx in range(len(nbrs)):
            for j_idx in range(i_idx + 1, len(nbrs)):
                ang = float(
                    rdMolTransforms.GetAngleDeg(conf_best, nbrs[i_idx], center_idx, nbrs[j_idx])
                )
                angles_deg.append(ang)
    min_ang = min(angles_deg) if angles_deg else 109.5
    max_ang = max(angles_deg) if angles_deg else 109.5
    geom_status = "pass" if max_cov_dev <= 0.30 else ("warn" if max_cov_dev <= 0.45 else "fail")
    add_gate(
        "G6_COVALENT_GEOMETRY",
        "6. Covalent Bond-Length & Angle Plausibility",
        geom_status,
        f"max |dR_cov|={max_cov_dev:.3f} A, angles=[{min_ang:.1f} deg, {max_ang:.1f} deg]",
        "max |dR_cov| <= 0.30 A",
        f"Audited {len(bonds_payload)} covalent bonds and {len(angles_deg)} valence angles.",
    )

    # Gate 7: 3D Chirality & E/Z Retention (Pre/Post Relaxation)
    st_ok = best_entry["stereo_ok_embed"] and best_entry["stereo_ok_final"]
    add_gate(
        "G7_3D_CHIRALITY_RETENTION",
        "7. 3D Chirality & Double-Bond Stereo Retention",
        "pass" if st_ok else "fail",
        f"embed={best_entry['stereo_ok_embed']}, final={best_entry['stereo_ok_final']}",
        "3D CIP matches 2D input before & after relaxation",
        f"3D-reperceived SMILES: {best_entry['smiles_from_3d']}",
    )

    # Gate 8: Stationary-Point Force Convergence & Finite Energetics
    finite_ef = bool(
        np.isfinite(best_entry["energy_ev"])
        and np.isfinite(pos_best).all()
        and np.isfinite(forces_best).all()
    )
    if not finite_ef:
        g8_status = "fail"
    elif do_relax and not inject_steric_clash:
        g8_status = "pass" if best_entry["converged"] else "warn"
    else:
        g8_status = "pass" if best_entry["fmax_ev_a"] < 5.0 else "warn"
    add_gate(
        "G8_STATIONARY_CONVERGENCE",
        "8. Potential Energy & Stationary-Point Force Check",
        g8_status,
        f"E={best_entry['energy_ev']:.4f} eV, fmax={best_entry['fmax_ev_a']:.4f} eV/A",
        f"Finite E/F; relaxed fmax <= {fmax:.3f} eV/A" if do_relax else "Finite single-point E/F",
        f"{calc_label} ({best_entry['steps']} LBFGS steps, {len(conf_records)} conformers ranked).",
    )

    # Gate 9: Live SE(3) Rotation/Translation Equivariance & Noether Momentum Conservation
    q_rot = _random_so3_matrix(seed=seed + 101)
    t_vec = np.array([2.35, -1.65, 3.10], dtype=np.float64)
    pos_se3 = pos_best @ q_rot.T + t_vec[None, :]

    at_orig = Atoms(symbols=symbols, positions=pos_best)
    at_orig.calc = calc
    e_orig = float(at_orig.get_potential_energy())
    f_orig = np.asarray(at_orig.get_forces(), dtype=np.float64)

    at_rot = Atoms(symbols=symbols, positions=pos_se3)
    at_rot.calc = calc
    e_rot = float(at_rot.get_potential_energy())
    f_rot = np.asarray(at_rot.get_forces(), dtype=np.float64)

    f_expected_rot = f_orig @ q_rot.T
    se3_e_err = abs(e_rot - e_orig)
    so3_f_err = float(np.linalg.norm(f_rot - f_expected_rot, axis=1).max())
    net_force = float(np.linalg.norm(f_orig.sum(axis=0)))
    com = pos_best.mean(axis=0, keepdims=True)
    net_torque = float(np.linalg.norm(np.cross(pos_best - com, f_orig).sum(axis=0)))

    se3_ok = (
        (se3_e_err < 1e-4) and (so3_f_err < 1e-4) and (net_force < 1e-3) and (net_torque < 1e-3)
    )
    add_gate(
        "G9_SE3_EQUIVARIANCE_CONSERVATION",
        "9. SE(3) Equivariance & Momentum Conservation",
        "pass" if se3_ok else "fail",
        f"|dE|={se3_e_err:.2e} eV, max|dF_rot|={so3_f_err:.2e} eV/A",
        "|E(RQ+t)-E(R)| < 1e-4 eV & |F(RQ+t)-F(R)Q| < 1e-4",
        f"Noether: |sum F_i|={net_force:.2e} eV/A, |sum r_i x F_i|={net_torque:.2e} eV.",
    )

    # Gate 10: O(3) Reflection Parity + Conservative Gradient F = -grad_R E
    pos_mir = mirror_positions(pos_best)
    at_mir = Atoms(symbols=symbols, positions=pos_mir)
    at_mir.calc = calc
    e_mir = float(at_mir.get_potential_energy())
    f_mir = np.asarray(at_mir.get_forces(), dtype=np.float64)
    f_mir_expected = mirror_positions(f_orig)
    parity_e_err = abs(e_mir - e_orig)
    parity_f_err = float(np.linalg.norm(f_mir - f_mir_expected, axis=1).max())

    mol_mir = Chem.Mol(mol_3d)
    conf_mir = mol_mir.GetConformer(best_entry["conf_id"])
    for idx, xyz in enumerate(pos_mir):
        conf_mir.SetAtomPosition(idx, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
    smi_mir_3d = stereo_from_3d(mol_mir, best_entry["conf_id"])
    mir_rel = categorize(smi_mir_3d, best_entry["smiles_from_3d"])

    # Central finite-difference directional derivative check along random unit vector
    u_dir = rng.normal(size=pos_best.shape)
    u_dir /= max(float(np.linalg.norm(u_dir)), 1e-12)
    delta_fd = 1e-3
    at_plus = Atoms(symbols=symbols, positions=pos_best + delta_fd * u_dir)
    at_plus.calc = calc
    at_minus = Atoms(symbols=symbols, positions=pos_best - delta_fd * u_dir)
    at_minus.calc = calc
    fd_grad = (float(at_plus.get_potential_energy()) - float(at_minus.get_potential_energy())) / (
        2.0 * delta_fd
    )
    anal_grad = float(-np.sum(f_orig * u_dir))
    grad_err = abs(fd_grad - anal_grad)

    g10_ok = (parity_e_err < 1e-4) and (parity_f_err < 1e-4) and (grad_err < 2e-3)
    add_gate(
        "G10_PARITY_AND_CONSERVATIVE_FORCE",
        "10. O(3) Reflection Parity & Conservative F = -grad E",
        "pass" if g10_ok else "fail",
        f"|dE_mir|={parity_e_err:.2e} eV, |F + dE/dR|={grad_err:.2e} eV/A",
        "|E(-x)-E(x)| < 1e-4 eV & conservative FD err < 2e-3",
        f"Mirror: {mir_rel} ({smi_mir_3d}); anal={anal_grad:.5f} vs FD={fd_grad:.5f} eV/A.",
    )

    cip_by_atom = {c["atom_idx"]: c["cip"] for c in cip_info["chiral_centers"]}
    atoms_payload = [
        {
            "idx": i,
            "symbol": symbols[i],
            "z": atomic_numbers[i],
            "x": round(float(pos_best[i, 0]), 5),
            "y": round(float(pos_best[i, 1]), 5),
            "z_coord": round(float(pos_best[i, 2]), 5),
            "fx": round(float(forces_best[i, 0]), 5),
            "fy": round(float(forces_best[i, 1]), 5),
            "fz": round(float(forces_best[i, 2]), 5),
            "f_norm": round(float(np.linalg.norm(forces_best[i])), 5),
            "cip": cip_by_atom.get(i),
        }
        for i in range(len(symbols))
    ]

    statuses = [g["status"] for g in gates]
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    n_pass = sum(s == "pass" for s in statuses)
    n_warn = sum(s == "warn" for s in statuses)
    n_fail = sum(s == "fail" for s in statuses)

    return {
        "ok": overall != "fail",
        "overall_status": overall,
        "summary_counts": {"pass": n_pass, "warn": n_warn, "fail": n_fail, "total": len(gates)},
        "input_smiles": raw_input,
        "standardized_smiles": std.smiles,
        "smiles_from_3d": best_entry["smiles_from_3d"],
        "mirror_3d_smiles": smi_mir_3d,
        "mirror_relation": mir_rel,
        "inchikey": standard_inchikey(std),
        "formula": formula,
        "molecular_weight": round(mw, 3),
        "potential_label": calc_label,
        "depiction_2d": png_data_url(std.mol),
        "cip_info": cip_info,
        "gates": gates,
        "equivariance_metrics": {
            "se3_energy_abs_err_ev": se3_e_err,
            "so3_force_max_err_ev_a": so3_f_err,
            "net_force_norm_ev_a": net_force,
            "net_torque_norm_ev": net_torque,
            "parity_energy_abs_err_ev": parity_e_err,
            "parity_force_max_err_ev_a": parity_f_err,
            "conservative_grad_fd_err_ev_a": grad_err,
        },
        "structure_3d": {
            "n_atoms": len(atoms_payload),
            "atoms": atoms_payload,
            "bonds": bonds_payload,
            "clashes": clashes,
            "d_min_a": round(d_min_global, 4),
            "energy_ev": round(best_entry["energy_ev"], 6),
            "energy_kcal": round(best_entry["energy_kcal"], 4),
            "fmax_ev_a": round(best_entry["fmax_ev_a"], 5),
        },
        "conformers": [
            {
                **{k: v for k, v in c.items() if k not in ("positions", "forces")},
                "coords": np.round(c["positions"], 5).tolist(),
                "forces_vec": np.round(c["forces"], 5).tolist(),
            }
            for c in sorted(conf_records, key=lambda r: r["energy_ev"])
        ],
        "exports": {
            "xyz": _format_xyz(
                symbols,
                pos_best,
                f"{std.smiles} | {calc_label} | E={best_entry['energy_ev']:.6f} eV",
            ),
            "extxyz": _format_extxyz(
                symbols, pos_best, forces_best, best_entry["energy_ev"], std.smiles
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - t_start, 3),
    }
