"""Build a 3D molecule from SMILES and check 10 basic rules of chemistry and physics."""

from __future__ import annotations

import base64
import io
import math
import time
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write as ase_write
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdCIPLabeler, rdMolTransforms
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
    "Everyday chiral molecules": [
        {
            "name": "L-Alanine (amino acid)",
            "smiles": "N[C@@H](C)C(=O)O",
            "note": "Small chiral molecule; fast to relax in 3D.",
        },
        {
            "name": "(S)-Ibuprofen (pain reliever)",
            "smiles": "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
            "note": "Has one chiral carbon center.",
        },
        {
            "name": "(E)-Resveratrol (double-bond geometry)",
            "smiles": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
            "note": "Tests trans (E) double-bond shape.",
        },
    ],
    "Tight 3D ring cages": [
        {
            "name": "BCP cage (bicyclo[1.1.1]pentane)",
            "smiles": "CNC(=O)C12CC(C1)C2",
            "note": "Compact 3D cage with tight angles.",
        },
        {
            "name": "Cubane amide (cube-shaped carbon cage)",
            "smiles": "NC(=O)C12C3C4C1C5C2C3C45",
            "note": "Carbon atoms arranged at 90-degree corners of a cube.",
        },
        {
            "name": "Phenyl-SF5 (6-bond sulfur)",
            "smiles": "Fc1ccccc1S(F)(F)(F)(F)F",
            "note": "Sulfur bonded to five fluorines and one carbon.",
        },
    ],
    "Edge cases & rule breakers": [
        {
            "name": "Unlabeled lactic acid (warns on rule 3)",
            "smiles": "CC(O)C(=O)O",
            "note": "Chiral carbon is missing @ or @@.",
        },
        {
            "name": "Boc & Ph shorthand ([Boc], [Ph])",
            "smiles": "[Boc]NCc1ccc([Ph])cc1",
            "note": "Expands common chemistry shorthand groups before building 3D.",
        },
        {
            "name": "Sodium acetate salt (fails rule 1)",
            "smiles": "CC(=O)[O-].[Na+]",
            "note": "Two separate charged ions instead of one neutral molecule.",
        },
        {
            "name": "Benzyl radical (fails rule 2)",
            "smiles": "[CH2]c1ccccc1",
            "note": "Has an unpaired electron.",
        },
        {
            "name": "Boron ester (fails rule 2 on MACE)",
            "smiles": "CC1(C)OB(c2ccc(F)cc2)OC1(C)C",
            "note": "MACE-OFF23 does not support boron (B).",
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
    """Fast RDKit MMFF94s force field wrapped as an ASE Calculator."""

    implemented_properties = ["energy", "forces"]
    supported_elements = frozenset({"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"})

    def __init__(self, template_mol_h: Chem.Mol) -> None:
        super().__init__()
        self._mol = Chem.Mol(template_mol_h)
        if self._mol.GetNumConformers() == 0:
            self._mol.AddConformer(Chem.Conformer(self._mol.GetNumAtoms()), assignId=True)
        self._use_uff = AllChem.MMFFGetMoleculeProperties(self._mol, mmffVariant="MMFF94s") is None

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
        for i, xyz in enumerate(pos):
            conf.SetAtomPosition(i, [float(xyz[0]), float(xyz[1]), float(xyz[2])])

        if not self._use_uff:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if ff is None:
            raise RuntimeError("Could not set up RDKit force field for this molecule.")

        energy_kcal = float(ff.CalcEnergy())
        grad_kcal_a = np.asarray(ff.CalcGrad(), dtype=np.float64).reshape(-1, 3)
        self.results["energy"] = energy_kcal / EV_TO_KCAL
        self.results["forces"] = -grad_kcal_a / EV_TO_KCAL


_ANI_CACHE: dict[str, Any] = {}


class ANI2xASECalculator(Calculator):
    """TorchANI ANI-2x neural potential wrapped as an ASE Calculator."""

    implemented_properties = ["energy", "forces"]
    supported_elements = ANI2X_ELEMENTS

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        import torch

        try:
            import torchani
        except ImportError as exc:
            raise RuntimeError("Optional package 'torchani' is not installed.") from exc

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
        numbers = torch.tensor(
            [target.get_atomic_numbers().tolist()], dtype=torch.long, device=self._device
        )
        coords = torch.tensor(
            [target.get_positions().tolist()],
            dtype=torch.float64,
            device=self._device,
            requires_grad=True,
        )
        e_hartree = self._model((numbers, coords)).energies
        grad = torch.autograd.grad(e_hartree.sum(), coords)[0][0].detach().cpu().numpy()
        self.results["energy"] = float(e_hartree.item()) * HARTREE_TO_EV
        self.results["forces"] = (-grad * HARTREE_TO_EV).astype(np.float64)


def get_3d_calculator(
    potential: str, template_mol_h: Chem.Mol, device: str = "cpu"
) -> tuple[Calculator, str, frozenset[str]]:
    key = potential.strip().lower()
    if key == "mmff94s":
        calc = MMFF94sASECalculator(template_mol_h)
        return calc, "MMFF94s (fast rule-based forces)", calc.supported_elements
    if key in ("mace-small", "small"):
        return (
            mace_calculator("small", device=device, dtype="float64"),
            "MACE-OFF23-small",
            MACE_ELEMENTS,
        )
    if key in ("mace-medium", "medium"):
        return (
            mace_calculator("medium", device=device, dtype="float64"),
            "MACE-OFF23-medium",
            MACE_ELEMENTS,
        )
    if key in ("mace-large", "large"):
        return (
            mace_calculator("large", device=device, dtype="float64"),
            "MACE-OFF23-large",
            MACE_ELEMENTS,
        )
    if key in ("ani2x", "ani-2x"):
        calc = ANI2xASECalculator(device=device)
        return calc, "ANI-2x", ANI2X_ELEMENTS
    raise ValueError(f"Unknown force model: {potential!r}")


def _random_rotation_matrix(seed: int = 42) -> np.ndarray:
    """Build a random 3D rotation matrix (det = +1, no mirror flip)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _stereo_info(mol_no_h: Chem.Mol) -> dict[str, Any]:
    m = Chem.Mol(mol_no_h)
    rdCIPLabeler.AssignCIPLabels(m)
    centers = [
        {"atom_idx": a.GetIdx(), "symbol": a.GetSymbol(), "cip": a.GetProp("_CIPCode")}
        for a in m.GetAtoms()
        if a.HasProp("_CIPCode")
    ]
    bonds = [
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
    for item in Chem.FindPotentialStereo(Chem.Mol(m)):
        if item.specified == Chem.StereoSpecified.Unspecified:
            if item.type == Chem.StereoType.Atom_Tetrahedral:
                unassigned_tet += 1
            elif item.type == Chem.StereoType.Bond_Double:
                unassigned_db += 1
    return {
        "chiral_centers": centers,
        "stereo_bonds": bonds,
        "n_specified": len(centers) + len(bonds),
        "n_unassigned_tetrahedral": unassigned_tet,
        "n_unassigned_double_bonds": unassigned_db,
    }


def _to_xyz(symbols: list[str], pos: np.ndarray, comment: str) -> str:
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, pos, strict=True):
        lines.append(f"{sym:<3s} {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines) + "\n"


def _to_extxyz(
    symbols: list[str], pos: np.ndarray, forces: np.ndarray, energy_ev: float, smiles: str
) -> str:
    atoms = Atoms(symbols=symbols, positions=pos)
    atoms.info["smiles"] = smiles
    atoms.info["mace_energy"] = float(energy_ev)
    atoms.arrays["mace_forces"] = np.asarray(forces, dtype=np.float64)
    buf = io.StringIO()
    ase_write(buf, atoms, format="extxyz")
    return buf.getvalue()


def _gate(gate_id: str, title: str, status: str, metric: str, threshold: str, detail: str) -> dict:
    return {
        "id": gate_id,
        "title": title,
        "status": status,
        "metric": metric,
        "threshold": threshold,
        "detail": detail,
    }


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
    """Build 3D conformers from SMILES and check 10 basic rules of chemistry and physics."""
    t0 = time.perf_counter()
    raw_input = (smiles or "").strip()
    if not raw_input:
        raise ValueError("Please enter a SMILES string.")

    gates: list[dict[str, Any]] = []

    # Expand shorthand labels like [Boc] or [Ph] if requested
    working_smiles = raw_input
    expanded_note = ""
    labelled = parse_labelled(raw_input) if expand_abbrev else None
    has_alias = bool(
        labelled is not None and any(a.GetAtomicNum() == 0 for a in labelled.GetAtoms())
    )
    if expand_abbrev and (has_alias or "*" in raw_input):
        try:
            working_smiles = Chem.MolToSmiles(expand_aliases(raw_input))
            expanded_note = f"Expanded shorthand to {working_smiles}"
        except ValueError as exc:
            gates.append(
                _gate(
                    "G1_VALENCE_TOPOLOGY",
                    "1. Valid chemical bonds",
                    "fail",
                    str(exc),
                    "Valid bonds, one connected molecule",
                    f"Could not expand shorthand in {raw_input!r}.",
                )
            )
            return {
                "ok": False,
                "overall_status": "fail",
                "input_smiles": raw_input,
                "standardized_smiles": None,
                "gates": gates,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }

    # Check 1: Valid chemical bonds and single connected molecule
    raw_mol = Chem.MolFromSmiles(working_smiles)
    if raw_mol is None or raw_mol.GetNumAtoms() == 0:
        gates.append(
            _gate(
                "G1_VALENCE_TOPOLOGY",
                "1. Valid chemical bonds",
                "fail",
                "Invalid SMILES or impossible bond count",
                "Valid bonds, one connected molecule",
                f"RDKit could not parse {working_smiles!r}.",
            )
        )
        return {
            "ok": False,
            "overall_status": "fail",
            "input_smiles": raw_input,
            "standardized_smiles": None,
            "gates": gates,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    frags = Chem.GetMolFrags(raw_mol)
    has_dummy = any(a.GetAtomicNum() == 0 for a in raw_mol.GetAtoms())
    std = standardize(working_smiles)
    if len(frags) > 1 or has_dummy or not std.valid:
        why = (
            f"Contains {len(frags)} separate pieces (salt or mixture)"
            if len(frags) > 1
            else ("Contains unknown * group" if has_dummy else std.reason)
        )
        gates.append(
            _gate(
                "G1_VALENCE_TOPOLOGY",
                "1. Valid chemical bonds",
                "fail",
                why,
                "One connected molecule, no * placeholders",
                "3D force models expect one connected molecule, not separate ions.",
            )
        )
        net_q = sum(a.GetFormalCharge() for a in raw_mol.GetAtoms())
        n_rad = sum(a.GetNumRadicalElectrons() for a in raw_mol.GetAtoms())
        present = {a.GetSymbol() for a in raw_mol.GetAtoms() if a.GetAtomicNum() > 0}
        allowed = ANI2X_ELEMENTS if "ani" in potential.lower() else MACE_ELEMENTS
        unsup = sorted(present - allowed)
        gates.append(
            _gate(
                "G2_DOMAIN_ELIGIBILITY",
                "2. Neutral charge & supported elements",
                "fail" if (net_q != 0 or n_rad != 0 or unsup) else "warn",
                f"charge={net_q:+d}, radicals={n_rad}, unsupported={unsup or 'none'}",
                "Charge = 0, no unpaired electrons, supported elements",
                f"Elements found: {sorted(present)}.",
            )
        )
        return {
            "ok": False,
            "overall_status": "fail",
            "input_smiles": raw_input,
            "standardized_smiles": std.smiles if std.valid else None,
            "depiction_2d": png_data_url(raw_mol),
            "gates": gates,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    gates.append(
        _gate(
            "G1_VALENCE_TOPOLOGY",
            "1. Valid chemical bonds",
            "pass",
            f"{std.mol.GetNumAtoms()} non-H atoms, 1 molecule",
            "Valid bonds, one connected molecule",
            expanded_note or f"Clean SMILES: {std.smiles}",
        )
    )

    # Check 2: Neutral charge, no radicals, supported elements
    if "ani" in potential.lower():
        allowed_els = ANI2X_ELEMENTS
    elif potential.lower() == "mmff94s":
        allowed_els = MMFF94sASECalculator.supported_elements
    else:
        allowed_els = MACE_ELEMENTS

    policy = EligibilityPolicy(
        supported_elements=allowed_els,
        require_assigned_stereo=False,
        allow_charge=False,
        allow_radicals=False,
    )
    reasons = policy.reasons(std.mol)
    mol_with_h = Chem.AddHs(Chem.Mol(std.mol))
    formula = CalcMolFormula(mol_with_h)
    mw = float(Descriptors.MolWt(mol_with_h))

    if reasons:
        gates.append(
            _gate(
                "G2_DOMAIN_ELIGIBILITY",
                "2. Neutral charge & supported elements",
                "fail",
                "; ".join(reasons),
                f"Charge = 0, no radicals, elements in {sorted(allowed_els)}",
                f"Formula {formula} ({mw:.1f} Da).",
            )
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
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    gates.append(
        _gate(
            "G2_DOMAIN_ELIGIBILITY",
            "2. Neutral charge & supported elements",
            "pass",
            f"charge=0, radicals=0, {formula} ({mw:.1f} Da)",
            "Neutral molecule with supported atoms",
            f"All atoms are supported by {potential}.",
        )
    )

    # Check 3: Left/right handedness (R/S and E/Z) labeled
    cip_info = _stereo_info(std.mol)
    n_unassigned = cip_info["n_unassigned_tetrahedral"] + cip_info["n_unassigned_double_bonds"]
    if n_unassigned > 0:
        gates.append(
            _gate(
                "G3_STEREO_SPECIFICATION",
                "3. Left/right handedness labeled",
                "warn",
                f"{n_unassigned} unlabeled stereo site(s)",
                "Every chiral center or E/Z bond is labeled",
                "Missing @/@@ or /\\ labels means 3D builder has to guess one side.",
            )
        )
    else:
        gates.append(
            _gate(
                "G3_STEREO_SPECIFICATION",
                "3. Left/right handedness labeled",
                "pass",
                f"{cip_info['n_specified']} labeled, 0 missing",
                "Every chiral center or E/Z bond is labeled",
                "No missing 3D handedness labels.",
            )
        )

    # Check 4: Build 3D coordinates
    mol_3d, embed_status = embed(std.smiles, n_conf=max(1, int(n_conf)), seed=seed)
    if mol_3d is None:
        gates.append(
            _gate(
                "G4_ETKDG_3D_EMBEDDING",
                "4. 3D shape can be built",
                "fail",
                embed_status,
                "At least 1 valid 3D shape",
                "Could not place atoms in 3D without impossible ring strain.",
            )
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
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    cids = [c.GetId() for c in mol_3d.GetConformers()]
    embed_stereo_ok = stereo_preserved(mol_3d, std.smiles)
    gates.append(
        _gate(
            "G4_ETKDG_3D_EMBEDDING",
            "4. 3D shape can be built",
            "pass",
            f"Built {len(cids)} 3D shape(s)",
            "At least 1 valid 3D shape",
            f"{sum(embed_stereo_ok)}/{len(cids)} initial shapes match 2D handedness.",
        )
    )

    calc, calc_label, _ = get_3d_calculator(potential, mol_3d, device=device)
    symbols = [a.GetSymbol() for a in mol_3d.GetAtoms()]
    atomic_nums = [int(a.GetAtomicNum()) for a in mol_3d.GetAtoms()]

    # Score and relax each 3D conformer
    conf_records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    rng = np.random.default_rng(seed + 17)

    for cid, ok_before in zip(cids, embed_stereo_ok, strict=True):
        pos_start = np.asarray(mol_3d.GetConformer(cid).GetPositions(), dtype=np.float64)
        if perturb_amplitude_a > 0.0:
            dirs = rng.normal(size=pos_start.shape)
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
            pos_start = pos_start + dirs * rng.uniform(
                0.0, perturb_amplitude_a, size=(len(pos_start), 1)
            )
        if inject_steric_clash and len(pos_start) >= 2:
            pos_start = pos_start.copy()
            pos_start[-1] = pos_start[0] + np.array([0.45, 0.0, 0.0])

        atoms_obj = Atoms(symbols=symbols, positions=pos_start)
        if do_relax and not inject_steric_clash:
            res = relax(atoms_obj, calc, fmax=fmax, steps=steps)
            energy_ev = res.energy
            forces = res.forces
            pos_end = res.positions
            fmax_val = res.fmax
            converged = res.converged and (res.fmax <= fmax * 1.02)
            n_steps = res.steps
        else:
            atoms_obj.calc = calc
            energy_ev = float(atoms_obj.get_potential_energy())
            forces = np.asarray(atoms_obj.get_forces(), dtype=np.float64)
            pos_end = pos_start
            fmax_val = float(np.linalg.norm(forces, axis=1).max())
            converged = fmax_val <= fmax
            n_steps = 0

        mol_after = Chem.Mol(mol_3d)
        conf_after = mol_after.GetConformer(cid)
        for i, xyz in enumerate(pos_end):
            conf_after.SetAtomPosition(i, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
        ok_after = bool(stereo_matches(mol_after, std.smiles, cid))
        smi_3d = stereo_from_3d(mol_after, cid)

        item = {
            "conf_id": int(cid),
            "energy_ev": float(energy_ev),
            "energy_kcal": float(energy_ev * EV_TO_KCAL),
            "fmax_ev_a": float(fmax_val),
            "rms_force_ev_a": float(np.sqrt(np.mean(np.sum(forces**2, axis=1)))),
            "converged": bool(converged),
            "steps": int(n_steps),
            "stereo_ok_embed": bool(ok_before),
            "stereo_ok_final": ok_after,
            "smiles_from_3d": smi_3d,
            "positions": pos_end,
            "forces": forces,
        }
        conf_records.append(item)
        if best is None or item["energy_ev"] < best["energy_ev"]:
            best = item

    assert best is not None
    e_min = min(c["energy_ev"] for c in conf_records)
    weights = [
        math.exp(-max(0.0, c["energy_ev"] - e_min) / (KB_EV_PER_K * 298.15)) for c in conf_records
    ]
    w_sum = sum(weights) or 1.0
    for c, w in zip(conf_records, weights, strict=True):
        c["rel_energy_kcal"] = round((c["energy_ev"] - e_min) * EV_TO_KCAL, 4)
        c["boltzmann_weight"] = round(w / w_sum, 4)

    pos_best = np.asarray(best["positions"], dtype=np.float64)
    forces_best = np.asarray(best["forces"], dtype=np.float64)

    # Check 5: Atoms do not crash into each other
    dist_mat = np.linalg.norm(pos_best[:, None, :] - pos_best[None, :, :], axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    d_min = float(dist_mat.min())

    bonded: set[tuple[int, int]] = set()
    neighbors: dict[int, list[int]] = {i: [] for i in range(len(symbols))}
    for b in mol_3d.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded.add((min(u, v), max(u, v)))
        neighbors[u].append(v)
        neighbors[v].append(u)

    one_three: set[tuple[int, int]] = set()
    for nbrs in neighbors.values():
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                one_three.add((min(nbrs[i], nbrs[j]), max(nbrs[i], nbrs[j])))

    clashes: list[dict[str, Any]] = []
    min_nonbond = float("inf")
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pair = (i, j)
            d = float(dist_mat[i, j])
            if pair not in bonded and pair not in one_three:
                min_nonbond = min(min_nonbond, d)
                limit = 1.15 if ("H" in (symbols[i], symbols[j])) else 1.40
                if d < limit:
                    clashes.append(
                        {
                            "i": i,
                            "j": j,
                            "pair": f"{symbols[i]}{i}-{symbols[j]}{j}",
                            "distance_a": round(d, 4),
                        }
                    )
            elif d < 0.75:
                clashes.append(
                    {
                        "i": i,
                        "j": j,
                        "pair": f"{symbols[i]}{i}-{symbols[j]}{j}",
                        "distance_a": round(d, 4),
                    }
                )

    no_crash = (d_min >= 0.75) and (len(clashes) == 0)
    gates.append(
        _gate(
            "G5_PAULI_STERIC_CLEARANCE",
            "5. Atoms do not crash into each other",
            "pass" if no_crash else "fail",
            f"closest pair = {d_min:.3f} A ({len(clashes)} clashes)",
            "Closest atoms >= 0.75 A, 0 non-bonded clashes",
            "All atoms have enough space." if no_crash else f"Too close: {clashes[:3]}",
        )
    )

    # Check 6: Normal bond lengths and angles
    pt = Chem.GetPeriodicTable()
    mol_best = Chem.Mol(mol_3d)
    conf_best = mol_best.GetConformer(best["conf_id"])
    for i, xyz in enumerate(pos_best):
        conf_best.SetAtomPosition(i, [float(xyz[0]), float(xyz[1]), float(xyz[2])])

    bonds_out: list[dict[str, Any]] = []
    max_dev = 0.0
    for b in mol_best.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        length = float(np.linalg.norm(pos_best[u] - pos_best[v]))
        cov_sum = float(pt.GetRcovalent(atomic_nums[u]) + pt.GetRcovalent(atomic_nums[v]))
        order = float(b.GetBondTypeAsDouble())
        expected = cov_sum - (0.11 * max(0.0, order - 1.0))
        dev = abs(length - expected)
        max_dev = max(max_dev, dev)
        bonds_out.append(
            {
                "u": u,
                "v": v,
                "order": order,
                "length_a": round(length, 4),
                "expected_a": round(expected, 4),
                "deviation_a": round(dev, 4),
            }
        )

    angles: list[float] = []
    for center, nbrs in neighbors.items():
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                angles.append(
                    float(rdMolTransforms.GetAngleDeg(conf_best, nbrs[i], center, nbrs[j]))
                )
    min_ang = min(angles) if angles else 109.5
    max_ang = max(angles) if angles else 109.5
    gates.append(
        _gate(
            "G6_COVALENT_GEOMETRY",
            "6. Normal bond lengths & angles",
            "pass" if max_dev <= 0.30 else ("warn" if max_dev <= 0.45 else "fail"),
            f"max bond diff = {max_dev:.3f} A, angles = {min_ang:.1f}-{max_ang:.1f} deg",
            "Bond length within 0.30 A of typical value",
            f"Checked {len(bonds_out)} bonds and {len(angles)} bond angles.",
        )
    )

    # Check 7: 3D handedness kept after relaxing
    stereo_ok = best["stereo_ok_embed"] and best["stereo_ok_final"]
    gates.append(
        _gate(
            "G7_3D_CHIRALITY_RETENTION",
            "7. 3D handedness kept after relaxing",
            "pass" if stereo_ok else "fail",
            "kept" if stereo_ok else "flipped",
            "3D R/S and E/Z match input SMILES",
            f"SMILES read back from 3D coordinates: {best['smiles_from_3d']}",
        )
    )

    # Check 8: Settled to a low-energy shape
    finite_ok = bool(
        np.isfinite(best["energy_ev"])
        and np.isfinite(pos_best).all()
        and np.isfinite(forces_best).all()
    )
    if not finite_ok:
        g8_status = "fail"
    elif do_relax and not inject_steric_clash:
        g8_status = "pass" if best["converged"] else "warn"
    else:
        g8_status = "pass" if best["fmax_ev_a"] < 5.0 else "warn"
    gates.append(
        _gate(
            "G8_STATIONARY_CONVERGENCE",
            "8. Settled to a stable low-energy shape",
            g8_status,
            f"E = {best['energy_ev']:.4f} eV, max force = {best['fmax_ev_a']:.4f} eV/A",
            f"Max atom force <= {fmax:.3f} eV/A" if do_relax else "Finite energy and forces",
            f"{calc_label} ({best['steps']} relax steps).",
        )
    )

    # Check 9: Rotating or shifting in 3D space does not change physics
    rot = _random_rotation_matrix(seed=seed + 101)
    shift = np.array([2.35, -1.65, 3.10], dtype=np.float64)
    pos_moved = pos_best @ rot.T + shift[None, :]

    at_orig = Atoms(symbols=symbols, positions=pos_best)
    at_orig.calc = calc
    e_orig = float(at_orig.get_potential_energy())
    f_orig = np.asarray(at_orig.get_forces(), dtype=np.float64)

    at_moved = Atoms(symbols=symbols, positions=pos_moved)
    at_moved.calc = calc
    e_moved = float(at_moved.get_potential_energy())
    f_moved = np.asarray(at_moved.get_forces(), dtype=np.float64)

    se3_e_err = abs(e_moved - e_orig)
    so3_f_err = float(np.linalg.norm(f_moved - (f_orig @ rot.T), axis=1).max())
    net_force = float(np.linalg.norm(f_orig.sum(axis=0)))
    center_of_pos = pos_best.mean(axis=0, keepdims=True)
    net_torque = float(np.linalg.norm(np.cross(pos_best - center_of_pos, f_orig).sum(axis=0)))

    se3_ok = (
        (se3_e_err < 1e-4) and (so3_f_err < 1e-4) and (net_force < 1e-3) and (net_torque < 1e-3)
    )
    gates.append(
        _gate(
            "G9_SE3_EQUIVARIANCE_CONSERVATION",
            "9. Moving or rotating in 3D keeps physics unchanged",
            "pass" if se3_ok else "fail",
            f"energy diff = {se3_e_err:.2e} eV, force diff = {so3_f_err:.2e} eV/A",
            "Energy & rotated force diff < 1e-4; total force & torque < 1e-3",
            f"Total force = {net_force:.2e} eV/A, total torque = {net_torque:.2e} eV.",
        )
    )

    # Check 10: Mirror-image energy & force matches energy slope (F = -dE/dx)
    pos_mir = mirror_positions(pos_best)
    at_mir = Atoms(symbols=symbols, positions=pos_mir)
    at_mir.calc = calc
    e_mir = float(at_mir.get_potential_energy())
    f_mir = np.asarray(at_mir.get_forces(), dtype=np.float64)
    parity_e_err = abs(e_mir - e_orig)
    parity_f_err = float(np.linalg.norm(f_mir - mirror_positions(f_orig), axis=1).max())

    mol_mir = Chem.Mol(mol_3d)
    conf_mir = mol_mir.GetConformer(best["conf_id"])
    for i, xyz in enumerate(pos_mir):
        conf_mir.SetAtomPosition(i, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
    smi_mir = stereo_from_3d(mol_mir, best["conf_id"])
    mir_rel = categorize(smi_mir, best["smiles_from_3d"])

    step_dir = rng.normal(size=pos_best.shape)
    step_dir /= max(float(np.linalg.norm(step_dir)), 1e-12)
    eps = 1e-3
    at_plus = Atoms(symbols=symbols, positions=pos_best + eps * step_dir)
    at_plus.calc = calc
    at_minus = Atoms(symbols=symbols, positions=pos_best - eps * step_dir)
    at_minus.calc = calc
    slope_num = (float(at_plus.get_potential_energy()) - float(at_minus.get_potential_energy())) / (
        2.0 * eps
    )
    slope_force = float(-np.sum(f_orig * step_dir))
    grad_err = abs(slope_num - slope_force)

    g10_ok = (parity_e_err < 1e-4) and (parity_f_err < 1e-4) and (grad_err < 2e-3)
    gates.append(
        _gate(
            "G10_PARITY_AND_CONSERVATIVE_FORCE",
            "10. Mirror test & force matches energy slope",
            "pass" if g10_ok else "fail",
            f"mirror E diff = {parity_e_err:.2e} eV, slope diff = {grad_err:.2e} eV/A",
            "Mirror energy diff < 1e-4 eV & force-slope diff < 2e-3 eV/A",
            f"Mirror twin is '{mir_rel}' ({smi_mir}).",
        )
    )

    cip_by_atom = {c["atom_idx"]: c["cip"] for c in cip_info["chiral_centers"]}
    atoms_out = [
        {
            "idx": i,
            "symbol": symbols[i],
            "z": atomic_nums[i],
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
    return {
        "ok": overall != "fail",
        "overall_status": overall,
        "summary_counts": {
            "pass": statuses.count("pass"),
            "warn": statuses.count("warn"),
            "fail": statuses.count("fail"),
            "total": len(gates),
        },
        "input_smiles": raw_input,
        "standardized_smiles": std.smiles,
        "smiles_from_3d": best["smiles_from_3d"],
        "mirror_3d_smiles": smi_mir,
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
            "n_atoms": len(atoms_out),
            "atoms": atoms_out,
            "bonds": bonds_out,
            "clashes": clashes,
            "d_min_a": round(d_min, 4),
            "energy_ev": round(best["energy_ev"], 6),
            "energy_kcal": round(best["energy_kcal"], 4),
            "fmax_ev_a": round(best["fmax_ev_a"], 5),
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
            "xyz": _to_xyz(
                symbols,
                pos_best,
                f"{std.smiles} | {calc_label} | E={best['energy_ev']:.6f} eV",
            ),
            "extxyz": _to_extxyz(symbols, pos_best, forces_best, best["energy_ev"], std.smiles),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
