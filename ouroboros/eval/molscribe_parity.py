"""MolScribe / MolNexTR exact-match scoring protocol for external SOTA comparison.

Reference protocol (Qian et al., JCIM 2023, arXiv:2205.14311; Chen et al., J. Cheminf. 2024,
arXiv:2403.03691 Table 2):
  1. Expand standard monovalent chemical abbreviations (``[Ph]``, ``[OMe]``, ``[Boc]``, ``[Me]``,
     ``[Et]``, ``[CO2Et]``, ...) into explicit molecular graphs.
  2. Collapse generic R-groups, aryl/halide placeholders, and isotope-tagged attachment points
     (``[R]``, ``[R1]``, ``[Ar]``, ``[X]``, ``[1*]``, ...) into unlabelled dummy atoms ``*``.
  3. Keep largest fragment and neutralize salt counterions (``ChargeParent`` when dummy-free).
  4. **Tetrahedral chirality (``@`` / ``@@``) is kept; double-bond ``E/Z`` (``/`` / ``\\``) is
     stripped** before canonicalizing SMILES with RDKit.

Ouroboros's primary ``ouroboros.eval.standardize`` protocol is **stricter** on two axes:
it requires exact ``E/Z`` double-bond geometry AND requires exact R-group label identity
(``[R1]`` != ``[R2]`` != ``*``). Reporting ``--protocol both`` removes scoring ambiguity when
comparing against published MolScribe / MolNexTR tables.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from ouroboros.data.rgroups import ABBREVIATIONS, _target_template
from ouroboros.eval.metrics import bootstrap_ci
from ouroboros.labels import ISOTOPE_LABELS, parse_labelled

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_GENERIC_BRACKET = re.compile(r"\[(?:R\d*|R'|R’|Ar|X|Y|Z|G|M|L|PG|\d*\*)\]")


@dataclass(frozen=True)
class MolScribeStd:
    mol: Chem.Mol | None
    smiles: str | None
    flat_smiles: str | None
    reason: str

    @property
    def valid(self) -> bool:
        return self.mol is not None and self.smiles is not None


@dataclass
class MolScribePairScore:
    valid: bool
    exact: bool
    exact_flat: bool
    tanimoto: float


def _parse_for_molscribe(smiles: str) -> Chem.Mol | None:
    s = smiles.strip()
    if not s:
        return None
    mol = parse_labelled(s)
    if mol is not None and mol.GetNumAtoms() > 0:
        return mol
    # Fallback: collapse bracketed generic R-groups directly to '*' and parse with RDKit
    collapsed = _GENERIC_BRACKET.sub("*", s)
    mol = Chem.MolFromSmiles(collapsed)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def _expand_known_abbreviations(mol: Chem.Mol) -> Chem.Mol:
    aliases: list[tuple[int, str]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = ISOTOPE_LABELS.get(atom.GetIsotope())
        if label is None and atom.HasProp("atomLabel"):
            label = atom.GetProp("atomLabel")
        if (
            label in ABBREVIATIONS
            and atom.GetDegree() == 1
            and atom.GetBonds()[0].GetBondType() == Chem.BondType.SINGLE
            and atom.GetFormalCharge() == 0
            and atom.GetNumExplicitHs() == 0
        ):
            aliases.append((atom.GetIdx(), label))
    if not aliases:
        return Chem.Mol(mol)

    rw = Chem.RWMol(mol)
    for idx, label in aliases:
        capped, _, _ = _target_template(label)
        rw.ReplaceAtom(idx, Chem.Atom(capped.GetAtomWithIdx(1)))
        mapping = {1: idx}
        for atom_idx in range(2, capped.GetNumAtoms()):
            mapping[atom_idx] = rw.AddAtom(Chem.Atom(capped.GetAtomWithIdx(atom_idx)))
        for bond in capped.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u == 0 or v == 0:
                continue
            rw.AddBond(mapping[u], mapping[v], bond.GetBondType())
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _strip_ez_keep_tetrahedral(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    for bond in out.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    for atom in out.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
            if atom.HasProp("atomLabel"):
                atom.ClearProp("atomLabel")
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    for bond in out.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    return out


def standardize_molscribe(smiles: str | None) -> MolScribeStd:
    """Standardize ``smiles`` under the MolScribe evaluation protocol."""
    if smiles is None or not smiles.strip():
        return MolScribeStd(None, None, None, "empty")
    mol = _parse_for_molscribe(smiles)
    if mol is None:
        return MolScribeStd(None, None, None, "unparsable")
    try:
        mol = _expand_known_abbreviations(mol)
        has_dummy = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
        if not has_dummy:
            mol = rdMolStandardize.ChargeParent(mol)
        else:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            mol = max(frags, key=lambda m: m.GetNumAtoms())
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        mol = _strip_ez_keep_tetrahedral(mol)
    except Exception:  # noqa: BLE001
        return MolScribeStd(None, None, None, "standardize_failed")
    if mol is None or mol.GetNumAtoms() == 0:
        return MolScribeStd(None, None, None, "standardize_failed")

    iso_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    flat_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat_mol)
    flat_smi = Chem.MolToSmiles(flat_mol, isomericSmiles=False, canonical=True)
    return MolScribeStd(mol, iso_smi, flat_smi, "ok")


def score_molscribe_pair(pred: str | None, ref: str) -> MolScribePairScore:
    p = standardize_molscribe(pred)
    r = standardize_molscribe(ref)
    if not r.valid or r.mol is None:
        raise ValueError(f"reference does not standardize under MolScribe protocol: {ref!r}")
    tan = 0.0
    if p.valid and p.mol is not None:
        fp_p = _MORGAN.GetFingerprint(p.mol)
        fp_r = _MORGAN.GetFingerprint(r.mol)
        tan = float(DataStructs.TanimotoSimilarity(fp_p, fp_r))
    return MolScribePairScore(
        valid=p.valid,
        exact=bool(p.valid and p.smiles == r.smiles),
        exact_flat=bool(p.valid and p.flat_smiles == r.flat_smiles),
        tanimoto=tan,
    )


def aggregate_molscribe(
    scores: Sequence[MolScribePairScore], n_boot: int = 1000, seed: int = 0
) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    exacts = [float(s.exact) for s in scores]
    mean_ex, lo, hi = bootstrap_ci(exacts, n_resamples=n_boot, seed=seed)
    return {
        "n": n,
        "exact": mean_ex,
        "exact_ci95": [lo, hi],
        "exact_flat": sum(float(s.exact_flat) for s in scores) / n,
        "valid": sum(float(s.valid) for s in scores) / n,
        "tanimoto": sum(s.tanimoto for s in scores) / n,
    }


__all__ = [
    "MolScribePairScore",
    "MolScribeStd",
    "aggregate_molscribe",
    "score_molscribe_pair",
    "standardize_molscribe",
]
