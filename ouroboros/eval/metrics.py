"""Recognition metrics. Every metric standardizes BOTH prediction and reference first
(``ouroboros.eval.standardize``), so the two are always compared under the same rules.

Per-sample metrics (``score_pair``):
  valid             prediction parses in RDKit (non-empty)
  exact             canonical SMILES equal, full stereochemistry (the headline metric)
  exact_flat        canonical SMILES equal after removing all stereo (constitution correct)
  inchikey          standard InChIKeys equal (stereo-preserving identity cross-check)
  tanimoto          Morgan r=2, 2048-bit Tanimoto (0 for invalid predictions)
  stereo_total      # defined stereo elements (tetrahedral + E/Z) in the reference
  stereo_correct    # of them reproduced by the prediction (0 if constitution wrong)

Aggregate (``aggregate``): means of the above plus
  invalid_rate                 = 1 - mean(valid)
  stereo_elem_acc_strict       = sum(stereo_correct) / sum(stereo_total)
  stereo_elem_acc_given_flat   = same, restricted to constitution-correct predictions
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdCIPLabeler, rdFingerprintGenerator

from ouroboros.eval.standardize import Std, flat_smiles, standard_inchikey, standardize

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass
class PairScore:
    valid: bool
    exact: bool
    exact_flat: bool
    inchikey: bool
    tanimoto: float
    stereo_total: int
    stereo_correct: int


def _stereo_elements(mol: Chem.Mol) -> tuple[list[int], list[int]]:
    """Indices of defined tetrahedral atoms and defined E/Z double bonds."""
    atoms, bonds = [], []
    for info in Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True, flagPossible=False):
        if info.specified != Chem.StereoSpecified.Specified:
            continue
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            atoms.append(info.centeredOn)
        elif info.type == Chem.StereoType.Bond_Double:
            bonds.append(info.centeredOn)
    return atoms, bonds


def _cip(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(m)
    return m


def _code(obj) -> str | None:
    return obj.GetProp("_CIPCode") if obj.HasProp("_CIPCode") else None


def _perm_parity(seq: list[int]) -> int:
    inv = sum(1 for i, j in itertools.combinations(range(len(seq)), 2) if seq[i] > seq[j])
    return inv % 2


_CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
_CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW


def _tet_same(r: Chem.Mol, p: Chem.Mol, a: int, match: tuple[int, ...]) -> bool:
    """Same configuration at ref atom ``a`` and pred atom ``match[a]``, by local parity.

    RDKit chiral tags are relative to the atom's bond order; we express the pred tag in the
    ref neighbour order (mapped through ``match``) and compare. Implicit H is handled the same
    way on both sides because the constitutions are identical.
    """
    ra, pa = r.GetAtomWithIdx(a), p.GetAtomWithIdx(match[a])
    if ra.GetChiralTag() not in (_CW, _CCW) or pa.GetChiralTag() not in (_CW, _CCW):
        return False
    ref_nbrs = [match[b.GetOtherAtomIdx(a)] for b in ra.GetBonds()]
    pred_nbrs = [b.GetOtherAtomIdx(match[a]) for b in pa.GetBonds()]
    if sorted(ref_nbrs) != sorted(pred_nbrs):
        return False
    flip = _perm_parity([pred_nbrs.index(x) for x in ref_nbrs])
    ptag = pa.GetChiralTag() if not flip else (_CW if pa.GetChiralTag() == _CCW else _CCW)
    return ptag == ra.GetChiralTag()


_CIS_TRANS = {Chem.BondStereo.STEREOCIS: 0, Chem.BondStereo.STEREOTRANS: 1}


def _db_relation(mol: Chem.Mol, bond: Chem.Bond, x: int, y: int) -> int | None:
    """0 = cis, 1 = trans between neighbour x (of one double-bond atom) and y (of the other)."""
    st = bond.GetStereo()
    if st not in _CIS_TRANS:
        return None
    su, sv = list(bond.GetStereoAtoms())
    u = bond.GetBeginAtomIdx()
    if mol.GetBondBetweenAtoms(x, u) is None:  # x hangs off v: swap roles
        x, y = y, x
    rel = _CIS_TRANS[st]
    rel ^= int(x != su)  # the other substituent on the same atom flips cis <-> trans
    rel ^= int(y != sv)
    return rel


def _db_same(r: Chem.Mol, p: Chem.Mol, b: int, match: tuple[int, ...]) -> bool:
    rb = r.GetBondWithIdx(b)
    pb = p.GetBondBetweenAtoms(match[rb.GetBeginAtomIdx()], match[rb.GetEndAtomIdx()])
    if pb is None:
        return False
    if rb.GetStereo() in _CIS_TRANS and pb.GetStereo() in _CIS_TRANS:
        x, y = rb.GetStereoAtoms()
        return _db_relation(p, pb, match[x], match[y]) == _CIS_TRANS[rb.GetStereo()]
    # E/Z-typed bonds (not produced by the SMILES parser, but possible after processing):
    # fall back to CIP labels, which are available on the CIP-annotated copies.
    return _code(rb) is not None and _code(rb) == _code(pb)


def stereo_element_counts(ref: Std, pred: Std) -> tuple[int, int]:
    """(total defined stereo elements in ref, number reproduced by pred).

    Tetrahedral centres and E/Z bonds are compared locally (chiral-tag parity / cis-trans
    relation of mapped neighbours), which also covers ring cis/trans (pseudo-asymmetric)
    centres. The atom correspondence comes from a stereo-agnostic substructure match between
    the two constitutionally identical molecules; for symmetric molecules every match is tried
    and the best one is used. If the constitution is wrong (or pred invalid), no element counts
    as reproduced.
    """
    if not ref.valid:
        return 0, 0
    ref_atoms, ref_bonds = _stereo_elements(ref.mol)
    total = len(ref_atoms) + len(ref_bonds)
    if total == 0 or not pred.valid or flat_smiles(ref) != flat_smiles(pred):
        return total, 0
    r, p = _cip(ref.mol), _cip(pred.mol)
    matches = p.GetSubstructMatches(r, uniquify=False, useChirality=False, maxMatches=1000)
    best = 0
    for match in matches:  # match[i] = index in p of atom i of r
        n = sum(_tet_same(r, p, a, match) for a in ref_atoms)
        n += sum(_db_same(r, p, b, match) for b in ref_bonds)
        best = max(best, n)
        if best == total:
            break
    return total, best


def tanimoto(a: Std, b: Std) -> float:
    if not (a.valid and b.valid):
        return 0.0
    fa, fb = _MORGAN.GetFingerprint(a.mol), _MORGAN.GetFingerprint(b.mol)
    return DataStructs.TanimotoSimilarity(fa, fb)


def score_pair(pred: str | None, ref: str) -> PairScore:
    p, r = standardize(pred), standardize(ref)
    if not r.valid:
        raise ValueError(f"reference does not standardize ({r.reason}): {ref!r}")
    total, correct = stereo_element_counts(r, p)
    return PairScore(
        valid=p.valid,
        exact=p.valid and p.smiles == r.smiles,
        exact_flat=p.valid and flat_smiles(p) == flat_smiles(r),
        inchikey=p.valid and standard_inchikey(p) == standard_inchikey(r),
        tanimoto=tanimoto(p, r),
        stereo_total=total,
        stereo_correct=correct,
    )


def aggregate(scores: Sequence[PairScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    arr = {k: np.array([getattr(s, k) for s in scores], dtype=float) for k in asdict(scores[0])}
    tot, cor, flat = arr["stereo_total"], arr["stereo_correct"], arr["exact_flat"].astype(bool)
    return {
        "n": n,
        "exact": arr["exact"].mean(),
        "exact_flat": arr["exact_flat"].mean(),
        "inchikey": arr["inchikey"].mean(),
        "tanimoto": arr["tanimoto"].mean(),
        "valid": arr["valid"].mean(),
        "invalid_rate": 1.0 - arr["valid"].mean(),
        "n_stereo_elements": int(tot.sum()),
        "stereo_elem_acc_strict": float(cor.sum() / tot.sum()) if tot.sum() else float("nan"),
        "stereo_elem_acc_given_flat": (
            float(cor[flat].sum() / tot[flat].sum()) if tot[flat].sum() else float("nan")
        ),
        "n_with_stereo": int((tot > 0).sum()),
        "exact_on_stereo_subset": (
            float(arr["exact"][tot > 0].mean()) if (tot > 0).any() else float("nan")
        ),
    }


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI of a per-sample metric."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


__all__ = [
    "PairScore",
    "score_pair",
    "aggregate",
    "bootstrap_ci",
    "stereo_element_counts",
    "tanimoto",
]
