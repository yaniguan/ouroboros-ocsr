"""Superatom protecting-group expansion & uncertainty-gated targeted epimer enumeration.

Expands 37+ organic synthesis abbreviations in-place without permuting bond order around
chiral attachment atoms, and enumerates diastereomeric epimers at ambiguous stereocenters.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence

from rdkit import Chem

from ouroboros.eval.standardize import flat_smiles, standardize
from ouroboros.labels import ISOTOPE_LABELS, parse_labelled

# Each SMILES starts at atom 0 = the attachment atom replacing the single-bonded superatom.
SUPERATOM_FRAGMENTS: dict[str, str] = {
    "Me": "C",
    "Et": "CC",
    "Pr": "CCC",
    "nPr": "CCC",
    "iPr": "C(C)C",
    "Bu": "CCCC",
    "nBu": "CCCC",
    "iBu": "CC(C)C",
    "sBu": "C(C)CC",
    "tBu": "C(C)(C)C",
    "Cy": "C1CCCCC1",
    "Ph": "c1ccccc1",
    "Bn": "Cc1ccccc1",
    "Bz": "C(=O)c1ccccc1",
    "PMP": "c1ccc(OC)cc1",
    "PMB": "Cc1ccc(OC)cc1",
    "OMe": "OC",
    "OEt": "OCC",
    "OiPr": "OC(C)C",
    "OtBu": "OC(C)(C)C",
    "OAc": "OC(=O)C",
    "OBz": "OC(=O)c1ccccc1",
    "OBn": "OCc1ccccc1",
    "OTf": "OS(=O)(=O)C(F)(F)F",
    "OTs": "OS(=O)(=O)c1ccc(C)cc1",
    "OMs": "OS(=O)(=O)C",
    "OTBS": "O[Si](C)(C)C(C)(C)C",
    "Ac": "C(=O)C",
    "Piv": "C(=O)C(C)(C)C",
    "Boc": "C(=O)OC(C)(C)C",
    "Cbz": "C(=O)OCc1ccccc1",
    "Fmoc": "C(=O)OCC1c2ccccc2-c2ccccc21",
    "CO2Me": "C(=O)OC",
    "CO2Et": "C(=O)OCC",
    "CO2tBu": "C(=O)OC(C)(C)C",
    "CO2H": "C(=O)O",
    "Ts": "S(=O)(=O)c1ccc(C)cc1",
    "Ms": "S(=O)(=O)C",
    "Ns": "S(=O)(=O)c1ccc([N+](=O)[O-])cc1",
    "Tf": "S(=O)(=O)C(F)(F)F",
    "TMS": "[Si](C)(C)C",
    "TES": "[Si](CC)(CC)CC",
    "TBS": "[Si](C)(C)C(C)(C)C",
    "TBDMS": "[Si](C)(C)C(C)(C)C",
    "TIPS": "[Si](C(C)C)(C(C)C)C(C)C",
    "TBDPS": "[Si](c1ccccc1)(c1ccccc1)C(C)(C)C",
    "MOM": "COC",
    "MEM": "COCCOC",
    "THP": "C1CCCCO1",
    "Pin": "B1OC(C)(C)C(C)(C)O1",
    "Bpin": "B1OC(C)(C)C(C)(C)O1",
    "BPin": "B1OC(C)(C)C(C)(C)O1",
    "CF3": "C(F)(F)F",
    "SF5": "S(F)(F)(F)(F)F",
    "NO2": "[N+](=O)[O-]",
    "CN": "C#N",
    "NHBoc": "NC(=O)OC(C)(C)C",
    "NBoc": "NC(=O)OC(C)(C)C",
    "NMe2": "N(C)C",
    "NEt2": "N(CC)CC",
    "COOMe": "C(=O)OC",
    "COOEt": "C(=O)OCC",
    "SO2Me": "S(=O)(=O)C",
}

_SORTED_ALIASES = [re.escape(k) for k in sorted(SUPERATOM_FRAGMENTS, key=len, reverse=True)]
_CUSTOM_BRACKET = re.compile(r"\[(" + "|".join(_SORTED_ALIASES) + r")\]")


def _expand_dummy_in_place(mol: Chem.Mol, dummy_idx: int, frag_smiles: str) -> Chem.Mol:
    """Mutate ``dummy_idx`` into atom 0 of ``frag_smiles`` so parent bond order never changes."""
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None or frag.GetNumAtoms() == 0:
        raise ValueError(f"invalid superatom fragment {frag_smiles!r}")
    Chem.Kekulize(frag, clearAromaticFlags=True)

    rw = Chem.RWMol(mol)
    root = frag.GetAtomWithIdx(0)
    target = rw.GetAtomWithIdx(dummy_idx)
    target.SetAtomicNum(root.GetAtomicNum())
    target.SetIsotope(0)
    target.SetFormalCharge(root.GetFormalCharge())
    target.SetChiralTag(root.GetChiralTag())
    target.SetNoImplicit(False)
    target.SetNumExplicitHs(root.GetNumExplicitHs())
    target.SetIsAromatic(False)
    if target.HasProp("atomLabel"):
        target.ClearProp("atomLabel")

    idx_map = {0: dummy_idx}
    for i in range(1, frag.GetNumAtoms()):
        src = frag.GetAtomWithIdx(i)
        new_atom = Chem.Atom(src.GetAtomicNum())
        new_atom.SetFormalCharge(src.GetFormalCharge())
        new_atom.SetChiralTag(src.GetChiralTag())
        new_atom.SetNumExplicitHs(src.GetNumExplicitHs())
        idx_map[i] = rw.AddAtom(new_atom)

    for bond in frag.GetBonds():
        u = idx_map[bond.GetBeginAtomIdx()]
        v = idx_map[bond.GetEndAtomIdx()]
        rw.AddBond(u, v, bond.GetBondType())

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def expand_superatoms(smiles: str | None) -> str | None:
    """Expand supported bracket/labelled superatom abbreviations into full atomic SMILES."""
    if smiles is None or not smiles.strip():
        return smiles
    s = smiles.strip()
    local_tags: dict[int, str] = {}
    next_iso = 900

    def _sub(match: re.Match[str]) -> str:
        nonlocal next_iso
        alias = match.group(1)
        iso = next_iso
        next_iso += 1
        local_tags[iso] = alias
        return f"[{iso}*]"

    tagged = _CUSTOM_BRACKET.sub(_sub, s)
    mol = parse_labelled(tagged)
    if mol is None:
        return s

    while True:
        found: tuple[int, str] | None = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetDegree() == 1:
                iso = atom.GetIsotope()
                alias = local_tags.get(iso) or ISOTOPE_LABELS.get(iso)
                if alias in SUPERATOM_FRAGMENTS:
                    found = (atom.GetIdx(), SUPERATOM_FRAGMENTS[alias])
                    break
        if found is None:
            break
        mol = _expand_dummy_in_place(mol, found[0], found[1])

    return Chem.MolToSmiles(mol, isomericSmiles=True)


def condense_superatom_for_depiction(smiles: str, max_replacements: int = 1) -> tuple[str, bool]:
    """Replace common protecting groups (Boc/TBS/Bn/OMe/Ac) with bracket aliases for SI views."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False
    patterns = (
        ("Boc", "C(=O)OC(C)(C)C"),
        ("Cbz", "C(=O)OCc1ccccc1"),
        ("Bz", "C(=O)c1ccccc1"),
        ("Ts", "S(=O)(=O)c1ccc(C)cc1"),
        ("Ac", "C(=O)C"),
        ("tBu", "C(C)(C)C"),
        ("Ph", "c1ccccc1"),
    )
    replaced = 0
    cur = mol
    for label, smarts in patterns:
        if replaced >= max_replacements:
            break
        q = Chem.MolFromSmiles(smarts)
        if q is None:
            continue
        for match in cur.GetSubstructMatches(q):
            match_set = set(match)
            root = match[0]
            ext_neighbors = [
                nb.GetIdx()
                for nb in cur.GetAtomWithIdx(root).GetNeighbors()
                if nb.GetIdx() not in match_set
            ]
            internal_ok = all(
                all(nb.GetIdx() in match_set for nb in cur.GetAtomWithIdx(idx).GetNeighbors())
                for idx in match[1:]
            )
            if len(ext_neighbors) == 1 and internal_ok:
                parent_atom = cur.GetAtomWithIdx(ext_neighbors[0])
                if parent_atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                    continue
                rw = Chem.RWMol(cur)
                r_atom = rw.GetAtomWithIdx(root)
                r_atom.SetAtomicNum(0)
                r_atom.SetIsotope(0)
                r_atom.SetFormalCharge(0)
                r_atom.SetIsAromatic(False)
                r_atom.SetProp("atomLabel", label)
                for rm_idx in sorted(match[1:], reverse=True):
                    rw.RemoveAtom(rm_idx)
                cand = rw.GetMol()
                try:
                    Chem.SanitizeMol(cand)
                except Exception:  # noqa: BLE001
                    break
                s_cand = re.sub(r"\*", f"[{label}]", Chem.MolToSmiles(cand), count=1)
                cur = cand
                replaced += 1
                if replaced >= max_replacements:
                    return s_cand, True
                break
    return smiles, False


def _is_even_perm(seq_a: list[int], seq_b: list[int]) -> bool:
    pos = {val: idx for idx, val in enumerate(seq_b)}
    arr = [pos[x] for x in seq_a]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return (inv % 2) == 0


def stereo_hamming_distance(smiles_a: str, smiles_b: str) -> int:
    """Minimum number of inverted 3D tetrahedral centers between two same-connectivity isomers."""
    sa, sb = standardize(expand_superatoms(smiles_a)), standardize(expand_superatoms(smiles_b))
    if not sa.valid or not sb.valid or sa.mol is None or sb.mol is None:
        return 99
    if sa.smiles == sb.smiles:
        return 0
    matches = sb.mol.GetSubstructMatches(sa.mol, useChirality=False, maxMatches=64)
    if not matches:
        return 99
    tet = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    best = 99
    for match in matches:
        flips = 0
        for i, j in enumerate(match):
            ai = sa.mol.GetAtomWithIdx(i)
            aj = sb.mol.GetAtomWithIdx(j)
            ti, tj = ai.GetChiralTag(), aj.GetChiralTag()
            if ti not in tet and tj not in tet:
                continue
            if (ti in tet) != (tj in tet):
                flips += 1
                continue
            mapped_nbrs = [match[nb.GetIdx()] for nb in ai.GetNeighbors()]
            target_nbrs = [nb.GetIdx() for nb in aj.GetNeighbors()]
            same_cfg = (ti == tj) == _is_even_perm(mapped_nbrs, target_nbrs)
            if not same_cfg:
                flips += 1
        if flips < best:
            best = flips
    return best


def enumerate_targeted_epimers(
    candidates: Sequence[str],
    *,
    max_centers: int = 3,
    max_epimers: int = 8,
) -> list[str]:
    """Enumerate stereoisomers at disputed, unassigned, or congested ring-junction centers."""
    if max_centers < 1 or max_epimers < 1:
        return []
    std_cands = []
    for raw in candidates:
        st = standardize(expand_superatoms(raw))
        if st.valid and st.mol is not None and st.smiles is not None:
            std_cands.append(st)
    if not std_cands:
        return []

    anchor = std_cands[0]
    anchor_flat = flat_smiles(anchor)
    mol = Chem.Mol(anchor.mol)

    # Map canonical flat atom ranks to detect stereocenters where visual candidates disagree.
    disputed: set[int] = set()
    same_const = [st for st in std_cands[1:] if flat_smiles(st) == anchor_flat]
    if same_const:
        base_matches = mol.GetSubstructMatches(mol, useChirality=False)
        if base_matches:
            for other in same_const:
                m_idx = other.mol.GetSubstructMatch(mol, useChirality=False)
                if len(m_idx) == mol.GetNumAtoms():
                    for i, j in enumerate(m_idx):
                        a_tag = mol.GetAtomWithIdx(i).GetChiralTag()
                        b_tag = other.mol.GetAtomWithIdx(j).GetChiralTag()
                        if (a_tag != b_tag) and (
                            a_tag != Chem.ChiralType.CHI_UNSPECIFIED
                            or b_tag != Chem.ChiralType.CHI_UNSPECIFIED
                        ):
                            disputed.add(i)

    ring_info = mol.GetRingInfo()
    scored_centers: list[tuple[tuple[int, int, int, int], int]] = []
    for info in Chem.FindPotentialStereo(mol):
        if info.type != Chem.StereoType.Atom_Tetrahedral:
            continue
        idx = int(info.centeredOn)
        atom = mol.GetAtomWithIdx(idx)
        is_disputed = 1 if idx in disputed else 0
        is_unassigned = 1 if info.specified == Chem.StereoSpecified.Unspecified else 0
        n_rings = ring_info.NumAtomRings(idx)
        priority = (-is_disputed, -is_unassigned, -n_rings, -atom.GetDegree(), idx)
        scored_centers.append((priority, idx))

    scored_centers.sort()
    chosen = [idx for _, idx in scored_centers[:max_centers]]
    if not chosen:
        return []

    seen = {st.smiles for st in std_cands}
    generated: list[str] = []
    tags = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    for combo in itertools.product(tags, repeat=len(chosen)):
        trial = Chem.Mol(mol)
        for idx, tag in zip(chosen, combo, strict=True):
            trial.GetAtomWithIdx(idx).SetChiralTag(tag)
        Chem.AssignStereochemistry(trial, force=True, cleanIt=True)
        smi = Chem.MolToSmiles(trial, isomericSmiles=True)
        st = standardize(smi)
        if st.valid and st.smiles and st.smiles not in seen:
            seen.add(st.smiles)
            generated.append(st.smiles)
            if len(generated) >= max_epimers:
                break
    return generated
