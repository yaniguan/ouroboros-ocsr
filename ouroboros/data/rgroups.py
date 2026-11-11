"""Collapse a pendant substituent into a labelled dummy, preserving the remaining graph."""
from __future__ import annotations

import functools
import random

from rdkit import Chem

from ouroboros.labels import LABEL_ISOTOPES, set_draw_labels

# Abbreviation patterns start at the atom attached by a single bond to the retained core.
ABBREVIATIONS = {
    'Ph': 'c1ccccc1',
    'OMe': 'OC',
    'OEt': 'OCC',
    'CO2Et': 'C(=O)OCC',
    'CO2Me': 'C(=O)OC',
    'Boc': 'C(=O)OC(C)(C)C',
    'Me': 'C',
    'Et': 'CC',
    'Pr': 'CCC',
    'iPr': 'C(C)C',
    'Bu': 'CCCC',
    'tBu': 'C(C)(C)C',
    'Cbz': 'C(=O)OCc1ccccc1',
    'Fmoc': 'C(=O)OCC1c2ccccc2-c2ccccc21',
    'Ac': 'C(=O)C',
    'Bn': 'Cc1ccccc1',
    'Bz': 'C(=O)c1ccccc1',
    'Ts': 'S(=O)(=O)c1ccc(C)cc1',
    'Ms': 'S(=O)(=O)C',
    'TBS': '[Si](C)(C)C(C)(C)C',
    'TBDMS': '[Si](C)(C)C(C)(C)C',
    'TMS': '[Si](C)(C)C',
    'PMP': 'c1ccc(OC)cc1',
    'CF3': 'C(F)(F)F',
    'CCl3': 'C(Cl)(Cl)Cl',
    'CO2H': 'C(=O)O',
    'COOH': 'C(=O)O',
    'CHO': 'C=O',
    'OPh': 'Oc1ccccc1',
    'OBn': 'OCc1ccccc1',
    'OAc': 'OC(=O)C',
    'OTs': 'OS(=O)(=O)c1ccc(C)cc1',
    'OMs': 'OS(=O)(=O)C',
    'OTBS': 'O[Si](C)(C)C(C)(C)C',
    'OTMS': 'O[Si](C)(C)C',
    'SMe': 'SC',
    'SEt': 'SCC',
    'SPh': 'Sc1ccccc1',
    'NHBoc': 'NC(=O)OC(C)(C)C',
    'NHAc': 'NC(=O)C',
    'NHCbz': 'NC(=O)OCc1ccccc1',
    'NHFmoc': 'NC(=O)OCC1c2ccccc2-c2ccccc21',
    'NMe2': 'N(C)C',
    'NEt2': 'N(CC)CC',
    'SO2Me': 'S(=O)(=O)C',
    'SO2Ph': 'S(=O)(=O)c1ccccc1',
}
GENERIC = ('R', 'R1', 'R2', 'R3', 'R4', 'X')


@functools.cache
def _target_template(name: str) -> tuple[Chem.Mol, str, str]:
    """Build a dummy-attached template so root valence and H counts match in-molecule use."""
    capped = Chem.MolFromSmiles('*' + ABBREVIATIONS[name])
    if capped is None or capped.GetNumAtoms() < 2:
        raise ValueError(f'Invalid abbreviation definition: {name}')
    atoms = list(range(1, capped.GetNumAtoms()))
    canon = Chem.MolFragmentToSmiles(capped, atomsToUse=atoms, canonical=True)
    rooted = Chem.MolFragmentToSmiles(capped, atomsToUse=atoms, rootedAtAtom=1)
    return capped, canon, rooted


def substitute(mol: Chem.Mol, rng: random.Random, label: str | None = None) -> Chem.Mol:
    """Replace one pendant group, or return a copy if no safe candidate exists.

    Known abbreviations collapse only their exact fragment. Generic labels collapse
    the smaller side of a single non-ring bond; at least two core atoms remain. Avoid
    bonds incident on stereo elements so the retained stereo remains meaningful.
    """
    choices = []
    labels = [label] if label else [*GENERIC, *ABBREVIATIONS]
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if any(
            mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            or any(
                x.GetStereo() != Chem.BondStereo.STEREONONE
                for x in mol.GetAtomWithIdx(i).GetBonds()
            )
            for i in (a, b)
        ):
            continue
        for root, core in ((a, b), (b, a)):
            side, stack = set(), [root]
            while stack:
                i = stack.pop()
                if i in side or i == core:
                    continue
                side.add(i)
                stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors())
            if mol.GetNumAtoms() - len(side) < 2:
                continue
            if any(mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                   or any(b.GetStereo() != Chem.BondStereo.STEREONONE
                          for b in mol.GetAtomWithIdx(i).GetBonds()) for i in side):
                continue
            atoms_sorted = sorted(side)
            fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms_sorted, canonical=True)
            rooted = None
            for name in labels:
                if name not in LABEL_ISOTOPES:
                    raise ValueError(f'Unsupported label: {name}')
                if name in GENERIC:
                    if len(side) <= mol.GetNumAtoms() - len(side):
                        choices.append((name, root, core, side))
                elif name in ABBREVIATIONS:
                    _, target_canon, target_rooted = _target_template(name)
                    if fragment == target_canon:
                        # Root must have the same rooted topology as the abbreviation.
                        if rooted is None:
                            rooted = Chem.MolFragmentToSmiles(
                                mol, atomsToUse=atoms_sorted, rootedAtAtom=root
                            )
                        if rooted == target_rooted:
                            choices.append((name, root, core, side))
    if not choices:
        return Chem.Mol(mol)
    name, root, core, side = rng.choice(choices)
    rw = Chem.RWMol(mol)
    atom = Chem.Atom(0)
    atom.SetIsotope(LABEL_ISOTOPES[name])
    new = rw.AddAtom(atom)
    rw.AddBond(core, new, Chem.BondType.SINGLE)
    for i in sorted(side, reverse=True):
        rw.RemoveAtom(i)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    set_draw_labels(out)
    return out


def expand_aliases(smiles: str) -> Chem.Mol:
    """Expand known aliases with one single bond. Reject generic R groups.

    Replace each alias atom with the attachment atom of its dummy-capped fragment. Keep the
    other atom indices and neighbor order so their stereochemistry stays the same.
    """
    from ouroboros.labels import ISOTOPE_LABELS, parse_labelled

    mol = parse_labelled(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError('invalid_smiles')
    aliases = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = ISOTOPE_LABELS.get(atom.GetIsotope())
        if label not in ABBREVIATIONS:
            raise ValueError(f'unsupported_alias: {label or "*"}')
        if (atom.GetDegree() != 1
                or atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE
                or atom.GetFormalCharge()
                or atom.GetNumExplicitHs()):
            raise ValueError(f'unsupported_attachment: {label}')
        aliases.append((atom.GetIdx(), label))
    rw = Chem.RWMol(mol)
    for idx, label in aliases:
        capped, _, _ = _target_template(label)
        rw.ReplaceAtom(idx, Chem.Atom(capped.GetAtomWithIdx(1)))
        mapping = {1: idx}
        for atom_idx in range(2, capped.GetNumAtoms()):
            atom = capped.GetAtomWithIdx(atom_idx)
            mapping[atom_idx] = rw.AddAtom(Chem.Atom(atom))
        for bond in capped.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u == 0 or v == 0:
                continue
            rw.AddBond(mapping[u], mapping[v], bond.GetBondType())
    expanded = rw.GetMol()
    try:
        Chem.SanitizeMol(expanded)
        Chem.AssignStereochemistry(expanded, cleanIt=True, force=True)
    except Exception as exc:
        raise ValueError('invalid_expansion') from exc
    return expanded
