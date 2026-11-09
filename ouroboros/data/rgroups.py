"""Collapse a pendant substituent into a labelled dummy, preserving the remaining graph."""
from __future__ import annotations

import random
from rdkit import Chem
from ouroboros.labels import LABEL_ISOTOPES, set_draw_labels

# Abbreviation patterns start at the atom attached to the retained core.
ABBREVIATIONS = {
    'Ph': 'c1ccccc1', 'OMe': 'OC', 'OEt': 'OCC', 'CO2Et': 'C(=O)OCC',
    'CO2Me': 'C(=O)OC', 'Boc': 'C(=O)OC(C)(C)C', 'Me': 'C', 'Et': 'CC',
}
GENERIC = ('R', 'R1', 'R2', 'R3', 'R4', 'X')


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
        if any(mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
               or any(x.GetStereo() != Chem.BondStereo.STEREONONE for x in mol.GetAtomWithIdx(i).GetBonds())
               for i in (a, b)):
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
            fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(side), canonical=True)
            for name in labels:
                if name not in LABEL_ISOTOPES:
                    raise ValueError(f'Unsupported label: {name}')
                if name in GENERIC:
                    if len(side) <= mol.GetNumAtoms() - len(side):
                        choices.append((name, root, core, side))
                elif name in ABBREVIATIONS:
                    target = Chem.MolFromSmiles(ABBREVIATIONS[name])
                    if fragment == Chem.MolToSmiles(target):
                        # Root must have the same rooted topology as the abbreviation.
                        rooted = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(side), rootedAtAtom=root)
                        if rooted == Chem.MolToSmiles(target, rootedAtAtom=0):
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

    Replace each alias atom with the first atom of its fragment. Keep the other
    atom indices and neighbor order so their stereochemistry stays the same.
    """
    from ouroboros.labels import parse_labelled, ISOTOPE_LABELS

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
        fragment = Chem.MolFromSmiles(ABBREVIATIONS[label])
        rw.ReplaceAtom(idx, Chem.Atom(fragment.GetAtomWithIdx(0)))
        mapping = [idx]
        for atom_idx in range(1, fragment.GetNumAtoms()):
            atom = fragment.GetAtomWithIdx(atom_idx)
            mapping.append(rw.AddAtom(Chem.Atom(atom)))
        for bond in fragment.GetBonds():
            rw.AddBond(mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()],
                       bond.GetBondType())
    expanded = rw.GetMol()
    try:
        Chem.SanitizeMol(expanded)
        Chem.AssignStereochemistry(expanded, cleanIt=True, force=True)
    except Exception as exc:
        raise ValueError('invalid_expansion') from exc
    return expanded
