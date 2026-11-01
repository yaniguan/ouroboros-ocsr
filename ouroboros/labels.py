"""Labelled pseudo-atoms shared by ingestion, scoring and synthetic drawing.

Reserved dummy isotopes are an internal graph representation only. Public SMILES use
bracket labels, which survive canonicalization and can be tokenized as single atoms.
Ar denotes the conventional aryl placeholder here (not elemental argon).
"""
from __future__ import annotations

import re
from rdkit import Chem

# Append only: isotope identities must remain stable across datasets/checkpoints.
LABELS = tuple(dict.fromkeys([
    'R', *[f'R{i}' for i in range(1, 100)], "R'", 'R’', 'X', 'Ar',
    'Ph', 'OMe', 'OEt', 'Me', 'Et', 'Pr', 'iPr', 'Bu', 'tBu', 'CO2Et',
    'CO2Me', 'Boc', 'Cbz', 'Fmoc', 'Ac', 'Bn', 'Bz', 'Ts', 'Ms',
    'TBS', 'TBDMS', 'TMS', 'PMP', 'TAGa', 'G', 'M', 'L', 'PG', 'Y', 'Z',
]))
LABEL_ISOTOPES = {label: 500 + i for i, label in enumerate(LABELS)}
ISOTOPE_LABELS = {v: k for k, v in LABEL_ISOTOPES.items()}
_BRACKET = re.compile(r'\[([^\]]+)\]')
_BARE = re.compile('(?:' + '|'.join(re.escape(x) for x in sorted(LABELS, key=len, reverse=True))
                   + r')(?![a-z0-9])')


def normalize_labels(smiles: str) -> str:
    """Bracket supported bare aliases without changing valid bracket atoms (Ru, Xe, etc.)."""
    head, sep, tail = smiles.partition('|')
    parts = re.split(r'(\[[^\]]+\])', head)
    return ''.join(p if p.startswith('[') else _BARE.sub(lambda m: '[' + m[0] + ']', p)
                   for p in parts) + sep + tail


def parse_labelled(smiles: str) -> Chem.Mol | None:
    s = normalize_labels(smiles)
    s = _BRACKET.sub(lambda m: f'[{LABEL_ISOTOPES[m[1]]}*]'
                     if m[1] in LABEL_ISOTOPES else m[0], s)
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.HasProp('atomLabel'):
                label = atom.GetProp('atomLabel')
                if label not in LABEL_ISOTOPES:
                    return None  # never silently erase an unsupported CX alias
                atom.SetIsotope(LABEL_ISOTOPES[label])
        set_draw_labels(mol)
    return mol


def set_draw_labels(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        label = ISOTOPE_LABELS.get(atom.GetIsotope()) if atom.GetAtomicNum() == 0 else None
        if label:
            atom.SetProp('atomLabel', label)


def labelled_smiles(mol: Chem.Mol) -> str:
    s = Chem.MolToSmiles(mol)
    return re.sub(r'\[(\d+)\*\]', lambda m: '[' + ISOTOPE_LABELS[int(m[1])] + ']'
                  if int(m[1]) in ISOTOPE_LABELS else m[0], s)
