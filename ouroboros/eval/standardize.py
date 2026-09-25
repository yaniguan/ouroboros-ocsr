"""Standardization used for ALL scoring (predictions and references alike).

Intended to match the protocol of arXiv:2608.09100 ("Real Data Closes Synthetic-to-Real Gap in
OCSR"). Verification status of each rule (the full text could not be read from the development
sandbox; the status below comes from search-engine snippets of the paper):

====================================================================  ==========================
Rule                                                                   Status vs. paper
====================================================================  ==========================
Parse with RDKit; canonicalize with RDKit                              confirmed (snippet)
Salts/solvates reduced to the neutral largest fragment                 confirmed (snippet)
  implemented as ``rdMolStandardize.ChargeParent`` (RDKit: "the        exact RDKit call UNVERIFIED
  uncharged version of the largest fragment"; includes RDKit Cleanup,
  i.e. functional-group normalization)
Exact match = equality of canonical SMILES with full stereochemistry   confirmed (snippet)
Identity cross-check via stereo-preserving (standard) InChIKey         confirmed (snippet)
Validity = fraction of outputs RDKit can parse                         confirmed (snippet)
Empty / zero-atom output counts as invalid                             UNVERIFIED (our choice)
No tautomer canonicalization                                           UNVERIFIED (our choice)
====================================================================  ==========================

Per-sample parity with the paper's scoring code is pending (TASK.md, Am1-C / U3). Until it is
established every results table must carry the footnote in ``PARITY_FOOTNOTE``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

PARITY_VERIFIED = False
PARITY_FOOTNOTE = "Scoring parity with arXiv:2608.09100 is unverified (see TASK.md Am1-C)."


@dataclass(frozen=True)
class Std:
    """A standardized molecule, or the reason it could not be produced."""

    mol: Chem.Mol | None
    smiles: str | None  # canonical isomeric SMILES of the standardized molecule
    reason: str  # "ok" | "empty" | "unparsable" | "standardize_failed"

    @property
    def valid(self) -> bool:
        return self.mol is not None


def parse(smiles: str | None) -> Chem.Mol | None:
    """RDKit parse; empty strings and zero-atom molecules count as unparsable."""
    if smiles is None or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def standardize(smiles: str | None) -> Std:
    if smiles is None or not smiles.strip():
        return Std(None, None, "empty")
    mol = parse(smiles)
    if mol is None:
        return Std(None, None, "unparsable")
    try:
        parent = rdMolStandardize.ChargeParent(mol)
        Chem.SanitizeMol(parent)
    except Exception:  # noqa: BLE001 - RDKit raises assorted errors
        return Std(None, None, "standardize_failed")
    if parent is None or parent.GetNumAtoms() == 0:
        return Std(None, None, "standardize_failed")
    return Std(parent, Chem.MolToSmiles(parent), "ok")


def standard_inchikey(std: Std) -> str | None:
    """Standard InChIKey of a standardized molecule (keeps tetrahedral and E/Z stereo layers)."""
    if not std.valid:
        return None
    key = Chem.MolToInchiKey(std.mol)
    return key or None


def flat_smiles(std: Std) -> str | None:
    """Canonical SMILES with all stereo removed (constitution only)."""
    if not std.valid:
        return None
    m = Chem.Mol(std.mol)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)
