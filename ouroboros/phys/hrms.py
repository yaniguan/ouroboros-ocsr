"""High-Resolution Mass Spectrometry (HRMS) exact-mass conservation & adduct likelihoods.

Eliminates non-isomeric constitutional OCSR hallucinations by checking candidate neutral
monoisotopic masses against common ESI/MALDI/EI ionization adducts within a ppm tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from ouroboros.eval.standardize import standardize
from ouroboros.phys.epimers import expand_superatoms

ELECTRON_MASS = 0.000548579909
PROTON_MASS = 1.007276466621

STANDARD_ADDUCTS: dict[str, float] = {
    "[M+H]+": PROTON_MASS,
    "[M+Na]+": 22.989220702,
    "[M+K]+": 38.963157906,
    "[M+NH4]+": 18.033825553,
    "[M-H]-": -PROTON_MASS,
    "[M]+.": -ELECTRON_MASS,
}


@dataclass(frozen=True)
class HRMSMatch:
    valid: bool
    formula: str | None
    exact_neutral_mass: float | None
    best_adduct: str | None
    calcd_mz: float | None
    obs_mz: float | None
    ppm_error: float | None
    within_window: bool
    formula_matches: bool | None
    loss: float


def exact_adduct_mz(molecule_or_mass: float | int | str, adduct: str = "[M+H]+") -> float | None:
    if adduct not in STANDARD_ADDUCTS:
        raise KeyError(f"unsupported adduct {adduct!r}; known: {sorted(STANDARD_ADDUCTS)}")
    if isinstance(molecule_or_mass, str):
        std = standardize(expand_superatoms(molecule_or_mass))
        if not std.valid or std.mol is None:
            return None
        neutral_mass = float(Descriptors.ExactMolWt(Chem.AddHs(std.mol)))
    else:
        neutral_mass = float(molecule_or_mass)
    return neutral_mass + STANDARD_ADDUCTS[adduct]


def score_hrms(
    smiles: str | None,
    obs_mz: float | None,
    *,
    adduct: str | None = None,
    expected_formula: str | None = None,
    window_ppm: float = 5.0,
    outside_penalty: float = 25.0,
) -> HRMSMatch:
    """Evaluate exact-mass adduct match and molecular formula consistency."""
    if window_ppm <= 0:
        raise ValueError("window_ppm must be positive")
    expanded = expand_superatoms(smiles) if smiles else None
    std = standardize(expanded)
    if not std.valid or std.mol is None or any(a.GetAtomicNum() == 0 for a in std.mol.GetAtoms()):
        return HRMSMatch(
            valid=False,
            formula=None,
            exact_neutral_mass=None,
            best_adduct=None,
            calcd_mz=None,
            obs_mz=obs_mz,
            ppm_error=None,
            within_window=False,
            formula_matches=False if expected_formula else None,
            loss=outside_penalty * 2.0,
        )

    mol_h = Chem.AddHs(std.mol)
    formula = rdMolDescriptors.CalcMolFormula(mol_h)
    neutral_mass = float(Descriptors.ExactMolWt(mol_h))
    formula_ok = None if expected_formula is None else (formula == expected_formula.strip())

    if obs_mz is None or obs_mz <= 0:
        f_loss = 0.0 if formula_ok in (None, True) else outside_penalty
        return HRMSMatch(
            valid=True,
            formula=formula,
            exact_neutral_mass=neutral_mass,
            best_adduct=adduct,
            calcd_mz=exact_adduct_mz(neutral_mass, adduct) if adduct in STANDARD_ADDUCTS else None,
            obs_mz=obs_mz,
            ppm_error=None,
            within_window=formula_ok in (None, True),
            formula_matches=formula_ok,
            loss=f_loss,
        )

    adducts = [adduct] if adduct in STANDARD_ADDUCTS else list(STANDARD_ADDUCTS)
    best_a = adducts[0]
    best_calcd = float(exact_adduct_mz(neutral_mass, best_a) or 0.0)
    best_ppm = abs(best_calcd - obs_mz) / obs_mz * 1e6
    for cand_a in adducts[1:]:
        calcd = float(exact_adduct_mz(neutral_mass, cand_a) or 0.0)
        ppm = abs(calcd - obs_mz) / obs_mz * 1e6
        if ppm < best_ppm:
            best_a, best_calcd, best_ppm = cand_a, calcd, ppm

    within = best_ppm <= window_ppm and (formula_ok in (None, True))
    if best_ppm <= window_ppm:
        loss = 0.0
    else:
        excess = min((best_ppm - window_ppm) / window_ppm, 20.0)
        loss = outside_penalty + 0.5 * excess**2
    if formula_ok is False:
        loss = max(loss, outside_penalty)

    return HRMSMatch(
        valid=True,
        formula=formula,
        exact_neutral_mass=neutral_mass,
        best_adduct=best_a,
        calcd_mz=best_calcd,
        obs_mz=obs_mz,
        ppm_error=best_ppm,
        within_window=within,
        formula_matches=formula_ok,
        loss=float(loss),
    )
