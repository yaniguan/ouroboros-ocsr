"""Deterministic zero-leakage ChemVeritas-600 catalog across 5 hard chemical strata.

Generates 600 unique connectivity scaffolds (120 per stratum; or 25 in ``smoke`` preset)
covering bridged/caged natural-product cores, dense contiguous stereocenters, strained
ring junctions, synthetic protecting-group superatoms, and post-2020 3D bioisosteres.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ouroboros.eval.standardize import flat_smiles, leakage_key, standardize

STRATA = (
    "S1_bridged_caged",
    "S2_dense_stereo",
    "S3_strained_bredt",
    "S4_superatom_si",
    "S5_medchem_bioisosteres",
)


@dataclass(frozen=True)
class CatalogEntry:
    mol_id: str
    stratum: str
    smiles: str
    flat_smiles: str
    inchikey: str
    inchikey14: str
    formula: str
    n_heavy_atoms: int
    n_stereocenters: int


_R_TAILS = (
    "C",
    "CC",
    "C(C)C",
    "C1CC1",
    "CF",
    "C(F)(F)F",
    "OC",
    "OCC",
    "C(=O)OC",
    "C(=O)N(C)C",
    "c1ccccc1",
    "c1ccc(F)cc1",
    "c1ccc(Cl)cc1",
    "c1ccc(OC)cc1",
    "c1ccncc1",
    "C#N",
    "CC#N",
    "S(=O)(=O)C",
    "C1CCC1",
    "Cc1ccccc1",
)

_S1_TEMPLATES = (
    # Bridged bicyclo[2.2.1]heptane (borneol), bicyclo[2.2.2]octane, tropane, quinuclidine
    "C[C@@]12CC[C@@H](C1)[C@H](O{R})[C@@H]2{Q}",
    "O=C1[C@@H]2CC[C@H](C2)[C@H]1{R}",
    "CN1[C@H]2CC[C@@H]1C[C@H](O{R})C2",
    "O[C@H]1[C@@H]2CC[C@@H](CC2)[C@@H]1{R}",
    "N12CC[C@@H](CC1)[C@H](O)[C@@H]2{R}",
    "C1[C@@H]2CC[C@H]1[C@@H](NC(=O){R})[C@H]2O",
    "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](O)[C@@H]2{R}",
    "O=C(O{Q})[C@H]1[C@@H]2CC[C@H](O2)[C@@H]1{R}",
)

_S2_TEMPLATES = (
    # Dense >=3 stereocenters: polyketide triols, pyranoses, decalins, spiro/alkenyl chains
    "C[C@H](O)[C@@H](C)[C@H](O)[C@@H](OC)C(=O){R}",
    "O[C@H]1[C@@H](O)[C@H](CO)[O][C@@H](O{R})[C@@H]1F",
    "C[C@H]1CC[C@@H]2[C@@H](CC[C@H](O)[C@]12C){R}",
    "C/C=C/[C@H](O)[C@@H](C)[C@H](NC(=O){R})OC",
    "O=C1O[C@H]({R})[C@@H](C)[C@H](O)[C@@H]1C",
    "C[C@@H]1O[C@]2(CC[C@H]({R})O2)C[C@H]1O",
    "N([C@H]1[C@@H](O)[C@H](F)C[C@@H]1{R})C(=O)C",
    "CO[C@H]1[C@H](C)C[C@@H]({R})[C@H](O)[C@@H]1C",
)

_S3_TEMPLATES = (
    # Strained fused/bridged small rings: bicyclo[3.2.0]/[3.1.0]/[4.1.0] & fused epoxides
    "O=C1C[C@@H]2[C@H]1[C@@H](O)[C@H]2{R}",
    "C[C@]12C[C@H]1[C@@H](O)C[C@H]2{R}",
    "O1[C@@H]2[C@H]1CC[C@@H]({R})[C@H]2O",
    "C1[C@@H]2[C@H]1[C@H](NC(=O){R})C[C@@H]2F",
    "CC1(C)[C@@H]2[C@H]1[C@@H]({R})CC(=O)C2",
    "O1C[C@]2(C1)C[C@H](O)[C@@H]({R})C2",
    "N1(C(=O)OC)[C@@H]2[C@H]1CCC[C@@H]2{R}",
    "F[C@]12C[C@@H]1[C@H](O)[C@@H]({R})C2",
)

_S4_PROTECTING_GROUPS = (
    "C(=O)OC(C)(C)C",  # Boc
    "C(=O)OCc1ccccc1",  # Cbz
    "C(=O)c1ccccc1",  # Bz
    "S(=O)(=O)c1ccc(C)cc1",  # Ts
    "C(=O)C(C)(C)C",  # Piv
    "C(=O)C",  # Ac
)

_S4_SCAFFOLDS = (
    "O[C@H]1CCN({PG})C[C@@H]1{R}",
    "C[C@H](N{PG})[C@@H](O)C(=O)O{R}",
    "O=C(O{R})[C@@H]1C[C@H](O)CN1{PG}",
    "COc1ccc(C[C@H](N{PG})[C@@H](O){R})cc1",
    "F[C@@H]1C[C@H](N{PG})[C@@H](C{R})C1",
)

_S5_BIOISOSTERE_TEMPLATES = (
    # BCP [1.1.1], BCH [2.1.1] (Nature 2022), 2-oxa-BCH (Nat Chem 2023), BCHep, cubane, spiro, SF5
    "O=C(N[C@H](C){R})C12CC(C1)(C2)c1ccc(F)cc1",  # BCP
    "O=C(O{Q})C12CC(C1)([C@H]2{R})C#N",  # BCH [2.1.1]
    "O=C(N{Q})C12COC(C1)({R})C2",  # 2-oxabicyclo[2.1.1]hexane
    "FC1(F)CC2(CC(C1)(C2){R})C(=O)OC",  # difluoro-BCHep [3.1.1]
    "O=C(N[C@@H](C){R})C12C3C4C1C5C2C3C45c1ccccc1",  # 1,4-cubane
    "O=C({R})N1CC2(C1)COC2",  # 2-oxa-6-azaspiro[3.3]heptane
    "FS(F)(F)(F)(F)c1ccc(C(=O)N[C@H](C)[C@@H](O){R})cc1",  # octahedral SF5
    "FC1(F)C[C@H]1C(=O)N[C@@H](C){R}",  # gem-difluorocyclopropane
)

_Q_SMALL = ("C", "CC", "C(C)C", "C1CC1", "CC(F)(F)F")


def _count_stereo(mol: Chem.Mol) -> int:
    return sum(
        1
        for s in Chem.FindPotentialStereo(mol)
        if s.specified == Chem.StereoSpecified.Specified
    )


def _generate_stratum(
    stratum: str, target_count: int, seen_key14: set[str], exclude_key14: set[str]
) -> list[CatalogEntry]:
    candidates: list[str] = []
    if stratum == "S1_bridged_caged":
        for tmpl in _S1_TEMPLATES:
            for r in _R_TAILS:
                if "{Q}" in tmpl:
                    for q in _Q_SMALL:
                        candidates.append(tmpl.format(R=r, Q=q))
                else:
                    candidates.append(tmpl.format(R=r))
    elif stratum == "S2_dense_stereo":
        for tmpl in _S2_TEMPLATES:
            for r in _R_TAILS:
                candidates.append(tmpl.format(R=r))
    elif stratum == "S3_strained_bredt":
        for tmpl in _S3_TEMPLATES:
            for r in _R_TAILS:
                candidates.append(tmpl.format(R=r))
    elif stratum == "S4_superatom_si":
        for scaf in _S4_SCAFFOLDS:
            for pg in _S4_PROTECTING_GROUPS:
                for r in _R_TAILS:
                    candidates.append(scaf.format(PG=pg, R=r))
    elif stratum == "S5_medchem_bioisosteres":
        for tmpl in _S5_BIOISOSTERE_TEMPLATES:
            for r in _R_TAILS:
                if "{Q}" in tmpl:
                    for q in _Q_SMALL:
                        candidates.append(tmpl.format(R=r, Q=q))
                else:
                    candidates.append(tmpl.format(R=r))
    else:
        raise ValueError(f"unknown stratum {stratum!r}")

    entries: list[CatalogEntry] = []
    prefix = stratum.split("_", 1)[0]
    for raw in candidates:
        if len(entries) >= target_count:
            break
        std = standardize(raw)
        if not std.valid or std.mol is None or std.smiles is None:
            continue
        key = leakage_key(std)
        if key is None or key[:14] in seen_key14 or key[:14] in exclude_key14:
            continue
        n_heavy = std.mol.GetNumHeavyAtoms()
        if not (9 <= n_heavy <= 42):
            continue
        n_st = _count_stereo(std.mol)
        if stratum == "S2_dense_stereo" and n_st < 3:
            continue
        if stratum in ("S1_bridged_caged", "S3_strained_bredt", "S4_superatom_si") and n_st < 2:
            continue

        seen_key14.add(key[:14])
        mol_id = f"CV-{prefix}-{len(entries) + 1:03d}"
        formula = rdMolDescriptors.CalcMolFormula(Chem.AddHs(std.mol))
        entries.append(
            CatalogEntry(
                mol_id=mol_id,
                stratum=stratum,
                smiles=std.smiles,
                flat_smiles=flat_smiles(std) or std.smiles,
                inchikey=key,
                inchikey14=key[:14],
                formula=formula,
                n_heavy_atoms=n_heavy,
                n_stereocenters=n_st,
            )
        )

    if len(entries) < target_count:
        raise RuntimeError(
            f"stratum {stratum} produced only {len(entries)}/{target_count} unique scaffolds"
        )
    return entries


def build_catalog(
    preset: str = "gold600", *, exclude_key14: set[str] | None = None
) -> list[CatalogEntry]:
    """Build deterministic ChemVeritas catalog: ``gold600`` (600) or ``smoke`` (25)."""
    per_stratum = {"gold600": 120, "medium100": 20, "smoke": 5}.get(preset)
    if per_stratum is None:
        raise ValueError(f"unknown preset {preset!r}; choose gold600, medium100, or smoke")

    seen: set[str] = set()
    excl = set(exclude_key14 or ())
    out: list[CatalogEntry] = []
    for stratum in STRATA:
        out.extend(_generate_stratum(stratum, per_stratum, seen, excl))
    return out
