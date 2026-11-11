"""Eligibility policies, 7-family post-2020 3D bioisostere detectors & matched control panel.

Aligns strictly with MACE-OFF23's neutral closed-shell 10-element domain
({H, C, N, O, F, P, S, Cl, Br, I}; Kovacs et al., JACS 2025, 10.1021/jacs.4c07099)
and covers modern 'Escape from Flatland 2.0' scaffolds (BCP, BCH, 2-oxa-BCH, BCHep,
cubane, azaspiro/oxetane, SF5) paired with heavy-atom-matched planar/aliphatic controls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rdkit import Chem

from ouroboros.eval.standardize import standardize
from ouroboros.geometry.eligibility import EligibilityPolicy
from ouroboros.geometry.splits import connectivity_key

MACE_OFF23_ELEMENTS: frozenset[str] = frozenset(
    {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
)
AIMNET2_ELEMENTS: frozenset[str] = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"}
)

MACE_OFF23_POLICY = EligibilityPolicy(
    supported_elements=MACE_OFF23_ELEMENTS,
    require_assigned_stereo=False,
    allow_charge=False,
    allow_radicals=False,
)

AIMNET2_POLICY = EligibilityPolicy(
    supported_elements=AIMNET2_ELEMENTS,
    require_assigned_stereo=False,
    allow_charge=True,
    allow_radicals=False,
)

BIOISOSTERE_FAMILIES: tuple[str, ...] = (
    "bcp_111",
    "bch_211",
    "oxa_bch_211",
    "bchep_311",
    "cubane",
    "azaspiro_oxetane",
    "sf5_polyfluoro",
)

_ALIAS_SMARTS: dict[str, str] = {
    "[SF5]": "S(F)(F)(F)(F)F",
    "[CF3]": "C(F)(F)F",
    "[Boc]": "C(=O)OC(C)(C)C",
    "[OMe]": "OC",
    "[OEt]": "OCC",
    "[Ac]": "C(=O)C",
    "[tBu]": "C(C)(C)C",
    "[CN]": "C#N",
}

_CUBANE_Q = Chem.MolFromSmarts("C12C3C4C1C5C2C3C45")
_BCP_Q = Chem.MolFromSmarts("[C;R2,R3]12[CH2,CH,C][C;R2,R3]([CH2,CH,C]1)[CH2,CH,C]2")
_OXA_BCH_Q = Chem.MolFromSmarts("[C;R2,R3]12[CH2,CH,C][O][C;R2,R3]([CH2,CH,C]1)[CH2,CH,C]2")
_BCH_Q = Chem.MolFromSmarts("[C;R2,R3]12[CH2,CH,C][CH2,CH,C][C;R2,R3]([CH2,CH,C]1)[CH2,CH,C]2")
_BCHEP_Q = Chem.MolFromSmarts(
    "[C;R2,R3]12[CH2,CH,C][CH2,CH,C][CH2,CH,C][C;R2,R3]([CH2,CH,C]1)[CH2,CH,C]2"
)
_SPIRO_SMALL_Q = Chem.MolFromSmarts("[C;X4;R2](@[R])(@[R])(@[r3,r4])@[r3,r4]")
_OXETANE_Q = Chem.MolFromSmarts("C1COC1")
_SF5_Q = Chem.MolFromSmarts("[S;X6](F)(F)(F)(F)F")
_GEM_F2_SMALL_Q = Chem.MolFromSmarts("F[C;r3,r4](F)")


@dataclass(frozen=True)
class BioisostereRecord:
    mol_id: str
    smiles: str
    connectivity_key: str
    group: str  # "bioisostere" | "control"
    family: str
    paired_id: str | None
    n_heavy_atoms: int
    eligible: bool
    rejection_reasons: tuple[str, ...]


def normalize_medchem_smiles(raw_smiles: str) -> str:
    """Expand common medicinal-chemistry bracket aliases and keep largest neutral parent."""
    s = raw_smiles.strip()
    for alias, repl in _ALIAS_SMARTS.items():
        if alias in s:
            s = re.sub(re.escape(alias), repl, s)
    if "." in s:
        parts = [p for p in s.split(".") if p]
        parts.sort(key=len, reverse=True)
        s = parts[0]
    return s


def check_mace_off23_eligibility(
    raw_smiles: str, *, policy: EligibilityPolicy = MACE_OFF23_POLICY
) -> tuple[bool, str | None, tuple[str, ...]]:
    """Standardize SMILES and enforce neutral closed-shell 10-element eligibility."""
    norm = normalize_medchem_smiles(raw_smiles)
    std = standardize(norm)
    if not std.valid or std.mol is None or std.smiles is None:
        return False, None, (std.reason or "invalid_smiles",)
    if any(a.GetAtomicNum() == 0 for a in std.mol.GetAtoms()):
        return False, std.smiles, ("unresolved_superatom",)
    reasons = tuple(policy.reasons(std.mol))
    return len(reasons) == 0, std.smiles, reasons


def detect_bioisostere_families(smiles: str) -> tuple[str, ...]:
    """Identify which post-2020 strained 3D bioisostere or hypervalent motifs a SMILES contains."""
    mol = Chem.MolFromSmiles(normalize_medchem_smiles(smiles))
    if mol is None:
        return ()
    hits: list[str] = []
    if _CUBANE_Q is not None and mol.HasSubstructMatch(_CUBANE_Q):
        hits.append("cubane")
    else:
        if _BCP_Q is not None and mol.HasSubstructMatch(_BCP_Q):
            hits.append("bcp_111")
        if _OXA_BCH_Q is not None and mol.HasSubstructMatch(_OXA_BCH_Q):
            hits.append("oxa_bch_211")
        elif _BCH_Q is not None and mol.HasSubstructMatch(_BCH_Q):
            hits.append("bch_211")
        if _BCHEP_Q is not None and mol.HasSubstructMatch(_BCHEP_Q):
            hits.append("bchep_311")

    if (_SPIRO_SMALL_Q is not None and mol.HasSubstructMatch(_SPIRO_SMALL_Q)) or (
        _OXETANE_Q is not None and mol.HasSubstructMatch(_OXETANE_Q)
    ):
        hits.append("azaspiro_oxetane")
    if (_SF5_Q is not None and mol.HasSubstructMatch(_SF5_Q)) or (
        _GEM_F2_SMALL_Q is not None and mol.HasSubstructMatch(_GEM_F2_SMALL_Q)
    ):
        hits.append("sf5_polyfluoro")
    return tuple(hits)


# Paired (family, bioisostere_template, matched_planar_or_aliphatic_control_template)
_PAIRED_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "bcp_111",
        "O=C(N{R})C12CC(C1)(C2)c1ccc(F)cc1",
        "O=C(N{R})c1ccc(-c2ccc(F)cc2)cc1",
    ),
    (
        "bch_211",
        "O=C(O{R})C12CCC(C1)(C#N)C2",
        "O=C(O{R})c1ccc(C#N)cc1",
    ),
    (
        "oxa_bch_211",
        "O=C(N{R})C12COC(C1)(C)C2",
        "O=C(N{R})C1CCOCC1",
    ),
    (
        "bchep_311",
        "O=C(O{R})C12CCCC(C1)(F)C2",
        "O=C(O{R})C1CCC(F)CC1",
    ),
    (
        "cubane",
        "O=C(N{R})C12C3C4C1C5C2C3C45C",
        "O=C(N{R})c1ccc(C)cc1",
    ),
    (
        "azaspiro_oxetane",
        "O=C({R})N1CC2(C1)COC2",
        "O=C({R})N1CCOCC1",
    ),
    (
        "sf5_polyfluoro",
        "O=C(N{R})c1ccc(S(F)(F)(F)(F)F)cc1",
        "O=C(N{R})c1ccc(C(C)(C)C)cc1",
    ),
)

_TAIL_CAPS: tuple[str, ...] = (
    "C",
    "CC",
    "C(C)C",
    "CC(F)F",
    "C1CC1",
    "CCOC",
    "Cc1ccccc1",
    "c1ccc(Cl)cc1",
)


def build_bioisostere_panel(preset: str = "smoke") -> list[BioisostereRecord]:
    """Construct connectivity-unique pairs of 3D bioisosteres and heavy-atom-matched controls."""
    if preset == "smoke":
        caps = _TAIL_CAPS[:2]  # 7 families * 2 caps = 14 bioisosteres + 14 controls = 28 scaffolds
    elif preset == "full":
        caps = _TAIL_CAPS  # 7 families * 8 caps = 56 bioisosteres + 56 controls = 112 scaffolds
    else:
        raise ValueError(f"unknown panel preset {preset!r}; expected 'smoke' or 'full'")

    records: list[BioisostereRecord] = []
    seen_keys: set[str] = set()
    pair_idx = 0
    for fam, bio_t, ctrl_t in _PAIRED_TEMPLATES:
        for cap in caps:
            pair_idx += 1
            ok_b, smi_b, r_b = check_mace_off23_eligibility(bio_t.format(R=cap))
            ok_c, smi_c, r_c = check_mace_off23_eligibility(ctrl_t.format(R=cap))
            if not (ok_b and ok_c and smi_b and smi_c):
                continue
            kb, kc = connectivity_key(smi_b), connectivity_key(smi_c)
            if kb in seen_keys or kc in seen_keys or kb == kc:
                continue
            seen_keys.update((kb, kc))
            bid = f"BIO-{pair_idx:03d}"
            cid = f"CTL-{pair_idx:03d}"
            mb = Chem.MolFromSmiles(smi_b)
            mc = Chem.MolFromSmiles(smi_c)
            records.append(
                BioisostereRecord(
                    mol_id=bid,
                    smiles=smi_b,
                    connectivity_key=kb,
                    group="bioisostere",
                    family=fam,
                    paired_id=cid,
                    n_heavy_atoms=mb.GetNumHeavyAtoms(),
                    eligible=True,
                    rejection_reasons=r_b,
                )
            )
            records.append(
                BioisostereRecord(
                    mol_id=cid,
                    smiles=smi_c,
                    connectivity_key=kc,
                    group="control",
                    family="planar_control",
                    paired_id=bid,
                    n_heavy_atoms=mc.GetNumHeavyAtoms(),
                    eligible=True,
                    rejection_reasons=r_c,
                )
            )
    return records
