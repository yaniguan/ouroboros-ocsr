"""Supporting Information (SI) experimental characterization block parser & formatter.

Extracts compound numbers, Hill formulas, HRMS adducts/m/z, 13C NMR peak lists (stripping
deuterated solvent peaks), 1H NMR multiplets with J-couplings, and optical rotations [alpha]_D
from ACS / RSC / Wiley / ChemRxiv organic synthesis experimental text.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ouroboros.phys.hrms import STANDARD_ADDUCTS
from ouroboros.phys.nmr import H1Multiplet, NMRPrediction
from ouroboros.phys.posterior import SIObservables

# Common residual 13C solvent peaks (ppm) stripped only when explicitly matching solvent header.
SOLVENT_C13_PPM: dict[str, float] = {
    "cdcl3": 77.16,
    "dmso": 39.52,
    "cd3od": 49.00,
    "acetone": 29.84,
    "c6d6": 128.06,
}

_FORMULA_RE = re.compile(r"\b(C\d+H\d+(?:[A-Z][a-z]?\d*)*)\b")
_ADDUCT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\[\s*M\s*\+\s*Na\s*\]\s*\+?", re.I), "[M+Na]+"),
    (re.compile(r"\[\s*M\s*\+\s*K\s*\]\s*\+?", re.I), "[M+K]+"),
    (re.compile(r"\[\s*M\s*\+\s*NH4\s*\]\s*\+?", re.I), "[M+NH4]+"),
    (re.compile(r"\[\s*M\s*-\s*H\s*\]\s*-?", re.I), "[M-H]-"),
    (re.compile(r"\[\s*M\s*\+\s*H\s*\]\s*\+?", re.I), "[M+H]+"),
    (re.compile(r"\[\s*M\s*\]\s*\+\.?", re.I), "[M]+."),
)
_HRMS_RE = re.compile(
    r"HRMS[^;\n]*?(?:calcd[^;\n]*?(?P<calcd>\d{2,4}\.\d{3,6}))?"
    r"[^;\n]*?(?:found|obsd|m/z\s*=)[^0-9;\n]*?(?P<found>\d{2,4}\.\d{3,6})",
    re.I,
)
_HRMS_SIMPLE_RE = re.compile(
    r"HRMS[^\n]*?(?P<m1>\d{2,4}\.\d{3,6})(?:[^\n]+?(?P<m2>\d{2,4}\.\d{3,6}))?",
    re.I,
)
_NMR_STOP = r"(?=\.\s+(?:[A-Z1¹\[]|$)|\.\s*$|;\s*(?:13\s*C|¹³C|1\s*H|¹H|HRMS|IR|MS|\[\s*[αa])|\n|$)"
_C13_BLOCK_RE = re.compile(
    r"(?:13\s*C|¹³C)\s*NMR\s*(?:\((?P<meta>[^)]*)\))?\s*[:\s]*(?:δ|delta)?\s*(?P<body>.+?)"
    + _NMR_STOP,
    re.I,
)
_H1_BLOCK_RE = re.compile(
    r"(?:1\s*H|¹H)\s*NMR\s*(?:\((?P<meta>[^)]*)\))?\s*[:\s]*(?:δ|delta)?\s*(?P<body>.+?)"
    + _NMR_STOP,
    re.I,
)
_ALPHA_RE = re.compile(r"\[\s*(?:α|alpha)\s*\]\s*_?D[^-+0-9\n]{0,20}([+-]?\d+(?:\.\d+)?)", re.I)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_H1_ITEM_RE = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*[–-]\s*-?\d+(?:\.\d+)?)?\s*\(([^)]+)\)")
_J_RE = re.compile(r"J\s*=\s*([0-9.,\s]+)\s*Hz", re.I)
_INT_H_RE = re.compile(r"(\d+)\s*H\b")
_MULT_RE = re.compile(r"^\s*([sdqtpmbr]+)\b", re.I)


@dataclass
class SICharacterization:
    compound_id: str
    molecular_formula: str | None = None
    hrms_obs_mz: float | None = None
    hrms_calcd_mz: float | None = None
    hrms_adduct: str | None = None
    c13_peaks_ppm: list[float] = field(default_factory=list)
    dept_counts: dict[str, int] | None = None
    h1_multiplets: list[H1Multiplet] = field(default_factory=list)
    optical_rotation_deg: float | None = None
    raw_text: str = ""

    @property
    def internal_hrms_ppm(self) -> float | None:
        if self.hrms_obs_mz and self.hrms_calcd_mz and self.hrms_obs_mz > 0:
            return abs(self.hrms_calcd_mz - self.hrms_obs_mz) / self.hrms_obs_mz * 1e6
        return None

    def to_observables(self) -> SIObservables:
        return SIObservables(
            formula=self.molecular_formula,
            hrms_obs_mz=self.hrms_obs_mz,
            hrms_adduct=self.hrms_adduct,
            c13_peaks_ppm=tuple(self.c13_peaks_ppm),
            dept_counts=self.dept_counts,
            h1_multiplets=tuple(self.h1_multiplets),
            optical_rotation_deg=self.optical_rotation_deg,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["h1_multiplets"] = [asdict(m) for m in self.h1_multiplets]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> SICharacterization:
        mults = [
            H1Multiplet(
                ppm=float(m["ppm"]),
                integration=int(m.get("integration", 1)),
                multiplicity=str(m.get("multiplicity", "m")),
                j_hz=tuple(float(x) for x in m.get("j_hz", ())),
            )
            for m in data.get("h1_multiplets", ())
        ]
        return cls(
            compound_id=str(data.get("compound_id", "unknown")),
            molecular_formula=data.get("molecular_formula"),
            hrms_obs_mz=float(data["hrms_obs_mz"]) if data.get("hrms_obs_mz") is not None else None,
            hrms_calcd_mz=(
                float(data["hrms_calcd_mz"]) if data.get("hrms_calcd_mz") is not None else None
            ),
            hrms_adduct=data.get("hrms_adduct"),
            c13_peaks_ppm=[float(x) for x in data.get("c13_peaks_ppm", ())],
            dept_counts=data.get("dept_counts"),
            h1_multiplets=mults,
            optical_rotation_deg=(
                float(data["optical_rotation_deg"])
                if data.get("optical_rotation_deg") is not None
                else None
            ),
            raw_text=str(data.get("raw_text", "")),
        )


def parse_si_block(text: str, compound_id: str = "1") -> SICharacterization:
    """Parse a synthetic organic chemistry experimental paragraph into SICharacterization."""
    adduct: str | None = None
    for pat, label in _ADDUCT_PATTERNS:
        if pat.search(text):
            adduct = label
            break

    calcd_mz: float | None = None
    obs_mz: float | None = None
    m_hrms = _HRMS_RE.search(text)
    if m_hrms:
        if m_hrms.group("calcd"):
            calcd_mz = float(m_hrms.group("calcd"))
        obs_mz = float(m_hrms.group("found"))
    else:
        m_simple = _HRMS_SIMPLE_RE.search(text)
        if m_simple:
            if m_simple.group("m2"):
                calcd_mz = float(m_simple.group("m1"))
                obs_mz = float(m_simple.group("m2"))
            else:
                obs_mz = float(m_simple.group("m1"))

    # Formula: prefer neutral formula before adduct; if adduct-ion formula is written, keep raw
    # only when explicitly marked as molecular formula, else leave formula None when adduct ion.
    formula: str | None = None
    mf_explicit = re.search(r"(?:Formula|Molecular\s+Formula)\s*[:=]?\s*(C\d+H\d+\w*)", text, re.I)
    if mf_explicit:
        formula = mf_explicit.group(1)
    elif adduct is None:
        m_f = _FORMULA_RE.search(text)
        if m_f:
            formula = m_f.group(1)

    c13_peaks: list[float] = []
    m_c13 = _C13_BLOCK_RE.search(text)
    if m_c13:
        meta = (m_c13.group("meta") or "").lower()
        solv_ppm = next((v for k, v in SOLVENT_C13_PPM.items() if k in meta), None)
        for tok in _FLOAT_RE.findall(m_c13.group("body")):
            val = float(tok)
            if -20.0 <= val <= 240.0:
                if solv_ppm is not None and abs(val - solv_ppm) < 0.08:
                    continue
                c13_peaks.append(val)

    h1_mults: list[H1Multiplet] = []
    m_h1 = _H1_BLOCK_RE.search(text)
    if m_h1:
        for shift_s, detail in _H1_ITEM_RE.findall(m_h1.group("body")):
            ppm = float(shift_s)
            m_mult = _MULT_RE.search(detail)
            mult = m_mult.group(1).lower() if m_mult else "m"
            m_nh = _INT_H_RE.search(detail)
            nh = int(m_nh.group(1)) if m_nh else 1
            j_vals: list[float] = []
            m_j = _J_RE.search(detail)
            if m_j:
                j_vals = [float(x) for x in _FLOAT_RE.findall(m_j.group(1))]
            h1_mults.append(
                H1Multiplet(ppm=ppm, integration=nh, multiplicity=mult, j_hz=tuple(j_vals))
            )

    alpha: float | None = None
    m_alpha = _ALPHA_RE.search(text)
    if m_alpha:
        alpha = float(m_alpha.group(1))

    return SICharacterization(
        compound_id=compound_id,
        molecular_formula=formula,
        hrms_obs_mz=obs_mz,
        hrms_calcd_mz=calcd_mz,
        hrms_adduct=adduct,
        c13_peaks_ppm=c13_peaks,
        h1_multiplets=h1_mults,
        optical_rotation_deg=alpha,
        raw_text=text.strip(),
    )


_HEADER_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(?:Compound\s+|Synthesis\s+of\s+)?([0-9]{1,3}[a-z]?|S[0-9]{1,3})\s*[:.)\-]\s*",
    re.I,
)


def parse_si_document(document_text: str) -> list[SICharacterization]:
    """Split a multi-compound Supporting Information text document and parse valid NMR/HRMS rows."""
    matches = list(_HEADER_SPLIT_RE.finditer(document_text))
    if not matches:
        single = parse_si_block(document_text, compound_id="1")
        has_data = bool(single.hrms_obs_mz or single.c13_peaks_ppm or single.h1_multiplets)
        return [single] if has_data else []

    records: list[SICharacterization] = []
    for idx, m in enumerate(matches):
        cid = m.group(1)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(document_text)
        rec = parse_si_block(document_text[start:end], compound_id=cid)
        if rec.hrms_obs_mz is not None or rec.c13_peaks_ppm or rec.h1_multiplets:
            records.append(rec)
    return records


def format_si_paragraph(
    compound_id: str,
    *,
    formula: str,
    neutral_mass: float,
    nmr: NMRPrediction,
    adduct: str = "[M+H]+",
    ppm_jitter: float = 0.0,
) -> tuple[SICharacterization, str]:
    """Format a realistic ACS-style SI experimental block from physical ground truth."""
    delta = STANDARD_ADDUCTS[adduct]
    calcd_mz = neutral_mass + delta
    obs_mz = calcd_mz * (1.0 + ppm_jitter * 1e-6)
    c13_list = [round(v, 1) for v in sorted(nmr.c13_shifts_ppm, reverse=True)]

    h1_items: list[H1Multiplet] = []
    h1_tokens: list[str] = []
    j_pool = list(reversed(nmr.vicinal_j_hh_hz))
    for i, ppm in enumerate(sorted(nmr.h1_shifts_ppm, reverse=True)):
        if i < len(j_pool):
            jv = round(j_pool[i], 1)
            mult = "d"
            j_tup = (jv,)
            h1_tokens.append(f"{ppm:.2f} (d, J = {jv:.1f} Hz, 1H)")
        else:
            mult = "s"
            j_tup = ()
            h1_tokens.append(f"{ppm:.2f} (s, 1H)")
        h1_items.append(
            H1Multiplet(ppm=round(ppm, 2), integration=1, multiplicity=mult, j_hz=j_tup)
        )

    c13_str = ", ".join(f"{v:.1f}" for v in c13_list)
    h1_str = ", ".join(h1_tokens)
    text = (
        f"Compound {compound_id}: Molecular Formula: {formula}. "
        f"1H NMR (400 MHz, CDCl3) δ {h1_str}. "
        f"13C NMR (101 MHz, CDCl3) δ {c13_str}. "
        f"HRMS (ESI-TOF) m/z: {adduct} calcd for {calcd_mz:.4f}; found {obs_mz:.4f}."
    )
    si = SICharacterization(
        compound_id=compound_id,
        molecular_formula=formula,
        hrms_obs_mz=round(obs_mz, 4),
        hrms_calcd_mz=round(calcd_mz, 4),
        hrms_adduct=adduct,
        c13_peaks_ppm=c13_list,
        dept_counts=dict(nmr.dept_counts),
        h1_multiplets=h1_items,
        raw_text=text,
    )
    return si, text
