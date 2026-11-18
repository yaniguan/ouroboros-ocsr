"""Map OCSR predictions to a molecule context for the lab reasoner (YLM).

Input: the predictions CSV written by ``scripts/predict_real.py``
(columns ``id,pred,subset`` and, with ``--min-margin``/``--max-disagreement``,
``accepted,rejection_reason,margin,disagreement``). The ``gold`` column is
never copied: the reasoner must not see reference answers.

Output: one JSON object per image, plus an optional short text block for a prompt.

    python -m lab.molecule_context runs/preds.csv --out runs/context.jsonl --prompt

Status rules (simple and strict; the reasoner should only trust ``ok``):
  invalid   RDKit cannot parse or standardize the prediction
  rejected  the confidence gate in predict_real.py rejected it
  review    parsed, but has R-group/alias pseudo-atoms or unassigned stereo
  ok        none of the above
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors

from ouroboros.eval.standardize import flat_smiles, standard_inchikey, standardize
from ouroboros.labels import ISOTOPE_LABELS

SCHEMA = "ouroboros.molecule_context/v1"
STATUSES = ("ok", "review", "rejected", "invalid")


@dataclass
class MoleculeContext:
    id: str
    status: str
    usable: bool
    pred_raw: str
    subset: str = ""
    smiles: str | None = None  # standardized, with stereo
    smiles_no_stereo: str | None = None
    inchikey: str | None = None
    formula: str | None = None
    heavy_atoms: int | None = None
    stereo_specified: int = 0
    stereo_unassigned: int = 0
    pseudo_atoms: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    gate: dict = field(default_factory=dict)
    molfile: str | None = None
    schema: str = SCHEMA

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _as_bool(text: str | None) -> bool | None:
    if text is None or str(text).strip() == "":
        return None
    return str(text).strip().lower() in ("true", "1", "yes")


def _as_float(text: str | None) -> float | None:
    if text is None or str(text).strip() in ("", "None"):
        return None
    return float(text)


def _stereo_counts(mol: Chem.Mol) -> tuple[int, int]:
    specified = unassigned = 0
    for info in Chem.FindPotentialStereo(mol):
        if info.specified == Chem.StereoSpecified.Specified:
            specified += 1
        else:
            unassigned += 1
    return specified, unassigned


def _molfile(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    return Chem.MolToMolBlock(m)


def from_prediction(row: dict, *, with_molfile: bool = False) -> MoleculeContext:
    """Build one context record from one predictions-CSV row (a dict)."""
    pred = (row.get("pred") or "").strip()
    ctx = MoleculeContext(id=str(row.get("id", "")), status="invalid", usable=False,
                          pred_raw=pred, subset=row.get("subset", "") or "")
    accepted = _as_bool(row.get("accepted"))
    if accepted is not None:
        ctx.gate = {"accepted": accepted,
                    "rejection_reason": row.get("rejection_reason", "") or "",
                    "margin": _as_float(row.get("margin")),
                    "disagreement": _as_float(row.get("disagreement"))}

    std = standardize(pred)
    if not std.valid:
        ctx.reasons.append(std.reason)
        return ctx

    mol = std.mol
    ctx.smiles = std.smiles
    ctx.smiles_no_stereo = flat_smiles(std)
    ctx.inchikey = standard_inchikey(std)
    ctx.formula = rdMolDescriptors.CalcMolFormula(mol)
    ctx.heavy_atoms = mol.GetNumHeavyAtoms()
    ctx.stereo_specified, ctx.stereo_unassigned = _stereo_counts(mol)
    ctx.pseudo_atoms = sorted({ISOTOPE_LABELS.get(a.GetIsotope(), "*")
                               for a in mol.GetAtoms() if a.GetAtomicNum() == 0})
    if with_molfile:
        ctx.molfile = _molfile(mol)

    if accepted is False:
        ctx.status = "rejected"
        ctx.reasons.extend(r for r in ctx.gate["rejection_reason"].split(";") if r)
        if not ctx.reasons:
            ctx.reasons.append("gate_rejected")
        return ctx
    if ctx.pseudo_atoms:
        ctx.reasons.append("pseudo_atoms")
    if ctx.stereo_unassigned:
        ctx.reasons.append("unassigned_stereo")
    ctx.status = "review" if ctx.reasons else "ok"
    ctx.usable = ctx.status == "ok"
    return ctx


def read_predictions(path: str | Path, *, with_molfile: bool = False) -> list[MoleculeContext]:
    with open(path, newline="") as f:
        return [from_prediction(r, with_molfile=with_molfile) for r in csv.DictReader(f)]


def prompt_block(contexts: list[MoleculeContext]) -> str:
    """Short, plain text for the reasoner. Only ``ok`` rows give a structure."""
    lines = ["Molecules read from figures (OCSR). Trust only status=ok."]
    for c in contexts:
        if c.status == "ok":
            lines.append(f"- {c.id}: ok | SMILES {c.smiles} | formula {c.formula}")
        elif c.status == "review":
            lines.append(f"- {c.id}: review ({', '.join(c.reasons)}) | SMILES {c.smiles}"
                         " | confirm with a human before use")
        else:
            lines.append(f"- {c.id}: {c.status} ({', '.join(c.reasons)}) | structure unknown")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("predictions", help="CSV from scripts/predict_real.py")
    p.add_argument("--out", help="write JSON lines here (default: stdout)")
    p.add_argument("--molfile", action="store_true", help="add a 2D molfile per molecule")
    p.add_argument("--prompt", action="store_true", help="also print the prompt block")
    a = p.parse_args(argv)
    contexts = read_predictions(a.predictions, with_molfile=a.molfile)
    text = "".join(c.to_json() + "\n" for c in contexts)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    counts = {s: sum(c.status == s for c in contexts) for s in STATUSES}
    print(json.dumps({"rows": len(contexts), **counts}), file=sys.stderr)
    if a.prompt:
        print(prompt_block(contexts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
