"""Image(s) or a predictions CSV -> SMILES -> conformers -> MACE-OFF labels -> extxyz dataset.

Training-set builder toward a unified force field: every recognised molecule contributes ETKDG
conformers labelled with MACE-OFF energies and forces.

    # from drawings, with a checkpoint or an exported HF folder (scripts/export_hf_release.py)
    python scripts/image_to_forcefield.py --images figs/*.png --model checkpoints/C-selected.pt \
        --out data/ff/ocsr.extxyz --n-conf 5 --dedupe-inchikey
    # from predict_real.py output (column ``pred``)
    python scripts/image_to_forcefield.py --predictions runs/v2/C_preds.csv \
        --out data/ff/ocsr.extxyz --only-valid --relax

Per input row: RDKit standardization (``ouroboros.eval.standardize``, the scoring standard) ->
``--n-conf`` ETKDGv3 conformers with stereo enforced (``ouroboros.geometry.conformers.embed``) ->
per conformer either a MACE-OFF single point on the ETKDG geometry (default, ``--no-relax``:
non-zero forces, the informative case for force training) or a LBFGS relaxation (``--relax``,
fmax 0.05 eV/Å; the relaxed minimum is written). If mace-torch is not installed (or
``--mace-model none``) geometry is still written and the energy is null.

Outputs (both appended, so repeated runs grow one dataset):
* ``--out`` extxyz (``ase.io.write``). ``info``: smiles, inchikey, source_image, source_id,
  conf_id, mace_model, relaxed, converged (relaxed frames only), stereo_ok, stereo_unassigned,
  has_energy and, when available,
  ``mace_energy`` (eV); ``arrays["mace_forces"]`` (eV/Å). Non-standard key names on purpose: ASE
  turns ``energy``/``forces`` into a calculator on read, and these are MACE-OFF labels (a
  distillation target), not DFT references. For MACE training use
  ``--energy_key mace_energy --forces_key mace_forces``. Frames without energy have no
  ``mace_energy`` key (``has_energy=F``) rather than a NaN a trainer could ingest.
* manifest JSONL (default ``<out stem>.manifest.jsonl``): one line per input row, including
  skipped rows with their reason — the authoritative SMILES record (ASE's extxyz string escaping
  is not guaranteed for backslashes in SMILES; join frame ``source_id``/``conf_id`` to manifest
  ``id``/``conformers[].conf_id``, or on inchikey).

Stereo: ETKDG enforces every specified stereo element and each conformer is re-checked from its
3D coordinates (``stereo_ok``; also after relaxation). Unassigned stereocentres / double bonds in
the predicted SMILES are counted (``n_unassigned_*``, ``stereo_unassigned``): ETKDG then picks an
arbitrary configuration, so such frames describe *a* stereoisomer, not necessarily the drawn one.

``--only-valid`` writes only frames that pass every check (stereo_ok; finite energy/forces when
labelled; converged when relaxed). Without it such frames are written with their flags. R-group /
pseudo-atom predictions (``*``, ``[R1]``...) and unparsable SMILES never yield geometry.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from ouroboros.eval.standardize import standard_inchikey, standardize
from ouroboros.geometry.batch import embed_many, evaluate_single_points
from ouroboros.geometry.conformers import (
    FMAX,
    embed,
    mace_calculator,
    relax,
    stereo_matches,
    to_atoms,
)

ROOT = Path(__file__).resolve().parents[1]
ENERGY_KEY = "mace_energy"  # eV
FORCES_KEY = "mace_forces"  # eV/Å
SMILES_COLS = ("pred", "prediction", "smiles")
IMAGE_COLS = ("source_image", "image", "image_path", "path")


# ------------------------------------------------------------------------------- inputs


def _first(row: dict, names) -> str | None:
    return next((row[n] for n in names if n in row and row[n] not in (None, "")), None)


def rows_from_csv(path: Path, smiles_col: str | None, id_col: str, image_col: str | None):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        cols = reader.fieldnames or []
        col = smiles_col or next((c for c in SMILES_COLS if c in cols), None)
        if col is None or col not in cols:
            raise SystemExit(f"{path}: no SMILES column {smiles_col or SMILES_COLS}; have {cols}")
        for n, r in enumerate(reader):
            rid = r.get(id_col) or str(n)
            image = r.get(image_col) if image_col else _first(r, IMAGE_COLS)
            if not image:  # predict_real.py CSVs identify the image by (subset, id)
                image = f"{r['subset']}/{rid}" if r.get("subset") else f"{path.name}#{rid}"
            yield {"id": rid, "smiles": r.get(col) or "", "source_image": image,
                   "accepted": r.get("accepted"), "rejection_reason": r.get("rejection_reason", "")}


def _predict_real():
    spec = importlib.util.spec_from_file_location("predict_real", ROOT / "scripts/predict_real.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(path: Path, device):
    """A training/inference checkpoint (.pt) or an exported HF folder -> (model, image_size)."""
    if Path(path).is_dir():
        from ouroboros.hf import CONFIG_FILE, load_pretrained

        size = json.loads((Path(path) / CONFIG_FILE).read_text())["data"]["image_size"]
        return load_pretrained(path, device=device), size
    model, cfg = _predict_real().load_checkpoint(path, device)
    return model, cfg["data"]["image_size"]


def rows_from_images(paths: list[Path], model_path: Path, beam: int, batch_size: int,
                     ink_crop=False, ink_threshold=250, ink_margin=8):
    import torch
    from PIL import Image

    from ouroboros.data.real import fit_to_canvas

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, size = load_model(model_path, dev)
    predict_batch = _predict_real().predict_batch
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        ink = []
        for p in batch:
            with Image.open(p) as im:
                fitted = fit_to_canvas(
                    im, size, ink_crop=ink_crop,
                    ink_threshold=ink_threshold, ink_margin=ink_margin,
                )
                gray = np.asarray(fitted, dtype=np.float32)
            ink.append(1.0 - torch.from_numpy(gray)[None] / 255.0)
        images = torch.stack(ink).to(dev)
        with torch.inference_mode():
            if beam == 1:
                preds = model.generate(images)
            else:
                preds = [p for p, _ in predict_batch([model], images, beam, [], True)]
        for p, smi in zip(batch, preds, strict=True):
            yield {"id": Path(p).stem, "smiles": smi, "source_image": str(p)}


# ----------------------------------------------------------------------------- chemistry


def unassigned_stereo(mol: Chem.Mol) -> dict:
    """Stereo elements RDKit considers possible but the SMILES leaves unspecified."""
    counts = {"n_unassigned_stereocentres": 0, "n_unassigned_stereobonds": 0,
              "n_unassigned_other": 0}
    for el in Chem.FindPotentialStereo(Chem.Mol(mol)):
        if el.specified != Chem.StereoSpecified.Unspecified:
            continue
        if el.type == Chem.StereoType.Atom_Tetrahedral:
            counts["n_unassigned_stereocentres"] += 1
        elif el.type == Chem.StereoType.Bond_Double:
            counts["n_unassigned_stereobonds"] += 1
        else:
            counts["n_unassigned_other"] += 1
    counts["stereo_unassigned"] = any(counts.values())
    return counts


def _finalize_conformer(
    mol: Chem.Mol,
    smiles: str,
    cid: int,
    positions: np.ndarray,
    energy: float | None = None,
    forces: np.ndarray | None = None,
    converged: bool | None = None,
    fmax: float | None = None,
    error: str | None = None,
) -> dict:
    out = {
        "conf_id": int(cid),
        "energy": energy,
        "forces": forces,
        "converged": converged,
        "fmax": fmax,
        "positions": positions,
        "error": error,
    }
    final = Chem.Mol(mol)
    conf = final.GetConformer(cid)
    for i, pos in enumerate(out["positions"]):
        conf.SetAtomPosition(i, [float(x) for x in pos])
    out["stereo_ok"] = bool(stereo_matches(final, smiles, cid))
    out["finite"] = out["energy"] is None or bool(
        math.isfinite(out["energy"])
        and np.isfinite(out["forces"]).all()
        and np.isfinite(out["positions"]).all()
    )
    return out


def label_conformer(mol: Chem.Mol, smiles: str, cid: int, calc, do_relax: bool) -> dict:
    """Positions (+ energy/forces if ``calc``) for one embedded conformer, with quality flags."""
    positions = mol.GetConformer(cid).GetPositions()
    energy = forces = converged = fmax = error = None
    if calc is not None:
        try:
            if do_relax:
                r = relax(to_atoms(mol, cid), calc, FMAX)
                energy, forces, positions = r.energy, r.forces, r.positions
                converged = bool(r.converged and r.fmax <= FMAX)
                fmax = r.fmax
            else:
                atoms = to_atoms(mol, cid)
                atoms.calc = calc
                forces = atoms.get_forces()
                energy = float(atoms.get_potential_energy())
                fmax = float(np.linalg.norm(forces, axis=1).max())
        except (RuntimeError, ValueError) as e:
            error = f"mace_failed: {e}"
    return _finalize_conformer(mol, smiles, cid, positions, energy, forces, converged, fmax, error)


def passes(c: dict, do_relax: bool) -> bool:
    if c["error"] or not c["stereo_ok"] or not c["finite"]:
        return False
    return not (do_relax and c["energy"] is not None and not c["converged"])


def _prepare_row(row: dict, a, calc, mace_name: str | None, seen: set[str]):
    rec = {
        "id": row["id"],
        "source_image": row["source_image"],
        "input_smiles": row["smiles"],
        "smiles": None,
        "inchikey": None,
        "status": "ok",
        "mace_model": mace_name,
        "relaxed": bool(getattr(a, "relax", False) and calc is not None),
        "n_frames": 0,
        "energy": None,
        "conformers": [],
    }
    if getattr(a, "respect_confidence", False):
        rec["confidence_accepted"] = row.get("accepted")
        rec["rejection_reason"] = row.get("rejection_reason", "")
        if str(row.get("accepted", "")).lower() not in ("true", "1"):
            return None, rec | {"status": "confidence_rejected"}
    std = standardize(row["smiles"])
    if not std.valid:
        return None, rec | {"status": f"invalid_smiles: {std.reason}"}
    if getattr(a, "expand_aliases", False):
        from ouroboros.data.rgroups import expand_aliases

        rec["recognized_smiles"] = std.smiles
        try:
            expanded = expand_aliases(row["smiles"])
        except ValueError as error:
            return None, rec | {"status": str(error)}
        rec["expanded_smiles"] = Chem.MolToSmiles(expanded)
        std = standardize(rec["expanded_smiles"])
        if not std.valid:
            return None, rec | {"status": "invalid_expansion"}
    rec["smiles"] = std.smiles
    if any(at.GetAtomicNum() == 0 for at in std.mol.GetAtoms()):
        return None, rec | {"status": "pseudo_atoms"}  # R-groups / abbreviations: no elements
    if (
        getattr(a, "geometry_eligibility", False)
        or getattr(a, "allowed_elements", None)
        or getattr(a, "require_assigned_stereo", False)
    ):
        from ouroboros.geometry.eligibility import EligibilityPolicy

        elements = getattr(a, "allowed_elements", None)
        policy = EligibilityPolicy(
            frozenset(elements) if elements is not None else None,
            require_assigned_stereo=getattr(a, "require_assigned_stereo", False),
        )
        rec["eligibility_reasons"] = policy.reasons(std.mol, calc)
        if rec["eligibility_reasons"]:
            return None, rec | {"status": "geometry_ineligible"}
    rec["inchikey"] = standard_inchikey(std)
    rec.update(unassigned_stereo(std.mol))
    if getattr(a, "dedupe_inchikey", False) and rec["inchikey"] and rec["inchikey"] in seen:
        return None, rec | {"status": "duplicate_inchikey"}
    return std, rec


def _expand_displacements(mol: Chem.Mol, smiles: str, a, rec: dict) -> tuple[Chem.Mol, dict]:
    sampling = getattr(a, "displaced_samples", 0)
    sample_metadata: dict[int, dict] = {}
    if not sampling:
        return mol, sample_metadata
    from ouroboros.geometry.displacements import displaced_conformers

    if getattr(a, "relax", False):
        raise ValueError("displaced samples require single-point labeling, not relaxation")
    for conf in mol.GetConformers():
        parent_id = conf.GetId()
        sample_metadata[parent_id] = dict(
            parent_conf_id=parent_id,
            sample_id=0,
            displacement_seed=a.seed,
            displacement_max=0.0,
        )
    amplitude = getattr(a, "max_displacement", 0.1)
    distance = getattr(a, "displacement_min_distance", 0.6)
    mol, trials = displaced_conformers(mol, smiles, sampling, amplitude, a.seed, distance)
    rec["displacement_samples"] = trials
    rec["sampling_parameters"] = dict(
        n_samples=sampling, amplitude=amplitude, min_distance=distance, seed=a.seed
    )
    for trial in trials:
        if trial["conf_id"] is not None:
            sample_metadata[trial["conf_id"]] = dict(
                parent_conf_id=trial["parent_conf_id"],
                sample_id=trial["sample_id"],
                displacement_seed=a.seed,
                displacement_max=trial["displacement_max"],
            )
    return mol, sample_metadata


def _assemble_frames(
    row: dict,
    std,
    mol: Chem.Mol,
    labelled: list[dict],
    sample_metadata: dict,
    a,
    mace_name: str | None,
    rec: dict,
    seen: set[str],
) -> tuple[list, dict]:
    from ase import Atoms

    only_valid = getattr(a, "only_valid", False)
    symbols = [at.GetSymbol() for at in mol.GetAtoms()]
    frames = []
    for c in labelled:
        ok = passes(c, rec["relaxed"])
        write = ok or not only_valid
        rec["conformers"].append(
            {k: c[k] for k in ("conf_id", "energy", "converged", "fmax", "stereo_ok", "error")}
            | {"written": write}
        )
        rec["conformers"][-1].update(sample_metadata.get(c["conf_id"], {}))
        if not write:
            continue
        atoms = Atoms(symbols=symbols, positions=np.asarray(c["positions"], dtype=float))
        atoms.info.update(
            smiles=std.smiles,
            inchikey=rec["inchikey"] or "",
            source_image=row["source_image"],
            source_id=str(row["id"]),
            conf_id=c["conf_id"],
            mace_model=mace_name or "none",
            relaxed=rec["relaxed"],
            stereo_ok=c["stereo_ok"],
            stereo_unassigned=rec["stereo_unassigned"],
            has_energy=c["energy"] is not None,
        )
        atoms.info.update(sample_metadata.get(c["conf_id"], {}))
        if rec["relaxed"]:
            atoms.info["converged"] = bool(c["converged"])
        if c["energy"] is not None:
            atoms.info[ENERGY_KEY] = float(c["energy"])
            atoms.arrays[FORCES_KEY] = np.asarray(c["forces"], dtype=float)
        frames.append(atoms)
    energies = [c["energy"] for c in rec["conformers"] if c["written"] and c["energy"] is not None]
    rec["n_frames"] = len(frames)
    rec["energy"] = min(energies) if energies else None  # lowest written conformer, eV
    if not frames:
        rec["status"] = "no_valid_conformer"
    elif rec["inchikey"]:
        seen.add(rec["inchikey"])
    return frames, rec


def process(row: dict, a, calc, mace_name: str | None, seen: set[str]) -> tuple[list, dict]:
    std, rec = _prepare_row(row, a, calc, mace_name, seen)
    if std is None:
        return [], rec
    attempts: list[dict] = []
    mol, reason = embed(std.smiles, a.n_conf, a.seed, diagnostics=attempts)
    if mol is None:
        return [], rec | {"status": reason}
    mol, sample_metadata = _expand_displacements(mol, std.smiles, a, rec)
    calc_batch = getattr(a, "calc_batch_size", 1)
    confs = list(mol.GetConformers())
    if not rec["relaxed"] and calc is not None and calc_batch > 1 and len(confs) > 1:
        atoms_list = [to_atoms(mol, conf.GetId()) for conf in confs]
        sp_results = evaluate_single_points(atoms_list, calc, batch_size=calc_batch)
        labelled = [
            _finalize_conformer(
                mol,
                std.smiles,
                conf.GetId(),
                conf.GetPositions(),
                sp.energy,
                sp.forces,
                None,
                sp.fmax,
                sp.error,
            )
            for conf, sp in zip(confs, sp_results, strict=True)
        ]
    else:
        labelled = [
            label_conformer(mol, std.smiles, conf.GetId(), calc, rec["relaxed"]) for conf in confs
        ]
    return _assemble_frames(row, std, mol, labelled, sample_metadata, a, mace_name, rec, seen)


def process_batch(
    rows: list[dict], a, calc, mace_name: str | None, seen: set[str]
) -> list[tuple[list, dict]]:
    """Process a chunk of distinct-InChIKey rows using parallel ETKDG embedding and batched
    single-point force-field evaluation while preserving exact sequential row/frame ordering."""
    if not rows:
        return []
    embed_workers = getattr(a, "embed_workers", 1)
    calc_batch = getattr(a, "calc_batch_size", 1)
    if getattr(a, "relax", False):
        raise ValueError("process_batch only supports single-point labeling (relax=False)")

    out: list[tuple[list, dict] | None] = [None] * len(rows)
    active_indices: list[int] = []
    prepared: list[tuple] = []
    for idx, row in enumerate(rows):
        std, rec = _prepare_row(row, a, calc, mace_name, seen)
        if std is None:
            out[idx] = ([], rec)
        else:
            active_indices.append(idx)
            prepared.append((row, std, rec))

    if not active_indices:
        return [item for item in out if item is not None]

    embedded = embed_many(
        [(std.smiles, a.n_conf, a.seed) for _, std, _ in prepared],
        workers=embed_workers,
        embed_fn=embed,
    )

    mol_entries: list[tuple[int, dict, object, Chem.Mol, dict, dict]] = []
    all_atoms: list = []
    conformer_spans: list[tuple[int, int]] = []
    for (idx, (row, std, rec)), (mol, reason, _) in zip(
        zip(active_indices, prepared, strict=True), embedded, strict=True
    ):
        if mol is None:
            out[idx] = ([], rec | {"status": reason})
            continue
        mol, sample_metadata = _expand_displacements(mol, std.smiles, a, rec)
        start = len(all_atoms)
        for conf in mol.GetConformers():
            all_atoms.append(to_atoms(mol, conf.GetId()))
        conformer_spans.append((start, len(all_atoms)))
        mol_entries.append((idx, row, std, mol, sample_metadata, rec))

    sp_all = evaluate_single_points(all_atoms, calc, batch_size=calc_batch)
    for (idx, row, std, mol, sample_metadata, rec), (start, end) in zip(
        mol_entries, conformer_spans, strict=True
    ):
        confs = list(mol.GetConformers())
        sp_slice = sp_all[start:end]
        labelled = [
            _finalize_conformer(
                mol,
                std.smiles,
                conf.GetId(),
                conf.GetPositions(),
                sp.energy,
                sp.forces,
                None,
                sp.fmax,
                sp.error,
            )
            for conf, sp in zip(confs, sp_slice, strict=True)
        ]
        out[idx] = _assemble_frames(
            row, std, mol, labelled, sample_metadata, a, mace_name, rec, seen
        )
    return [item for item in out if item is not None]


def _iter_row_chunks(rows, a):
    """Group input rows into bounded chunks, flushing before any intra-chunk duplicate InChIKey
    when ``--dedupe-inchikey`` is enabled so deduplication matches sequential streaming."""
    target = max(getattr(a, "embed_workers", 1) * 4, getattr(a, "calc_batch_size", 1), 8)
    dedupe = getattr(a, "dedupe_inchikey", False)
    chunk: list[dict] = []
    chunk_keys: set[str] = set()
    for row in rows:
        key = None
        if dedupe and row.get("smiles"):
            std = standardize(row["smiles"])
            if std.valid:
                key = standard_inchikey(std)
        if key and key in chunk_keys and chunk:
            yield chunk
            chunk = []
            chunk_keys = set()
        chunk.append(row)
        if key:
            chunk_keys.add(key)
        if len(chunk) >= target:
            yield chunk
            chunk = []
            chunk_keys = set()
    if chunk:
        yield chunk


def load_calculator(model: str, device: str | None):
    if model == "none":
        return None, None
    try:
        return mace_calculator(model, device), f"MACE-OFF23 {model}"
    except ImportError as e:  # mace-torch not installed: geometry only
        print(f"warning: MACE unavailable ({e}); writing geometry only, energy null")
        return None, None


def _seen_inchikeys(manifest: Path) -> set[str]:
    if not manifest.is_file():
        return set()
    recs = (json.loads(x) for x in manifest.read_text().splitlines() if x.strip())
    return {r["inchikey"] for r in recs if r.get("inchikey") and r.get("n_frames")}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, nargs="+", help="image files (needs --model)")
    src.add_argument("--predictions", type=Path, help="CSV/TSV with a SMILES column")
    ap.add_argument("--model", type=Path, help="checkpoint .pt or exported HF folder")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smiles-col", default=None, help=f"default: first of {SMILES_COLS}")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--image-col", default=None, help=f"default: first of {IMAGE_COLS}")
    ap.add_argument("--out", type=Path, required=True, help="extxyz file (appended)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="JSONL (appended); default <out stem>.manifest.jsonl")
    ap.add_argument("--n-conf", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relax", action=argparse.BooleanOptionalAction, default=False,
                    help="relax each conformer with MACE (default: single points on ETKDG)")
    ap.add_argument("--only-valid", action="store_true",
                    help="write only frames passing stereo / finiteness / convergence checks")
    ap.add_argument("--dedupe-inchikey", action="store_true",
                    help="skip molecules already in this run or in an existing manifest")
    ap.add_argument("--mace-model", default="small", help="small | medium | large | none")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ink-crop", action="store_true")
    ap.add_argument("--ink-threshold", type=int, default=250)
    ap.add_argument("--ink-margin", type=int, default=8)
    ap.add_argument("--respect-confidence", action="store_true",
                    help="require accepted=True in predictions CSV; rejected rows stay in manifest")
    ap.add_argument(
        "--expand-aliases",
        action="store_true",
        help="expand known aliases before making 3D coordinates; reject generic R groups",
    )
    ap.add_argument("--geometry-eligibility", action="store_true",
                    help="reject net charge, radicals, and unsupported elements")
    ap.add_argument("--allowed-elements", nargs="+", default=None,
                    help="list supported element symbols and turn on geometry checks")
    ap.add_argument("--require-assigned-stereo", action="store_true",
                    help="turn on geometry checks and reject unassigned stereo")
    ap.add_argument(
        "--displaced-samples",
        type=int,
        default=0,
        help="random coordinate changes to try per ETKDG conformer (default: off)",
    )
    ap.add_argument(
        "--max-displacement", type=float, default=0.1, help="maximum atom movement in Angstrom"
    )
    ap.add_argument(
        "--displacement-min-distance",
        type=float,
        default=0.6,
        help="minimum distance between atoms in Angstrom",
    )
    ap.add_argument(
        "--calc-batch-size",
        type=int,
        default=1,
        help="conformers per single-point calculator forward pass (default: 1)",
    )
    ap.add_argument(
        "--embed-workers",
        type=int,
        default=1,
        help="worker threads for parallel ETKDG conformer embedding (default: 1)",
    )
    a = ap.parse_args(argv)
    if a.respect_confidence and a.images:
        ap.error("--respect-confidence requires --predictions with acceptance fields")
    if a.allowed_elements:
        valid_symbols = {Chem.GetPeriodicTable().GetElementSymbol(z) for z in range(1, 119)}
        if not set(a.allowed_elements) <= valid_symbols:
            ap.error("--allowed-elements requires element symbols")
    if (a.displaced_samples < 0 or not math.isfinite(a.max_displacement) or a.max_displacement < 0
            or not math.isfinite(a.displacement_min_distance) or a.displacement_min_distance < 0):
        ap.error("displacement count, amplitude and distance must be finite and nonnegative")
    if a.displaced_samples and a.relax:
        ap.error("--displaced-samples requires single points; omit --relax")
    if a.calc_batch_size < 1 or a.embed_workers < 1:
        ap.error("--calc-batch-size and --embed-workers must be positive")
    if a.calc_batch_size > 1 and a.relax:
        ap.error("--calc-batch-size > 1 requires single points; omit --relax")
    if a.embed_workers > 1 and a.relax:
        ap.error("--embed-workers > 1 requires single points; omit --relax")
    if a.images and a.model is None:
        ap.error("--images needs --model")
    if a.n_conf < 1 or a.beam < 1:
        ap.error("--n-conf and --beam must be positive")
    from ase.io import write

    manifest = a.manifest or a.out.with_name(a.out.stem + ".manifest.jsonl")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    seen = _seen_inchikeys(manifest) if a.dedupe_inchikey else set()
    calc, mace_name = load_calculator(a.mace_model, a.device)
    rows = (rows_from_images(a.images, a.model, a.beam, a.batch_size,
                             a.ink_crop, a.ink_threshold, a.ink_margin) if a.images
            else rows_from_csv(a.predictions, a.smiles_col, a.id_col, a.image_col))
    t0, counts, n_frames = time.time(), {}, 0
    batched = a.calc_batch_size > 1 or a.embed_workers > 1
    with open(manifest, "a") as mf:
        if batched:
            for chunk in _iter_row_chunks(rows, a):
                results = process_batch(chunk, a, calc, mace_name, seen)
                chunk_frames = []
                for frames, rec in results:
                    chunk_frames.extend(frames)
                    n_frames += len(frames)
                    status = rec["status"].split(":")[0]
                    counts[status] = counts.get(status, 0) + 1
                    mf.write(json.dumps(rec) + "\n")
                if chunk_frames:
                    write(a.out, chunk_frames, format="extxyz", append=True)
                mf.flush()
        else:
            for row in rows:
                frames, rec = process(row, a, calc, mace_name, seen)
                if frames:
                    write(a.out, frames, format="extxyz", append=True)
                n_frames += len(frames)
                status = rec["status"].split(":")[0]
                counts[status] = counts.get(status, 0) + 1
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
    summary = {"out": str(a.out), "manifest": str(manifest), "frames_written": n_frames,
               "rows": counts, "mace_model": mace_name,
               "relaxed": bool(a.relax and calc is not None),
               "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
