"""ChemVeritas-Bench multimodal dataset & task builder (T1-T5 + SHA256 manifest receipt).

Renders 4 visual regimes per molecule (V0_clean, V1_rotated_offgrid, V2_crowded_si_scan,
V3_condensed_layout), synthesizes 5 typed hard negatives, simulates/formats paired SI
analytical text (HRMS + 13C/1H NMR), and writes standardized task JSONL files.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from ouroboros.bench.catalog import CatalogEntry, build_catalog
from ouroboros.data.render import RenderStyle, encode_png, render
from ouroboros.eval.standardize import flat_smiles, standardize
from ouroboros.geometry.propagation import categorize, mirror_smiles
from ouroboros.labels import parse_labelled
from ouroboros.phys.epimers import condense_superatom_for_depiction, enumerate_targeted_epimers
from ouroboros.phys.nmr import predict_nmr
from ouroboros.phys.strain import evaluate_3d_strain
from ouroboros.si.parser import format_si_paragraph

VIEWS = ("V0_clean", "V1_rotated_offgrid", "V2_crowded_si_scan", "V3_condensed_layout")
OFFGRID_ANGLES = (37.0, 73.0, 127.0, 163.0, 217.0, 253.0, 307.0, 341.0)


@dataclass(frozen=True)
class HardNegative:
    role: str
    smiles: str
    category: str
    formula: str
    min_energy_kcal: float | None


def _make_constitutional_isomer_same_formula(smiles: str) -> str | None:
    """Create a valid constitutional isomer with the EXACT same molecular formula."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    target_formula = rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol))
    orig_flat = flat_smiles(standardize(smiles))

    # Regioisomeric transformations that strictly conserve elemental composition.
    transforms = (
        ("CC(=O)O", "COC(=O)"),
        ("C(=O)OC", "OC(=O)C"),
        ("c1ccc(F)cc1", "c1cccc(F)c1"),
        ("c1ccc(Cl)cc1", "c1cccc(Cl)c1"),
        ("c1ccc(OC)cc1", "c1cccc(OC)c1"),
        ("CC", "C(C)"),
        ("OC", "CO"),
    )
    for src, dst in transforms:
        if src in smiles:
            cand = smiles.replace(src, dst, 1)
            st = standardize(cand)
            if st.valid and st.mol is not None and flat_smiles(st) != orig_flat:
                if rdMolDescriptors.CalcMolFormula(Chem.AddHs(st.mol)) == target_formula:
                    return st.smiles

    # Graph-level substituent walk: move a terminal methyl/F/Cl/OH to another CH atom.
    rw_base = Chem.Mol(mol)
    terminals = [
        a.GetIdx()
        for a in rw_base.GetAtoms()
        if a.GetDegree() == 1
        and a.GetAtomicNum() in (6, 8, 9, 17)
        and a.GetTotalNumHs() in (0, 1, 3)
        and a.GetNeighbors()[0].GetAtomicNum() == 6
        and a.GetNeighbors()[0].GetTotalNumHs() >= 0
    ]
    acceptors = [
        a.GetIdx()
        for a in rw_base.GetAtoms()
        if a.GetAtomicNum() == 6
        and a.GetTotalNumHs() >= 1
        and a.GetHybridization() in (Chem.HybridizationType.SP3, Chem.HybridizationType.SP2)
    ]
    for t_idx in terminals:
        parent_idx = rw_base.GetAtomWithIdx(t_idx).GetNeighbors()[0].GetIdx()
        bond = rw_base.GetBondBetweenAtoms(t_idx, parent_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        for acc_idx in acceptors:
            if acc_idx in (t_idx, parent_idx):
                continue
            rw = Chem.RWMol(rw_base)
            rw.RemoveBond(t_idx, parent_idx)
            rw.AddBond(t_idx, acc_idx, Chem.BondType.SINGLE)
            try:
                cand_mol = rw.GetMol()
                Chem.SanitizeMol(cand_mol)
            except Exception:  # noqa: BLE001
                continue
            st = standardize(Chem.MolToSmiles(cand_mol))
            if st.valid and st.mol is not None and flat_smiles(st) != orig_flat:
                if rdMolDescriptors.CalcMolFormula(Chem.AddHs(st.mol)) == target_formula:
                    return st.smiles
    return None


def _make_near_mass_decoy(smiles: str) -> str:
    """Create a plausible constitutional decoy differing in exact mass (caught by HRMS)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Swap a terminal O-CH3 -> O-CH2CH3 (+CH2), F -> Cl, or add a methyl on a CH carbon.
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 6 and a.GetTotalNumHs() >= 1 and not a.GetIsAromatic():
                rw = Chem.RWMol(mol)
                new_c = rw.AddAtom(Chem.Atom(6))
                rw.AddBond(a.GetIdx(), new_c, Chem.BondType.SINGLE)
                cand = rw.GetMol()
                try:
                    Chem.SanitizeMol(cand)
                    st = standardize(Chem.MolToSmiles(cand))
                    if st.valid and st.smiles and st.smiles != smiles:
                        return st.smiles
                except Exception:  # noqa: BLE001
                    pass
    return "CC(=O)Oc1ccccc1C(=O)O"


def synthesize_hard_negatives(
    truth_smiles: str, *, n_conf: int = 4, seed: int = 0
) -> list[HardNegative]:
    """Build 5 typed hard negatives: enantiomer, 2 epimers, same-formula isomer, near-mass decoy."""
    out: list[HardNegative] = []
    seen = {truth_smiles}

    def _add(role: str, smi: str | None) -> bool:
        if not smi:
            return False
        st = standardize(smi)
        if not st.valid or st.mol is None or st.smiles is None or st.smiles in seen:
            return False
        seen.add(st.smiles)
        sr = evaluate_3d_strain(st.smiles, n_conf=n_conf, seed=seed, relax_steps=80)
        formula = rdMolDescriptors.CalcMolFormula(Chem.AddHs(st.mol))
        out.append(
            HardNegative(
                role=role,
                smiles=st.smiles,
                category=categorize(st.smiles, truth_smiles),
                formula=formula,
                min_energy_kcal=sr.min_energy_kcal,
            )
        )
        return True

    # 1. Enantiomer
    _add("enantiomer", Chem.CanonSmiles(mirror_smiles(truth_smiles)))

    # 2 & 3. Diastereomeric epimers (sorted so strained epimer has highest 3D energy)
    epimers = [
        ep
        for ep in enumerate_targeted_epimers([truth_smiles], max_centers=3, max_epimers=8)
        if ep not in seen and categorize(ep, truth_smiles) == "diastereomer"
    ]
    ep_scored = []
    for ep in epimers[:4]:
        sr = evaluate_3d_strain(ep, n_conf=n_conf, seed=seed, relax_steps=80)
        e_val = sr.min_energy_kcal if sr.min_energy_kcal is not None else 999.0
        ep_scored.append((e_val, ep, sr))
    ep_scored.sort(key=lambda t: t[0])
    if ep_scored:
        _add("diastereomer_epimer", ep_scored[0][1])
    if len(ep_scored) >= 2:
        _add("diastereomer_strained", ep_scored[-1][1])

    # 4. Same-formula constitutional isomer (HRMS = 0 ppm decoy!)
    _add("constitutional_same_formula", _make_constitutional_isomer_same_formula(truth_smiles))

    # 5. Additional epimer or near-mass decoy (eliminated by HRMS +/- 5 ppm)
    if len(ep_scored) >= 3:
        _add("diastereomer_epimer_2", ep_scored[1][1])
    _add("near_mass_decoy", _make_near_mass_decoy(truth_smiles))

    # Top up to 6 so T2 still has >=4 achiral-resolvable negatives even after holding out enantiomer
    while len(out) < 6:
        extra = _make_near_mass_decoy(out[-1].smiles if out else truth_smiles)
        if not _add(f"extra_decoy_{len(out)}", extra):
            break
    return out[:6]


def _render_four_views(
    smiles: str, out_img_dir: Path, mol_id: str, image_size: int, seed: int
) -> dict[str, dict]:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse {smiles!r}")
    rng = random.Random(seed)
    views_meta: dict[str, dict] = {}

    # V0: clean ACS style
    s0 = RenderStyle(bond_line_width=2.0, bond_length=26.0, font_scale=0.62, palette="bw")
    r0 = render(mol, s0, size=image_size, seed=seed, apply_postprocess=False)
    p0 = out_img_dir / f"{mol_id}_V0_clean.png"
    p0.write_bytes(encode_png(r0.image))
    views_meta["V0_clean"] = {"path": f"images/{p0.name}", "angle_deg": 0.0, "condensed": False}

    # V1: off-grid planar rotation (preserves SO(2) wedge chirality!)
    angle = OFFGRID_ANGLES[seed % len(OFFGRID_ANGLES)]
    rot_img = r0.image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    p1 = out_img_dir / f"{mol_id}_V1_rotated_offgrid.png"
    p1.write_bytes(encode_png(rot_img))
    views_meta["V1_rotated_offgrid"] = {
        "path": f"images/{p1.name}",
        "angle_deg": angle,
        "condensed": False,
    }

    # V2: crowded scan degradation (compressed bonds + blur + noise + JPEG artefacts)
    s2 = RenderStyle(
        bond_line_width=1.6,
        bond_length=17.5,
        font_scale=0.55,
        palette="bw",
        coordgen=True,
        blur=0.85,
        noise_std=0.035,
        jpeg_quality=42,
        real_style=True,
    )
    r2 = render(mol, s2, size=image_size, seed=seed + 17, apply_postprocess=True)
    p2 = out_img_dir / f"{mol_id}_V2_crowded_si_scan.png"
    p2.write_bytes(encode_png(r2.image))
    views_meta["V2_crowded_si_scan"] = {
        "path": f"images/{p2.name}",
        "angle_deg": 0.0,
        "condensed": False,
    }

    # V3: condensed superatom layout (or alternate explicit-methyl publication layout)
    cond_smi, did_condense = condense_superatom_for_depiction(smiles)
    cond_mol = parse_labelled(cond_smi) if did_condense else None
    s3 = RenderStyle(
        bond_line_width=2.2,
        bond_length=23.0,
        font_scale=0.65,
        palette="bw",
        explicit_methyl=not did_condense,
        coordgen=rng.random() < 0.5,
    )
    r3 = render(cond_mol or mol, s3, size=image_size, seed=seed + 31, apply_postprocess=False)
    p3 = out_img_dir / f"{mol_id}_V3_condensed_layout.png"
    p3.write_bytes(encode_png(r3.image))
    views_meta["V3_condensed_layout"] = {
        "path": f"images/{p3.name}",
        "angle_deg": 0.0,
        "condensed": did_condense,
        "drawn_smiles": cond_smi if did_condense else smiles,
    }
    return views_meta


def build_chemveritas_benchmark(
    out_dir: Path,
    *,
    preset: str = "smoke",
    image_size: int = 384,
    n_conf: int = 4,
    seed: int = 2026,
    render_images: bool = True,
    exclude_key14: set[str] | None = None,
) -> dict:
    """Materialize ChemVeritas molecules, images, SI spectra, hard negatives, and T1-T5 JSONLs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[CatalogEntry] = build_catalog(preset=preset, exclude_key14=exclude_key14)
    rng = random.Random(seed)

    mol_rows: list[dict] = []
    t1_rows: list[dict] = []
    t2_rows: list[dict] = []
    t3_rows: list[dict] = []
    t4_rows: list[dict] = []
    t5_rows: list[dict] = []

    adduct_cycle = ("[M+H]+", "[M+Na]+", "[M+H]+", "[M+NH4]+")
    for idx, entry in enumerate(entries):
        item_seed = seed + idx * 37
        strain = evaluate_3d_strain(entry.smiles, n_conf=n_conf, seed=0, relax_steps=80)
        nmr = predict_nmr(entry.smiles, strain=strain, n_conf=n_conf, seed=0)
        mol_h = Chem.AddHs(Chem.MolFromSmiles(entry.smiles))
        exact_mass = float(Descriptors.ExactMolWt(mol_h))
        adduct = adduct_cycle[idx % len(adduct_cycle)]
        jitter_ppm = rng.uniform(-1.6, 1.6)
        si_obj, si_text = format_si_paragraph(
            entry.mol_id,
            formula=entry.formula,
            neutral_mass=exact_mass,
            nmr=nmr,
            adduct=adduct,
            ppm_jitter=jitter_ppm,
        )
        negs = synthesize_hard_negatives(entry.smiles, n_conf=n_conf, seed=0)
        views = (
            _render_four_views(
                entry.smiles, out_dir / "images", entry.mol_id, image_size, item_seed
            )
            if render_images
            else {v: {"path": f"images/{entry.mol_id}_{v}.png"} for v in VIEWS}
        )

        rpf_target = {
            "step1_formula": entry.formula,
            "step1_hrms_adduct": adduct,
            "step2_c13_sym_signals": nmr.c13_sym_count,
            "step3_cip_preserved": strain.cip_preserved,
            "step4_truth_smiles": entry.smiles,
        }

        mol_rec = {
            "mol_id": entry.mol_id,
            "stratum": entry.stratum,
            "truth_smiles": entry.smiles,
            "flat_smiles": entry.flat_smiles,
            "inchikey": entry.inchikey,
            "inchikey14": entry.inchikey14,
            "formula": entry.formula,
            "n_heavy_atoms": entry.n_heavy_atoms,
            "n_stereocenters": entry.n_stereocenters,
            "truth_min_energy_kcal": strain.min_energy_kcal,
            "c13_sym_count": nmr.c13_sym_count,
            "views": views,
            "si": si_obj.to_dict(),
            "si_text": si_text,
            "hard_negatives": [asdict(n) for n in negs],
            "veritas_rpf_target": rpf_target,
        }
        mol_rows.append(mol_rec)

        # T1: Visual OCSR across all 4 views (2,400 instances for gold600)
        for view_name, vmeta in views.items():
            t1_rows.append(
                {
                    "task_id": f"{entry.mol_id}_{view_name}_T1",
                    "task": "T1_visual_ocsr",
                    "mol_id": entry.mol_id,
                    "stratum": entry.stratum,
                    "view": view_name,
                    "image": vmeta["path"],
                    "truth_smiles": entry.smiles,
                    "truth_energy_kcal": strain.min_energy_kcal,
                }
            )

        # T2: Multimodal SI Disambiguation (crowded scan + SI spectra + achiral-resolvable decoys;
        # pure enantiomers are isolated in T5 where achiral parity abstention is evaluated).
        t2_negs = [n.smiles for n in negs if n.category != "enantiomer"][:5]
        pool = [entry.smiles] + t2_negs
        rng.shuffle(pool)
        ans_idx = pool.index(entry.smiles)
        t2_rows.append(
            {
                "task_id": f"{entry.mol_id}_T2",
                "task": "T2_multimodal_si_disambig",
                "mol_id": entry.mol_id,
                "stratum": entry.stratum,
                "view": "V2_crowded_si_scan",
                "image": views["V2_crowded_si_scan"]["path"],
                "si_text": si_text,
                "si": si_obj.to_dict(),
                "candidates": pool,
                "answer_index": ans_idx,
                "truth_smiles": entry.smiles,
                "veritas_rpf_target": rpf_target,
            }
        )

        # T3: Silent Stereochemical Corruption Audit (50% truth, 50% diastereomer/isomer negative)
        probe_truth = (idx % 2) == 0
        probe_smi = entry.smiles if probe_truth else t2_negs[0]
        t3_rows.append(
            {
                "task_id": f"{entry.mol_id}_T3",
                "task": "T3_silent_error_audit",
                "mol_id": entry.mol_id,
                "stratum": entry.stratum,
                "view": "V2_crowded_si_scan",
                "image": views["V2_crowded_si_scan"]["path"],
                "si_text": si_text,
                "si": si_obj.to_dict(),
                "proposed_smiles": probe_smi,
                "expected_category": categorize(probe_smi, entry.smiles),
                "expected_accept": probe_truth,
                "truth_smiles": entry.smiles,
            }
        )

        # T4: 3D Thermodynamic Stereoisomer Stability Ranking
        iso_pairs = [
            (entry.smiles, strain.min_energy_kcal)
        ] + [
            (n.smiles, n.min_energy_kcal)
            for n in negs
            if n.category == "diastereomer" and n.min_energy_kcal is not None
        ]
        if len(iso_pairs) >= 2 and strain.min_energy_kcal is not None:
            iso_shuffled = list(iso_pairs)
            rng.shuffle(iso_shuffled)
            ranked_truth = [
                s for s, _ in sorted(iso_shuffled, key=lambda kv: float(kv[1] or 0.0))
            ]
            t4_rows.append(
                {
                    "task_id": f"{entry.mol_id}_T4",
                    "task": "T4_thermodynamic_ranking",
                    "mol_id": entry.mol_id,
                    "stratum": entry.stratum,
                    "isomer_smiles": [s for s, _ in iso_shuffled],
                    "reference_energies_kcal": [float(e) for _, e in iso_shuffled],
                    "truth_lowest_smiles": ranked_truth[0],
                    "truth_ranking": ranked_truth,
                }
            )

        # T5: SO(2) vs O(3) Epistemic Parity Abstention Track (pure enantiomers + achiral SI only)
        en_neg = next((n for n in negs if n.category == "enantiomer"), None)
        if en_neg is not None:
            pair = [entry.smiles, en_neg.smiles]
            rng.shuffle(pair)
            t5_rows.append(
                {
                    "task_id": f"{entry.mol_id}_T5",
                    "task": "T5_epistemic_parity_abstention",
                    "mol_id": entry.mol_id,
                    "stratum": entry.stratum,
                    "si_text": si_text,
                    "si": si_obj.to_dict(),
                    "enantiomer_pair": pair,
                    "visual_occluded": True,
                    "expected_action": "ABSTAIN_PARITY",
                }
            )

    files_written: dict[str, str] = {}
    for fname, rows in (
        ("molecules.jsonl", mol_rows),
        ("task_T1_visual_ocsr.jsonl", t1_rows),
        ("task_T2_multimodal_si_disambig.jsonl", t2_rows),
        ("task_T3_silent_error_audit.jsonl", t3_rows),
        ("task_T4_thermodynamic_ranking.jsonl", t4_rows),
        ("task_T5_epistemic_parity_abstention.jsonl", t5_rows),
    ):
        fpath = out_dir / fname
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        fpath.write_text(payload, encoding="utf-8")
        files_written[fname] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    manifest = {
        "benchmark": "ChemVeritas-Bench",
        "version": "1.0.0",
        "preset": preset,
        "seed": seed,
        "n_molecules": len(mol_rows),
        "n_t1_visual_instances": len(t1_rows),
        "n_t2_multimodal_instances": len(t2_rows),
        "n_t3_audit_instances": len(t3_rows),
        "n_t4_thermo_instances": len(t4_rows),
        "n_t5_parity_instances": len(t5_rows),
        "unique_inchikey14": len({r["inchikey14"] for r in mol_rows}),
        "sha256_files": files_written,
    }
    (out_dir / "manifest_receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
