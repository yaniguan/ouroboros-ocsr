"""ChemVeritas-Bench stratified scoring, Veritas-RPF fidelity & bootstrap 95% CIs.

Enforces Constitution Principle I: rendered (V0) and degraded/rotated (V1-V3) views are
always reported in separate entries with sample counts and bootstrap 95% CIs, never averaged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ouroboros.eval.standardize import flat_smiles, standardize
from ouroboros.geometry.propagation import categorize, same_formula
from ouroboros.phys.epimers import expand_superatoms
from ouroboros.phys.posterior import area_under_risk_coverage, expected_calibration_error
from ouroboros.phys.strain import evaluate_3d_strain


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float | int]:
    """Mean and percentile bootstrap (1 - alpha) confidence interval."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"n": n, "mean": mean, "ci_low": float(lo), "ci_high": float(hi)}


def score_veritas_rpf(pred: dict, target: dict) -> dict[str, float]:
    """4-step Reasoning Path Fidelity (RPF) rubric inspired by SUPERChem & FrontierScience."""
    s1_f = str(pred.get("step1_formula") or "").strip() == str(target["step1_formula"]).strip()
    s1_a = (
        str(pred.get("step1_hrms_adduct") or "").strip()
        == str(target["step1_hrms_adduct"]).strip()
    )
    step1 = 0.5 * float(s1_f) + 0.5 * float(s1_a)

    pred_c13 = pred.get("step2_c13_sym_signals")
    step2 = (
        1.0
        if (pred_c13 is not None and int(pred_c13) == int(target["step2_c13_sym_signals"]))
        else 0.0
    )

    pred_cip = pred.get("step3_cip_preserved")
    step3 = (
        1.0
        if (pred_cip is not None and bool(pred_cip) == bool(target["step3_cip_preserved"]))
        else 0.0
    )

    st_pred = standardize(expand_superatoms(pred.get("smiles")))
    st_true = standardize(target["step4_truth_smiles"])
    step4 = 1.0 if (st_pred.valid and st_pred.smiles == st_true.smiles) else 0.0

    # Strict prefix fidelity: credit final step fully only when intermediate deductions hold.
    rpf_mean = 0.25 * (step1 + step2 + step3 + step4)
    strict_chain = 1.0 if (step1 >= 0.5 and step2 == 1.0 and step3 == 1.0 and step4 == 1.0) else 0.0
    return {
        "step1_formula_hrms": step1,
        "step2_c13_symmetry": step2,
        "step3_cip_sanity": step3,
        "step4_stereo_exact": step4,
        "rpf_score": rpf_mean,
        "strict_chain_exact": strict_chain,
    }


def evaluate_t1_visual_predictions(
    items: Sequence[dict],
    predictions: dict[str, dict],
    *,
    compute_3d_energy_error: bool = False,
    seed: int = 0,
) -> dict:
    """Score T1 visual OCSR predictions separately by view (never averaging V0 with V1-V3)."""
    by_view: dict[str, list[dict]] = {}
    for item in items:
        by_view.setdefault(item["view"], []).append(item)

    report: dict[str, dict] = {}
    for view, v_items in sorted(by_view.items()):
        valid_vec, flat_vec, exact_vec, confs = [], [], [], []
        cats: dict[str, int] = {
            "correct": 0,
            "enantiomer": 0,
            "diastereomer": 0,
            "constitutional": 0,
            "invalid": 0,
        }
        abs_de: list[float] = []
        for it in v_items:
            p = predictions.get(it["task_id"], {})
            raw_smi = expand_superatoms(p.get("smiles"))
            st_p = standardize(raw_smi)
            st_t = standardize(it["truth_smiles"])
            cat = categorize(raw_smi, it["truth_smiles"])
            cats[cat] = cats.get(cat, 0) + 1
            valid_vec.append(1.0 if st_p.valid else 0.0)
            flat_vec.append(1.0 if (st_p.valid and flat_smiles(st_p) == flat_smiles(st_t)) else 0.0)
            exact_vec.append(1.0 if cat == "correct" else 0.0)
            confs.append(float(p.get("confidence", 1.0)))

            if (
                compute_3d_energy_error
                and st_p.valid
                and st_p.smiles
                and it.get("truth_energy_kcal") is not None
                and same_formula(st_p.smiles, it["truth_smiles"])
            ):
                sr = evaluate_3d_strain(st_p.smiles, n_conf=4, seed=seed, relax_steps=40)
                if sr.min_energy_kcal is not None:
                    abs_de.append(abs(sr.min_energy_kcal - float(it["truth_energy_kcal"])))

        bool_exact = [bool(x) for x in exact_vec]
        report[view] = {
            "validity": bootstrap_ci(valid_vec, seed=seed),
            "flat_connectivity_match": bootstrap_ci(flat_vec, seed=seed),
            "stereo_exact_match": bootstrap_ci(exact_vec, seed=seed),
            "silent_stereo_gap": float(np.mean(flat_vec) - np.mean(exact_vec)),
            "categories": cats,
            "ece": expected_calibration_error(confs, bool_exact),
            "aurc": area_under_risk_coverage(confs, bool_exact),
            "isomeric_abs_delta_e_kcal": bootstrap_ci(abs_de, seed=seed) if abs_de else None,
        }
    return report


def evaluate_t2_multimodal_predictions(
    items: Sequence[dict], predictions: dict[str, dict], *, seed: int = 0
) -> dict:
    """Score T2 multimodal SI disambiguation including Veritas-RPF and stratum breakdown."""
    exact_vec, flat_vec, rpf_vec, chain_vec, confs = [], [], [], [], []
    by_stratum: dict[str, list[float]] = {}
    for it in items:
        p = predictions.get(it["task_id"], {})
        smi = expand_superatoms(p.get("smiles"))
        if not smi and p.get("selected_index") is not None:
            idx = int(p["selected_index"])
            if 0 <= idx < len(it["candidates"]):
                smi = it["candidates"][idx]
        p_norm = dict(p, smiles=smi)
        st_p = standardize(smi)
        st_t = standardize(it["truth_smiles"])
        is_exact = bool(st_p.valid and st_p.smiles == st_t.smiles)
        is_flat = bool(st_p.valid and flat_smiles(st_p) == flat_smiles(st_t))
        rpf = score_veritas_rpf(p_norm, it["veritas_rpf_target"])

        exact_vec.append(1.0 if is_exact else 0.0)
        flat_vec.append(1.0 if is_flat else 0.0)
        rpf_vec.append(rpf["rpf_score"])
        chain_vec.append(rpf["strict_chain_exact"])
        confs.append(float(p.get("confidence", 1.0)))
        by_stratum.setdefault(it["stratum"], []).append(1.0 if is_exact else 0.0)

    bool_exact = [bool(x) for x in exact_vec]
    return {
        "stereo_exact_match": bootstrap_ci(exact_vec, seed=seed),
        "flat_connectivity_match": bootstrap_ci(flat_vec, seed=seed),
        "veritas_rpf": bootstrap_ci(rpf_vec, seed=seed),
        "strict_rpf_chain": bootstrap_ci(chain_vec, seed=seed),
        "ece": expected_calibration_error(confs, bool_exact),
        "aurc": area_under_risk_coverage(confs, bool_exact),
        "per_stratum_exact": {
            s: bootstrap_ci(vals, seed=seed) for s, vals in sorted(by_stratum.items())
        },
    }


def evaluate_t3_audit_predictions(
    items: Sequence[dict], predictions: dict[str, dict], *, seed: int = 0
) -> dict:
    """Score T3 silent stereochemical corruption audit (accept/reject + error category)."""
    acc_vec, cat_vec = [], []
    for it in items:
        p = predictions.get(it["task_id"], {})
        pred_accept = bool(p.get("accept", False))
        pred_cat = str(p.get("category") or "invalid").lower()
        acc_vec.append(1.0 if pred_accept == bool(it["expected_accept"]) else 0.0)
        cat_vec.append(1.0 if pred_cat == str(it["expected_category"]).lower() else 0.0)
    return {
        "accept_reject_accuracy": bootstrap_ci(acc_vec, seed=seed),
        "error_category_accuracy": bootstrap_ci(cat_vec, seed=seed),
    }


def evaluate_t4_thermo_predictions(
    items: Sequence[dict], predictions: dict[str, dict], *, seed: int = 0
) -> dict:
    """Score T4 3D thermodynamic stereoisomer ranking (lowest-energy hit & exact order)."""
    low_vec, order_vec = [], []
    for it in items:
        p = predictions.get(it["task_id"], {})
        ranked = [standardize(expand_superatoms(s)).smiles or s for s in p.get("ranked_smiles", ())]
        lowest = standardize(expand_superatoms(p.get("lowest_smiles"))).smiles
        if not lowest and ranked:
            lowest = ranked[0]
        truth_low = standardize(it["truth_lowest_smiles"]).smiles
        truth_rank = [standardize(s).smiles or s for s in it["truth_ranking"]]
        low_vec.append(1.0 if lowest == truth_low else 0.0)
        order_vec.append(1.0 if ranked == truth_rank else 0.0)
    return {
        "lowest_isomer_accuracy": bootstrap_ci(low_vec, seed=seed),
        "exact_ranking_accuracy": bootstrap_ci(order_vec, seed=seed),
    }


def evaluate_t5_parity_abstention(
    items: Sequence[dict], predictions: dict[str, dict], *, seed: int = 0
) -> dict:
    """Score T5 epistemic parity honesty: models MUST abstain on achiral-only enantiomer pairs."""
    abstain_vec = []
    for it in items:
        p = predictions.get(it["task_id"], {})
        act = str(p.get("action") or "").upper()
        abstain_vec.append(1.0 if act == "ABSTAIN_PARITY" else 0.0)
    return {
        "epistemic_abstention_rate": bootstrap_ci(abstain_vec, seed=seed),
        "hallucinated_chirality_rate": float(1.0 - np.mean(abstain_vec)) if abstain_vec else 0.0,
    }
