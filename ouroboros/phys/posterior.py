"""Symmetry-complete Bayesian posterior OCSR reranking, ablations (A0-A6) & epistemic calibration.

Combines SO(2) reflection-free visual log-probabilities with E(3)-invariant 3D strain, HRMS,
and 13C/1H NMR inversion, enforcing parity abstention when achiral observables cannot resolve
pure enantiomers with tied visual priors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from ouroboros.decode.rerank import normalized
from ouroboros.eval.standardize import flat_smiles, standardize
from ouroboros.geometry.propagation import mirror_smiles
from ouroboros.phys.epimers import (
    enumerate_targeted_epimers,
    expand_superatoms,
    stereo_hamming_distance,
)
from ouroboros.phys.hrms import score_hrms
from ouroboros.phys.nmr import H1Multiplet, NMRForwardPredictor, score_nmr
from ouroboros.phys.strain import StrainResult, evaluate_3d_strain


@dataclass(frozen=True)
class SIObservables:
    formula: str | None = None
    hrms_obs_mz: float | None = None
    hrms_adduct: str | None = None
    c13_peaks_ppm: tuple[float, ...] = ()
    dept_counts: dict[str, int] | None = None
    h1_multiplets: tuple[H1Multiplet, ...] = ()
    optical_rotation_deg: float | None = None


@dataclass(frozen=True)
class PosteriorWeights:
    alpha_visual: float = 1.0
    beta_strain: float = 0.35
    gamma_hrms: float = 1.0
    gamma_nmr_c13: float = 1.0
    gamma_nmr_h1_j: float = 1.0
    hrms_window_ppm: float = 5.0
    parity_margin_threshold: float = 0.05
    augment_epimers: bool = False
    epimer_visual_backoff: float = -0.65


ABLATION_PRESETS: dict[str, PosteriorWeights] = {
    "A0_visual_top1": PosteriorWeights(
        alpha_visual=1.0, beta_strain=0.0, gamma_hrms=0.0, gamma_nmr_c13=0.0, gamma_nmr_h1_j=0.0
    ),
    "A1_rdkit_valid": PosteriorWeights(
        alpha_visual=1.0, beta_strain=0.0, gamma_hrms=0.0, gamma_nmr_c13=0.0, gamma_nmr_h1_j=0.0
    ),
    "A2_hrms": PosteriorWeights(
        alpha_visual=1.0, beta_strain=0.0, gamma_hrms=1.0, gamma_nmr_c13=0.0, gamma_nmr_h1_j=0.0
    ),
    "A3_strain_3d": PosteriorWeights(
        alpha_visual=1.0, beta_strain=0.35, gamma_hrms=1.0, gamma_nmr_c13=0.0, gamma_nmr_h1_j=0.0
    ),
    "A4_nmr_13c": PosteriorWeights(
        alpha_visual=1.0, beta_strain=0.35, gamma_hrms=1.0, gamma_nmr_c13=1.0, gamma_nmr_h1_j=0.0
    ),
    "A5_nmr_13c_1h_j": PosteriorWeights(
        alpha_visual=1.0, beta_strain=0.35, gamma_hrms=1.0, gamma_nmr_c13=1.0, gamma_nmr_h1_j=1.0
    ),
    "A6_full_posterior": PosteriorWeights(
        alpha_visual=1.0,
        beta_strain=0.35,
        gamma_hrms=1.0,
        gamma_nmr_c13=1.0,
        gamma_nmr_h1_j=1.0,
        augment_epimers=True,
    ),
}


@dataclass
class CandidateScore:
    raw_smiles: str
    canonical_smiles: str | None
    valid: bool
    from_epimer_expansion: bool
    visual_score: float
    hrms_ppm: float | None = None
    hrms_adduct: str | None = None
    hrms_loss: float = 0.0
    min_energy_kcal: float | None = None
    rel_isomer_strain_kcal: float = 0.0
    strain_loss: float = 0.0
    cip_preserved: bool = True
    has_steric_clash: bool = False
    c13_sym_count: int | None = None
    c13_mae_ppm: float | None = None
    h1_mae_ppm: float | None = None
    j_hh_mae_hz: float | None = None
    nmr_loss: float = 0.0
    log_posterior: float = float("-inf")
    posterior_prob: float = 0.0


@dataclass
class RerankDecision:
    selected_smiles: str
    canonical_smiles: str | None
    posterior_prob: float
    margin: float
    epistemic_status: str  # "confident" | "abstain_enantiomer_underdetermined" | "low_confidence"
    ablation: str
    candidates: list[CandidateScore] = field(default_factory=list)


def _are_enantiomers(s1: str | None, s2: str | None) -> bool:
    if not s1 or not s2 or s1 == s2:
        return False
    m1, m2 = standardize(s1), standardize(s2)
    if not m1.valid or not m2.valid or flat_smiles(m1) != flat_smiles(m2):
        return False
    mir = Chem.CanonSmiles(mirror_smiles(m1.smiles))
    return mir == m2.smiles and mir != m1.smiles


def rerank_candidates(
    candidates: Sequence[tuple[str, float]],
    observables: SIObservables | None = None,
    *,
    ablation: str = "A6_full_posterior",
    weights: PosteriorWeights | None = None,
    normalize_length: bool = False,
    n_conf: int = 8,
    seed: int = 0,
    calc=None,
    nmr_predictor: NMRForwardPredictor | None = None,
) -> RerankDecision:
    """Rank (smiles, visual_logprob) candidates under the joint Bayesian physics+NMR posterior."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    w = weights or ABLATION_PRESETS[ablation]
    obs = observables or SIObservables()

    if ablation == "A0_visual_top1":
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        top_raw, top_lp = ordered[0]
        st = standardize(expand_superatoms(top_raw))
        cs = [
            CandidateScore(
                raw_smiles=s,
                canonical_smiles=standardize(expand_superatoms(s)).smiles,
                valid=standardize(expand_superatoms(s)).valid,
                from_epimer_expansion=False,
                visual_score=float(lp),
                log_posterior=float(lp),
            )
            for s, lp in ordered
        ]
        probs = _softmax([c.log_posterior for c in cs])
        for c, p in zip(cs, probs, strict=True):
            c.posterior_prob = float(p)
        margin = float(cs[0].log_posterior - cs[1].log_posterior) if len(cs) > 1 else float("inf")
        return RerankDecision(
            selected_smiles=st.smiles or top_raw,
            canonical_smiles=st.smiles,
            posterior_prob=float(probs[0]),
            margin=margin,
            epistemic_status="confident",
            ablation=ablation,
            candidates=cs,
        )

    # Deduplicate by standardized canonical SMILES while keeping best visual score.
    pooled: dict[str, tuple[str, float, bool]] = {}
    invalid_entries: list[tuple[str, float]] = []
    for raw, lp in candidates:
        v_score = normalized(raw, lp) if normalize_length else float(lp)
        if not np.isfinite(v_score):
            v_score = float(lp)
        expanded = expand_superatoms(raw)
        st = standardize(expanded)
        if not st.valid or st.smiles is None:
            invalid_entries.append((raw, v_score))
            continue
        prev = pooled.get(st.smiles)
        if prev is None or v_score > prev[1]:
            pooled[st.smiles] = (raw, v_score, False)

    if w.augment_epimers and pooled:
        ordered_canon = [
            k for k, _ in sorted(pooled.items(), key=lambda kv: kv[1][1], reverse=True)
        ]
        # Prefer HRMS-compatible scaffolds before enumerating epimers (zero 3D cost pre-filter).
        hrms_valid_seeds = [
            k
            for k in ordered_canon
            if w.gamma_hrms <= 0.0
            or score_hrms(
                k,
                obs.hrms_obs_mz,
                adduct=obs.hrms_adduct,
                expected_formula=obs.formula,
                window_ppm=w.hrms_window_ppm,
            ).within_window
        ]
        seeds = hrms_valid_seeds or ordered_canon
        for ep_smi in enumerate_targeted_epimers(seeds, max_centers=3, max_epimers=8):
            if ep_smi not in pooled:
                best_ep_v = max(
                    pooled[s][1]
                    + w.epimer_visual_backoff * max(1, stereo_hamming_distance(s, ep_smi))
                    for s in seeds[:3]
                )
                pooled[ep_smi] = (ep_smi, best_ep_v, True)

    if not pooled:
        raw, v_score = max(invalid_entries, key=lambda x: x[1])
        only = CandidateScore(
            raw_smiles=raw,
            canonical_smiles=None,
            valid=False,
            from_epimer_expansion=False,
            visual_score=v_score,
            log_posterior=v_score,
            posterior_prob=1.0,
        )
        return RerankDecision(
            selected_smiles=raw,
            canonical_smiles=None,
            posterior_prob=1.0,
            margin=0.0,
            epistemic_status="low_confidence",
            ablation=ablation,
            candidates=[only],
        )

    need_strain = (w.beta_strain > 0.0) or (w.gamma_nmr_c13 > 0.0) or (w.gamma_nmr_h1_j > 0.0)
    strains: dict[str, StrainResult] = {}
    if need_strain:
        for canon in pooled:
            strains[canon] = evaluate_3d_strain(canon, n_conf=n_conf, seed=seed, calc=calc)

    # Compute stereoisomer-family energy floors (continuous anti-Bredt / epimer strain relative to
    # the lowest-energy isomer sharing the same flat connectivity).
    family_floor: dict[str, float] = {}
    for canon, sr in strains.items():
        if sr.embed_ok and sr.min_energy_kcal is not None:
            fk = flat_smiles(standardize(canon)) or canon
            family_floor[fk] = min(family_floor.get(fk, float("inf")), sr.min_energy_kcal)

    scored: list[CandidateScore] = []
    for canon, (raw, v_score, is_ep) in pooled.items():
        h_res = (
            score_hrms(
                canon,
                obs.hrms_obs_mz,
                adduct=obs.hrms_adduct,
                expected_formula=obs.formula,
                window_ppm=w.hrms_window_ppm,
            )
            if w.gamma_hrms > 0.0
            else None
        )
        h_loss = h_res.loss if h_res is not None else 0.0

        sr = strains.get(canon)
        rel_iso = 0.0
        s_loss = 0.0
        if sr is not None and w.beta_strain > 0.0:
            fk = flat_smiles(standardize(canon)) or canon
            if sr.embed_ok and sr.min_energy_kcal is not None and fk in family_floor:
                rel_iso = max(0.0, sr.min_energy_kcal - family_floor[fk])
            s_loss = rel_iso + sr.intrinsic_penalty_kcal

        n_loss = 0.0
        n_res = None
        if w.gamma_nmr_c13 > 0.0 or w.gamma_nmr_h1_j > 0.0:
            n_res = score_nmr(
                canon,
                obs_c13_ppm=obs.c13_peaks_ppm if w.gamma_nmr_c13 > 0.0 else (),
                obs_dept_counts=obs.dept_counts if w.gamma_nmr_c13 > 0.0 else None,
                obs_h1_multiplets=obs.h1_multiplets if w.gamma_nmr_h1_j > 0.0 else (),
                strain=sr,
                predictor=nmr_predictor,
                use_h1_and_j=(w.gamma_nmr_h1_j > 0.0),
            )
            c13_part = 0.0
            if w.gamma_nmr_c13 > 0.0:
                c13_part = n_res.signal_count_penalty + n_res.dept_penalty
                if n_res.c13_mae_ppm is not None:
                    c13_part += 0.45 * n_res.c13_mae_ppm + 0.35 * (n_res.c13_max_err_ppm or 0.0)
            h1_part = 0.0
            if w.gamma_nmr_h1_j > 0.0:
                if n_res.h1_mae_ppm is not None:
                    h1_part += 1.4 * n_res.h1_mae_ppm
                if n_res.j_hh_mae_hz is not None:
                    h1_part += 0.35 * n_res.j_hh_mae_hz
            n_loss = w.gamma_nmr_c13 * c13_part + w.gamma_nmr_h1_j * h1_part

        log_post = (
            w.alpha_visual * v_score
            - w.beta_strain * s_loss
            - w.gamma_hrms * h_loss
            - n_loss
        )
        scored.append(
            CandidateScore(
                raw_smiles=raw,
                canonical_smiles=canon,
                valid=True,
                from_epimer_expansion=is_ep,
                visual_score=v_score,
                hrms_ppm=h_res.ppm_error if h_res else None,
                hrms_adduct=h_res.best_adduct if h_res else None,
                hrms_loss=h_loss,
                min_energy_kcal=sr.min_energy_kcal if sr else None,
                rel_isomer_strain_kcal=rel_iso,
                strain_loss=s_loss,
                cip_preserved=sr.cip_preserved if sr else True,
                has_steric_clash=sr.has_steric_clash if sr else False,
                c13_sym_count=n_res.c13_sym_count if n_res else None,
                c13_mae_ppm=n_res.c13_mae_ppm if n_res else None,
                h1_mae_ppm=n_res.h1_mae_ppm if n_res else None,
                j_hh_mae_hz=n_res.j_hh_mae_hz if n_res else None,
                nmr_loss=n_loss,
                log_posterior=float(log_post),
            )
        )

    scored.sort(key=lambda c: (c.log_posterior, c.visual_score), reverse=True)
    probs = _softmax([c.log_posterior for c in scored])
    for c, p in zip(scored, probs, strict=True):
        c.posterior_prob = float(p)

    top = scored[0]
    margin = (
        float(top.log_posterior - scored[1].log_posterior) if len(scored) > 1 else float("inf")
    )
    status = "confident"
    if (
        len(scored) > 1
        and _are_enantiomers(top.canonical_smiles, scored[1].canonical_smiles)
        and abs(top.visual_score - scored[1].visual_score) < w.parity_margin_threshold
        and obs.optical_rotation_deg is None
    ):
        status = "abstain_enantiomer_underdetermined"
    elif top.posterior_prob < 0.30:
        status = "low_confidence"

    return RerankDecision(
        selected_smiles=top.canonical_smiles or top.raw_smiles,
        canonical_smiles=top.canonical_smiles,
        posterior_prob=top.posterior_prob,
        margin=margin,
        epistemic_status=status,
        ablation=ablation,
        candidates=scored,
    )


def _softmax(logits: Sequence[float]) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(len(arr), 1.0 / len(arr))
    m = float(np.max(arr[finite]))
    exps = np.where(finite, np.exp(np.clip(arr - m, -80.0, 0.0)), 0.0)
    total = float(exps.sum())
    return exps / total if total > 0 else np.full(len(arr), 1.0 / len(arr))


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Standard ECE across [0, 1] probability bins."""
    if not confidences:
        return 0.0
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    acc = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        if np.any(mask):
            ece += (mask.sum() / n) * abs(float(acc[mask].mean() - conf[mask].mean()))
    return float(ece)


def area_under_risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Area Under Risk-Coverage Curve (AURC; lower = better selective abstention)."""
    if not confidences:
        return 0.0
    order = np.argsort(-np.asarray(confidences, dtype=float), kind="mergesort")
    err = 1.0 - np.asarray(correct, dtype=float)[order]
    cum_risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(np.mean(cum_risk))
