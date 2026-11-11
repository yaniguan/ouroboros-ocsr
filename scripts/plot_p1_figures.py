"""Generate all 5 publication-grade vector PDF figures for Paper P1.

Reads deterministic JSON receipts from ``benchmarks/p1_chemveritas/`` and writes vector
PDF figures under ``benchmarks/p1_chemveritas/figures/`` with strict view segregation
(V0 never averaged with V1-V3) and 1,000-sample bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path("benchmarks/p1_chemveritas")
FIG_DIR = BENCH_DIR / "figures"

# Okabe-Ito colorblind-safe publication palette
PAL = {
    "navy": "#1f4e79",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "teal": "#009E73",
    "amber": "#E69F00",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "slate": "#59656d",
    "gray": "#8c969e",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _err_asym(ci_dict: dict) -> tuple[float, float]:
    m = float(ci_dict["mean"])
    return max(0.0, m - float(ci_dict["ci_low"])), max(0.0, float(ci_dict["ci_high"]) - m)


def plot_fig1_closed_loop(ocsr: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    views = [
        "V0_clean",
        "V0_rot90",
        "V1_rotated_offgrid",
        "V2_crowded_si_scan",
        "V3_condensed_layout",
    ]
    v_labels = [
        "V0 Clean\n(0 deg)",
        "V0 Rot90\n(90 deg)",
        "V1 Off-Grid\n(37-341 deg)",
        "V2 Crowded\nSI Scan",
        "V3 Condensed\nSuperatom",
    ]
    arm_data = ocsr["arms"]["Ouroboros-AC"]

    abls = [
        ("A0_visual_top1", "A0: Visual Top-1", PAL["slate"]),
        ("A2_hrms", "A2: + HRMS (5 ppm)", PAL["amber"]),
        ("A3_strain_3d", "A3: + MACE 3D Strain", PAL["sky"]),
        ("A6_full_posterior", "A6: + 13C/1H NMR + Epimers", PAL["teal"]),
    ]
    x = np.arange(len(views))
    width = 0.19
    for idx, (abl_key, label, color) in enumerate(abls):
        means, lo_e, hi_e = [], [], []
        for v in views:
            ci = arm_data[v][abl_key]["stereo_exact_match"]
            means.append(100.0 * float(ci["mean"]))
            lo, hi = _err_asym(ci)
            lo_e.append(100.0 * lo)
            hi_e.append(100.0 * hi)
        offset = (idx - 1.5) * width
        ax1.bar(
            x + offset,
            means,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            yerr=[lo_e, hi_e],
            capsize=2.2,
            error_kw={"elinewidth": 0.8, "ecolor": "#222222"},
            label=label,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(v_labels)
    ax1.set_ylabel("Unconditional Exact Stereoisomer (%) [95% CI]")
    ax1.set_ylim(0, 28)
    ax1.set_title("(a) End-to-End Ouroboros-AC (Beam-5) Closed-Loop Repair (n=100/view)")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): Stereochemical resolution among flat-connectivity-correct scaffolds (A0 vs A6)
    bar_specs = [
        ("V0_clean", "A0_visual_top1", "V0\nA0"),
        ("V0_clean", "A6_full_posterior", "V0\nA6"),
        ("V1_rotated_offgrid", "A0_visual_top1", "V1\nA0"),
        ("V1_rotated_offgrid", "A6_full_posterior", "V1\nA6"),
        ("V2_crowded_si_scan", "A0_visual_top1", "V2\nA0"),
        ("V2_crowded_si_scan", "A6_full_posterior", "V2\nA6"),
        ("V3_condensed_layout", "A0_visual_top1", "V3\nA0"),
        ("V3_condensed_layout", "A6_full_posterior", "V3\nA6"),
    ]
    cat_order = [
        ("correct", "Exact Stereoisomer", PAL["teal"]),
        ("enantiomer", "Enantiomer (Achiral Parity)", PAL["sky"]),
        ("diastereomer", "Silent Diastereomer Flip", PAL["vermilion"]),
    ]
    xb = np.arange(len(bar_specs))
    bottoms = np.zeros(len(bar_specs))
    flat_totals = []
    for v_k, a_k, _ in bar_specs:
        cats = arm_data[v_k][a_k]["categories"]
        flat_totals.append(
            cats.get("correct", 0) + cats.get("enantiomer", 0) + cats.get("diastereomer", 0)
        )

    for c_key, c_lbl, c_col in cat_order:
        vals = []
        for idx_b, (v_k, a_k, _) in enumerate(bar_specs):
            cats = arm_data[v_k][a_k]["categories"]
            tot = max(1, flat_totals[idx_b])
            vals.append(100.0 * cats.get(c_key, 0) / tot)
        vals_arr = np.asarray(vals)
        ax2.bar(
            xb,
            vals_arr,
            bottom=bottoms,
            width=0.62,
            color=c_col,
            edgecolor="white",
            linewidth=0.5,
            label=c_lbl,
        )
        bottoms += vals_arr

    for idx_b, n_f in enumerate(flat_totals):
        ax2.text(idx_b, 101.8, f"n={n_f}", ha="center", va="bottom", fontsize=6.8, color="#222222")

    ax2.set_xticks(xb)
    ax2.set_xticklabels([b[2] for b in bar_specs])
    ax2.set_ylabel("Share of Flat-Correct Predictions (%)")
    ax2.set_ylim(0, 114)
    ax2.set_title("(b) Diastereomer Corruption Elimination (Flat-Matched)")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92, ncol=1)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_closed_loop_neural_ocsr.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_rotation_equivariance(ocsr: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))

    arms = ["Ouroboros-A", "Ouroboros-B", "Ouroboros-C", "Ouroboros-AC"]
    arm_labels = [
        "Arm A\n(ResNet)",
        "Arm B\n(Aug ResNet)",
        "Arm C\n(C8 Steerable)",
        "Arm AC\n(Ensemble)",
    ]
    x = np.arange(len(arms))
    w = 0.24

    series = [
        ("rot90_inv", "90 deg Canonical SMILES Invariance", PAL["navy"]),
        ("V0_clean", "V0 Clean Flat Skeleton (0 deg)", PAL["sky"]),
        ("V0_rot90", "V0 Rot90 Flat Skeleton (90 deg)", PAL["vermilion"]),
    ]
    for idx, (s_key, lbl, col) in enumerate(series):
        vals, lo_e, hi_e = [], [], []
        for arm in arms:
            if s_key == "rot90_inv":
                ci = ocsr["rot90_c8_invariance"][arm]["canonical_smiles_invariance"]
            else:
                ci = ocsr["arms"][arm][s_key]["A0_visual_top1"]["flat_connectivity_match"]
            vals.append(100.0 * float(ci["mean"]))
            lo, hi = _err_asym(ci)
            lo_e.append(100.0 * lo)
            hi_e.append(100.0 * hi)
        ax1.bar(
            x + (idx - 1.0) * w,
            vals,
            width=w,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            yerr=[lo_e, hi_e],
            capsize=2.0,
            error_kw={"elinewidth": 0.8},
            label=lbl,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(arm_labels)
    ax1.set_ylabel("Invariance / Skeleton Match (%) [95% CI]")
    ax1.set_ylim(0, 108)
    ax1.set_title("(a) Planar C8 Rotation Invariance vs ResNet Collapse (n=100)")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): V0_clean & Off-grid V1 progression across A0-A6 for Ouroboros-C & AC
    ladder = [
        "A0_visual_top1",
        "A1_rdkit_valid",
        "A2_hrms",
        "A3_strain_3d",
        "A4_nmr_13c",
        "A5_nmr_13c_1h_j",
        "A6_full_posterior",
    ]
    short_l = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    xl = np.arange(len(ladder))
    lines = [
        (
            "Ouroboros-AC",
            "V0_clean",
            "Arm AC | V0 Clean (0 deg)",
            PAL["navy"],
            "s",
            "-",
        ),
        (
            "Ouroboros-C",
            "V1_rotated_offgrid",
            "Arm C | V1 Off-Grid (37-341 deg)",
            PAL["blue"],
            "o",
            "-",
        ),
        (
            "Ouroboros-AC",
            "V1_rotated_offgrid",
            "Arm AC | V1 Off-Grid (37-341 deg)",
            PAL["teal"],
            "^",
            "--",
        ),
        (
            "Ouroboros-AC",
            "V3_condensed_layout",
            "Arm AC | V3 Superatom Layout",
            PAL["amber"],
            "D",
            ":",
        ),
    ]
    for arm, vw, lbl, col, mk, ls in lines:
        ys = [
            100.0 * float(ocsr["arms"][arm][vw][ab]["stereo_exact_match"]["mean"]) for ab in ladder
        ]
        ax2.plot(
            xl, ys, marker=mk, linestyle=ls, color=col, linewidth=1.8, markersize=5.2, label=lbl
        )

    ax2.set_xticks(xl)
    ax2.set_xticklabels(short_l)
    ax2.set_xlabel("Bayesian Physics & Spectroscopy Ablation Stage")
    ax2.set_ylabel("Exact Stereoisomer Match (%)")
    ax2.set_ylim(0, 22)
    ax2.set_title("(b) Stepwise Closed-Loop Stereoisomer Recovery (n=100/view)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax2.grid(True, linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_rotation_equivariance_views.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_mace_strain_audit(audit: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    strata = [
        "S1_bridged_caged",
        "S2_dense_stereo",
        "S3_strained_bredt",
        "S4_superatom_si",
        "S5_medchem_bioisosteres",
    ]
    s_labels = [
        "S1 Bridged\nCages",
        "S2 Dense\nStereo",
        "S3 Strained\nBredt",
        "S4 Superatom\nSI",
        "S5 MedChem\nBioisostere",
    ]
    by_s: dict[str, list[float]] = {s: [] for s in strata}
    for rec in audit["pairs"]:
        by_s[rec["stratum"]].append(float(rec["delta_e_mace_kcal"]))

    data_s = [by_s[s] for s in strata]
    pos = np.arange(len(strata))
    bp = ax1.boxplot(
        data_s,
        positions=pos,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.6},
    )
    cols = [PAL["navy"], PAL["blue"], PAL["vermilion"], PAL["amber"], PAL["teal"]]
    for patch, col in zip(bp["boxes"], cols, strict=True):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#222222")

    rng = np.random.default_rng(7)
    for idx, vals in enumerate(data_s):
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax1.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            color=cols[idx],
            edgecolor="#111111",
            linewidth=0.4,
            s=26,
            zorder=3,
        )

    ax1.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(s_labels)
    ax1.set_ylabel(r"$\Delta E_{\mathrm{strain}}^{\mathrm{MACE}}$ (Corrupt $-$ Isolable, kcal/mol)")
    ax1.set_title("(a) MACE-OFF23-medium 3D Strain Penalty by Stratum (N=50 Pairs)")
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): ROC curves (MACE-OFF23-medium vs MMFF94s)
    rm = audit["roc_mace"]
    rf = audit["roc_mmff"]
    ax2.plot(
        rm["fpr"],
        rm["tpr"],
        color=PAL["teal"],
        linewidth=2.2,
        label=f"MACE-OFF23-medium (AUC = {rm['auc']:.3f})",
    )
    ax2.plot(
        rf["fpr"],
        rf["tpr"],
        color=PAL["amber"],
        linestyle="--",
        linewidth=1.8,
        label=f"MMFF94s Force Field (AUC = {rf['auc']:.3f})",
    )
    ax2.plot(
        [0, 1],
        [0, 1],
        color=PAL["gray"],
        linestyle=":",
        linewidth=1.0,
        label="Random (AUC = 0.500)",
    )
    ax2.set_xlabel("False Positive Rate (Isolable Flagged)")
    ax2.set_ylabel("True Positive Rate (Silent Corruption Caught)")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.03)
    ax2.set_title("(b) 3D Strain Discrimination of Silent Corruptions")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92)
    ax2.grid(True, linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_mace_strain_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_gold600_ablation(g600: dict) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 3.65))

    abls = [
        "A0_visual_top1",
        "A1_rdkit_valid",
        "A2_hrms",
        "A3_strain_3d",
        "A4_nmr_13c",
        "A5_nmr_13c_1h_j",
        "A6_full_posterior",
    ]
    short = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    x = np.arange(len(abls))
    rep = g600["ablations"]

    t2_m = [100.0 * rep[a]["T2"]["stereo_exact_match"]["mean"] for a in abls]
    rpf_m = [100.0 * rep[a]["T2"]["veritas_rpf"]["mean"] for a in abls]
    t3_m = [100.0 * rep[a]["T3"]["accept_reject_accuracy"]["mean"] for a in abls]
    t4_m = [100.0 * rep[a]["T4"]["lowest_isomer_accuracy"]["mean"] for a in abls]

    ax1.plot(x, t2_m, "o-", color=PAL["teal"], linewidth=2.0, label="T2 Stereo Exact (n=600)")
    ax1.plot(x, rpf_m, "s--", color=PAL["blue"], linewidth=1.8, label="T2 Veritas-RPF (n=600)")
    ax1.plot(x, t3_m, "^-", color=PAL["amber"], linewidth=1.8, label="T3 Corruption Audit (n=600)")
    ax1.plot(
        x,
        t4_m,
        "D:",
        color=PAL["purple"],
        linewidth=1.8,
        label="T4 Thermo Lowest (n="
        + str(rep["A6_full_posterior"]["T4"]["lowest_isomer_accuracy"]["n"])
        + ")",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short)
    ax1.set_xlabel("Ablation Preset")
    ax1.set_ylabel("Accuracy / Score (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("(a) Gold-600 Multi-Task Progression")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # Panel (b): Veritas-RPF 4-step breakdown across A0, A2, A4, A6
    step_keys = [
        "step1_formula_hrms",
        "step2_c13_symmetry",
        "step3_cip_sanity",
        "step4_stereo_exact",
    ]
    step_lbls = [
        "S1: Formula\n& HRMS",
        "S2: 13C\nSymmetry",
        "S3: 3D CIP\nSanity",
        "S4: Exact\nStereo",
    ]
    xs = np.arange(len(step_keys))
    ws = 0.20
    sel_abls = [
        ("A0_visual_top1", "A0 Visual", PAL["slate"]),
        ("A2_hrms", "A2 +HRMS", PAL["amber"]),
        ("A4_nmr_13c", "A4 +Strain+13C", PAL["sky"]),
        ("A6_full_posterior", "A6 Full Posterior", PAL["teal"]),
    ]
    for idx, (ak, albl, acol) in enumerate(sel_abls):
        vals = [100.0 * rep[ak]["T2"]["rpf_steps"][sk] for sk in step_keys]
        ax2.bar(
            xs + (idx - 1.5) * ws,
            vals,
            width=ws,
            color=acol,
            edgecolor="white",
            linewidth=0.5,
            label=albl,
        )
    ax2.set_xticks(xs)
    ax2.set_xticklabels(step_lbls)
    ax2.set_ylabel("Checkpoint Fidelity (%)")
    ax2.set_ylim(0, 108)
    ax2.set_title("(b) Veritas-RPF 4-Step Reasoning Fidelity")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.92)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (c): Calibration (ECE & AURC) + T5 Epistemic Parity Abstention
    ece_v = [100.0 * float(rep[a]["T2"]["ece"]) for a in abls]
    aurc_v = [100.0 * float(rep[a]["T2"]["aurc"]) for a in abls]
    t5_abs = [100.0 * float(rep[a]["T5"]["epistemic_abstention_rate"]["mean"]) for a in abls]

    ax3.plot(
        x, ece_v, "o-", color=PAL["vermilion"], linewidth=1.9, label="T2 ECE (%) [lower better]"
    )
    ax3.plot(
        x, aurc_v, "s--", color=PAL["amber"], linewidth=1.9, label="T2 AURC (%) [lower better]"
    )
    ax3.plot(x, t5_abs, "^-", color=PAL["navy"], linewidth=2.0, label="T5 Parity Abstention (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short)
    ax3.set_xlabel("Ablation Preset")
    ax3.set_ylabel("Calibration Metric / Abstention (%)")
    ax3.set_ylim(-3, 106)
    ax3.set_title("(c) Calibration & SO(2)/O(3) Parity Honesty")
    ax3.legend(loc="center right", frameon=True, framealpha=0.92)
    ax3.grid(True, linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_gold600_ablation_rpf.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig5_tier2_exp_nmr(t2_exp: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))

    abls = [
        "A0_visual_top1",
        "A1_rdkit_valid",
        "A2_hrms",
        "A3_strain_3d",
        "A4_nmr_13c",
        "A5_nmr_13c_1h_j",
        "A6_full_posterior",
    ]
    short = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    x = np.arange(len(abls))
    w = 0.34

    for idx, (src_key, lbl, col) in enumerate(
        [
            ("mmff94s", "MMFF94s + 3D NMR + Karplus", PAL["blue"]),
            ("mace_off23_medium", "MACE-OFF23-medium//MMFF94s + 3D NMR", PAL["teal"]),
        ]
    ):
        means, lo_e, hi_e = [], [], []
        for a in abls:
            ci = t2_exp[src_key]["ablations"][a]["exact_stereoisomer_accuracy"]
            means.append(100.0 * float(ci["mean"]))
            lo, hi = _err_asym(ci)
            lo_e.append(100.0 * lo)
            hi_e.append(100.0 * hi)
        ax1.bar(
            x + (idx - 0.5) * w,
            means,
            width=w,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            yerr=[lo_e, hi_e],
            capsize=2.3,
            error_kw={"elinewidth": 0.85},
            label=lbl,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(short)
    ax1.set_xlabel("Bayesian Physics & Spectroscopy Ablation")
    ax1.set_ylabel("Exact Literature Isomer Recovery (%) [95% CI]")
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Authentic Published 13C/1H/HRMS Panel (N=30, 15 Pairs)")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): Experimental vs Boltzmann-predicted 13C parity plot
    pts = t2_exp["c13_parity_points"]
    exp_v = np.asarray([p["exp_ppm"] for p in pts], dtype=float)
    pred_v = np.asarray([p["pred_ppm"] for p in pts], dtype=float)
    st = t2_exp["c13_forward_stats"]

    ax2.scatter(
        exp_v,
        pred_v,
        s=20,
        alpha=0.78,
        color=PAL["navy"],
        edgecolor="white",
        linewidth=0.3,
        label=f"Symmetry-unique 13C signals (n={len(pts)})",
    )
    lims = [min(exp_v.min(), pred_v.min()) - 5.0, max(exp_v.max(), pred_v.max()) + 5.0]
    ax2.plot(lims, lims, "--", color=PAL["vermilion"], linewidth=1.3, label="Parity (y = x)")
    ax2.fill_between(
        lims,
        [z - 5.0 for z in lims],
        [z + 5.0 for z in lims],
        color=PAL["sky"],
        alpha=0.15,
        label=r"$\pm 5$ ppm band",
    )
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(r"Experimental $\delta(^{13}\mathrm{C})$ in $\mathrm{CDCl}_3$ (ppm)")
    ax2.set_ylabel(r"3D Boltzmann-Predicted $\delta(^{13}\mathrm{C})$ (ppm)")
    ax2.set_title(f"(b) 13C Forward Accuracy: MAE={st['mae_ppm']:.2f} ppm, r={st['pearson_r']:.3f}")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax2.grid(True, linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_tier2_exp_nmr.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    _setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t2_exp = json.loads((BENCH_DIR / "tier2_exp_nmr_validation.json").read_text(encoding="utf-8"))
    audit = json.loads((BENCH_DIR / "mace_strain_audit.json").read_text(encoding="utf-8"))
    ocsr = json.loads((BENCH_DIR / "neural_ocsr_closed_loop.json").read_text(encoding="utf-8"))
    g600 = json.loads((BENCH_DIR / "gold600_ablation_receipt.json").read_text(encoding="utf-8"))

    plot_fig1_closed_loop(ocsr)
    plot_fig2_rotation_equivariance(ocsr)
    plot_fig3_mace_strain_audit(audit)
    plot_fig4_gold600_ablation(g600)
    plot_fig5_tier2_exp_nmr(t2_exp)
    print(f"Generated 5 vector PDF figures under {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
