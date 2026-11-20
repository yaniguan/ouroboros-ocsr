"""Check that numbers quoted in papers/p1/main.tex match the committed P1 receipts."""

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "p1_chemveritas"
TEX = ROOT / "papers" / "p1" / "main.tex"

pytestmark = pytest.mark.skipif(
    not (TEX.exists() and (BENCH / "gold600_summary_table.csv").exists()),
    reason="P1 manuscript or receipts not present",
)

# Gold-600 table row label in main.tex -> ablation name in gold600_summary_table.csv
ROWS = {
    "A0": "A0_visual_top1",
    "A1": "A1_rdkit_valid",
    "A2": "A2_hrms",
    "A3": "A3_strain_3d",
    "A4": "A4_nmr_13c",
    "A5": "A5_nmr_13c_1h_j",
    "A6": "A6_full_posterior",
}
# table columns after the T2 exact-match cell
COLS = ["T2_veritas_rpf", "T2_strict_chain", "T2_ece", "T2_aurc",
        "T3_audit_acc", "T4_lowest_acc", "T5_abstain_rate"]


def pct(x):
    return f"{100 * float(x):.1f}"


def _tex():
    return TEX.read_text(encoding="utf-8")


def _gold600_rows():
    with open(BENCH / "gold600_summary_table.csv", encoding="utf-8") as f:
        return {r["ablation"]: r for r in csv.DictReader(f)}


def _gold600_table(tex):
    """Return the text of the Gold-600 table (the Tier-2 table reuses the A0-A6 row labels)."""
    start = tex.index(r"\label{tab:gold600}")
    return tex[start:tex.index(r"\end{table", start)]


def _table_cells(tex, label):
    """Return the numeric cells of the Gold-600 table row that starts with \\texttt{label}:."""
    tex = _gold600_table(tex)
    m = re.search(r"\\texttt\{" + label + r"\}:(?:[^&\\]|\\[^\\])*&(.*?)\\\\", tex)
    assert m, f"row {label} not found in main.tex"
    cells = re.split(r"(?<!\\)&", m.group(1))
    return [re.sub(r"\\textbf\{|\}", "", c).strip() for c in cells]


@pytest.mark.parametrize("label", sorted(ROWS))
def test_gold600_table_matches_csv(label):
    row = _gold600_rows()[ROWS[label]]
    cells = _table_cells(_tex(), label)
    exact = f"{pct(row['T2_stereo_exact'])} ({pct(row['T2_ci_low'])}--{pct(row['T2_ci_high'])})"
    assert cells[0] == exact, f"{label} T2 exact: tex {cells[0]!r} vs receipt {exact!r}"
    for col, cell in zip(COLS, cells[1:]):
        # the CSV keeps 4 digits, so allow half a unit of rounding in the last quoted digit
        assert abs(float(cell) - 100 * float(row[col])) <= 0.051, f"{label} {col}: tex {cell!r} vs receipt {row[col]!r}"


def test_tier2_accuracy_matches_receipt():
    rec = json.loads((BENCH / "tier2_exp_nmr_validation.json").read_text(encoding="utf-8"))
    tex = _tex()
    for calc in ("mmff94s", "mace_off23_medium"):
        a = rec["closed_pool_a6_accuracy"][calc]
        quoted = f"{pct(a['mean'])}\\% [{pct(a['ci_low'])}, {pct(a['ci_high'])}]"
        assert quoted in tex, f"{calc}: {quoted!r} not in main.tex"


def test_tier2_c13_stats_match_receipt():
    rec = json.loads((BENCH / "tier2_exp_nmr_validation.json").read_text(encoding="utf-8"))
    stats = rec["c13_forward_stats"]
    tex = _tex()
    assert re.search(r"MAE\}? ?\$?= " + re.escape(f"{stats['mae_ppm']:.2f}"), tex)
    assert re.search(r"RMSE\}? ?\$?= " + re.escape(f"{stats['rmse_ppm']:.2f}"), tex)
    assert rec["n_compounds"] == 30 and "N=30" in tex
    assert rec["n_isomer_pairs"] == 15 and "15 DOI-cited isomer pairs" in tex


def test_strain_pair_counts_match_receipt():
    rec = json.loads((BENCH / "mace_strain_audit.json").read_text(encoding="utf-8"))
    assert rec["n_pairs"] == 50 and "N=50" in _tex()
    assert rec["n_embed_or_cip_failures_on_corrupt"] == 30
