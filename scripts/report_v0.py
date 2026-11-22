"""Regenerate the v0 comparison from measured summaries; fail if any arm is missing."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write-docs",
        action="store_true",
        help="also write benchmarks/v0.1.0-alpha/summary-generated.md",
    )
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    reports = root / "benchmarks/v0.1.0-alpha"
    rows = []
    curves = {}
    for arm in ("A", "B", "C"):
        report = json.loads((reports / f"{arm}-summary.json").read_text(encoding="utf-8"))
        full = [e for e in report["entries"] if e["set"] == "rendered_test"]
        sweep = sorted(
            [e for e in report["entries"] if e["set"] == "rendered_test:sweep"],
            key=lambda e: e["angle"],
        )
        assert len(full) == 1 and full[0]["n"] == 10000 and full[0]["angle"] == 0
        assert [e["angle"] for e in sweep] == list(range(0, 360, 15))
        assert all(e["n"] == 2000 for e in sweep)
        f = full[0]
        rows.append(
            {
                "arm": arm,
                "step": report["step"],
                "exact": f["exact"],
                "ci95_low": f["exact_ci95"][0],
                "ci95_high": f["exact_ci95"][1],
                "rotation_mean": sum(e["exact"] for e in sweep) / len(sweep),
                "rotation_min": min(e["exact"] for e in sweep),
                "invalid_rate": f["invalid_rate"],
                "exact_flat": f["exact_flat"],
                "stereo_subset_exact": f["exact_on_stereo_subset"],
                "stereo_element_accuracy": f["stereo_elem_acc_strict"],
            }
        )
        curves[arm] = sweep
    with (reports / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (reports / "rotation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "angle", "n", "exact", "invalid_rate"])
        for arm, sweep in curves.items():
            for e in sweep:
                writer.writerow([arm, e["angle"], e["n"], e["exact"], e["invalid_rate"]])
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    labels = {"A": "A · CNN", "B": "B · CNN + rotation augmentation", "C": "C · C8 steerable"}
    for arm, color in zip(curves, ("#64748b", "#c77713", "#007f76"), strict=True):
        sweep = curves[arm]
        ax.plot(
            [e["angle"] for e in sweep],
            [100 * e["exact"] for e in sweep],
            label=labels[arm],
            color=color,
            marker="o",
            markersize=3,
        )
    ax.set(
        xlabel="Input rotation (degrees)",
        ylabel="Stereo-aware exact match (%)",
        title="Ouroboros v0 · 2,000 fixed synthetic test molecules · seed 0",
        xticks=list(range(0, 361, 45)),
        ylim=(0, 100),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(reports / "rotation.svg")
    fig.savefig(reports / "rotation.png", dpi=180)
    plt.close(fig)
    lines = [
        "# v0.1.0-alpha measured results\n",
        "Single seed, synthetic-only pilot. Checkpoints selected by validation before test.\n",
        "| Arm | Full-test exact (95% CI) | 24-angle mean | Worst angle | Invalid SMILES |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['exact']:.2%} ({r['ci95_low']:.2%}–{r['ci95_high']:.2%}) "
            f"| {r['rotation_mean']:.2%} | {r['rotation_min']:.2%} | {r['invalid_rate']:.2%} |"
        )
    lines += [
        "\nFull test: 10,000 molecules at 0°. Rotation mean/minimum: the same 2,000 molecules",
        "at 24 angles, 0–345° in 15° steps. Intervals are per-set, not seed variability.\n",
        "![Rotation comparison](../benchmarks/v0.1.0-alpha/rotation.svg)\n",
        "## Stereochemistry\n",
        "| Arm | Exact ignoring stereo | Stereo-subset exact (n=4,000) | Stereo-element accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['exact_flat']:.2%} | {r['stereo_subset_exact']:.2%} "
            f"| {r['stereo_element_accuracy']:.2%} |"
        )
    lines += [
        "\nStereo-element accuracy uses the evaluator's strict denominator (5,282 elements).",
        "Connectivity correctness and SMILES validity do not establish correct stereochemistry.\n",
        "## Interpretation boundaries\n",
        "These results describe this training budget, renderer distribution and seed only.",
        "A/B share the same architecture; B adds random rotation. C has fewer encoder parameters",
        "under a FLOP-matching design. Hardware and wall times are recorded separately in",
        "[MODEL_CARD.md](../../MODEL_CARD.md) and [v0-results.md](../../docs/v0-results.md).",
        "No real-document or paper parity claim follows from these results.\n",
        "Raw aggregate summaries and CSVs: `benchmarks/v0.1.0-alpha/`. Rebuild with",
        "`python scripts/report_v0.py`; missing arms or incomplete angle grids are rejected.",
    ]
    if a.write_docs:
        (reports / "summary-generated.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
