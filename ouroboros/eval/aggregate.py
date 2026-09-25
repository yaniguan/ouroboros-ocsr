"""Aggregate sweep runs into a results table and plots (Phase 4, Am1-D).

Input: run directories, each with ``config.yaml`` (``meta``: arm, size, real_fraction, seed) and
``eval/predictions.jsonl`` from ``ouroboros.eval.evaluate.run_eval``.

Rules enforced here:
* every eval set is its own column; rendered and real-document sets are never averaged
  together (``pool_sets`` raises ``MixedKindsError``);
* per-set sample counts are reported;
* exact match: mean ± std over seeds, plus a 95% bootstrap CI (>= 1000 resamples) that resamples
  test items (paired across seeds) and recomputes the seed-mean;
* the scoring-parity footnote is attached while parity with arXiv:2608.09100 is unverified.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ouroboros.eval.standardize import PARITY_FOOTNOTE, PARITY_VERIFIED

ARM_ORDER = ["A", "B", "C", "C+", "D", "E"]
# Categorical slots 1-6 of the validated reference palette (light surface), in fixed order, so an
# arm keeps its colour in every figure. Markers are a second, colour-independent encoding.
ARM_COLORS = dict(
    zip(ARM_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], strict=True)
)
ARM_MARKERS = dict(zip(ARM_ORDER, ["o", "s", "^", "D", "v", "P"], strict=True))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


class MixedKindsError(ValueError):
    """Raised when asked to combine rendered and real-document sets into one number."""


def load_runs(run_dirs: list[str | Path]) -> list[dict]:
    runs = []
    for d in map(Path, run_dirs):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        meta = cfg.get("meta", {})
        recs = [json.loads(x) for x in (d / "eval" / "predictions.jsonl").read_text().splitlines()]
        runs.append(
            {
                "dir": str(d),
                "arm": meta["arm"],
                "size": int(meta["size"]),
                "real_fraction": float(meta.get("real_fraction", 0.0)),
                "seed": int(meta.get("seed", cfg.get("seed", 0))),
                "records": recs,
            }
        )
    return runs


def pool_sets(set_kinds: dict[str, str], sets: list[str]) -> list[str]:
    """Validate a request to average several eval sets; only same-kind sets may be pooled."""
    kinds = {set_kinds[s] for s in sets}
    if len(kinds) > 1:
        raise MixedKindsError(
            f"refusing to average rendered and real-document sets together: {sorted(sets)}"
        )
    return sets


def _exact_matrix(runs: list[dict], set_name: str, angle: float) -> tuple[np.ndarray, list[int]]:
    """[n_seeds, n_items] exact-match matrix aligned on item index."""
    per_seed, seeds = [], []
    for r in sorted(runs, key=lambda r: r["seed"]):
        d = {
            x["index"]: float(x["exact"])
            for x in r["records"]
            if x["set"] == set_name and x["angle"] == angle
        }
        if d:
            per_seed.append(d)
            seeds.append(r["seed"])
    if not per_seed:
        return np.zeros((0, 0)), []
    common = sorted(set.intersection(*(set(d) for d in per_seed)))
    return np.array([[d[i] for i in common] for d in per_seed]), seeds


def seed_mean_ci(m: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over seeds of per-seed exact match; CI by resampling items (paired across seeds)."""
    if m.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.shape[1], size=(n_boot, m.shape[1]))
    boot = m[:, idx].mean(axis=2).mean(axis=0)  # [n_boot]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(m.mean()), float(lo), float(hi)


def results_table(runs: list[dict], angle: float = 0.0, n_boot: int = 1000) -> dict:
    """Rows = (arm, size, real_fraction); one column block per eval set (never pooled)."""
    if n_boot < 1000:
        raise ValueError("bootstrap CIs require >= 1000 resamples")
    set_kinds: dict[str, str] = {}
    for r in runs:
        for x in r["records"]:
            prev = set_kinds.setdefault(x["set"], x["kind"])
            if prev != x["kind"]:
                raise MixedKindsError(f"set {x['set']} appears as both {prev} and {x['kind']}")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["arm"], r["size"], r["real_fraction"])].append(r)
    rows = []
    for (arm, size, frac), rs in sorted(
        groups.items(), key=lambda kv: (ARM_ORDER.index(kv[0][0]), kv[0][1], kv[0][2])
    ):
        row = {"arm": arm, "size": size, "real_fraction": frac, "n_seeds": len(rs), "sets": {}}
        for s in sorted(set_kinds):
            m, seeds = _exact_matrix(rs, s, angle)
            if m.size == 0:
                continue
            mean, lo, hi = seed_mean_ci(m, n_boot)
            per_seed = m.mean(axis=1)
            row["sets"][s] = {
                "kind": set_kinds[s],
                "n_items": int(m.shape[1]),
                "seeds": seeds,
                "exact_mean": mean,
                "exact_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
                "exact_ci95": [lo, hi],
            }
        rows.append(row)
    return {
        "angle": angle,
        "set_kinds": set_kinds,
        "rows": rows,
        "n_boot": n_boot,
        "footnote": None if PARITY_VERIFIED else PARITY_FOOTNOTE,
    }


def table_markdown(table: dict) -> str:
    """Markdown table; rendered-set columns first, then real-document columns, never merged."""
    sk = table["set_kinds"]
    sets = sorted(sk, key=lambda s: (sk[s] != "rendered", s))
    head = ["arm", "train size", "real frac.", "seeds"] + [f"{s} ({sk[s]})" for s in sets]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in table["rows"]:
        cells = [r["arm"], f"{r['size']:,}", f"{r['real_fraction']:g}", str(r["n_seeds"])]
        for s in sets:
            c = r["sets"].get(s)
            if c is None:
                cells.append("—")
                continue
            std = "" if np.isnan(c["exact_std"]) else f" ± {100 * c['exact_std']:.1f}"
            ci = f" [{100 * c['exact_ci95'][0]:.1f}, {100 * c['exact_ci95'][1]:.1f}]"
            cells.append(f"{100 * c['exact_mean']:.1f}{std}{ci} (n={c['n_items']})")
        lines.append("| " + " | ".join(cells) + " |")
    notes = [
        "",
        "Exact match (%, full stereochemistry): mean ± std over seeds, [95% bootstrap CI over "
        f"test items, {table['n_boot']} resamples], n = test items. Rendered and real-document "
        "sets are separate columns and are never averaged.",
    ]
    if table["footnote"]:
        notes.append(f"*{table['footnote']}*")
    return "\n".join(lines + notes) + "\n"


# ------------------------------------------------------------------------------------ plots


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def _line(ax, arm, xs, ys, lo, hi):
    c = ARM_COLORS.get(arm, INK2)
    ax.fill_between(xs, lo, hi, color=c, alpha=0.10, linewidth=0)
    ax.plot(
        xs,
        ys,
        color=c,
        linewidth=2,
        solid_capstyle="round",
        marker=ARM_MARKERS.get(arm, "o"),
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=f"arm {arm}",
    )
    ax.annotate(  # direct label at the line end (text in ink, not series colour)
        arm,
        (xs[-1], ys[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )


def _fmt_size(x: float) -> str:
    return f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k"


def _fig_legend(fig, ax) -> None:
    """Legend above the plot area (never over the data); markers repeat the arm identity."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            fontsize=8,
            labelcolor=INK,
            bbox_to_anchor=(0.5, 1.0),
        )


def plot_curves(table_by_x: dict, x_key: str, xlabel: str, out: Path, logx: bool) -> list[Path]:
    """One figure per eval set: exact match vs ``x_key`` per arm, CI band. ``table_by_x`` is the
    output of ``results_table``; rows sharing an arm are joined along ``x_key``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    sk = table_by_x["set_kinds"]
    other = "real_fraction" if x_key == "size" else "size"
    for s in sorted(sk):
        slices = sorted({r[other] for r in table_by_x["rows"]})
        fig, axes = plt.subplots(1, len(slices), figsize=(4.2 * len(slices), 3.4), squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, sl in zip(axes[0], slices, strict=True):
            for arm in ARM_ORDER:
                pts = sorted(
                    (r[x_key], r["sets"][s])
                    for r in table_by_x["rows"]
                    if r["arm"] == arm and r[other] == sl and s in r["sets"]
                )
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [100 * p[1]["exact_mean"] for p in pts]
                lo = [100 * p[1]["exact_ci95"][0] for p in pts]
                hi = [100 * p[1]["exact_ci95"][1] for p in pts]
                _line(ax, arm, xs, ys, lo, hi)
            lab = "train size" if other == "size" else "real fraction"
            _style_axes(ax, f"{s} ({sk[s]}) · {lab} = {sl:g}", xlabel, "exact match (%)")
            if logx:
                ax.set_xscale("log")
                xs_all = sorted({r[x_key] for r in table_by_x["rows"]})
                ax.set_xticks(xs_all, [_fmt_size(x) for x in xs_all])
                ax.minorticks_off()
        _fig_legend(fig, axes[0][0])
        if table_by_x["footnote"]:
            fig.text(0.01, 0.005, table_by_x["footnote"], fontsize=7, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 0.88))
        p = out / f"{x_key}_{s}.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    return paths


def rotation_sweep(runs: list[dict], n_boot: int = 1000) -> dict:
    """Per (arm, size, real_fraction, set): exact vs angle with seed-mean bootstrap CIs."""
    from ouroboros.eval.evaluate import angle_flags

    out: dict = {"curves": [], "footnote": None if PARITY_VERIFIED else PARITY_FOOTNOTE}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["arm"], r["size"], r["real_fraction"])].append(r)
    for (arm, size, frac), rs in groups.items():
        sets = sorted({(x["set"], x["kind"]) for r in rs for x in r["records"]})
        for s, kind in sets:
            angles = sorted({x["angle"] for r in rs for x in r["records"] if x["set"] == s})
            pts = []
            for a in angles:
                m, _ = _exact_matrix(rs, s, a)
                mean, lo, hi = seed_mean_ci(m, n_boot)
                pts.append({"angle": a, **angle_flags(a), "exact": mean, "ci95": [lo, hi]})
            on = [p["exact"] for p in pts if p["pixel_exact"]]
            off = [p["exact"] for p in pts if not p["pixel_exact"]]
            out["curves"].append(
                {
                    "arm": arm,
                    "size": size,
                    "real_fraction": frac,
                    "set": s,
                    "kind": kind,
                    "points": pts,
                    "mean_on_grid_90": float(np.mean(on)) if on else float("nan"),
                    "mean_off_grid_90": float(np.mean(off)) if off else float("nan"),
                }
            )
    return out


def plot_rotation(sweep: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    keys = sorted({(c["set"], c["kind"], c["size"], c["real_fraction"]) for c in sweep["curves"]})
    for s, kind, size, frac in keys:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        fig.patch.set_facecolor(SURFACE)
        for arm in ARM_ORDER:
            for c in sweep["curves"]:
                if (c["arm"], c["set"], c["size"], c["real_fraction"]) != (arm, s, size, frac):
                    continue
                xs = [p["angle"] for p in c["points"]]
                ys = [100 * p["exact"] for p in c["points"]]
                lo = [100 * p["ci95"][0] for p in c["points"]]
                hi = [100 * p["ci95"][1] for p in c["points"]]
                _line(ax, arm, xs, ys, lo, hi)
        for a in (0, 90, 180, 270):  # pixel-exact (C4) angles
            ax.axvline(a, color=GRID, linewidth=1, zorder=0)
        _style_axes(
            ax,
            f"rotation sweep · {s} ({kind}) · size {size:,} · real {frac:g}",
            "rotation angle (deg, CCW); vertical lines = 90° grid",
            "exact match (%)",
        )
        ax.set_xticks(range(0, 360, 45))
        _fig_legend(fig, ax)
        if sweep["footnote"]:
            fig.text(0.01, 0.005, sweep["footnote"], fontsize=7, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 0.86))
        p = out / f"rotation_{s}_{size}_{frac:g}.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    return paths
