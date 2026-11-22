#!/usr/bin/env python3
"""Reproducible GPU-profiling analysis for WignerFlow / ouroboros-ocsr / YLM Colab runs.

Inputs are the artifacts the repos' profilers write (summary.json, key_averages.txt,
nvidia-smi CSV / gpu_utilization.jsonl / gpu_samples.csv, optional chrome trace.json).
Usage:
    python scripts/analyze_profiling.py --raw RAW_DIR --out OUT_DIR [--manifest manifest.json]
The manifest maps run names to files inside RAW_DIR (see DEFAULT_MANIFEST).
Only numbers found in the files are reported; missing items become null.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import statistics
from pathlib import Path

UNIT = {"us": 1e-3, "ms": 1.0, "s": 1e3}
ELEMENTWISE = {
    "aten::add_",
    "aten::fill_",
    "aten::copy_",
    "aten::mul",
    "aten::add",
    "aten::zero_",
}


def _load_default_manifest() -> dict:
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo / "docs" / "profiling" / "manifest.json",
        Path(__file__).parent / "manifest.json",
    )
    for cand in candidates:
        if cand.is_file():
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return {}


DEFAULT_MANIFEST = _load_default_manifest()


def to_ms(tok: str) -> float | None:
    m = re.fullmatch(r"(-?[\d.]+)(us|ms|s)", tok.strip())
    return round(float(m.group(1)) * UNIT[m.group(2)], 6) if m else None


def parse_key_averages(text: str, top: int = 10) -> dict:
    """Parse a torch.profiler key_averages().table() dump."""
    lines = text.splitlines()
    out = {
        "self_cpu_total_ms": None,
        "self_cuda_total_ms": None,
        "top_cuda": [],
        "top_cpu": [],
    }
    for ln in lines:
        m = re.match(r"Self (CPU|CUDA) time total:\s*(\S+)", ln)
        if m:
            out[f"self_{m.group(1).lower()}_total_ms"] = to_ms(m.group(2))
    header = next((i for i, line in enumerate(lines) if line.strip().startswith("Name")), None)
    if header is None:
        return out
    cols = [c.strip() for c in re.split(r"\s{2,}", lines[header].strip())]
    rows = []
    for ln in lines[header + 2 :]:
        if ln.startswith("---") or not ln.strip():
            break
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) < len(cols):
            continue
        name = " ".join(parts[: len(parts) - len(cols) + 1])
        rec = dict(zip(cols[1:], parts[len(parts) - len(cols) + 1 :], strict=True))
        rows.append(
            {
                "name": name,
                "self_cpu_ms": to_ms(rec.get("Self CPU", "")),
                "self_cuda_ms": to_ms(rec.get("Self CUDA", "")),
                "calls": int(rec.get("# of Calls", "0") or 0),
            }
        )
    real = [
        r
        for r in rows
        if not r["name"].startswith("profile_step") and not r["name"].startswith("ProfilerStep")
    ]
    out["top_cuda"] = sorted(
        [r for r in real if r["self_cuda_ms"]], key=lambda r: -r["self_cuda_ms"]
    )[:top]
    out["top_cpu"] = sorted(
        [r for r in real if r["self_cpu_ms"]], key=lambda r: -r["self_cpu_ms"]
    )[:top]
    launch = [r for r in rows if r["name"].startswith("cudaLaunchKernel")]
    out["cudaLaunchKernel_calls"] = sum(r["calls"] for r in launch) or None
    return out


def util_series(path: str) -> list[float]:
    """Return GPU utilization % samples from JSONL or nvidia-smi CSV telemetry."""
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line)["utilization_percent"] for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f)
        head_raw = next(rd, None)
        if not head_raw:
            return []
        head = [h.strip() for h in head_raw]
        idx = next((i for i, h in enumerate(head) if h.startswith("util")), None)
        if idx is None:
            return []
        vals = []
        for r in rd:
            try:
                vals.append(float(r[idx].replace("%", "").strip()))
            except (ValueError, IndexError):
                pass
        return vals


def trace_gpu_busy(path: str) -> dict | None:
    """Union of GPU kernel/memcpy intervals vs. trace span -> GPU-busy fraction."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ev = [e for e in d.get("traceEvents", []) if e.get("ph") == "X"]
    k = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in ev
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    )
    if not k:
        return None
    busy = 0.0
    cs, ce = k[0][:2]
    for s, e, _ in k[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = max(e for _, e, _ in k) - k[0][0]
    by: collections.Counter[str] = collections.Counter()
    for s, e, n in k:
        by[n[:80]] += e - s
    rt: collections.Counter[str] = collections.Counter()
    for e in ev:
        if e.get("cat") == "cuda_runtime":
            rt[e["name"]] += e["dur"]
    return {
        "span_ms": span / 1e3,
        "busy_ms": busy / 1e3,
        "busy_frac": busy / span if span > 0 else 0.0,
        "n_kernels": len(k),
        "mean_kernel_us": sum(e - s for s, e, _ in k) / len(k),
        "top_kernels_ms": [[n, v / 1e3] for n, v in by.most_common(8)],
        "cuda_runtime_ms": {n: v / 1e3 for n, v in rt.most_common(5)},
    }


def stats(v: list[float]) -> dict | None:
    if not v:
        return None
    return {
        "mean": statistics.mean(v),
        "median": statistics.median(v),
        "max": max(v),
        "n": len(v),
        "frac_below_10pct": sum(x < 10 for x in v) / len(v),
    }


def verdict(run: dict) -> tuple[str, list[str]]:
    """Rule-based bound classification from the extracted evidence."""
    ev: list[str] = []
    util = run.get("gpu_util") or {}
    u = util.get("mean")
    ka = run.get("key_averages") or {}
    cpu, gpu = ka.get("self_cpu_total_ms"), ka.get("self_cuda_total_ms")
    ph = run.get("phases_ms_per_step") or {}
    tot = sum(ph.values()) or None
    if tot and ph.get("data", 0) / tot > 0.2:
        return "data-loader / input bound", [f"data phase {100 * ph['data'] / tot:.0f}% of step"]
    if cpu and gpu and cpu / gpu > 2 and (u is None or u < 50):
        ev.append(f"self CPU {cpu:.0f} ms vs self CUDA {gpu:.0f} ms (ratio {cpu / gpu:.1f})")
        if u is not None:
            ev.append(f"mean GPU util {u:.0f}%")
        return "CPU / kernel-launch (Python overhead) bound", ev
    if gpu and ka.get("top_cuda"):
        ew = sum(r["self_cuda_ms"] for r in ka["top_cuda"] if r["name"] in ELEMENTWISE)
        if ew / gpu > 0.4 and (u is None or u >= 60):
            notes = [f"elementwise add_/fill_/copy_/mul = {100 * ew / gpu:.0f}% of self CUDA time"]
            if u is not None:
                notes.append(f"mean GPU util {u:.0f}%")
            return "GPU memory-bandwidth bound (elementwise ops)", notes
    if run.get("trace") and run["trace"].get("busy_frac", 1) < 0.6:
        t = run["trace"]
        msg = (
            f"GPU busy {100 * t['busy_frac']:.0f}% of trace span, "
            f"{t['n_kernels']} kernels, mean {t['mean_kernel_us']:.0f} us"
        )
        return "kernel-launch / small-kernel bound", [msg]
    if u is not None and u < 50:
        return (
            "likely CPU / launch bound (GPU idle >50%, no op table for this segment)",
            [f"mean GPU util {u:.0f}%"],
        )
    if u is not None and u >= 75:
        return "GPU compute bound", [f"mean GPU util {u:.0f}%"]
    return "mixed (GPU partially idle)", [f"mean GPU util {None if u is None else round(u)}"]


def load(raw: str, name: str) -> str | None:
    p = os.path.join(raw, name)
    return p if os.path.exists(p) else None


def build(raw: str, manifest: dict) -> dict:
    data = {}
    for rid, m in manifest.items():
        run = {k: m[k] for k in ("repo", "label", "gpu", "gpu_mem_gb", "notes") if k in m}
        sum_path = load(raw, m["summary"]) if m.get("summary") else None
        if sum_path:
            with open(sum_path, encoding="utf-8") as f:
                s = json.load(f)
            for path in m.get("summary_path", []):
                s = s[path]
            run["summary"] = (
                {k: v for k, v in s.items() if not k.endswith("samples")}
                if isinstance(s, dict)
                else s
            )
            if isinstance(s, dict) and "phase_seconds_per_step" in s:
                run["phases_ms_per_step"] = {
                    k: v * 1e3 for k, v in s["phase_seconds_per_step"].items()
                }
            if isinstance(s, dict) and isinstance(s.get("phases"), dict):
                run["phases_ms_per_step"] = {k: v["mean_ms"] for k, v in s["phases"].items()}
            if isinstance(s, dict) and "gpu_utilization_samples" in s:
                run["gpu_util"] = stats(s["gpu_utilization_samples"])
        ka_path = load(raw, m["key_averages"]) if m.get("key_averages") else None
        if ka_path:
            with open(ka_path, encoding="utf-8") as f:
                run["key_averages"] = parse_key_averages(f.read())
        csv_path = load(raw, m["gpu_csv"]) if m.get("gpu_csv") else None
        if csv_path:
            run["gpu_util"] = stats(util_series(csv_path))
        tr_path = load(raw, m["trace"]) if m.get("trace") else None
        if tr_path:
            run["trace"] = trace_gpu_busy(tr_path)
        run["bound"], run["evidence"] = verdict(run)
        data[rid] = run
    return data


def plot(data: dict, out: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    paths: list[str] = []
    colors = {
        "data": "#8dd3c7",
        "data_loading": "#8dd3c7",
        "host_to_device": "#ffffb3",
        "neighbor_list": "#bebada",
        "forward": "#80b1d3",
        "backward": "#fb8072",
        "optimizer": "#fdb462",
    }
    for repo in sorted({r["repo"] for r in data.values()}):
        runs = {k: v for k, v in data.items() if v["repo"] == repo and v.get("phases_ms_per_step")}
        if runs:
            fig, ax = plt.subplots(figsize=(9, 0.6 * len(runs) + 1.8))
            for i, v in enumerate(runs.values()):
                left = 0.0
                existing = ax.get_legend_handles_labels()[1]
                for ph, ms in v["phases_ms_per_step"].items():
                    lbl = ph if i == 0 or ph not in existing else None
                    ax.barh(i, ms, left=left, color=colors.get(ph, "#ccc"), label=lbl)
                    left += ms
                ax.text(left, i, f" {left:.0f} ms", va="center", fontsize=8)
            ax.set_yticks(range(len(runs)))
            ax.set_yticklabels([v["label"] for v in runs.values()], fontsize=8)
            ax.set_xlabel("ms per step (CUDA-synchronized phase timers)")
            ax.set_title(f"{repo}: step-time breakdown")
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles, strict=True))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right")
            fig.tight_layout()
            p = os.path.join(out, f"{repo}_step_breakdown.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths.append(p)
        kas = {
            k: v
            for k, v in data.items()
            if v["repo"] == repo and v.get("key_averages", {}).get("top_cuda")
        }
        for k, v in kas.items():
            top = v["key_averages"]["top_cuda"][:10][::-1]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh(
                [t["name"][:55] for t in top],
                [t["self_cuda_ms"] for t in top],
                color="#80b1d3",
            )
            ax.set_xlabel("Self CUDA time (ms, profiled window)")
            ax.set_title(f'{v["label"]}: top CUDA ops/kernels')
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            p = os.path.join(out, f"{k}_top_cuda.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths.append(p)
        for k, v in data.items():
            if v["repo"] == repo and v.get("trace"):
                top = v["trace"]["top_kernels_ms"][::-1]
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.barh([n[:55] for n, _ in top], [ms for _, ms in top], color="#fb8072")
                ax.set_xlabel("GPU time in trace window (ms)")
                ax.tick_params(axis="y", labelsize=7)
                busy_pct = 100 * v["trace"]["busy_frac"]
                span_ms = v["trace"]["span_ms"]
                ax.set_title(
                    f"{v['label'][:40]}: top kernels (GPU busy {busy_pct:.0f}% of {span_ms:.0f} ms)"
                )
                fig.tight_layout()
                p = os.path.join(out, f"{k}_trace_kernels.png")
                fig.savefig(p, dpi=130)
                plt.close(fig)
                paths.append(p)

    runs_util = [(k, v) for k, v in data.items() if v.get("gpu_util")]
    if runs_util:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bar_colors = [
            "#80b1d3"
            if v["repo"] == "WignerFlow"
            else "#fb8072"
            if v["repo"] == "ouroboros-ocsr"
            else "#b3de69"
            for _, v in runs_util
        ]
        ax.bar(
            range(len(runs_util)),
            [v["gpu_util"]["mean"] for _, v in runs_util],
            color=bar_colors,
        )
        ax.set_xticks(range(len(runs_util)))
        ax.set_xticklabels(
            [v["label"] for _, v in runs_util], rotation=40, ha="right", fontsize=7
        )
        ax.set_ylabel("mean GPU util % (nvidia-smi)")
        ax.axhline(75, ls="--", c="gray")
        ax.set_title("GPU utilization by run (blue=WignerFlow, red=ocsr, green=YLM)")
        fig.tight_layout()
        p = os.path.join(out, "cross_gpu_util.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    runs_ka = [
        (k, v)
        for k, v in data.items()
        if v.get("key_averages", {}).get("self_cuda_total_ms")
    ]
    if runs_ka:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(runs_ka))
        cpu_s = [(v["key_averages"].get("self_cpu_total_ms") or 0.0) / 1e3 for _, v in runs_ka]
        cuda_s = [(v["key_averages"].get("self_cuda_total_ms") or 0.0) / 1e3 for _, v in runs_ka]
        ax.bar([i - 0.2 for i in x], cpu_s, 0.4, label="self CPU time (s)")
        ax.bar([i + 0.2 for i in x], cuda_s, 0.4, label="self CUDA time (s)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([v["label"] for _, v in runs_ka], rotation=20, ha="right", fontsize=8)
        ax.legend()
        ax.set_title("torch.profiler window: host (CPU) vs device (CUDA) time")
        fig.tight_layout()
        p = os.path.join(out, "cross_cpu_vs_cuda.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    runs_mem = [(k, v) for k, v in data.items() if v.get("peak_mem_gb")]
    if runs_mem:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            [v["label"] for _, v in runs_mem],
            [v["peak_mem_gb"] for _, v in runs_mem],
            color="#bebada",
        )
        for i, (_, v) in enumerate(runs_mem):
            if v.get("gpu_mem_gb"):
                ax.plot([v["gpu_mem_gb"]] * 2, [i - 0.4, i + 0.4], "k--")
        ax.set_xlabel("peak torch.cuda.max_memory_allocated (GB); dashed = card capacity")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        p = os.path.join(out, "cross_peak_memory.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)
    if a.manifest:
        with open(a.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = DEFAULT_MANIFEST
    data = build(a.raw, manifest)
    for v in data.values():
        s = v.get("summary") or {}
        pk = (
            s.get("cuda_peak_bytes")
            or s.get("peak_mem_bytes")
            or (s.get("peak_memory") or {}).get("allocated_bytes")
        )
        if pk:
            v["peak_mem_gb"] = pk / 1e9
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)
    for p in plot(data, os.path.join(a.out, "charts")):
        print(p)


if __name__ == "__main__":
    main()
