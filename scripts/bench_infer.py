"""CPU latency (p50/p95 per image) and accuracy of ouroboros.infer.OCSRPredictor on real images.

    python scripts/bench_infer.py --manifest real_data/subset100/manifest.csv \
        --out benchmarks/cpu-api/bench.json

The manifest is the one written by scripts/fetch_real_subset.py (image paths relative to it).
Rows whose gold SMILES does not standardize are skipped. Each request is timed end to end
(decode the file bytes, resize, encode, greedy decode, rescore, RDKit). Wall time includes
waiting for a busy shared CPU; process CPU time (1 thread) is the cost on an idle core.
Confidence threshold: the default is checked on even rows and reported on odd rows.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouroboros.eval.standardize import standardize  # noqa: E402
from ouroboros.infer import ROOT, OCSRPredictor  # noqa: E402

VARIANTS = {"A_fp32": ("ouroboros-A.pt", False), "A_int8": ("ouroboros-A.pt", True),
            "C_fp32": ("ouroboros-C.pt", False)}
THRESHOLDS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]


def cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        with open(cpuinfo, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _load_avg():
    return os.getloadavg() if hasattr(os, "getloadavg") else None


def q(xs):
    if not xs:
        return {"p50": None, "p95": None, "max": None}
    return {"p50": float(np.percentile(xs, 50)), "p95": float(np.percentile(xs, 95)),
            "max": max(xs)}


def selective(rows, tau):
    if not rows:
        return {"coverage": 0.0, "precision": None}
    acc = [r for r in rows if r["valid"] and r["confidence"] >= tau]
    precision = sum(r["correct"] for r in acc) / len(acc) if acc else None
    return {"coverage": len(acc) / len(rows), "precision": precision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=0, help="first n usable rows (0 = all)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out")
    args = ap.parse_args()
    base = Path(args.manifest).parent
    rows = []
    with open(args.manifest, encoding="utf-8", newline="") as mf:
        for r in csv.DictReader(mf):
            gold = standardize(r["smiles"])
            if gold.valid:
                rows.append({"path": base / r["image"], "gold": gold.smiles, "source": r["source"]})
    rows = rows[:args.n] if args.n else rows
    if not rows:
        ap.error("no valid rows in manifest")
    rep = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "threads": args.threads,
           "load_avg_start": _load_avg(), "n_images": len(rows), "variants": {}}
    preds = {}
    for name in args.variants:
        ckpt, int8 = VARIANTS[name]
        p = OCSRPredictor(ROOT / "checkpoints" / ckpt, threads=args.threads, int8=int8)
        out, wall, cpu = [], [], []
        for r in rows:
            data = r["path"].read_bytes()
            t, c = time.perf_counter(), time.process_time()
            res = p.predict(data)
            wall.append((time.perf_counter() - t) * 1e3)
            cpu.append((time.process_time() - c) * 1e3)
            out.append({"correct": res["canonical_smiles"] == r["gold"], "valid": res["valid"],
                        "confidence": res["confidence"], "source": r["source"],
                        "smiles": res["canonical_smiles"]})
        preds[name] = out
        v = {"latency_ms": {"wall": q(wall), "cpu": q(cpu)},
             "exact": float(np.mean([o["correct"] for o in out])),
             "valid": float(np.mean([o["valid"] for o in out])),
             "exact_by_source": {
                 s: float(np.mean([o["correct"] for o in out if o["source"] == s]))
                 for s in sorted({o["source"] for o in out})},
             "selective": {str(t): {"calibration_even_rows": selective(out[0::2], t),
                                    "held_out_odd_rows": selective(out[1::2], t)}
                           for t in THRESHOLDS}}
        if name != "A_fp32" and "A_fp32" in preds:
            pairs = zip(out, preds["A_fp32"], strict=True)
            same = [a["smiles"] == b["smiles"] for a, b in pairs]
            v["same_smiles_as_A_fp32"] = float(np.mean(same))
        rep["variants"][name] = v
        print(name, json.dumps({k: v[k] for k in ("latency_ms", "exact", "valid")}), flush=True)
    rep["load_avg_end"] = _load_avg()
    text = json.dumps(rep, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
