"""Score a model on the paired diagnostic set from ``make_diagnostics.py``.

    python scripts/eval_diagnostics.py --diag diag --weights ouroboros-C.pt --out r.json

Per base molecule the model reads the base drawing, three identity-preserving views, and up to three
identity-changing edits. Reported (exact match after standardization, base molecules with a pair):
  preserve:  both_correct         base and view both exactly right
             consistent_wrong     same prediction on both, but wrong (consistency is not
                                  correctness)
             consistent           same prediction on both, right or wrong
  edit:      both_correct         base and edited drawing both exactly right
             responds             base and edited drawing both right (same as both_correct)
             insensitive          prediction did not change although the molecule did
             wrong_other          prediction changed, but not to the edited molecule
Rates are over pairs, with Wilson 95% intervals; the unit is the base molecule, not the image.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ouroboros.demo.rotation import RotationLab
from ouroboros.eval.metrics import score_pair

PRESERVE = ("redraw", "rot45", "rot90")
EDITS = ("stereo_flip", "bond_order", "atom_swap")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def predict(lab: RotationLab, paths: list[Path], batch: int = 32) -> list[str]:
    preds = []
    for s in range(0, len(paths), batch):
        imgs = []
        for p in paths[s : s + batch]:
            gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            imgs.append(torch.from_numpy(1.0 - gray / 255.0)[None])
        with torch.inference_mode():
            preds += lab.model.generate(torch.stack(imgs))
    return preds


def summarize(rows: list[dict], preds: dict[str, str]) -> dict:
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pair[r["pair"]][r["kind"]] = r
    out: dict[str, dict] = {}
    for kind in (*PRESERVE, *EDITS):
        counts = defaultdict(int)
        for kinds in by_pair.values():
            if kind not in kinds or "base" not in kinds:
                continue
            b, v = kinds["base"], kinds[kind]
            pb, pv = preds[b["image"]], preds[v["image"]]
            ok_b = score_pair(pb, b["truth"]).exact
            ok_v = score_pair(pv, v["truth"]).exact
            same = pb == pv
            counts["pairs"] += 1
            counts["both_correct"] += ok_b and ok_v
            counts["base_correct"] += ok_b
            counts["consistent"] += same
            if kind in PRESERVE:
                counts["consistent_wrong"] += same and not ok_v
            else:
                counts["responds"] += ok_b and ok_v
                counts["insensitive"] += same
                counts["wrong_other"] += (not same) and not ok_v
        n = counts.pop("pairs", 0)
        entry = {"pairs": n}
        for k, v in counts.items():
            entry[k] = round(v / n, 4) if n else None
            entry[k + "_ci95"] = wilson(int(v), n)
        if kind in EDITS and counts.get("base_correct"):
            entry["responds_given_base_correct"] = round(
                counts["responds"] / counts["base_correct"], 4
            )
        out[kind] = entry
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--diag", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    lab = RotationLab.from_checkpoint(a.weights, a.device)
    rows = [json.loads(x) for x in (a.diag / "manifest.jsonl").read_text().splitlines()]
    paths = [a.diag / r["image"] for r in rows]
    preds = dict(zip((r["image"] for r in rows), predict(lab, paths), strict=True))
    result = {"weights": str(a.weights.name), "diag": a.diag.name, "kinds": summarize(rows, preds)}
    a.out.write_text(json.dumps(result, indent=1))
    for kind, e in result["kinds"].items():
        print(kind, {k: v for k, v in e.items() if not k.endswith("ci95")})


if __name__ == "__main__":
    main()
