"""Evaluate a run: angle 0 on every configured eval set, plus the rotation sweep on a fixed
prefix of each set (stored as set ``<name>:sweep``). Writes ``<run>/eval/predictions.jsonl`` and
``<run>/eval/summary.json``.

    python scripts/evaluate.py --run runs/A_200k_f0_s0 [--set eval.sweep_max_samples=500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ouroboros.data.loader import ShardDataset
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.eval.evaluate import run_eval, summarize
from ouroboros.model import build_model, load_model_state
from ouroboros.train.config import load_config


class _Prefix:
    def __init__(self, ds, n):
        self.ds, self.n = ds, min(n, len(ds))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.ds[i]

    def meta(self, i):
        return self.ds.meta(i)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None, help="default: <run>/ckpt/last.pt")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run)
    cfg = load_config(run / "config.yaml", a.set)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SmilesTokenizer.load(cfg["data"]["vocab"])
    model = build_model(cfg, tok)
    st = torch.load(a.ckpt or run / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    load_model_state(model, st["model"])
    model.to(dev).eval()
    if hasattr(model.encoder, "export"):  # steerable: pure-PyTorch inference copy
        model.encoder = model.encoder.export().to(dev)

    def generate(images):
        with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            return model.generate(images.to(dev))

    e = cfg["eval"]
    size = cfg["data"]["image_size"]
    full, sweep = {}, {}
    for name, s in e["sets"].items():
        ds = ShardDataset(s["root"], s["split"], image_size=size, max_samples=s.get("max_samples"))
        full[name] = (ds, s["kind"])
        sweep[f"{name}:sweep"] = (_Prefix(ds, e["sweep_max_samples"]), s["kind"])
    out = run / "eval"
    run_eval(generate, full, out / "full", angles=[0.0], batch_size=e["batch_size"], device=dev)
    parts = [out / "full" / "predictions.jsonl"]
    if not a.no_sweep:
        angles = list(range(0, 360, e["sweep_angles_step"]))
        run_eval(
            generate, sweep, out / "sweep", angles=angles, batch_size=e["batch_size"], device=dev
        )
        parts.append(out / "sweep" / "predictions.jsonl")
    rows = [json.loads(x) for p in parts for x in p.read_text().splitlines()]
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    summary["step"] = st["step"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for en in summary["entries"]:
        if en["angle"] == 0.0:
            print(
                f"{en['set']:<28} {en['kind']:<9} n={en['n']:<6} exact={en['exact']:.4f} "
                f"CI95=[{en['exact_ci95'][0]:.4f}, {en['exact_ci95'][1]:.4f}] "
                f"invalid={en['invalid_rate']:.4f}"
            )
    if summary["footnote"]:
        print(summary["footnote"])


if __name__ == "__main__":
    main()
