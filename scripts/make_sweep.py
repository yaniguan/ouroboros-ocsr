"""Expand configs/sweep.yaml into one config per run (configs/sweep/<run_id>.yaml) + index.csv.

python scripts/make_sweep.py [--sweep configs/sweep.yaml] [--out configs/sweep]
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import yaml

from ouroboros.train.config import deep_update, load_config, set_dotted


def run_id(arm: str, size: int, frac: float, seed: int) -> str:
    return f"{arm.replace('+', 'plus')}_n{size // 1000}k_f{frac:g}_s{seed}"


def expand(sweep_path: str | Path, with_optional: bool = False) -> list[dict]:
    sweep_path = Path(sweep_path)
    sw = yaml.safe_load(sweep_path.read_text())
    base = load_config(sweep_path.parent / sw["base"])
    runs = []
    arms = dict(sw["arms"])
    if with_optional:
        arms.update(sw.get("optional_arms", {}))
    for arm, overrides in arms.items():
        for key in overrides:
            if key.startswith("model.decoder") or key == "model":
                raise ValueError(f"arm {arm} overrides {key}: the decoder must be identical")
        for size in sw["sizes"]:
            for frac in sw["real_fractions"]:
                if sw.get("dedupe_full_real") and frac == 1.0 and size != min(sw["sizes"]):
                    continue
                for seed in sw["seeds"]:
                    cfg = copy.deepcopy(base)
                    for k, v in overrides.items():
                        if isinstance(v, dict):
                            parts = k.split(".")
                            node = cfg
                            for p in parts[:-1]:
                                node = node.setdefault(p, {})
                            node[parts[-1]] = deep_update({}, v)  # replace, not merge
                        else:
                            set_dotted(cfg, k, v)
                    cfg["seed"] = seed
                    cfg["data"]["synthetic_max_samples"] = size
                    cfg["data"]["real_fraction"] = float(frac)
                    cfg["data"]["real_roots"] = list(sw["real_roots"]) if frac > 0 else []
                    cfg["train"]["steps"] = int(sw["steps"][size])
                    rid = run_id(arm, size, frac, seed)
                    cfg["meta"] = {
                        "run_id": rid,
                        "arm": arm,
                        "size": size,
                        "real_fraction": float(frac),
                        "seed": seed,
                        "real_fraction_is_placeholder": bool(
                            sw.get("real_fractions_are_placeholders", False)
                        ),
                    }
                    runs.append(cfg)
    decoders = {yaml.safe_dump(r["model"]["decoder"]) for r in runs}
    assert len(decoders) == 1, "decoder configs differ across runs"
    return runs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--out", default="configs/sweep")
    ap.add_argument("--with-optional", action="store_true", help="also expand optional_arms (E)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.yaml"):
        old.unlink()
    runs = expand(a.sweep, a.with_optional)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "arm", "size", "real_fraction", "seed", "steps", "config"])
        for cfg in runs:
            m = cfg["meta"]
            path = out / f"{m['run_id']}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            w.writerow(
                [
                    m["run_id"],
                    m["arm"],
                    m["size"],
                    m["real_fraction"],
                    m["seed"],
                    cfg["train"]["steps"],
                    str(path),
                ]
            )
    print(f"{len(runs)} run configs -> {out}")


if __name__ == "__main__":
    main()
