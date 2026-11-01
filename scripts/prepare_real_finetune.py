"""Append-only vocabulary + run config for a real-data fine-tune of a released checkpoint.

    python scripts/prepare_real_finetune.py --ckpt checkpoints/ouroboros-A.pt \
        --config configs/finetune-real-pmc.yaml --synthetic-root /content/data/synth-aug/shards \
        --real-root /content/data/real/pmc --out-dir /content/drive/.../A-lr1.5e-05 \
        [--set train.lr=1e-5 train.steps=2000 ...]

Writes ``<out-dir>/vocab.json`` and ``<out-dir>/finetune.yaml`` and prints a JSON summary.

* vocabulary = the checkpoint's vocabulary (IDs unchanged) + label/alias tokens + every token of
  the synthetic train/val labels and the REAL TRAIN labels, appended in a deterministic order
  (frequency, then token). Real val/test labels never shape the vocabulary.
* ``model`` is replaced by the checkpoint's exact architecture; ``image_size`` by its own.
* labels longer than the decoder limit are an error here (``prepare_pmc.py`` and the
  synthetic builder's ``--max-tokens`` already drop them, with counts).
* ``validation.samples`` is clipped to the number of available real val rows.
* ``train.amp`` = bf16 when CUDA supports it, otherwise none (fp32), unless overridden.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import yaml

from ouroboros.data.loader import ShardDataset, list_shards
from ouroboros.decode.tokenizer import SmilesTokenizer, tokenize
from ouroboros.train.config import _scalar, load_config, set_dotted


def label_tokens(root: str | Path, split: str, counts: Counter) -> int:
    try:
        list_shards(root, split)
    except FileNotFoundError:
        return 0
    ds = ShardDataset(root, split)
    for i in range(len(ds)):
        counts.update(tokenize(ds.meta(i)["smiles"]))
    return len(ds)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/finetune-real-pmc.yaml")
    ap.add_argument("--synthetic-root", required=True)
    ap.add_argument("--real-root", required=True, help="ingested real shards (train + val)")
    ap.add_argument("--exclude-keys", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-5")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.ckpt).resolve()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    tok = SmilesTokenizer(list(state["vocab"]))
    n_base = len(tok)
    tok.add_label_tokens()
    n_labels = len(tok) - n_base
    counts: Counter = Counter()
    n_synth = label_tokens(a.synthetic_root, "train", counts)
    n_synth += label_tokens(a.synthetic_root, "val", counts)
    n_real = label_tokens(a.real_root, "train", counts)
    added = tok.extend(t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    vocab_path = out / "vocab.json"
    tok.save(vocab_path)

    cfg = load_config(a.config)
    cfg["model"] = state["config"]["model"]
    max_len = cfg["model"]["decoder"]["max_len"]
    cfg["data"].update(
        vocab=str(vocab_path),
        synthetic_root=str(a.synthetic_root),
        real_roots=[str(a.real_root)],
        image_size=state["config"]["data"]["image_size"],
    )
    if a.exclude_keys is not None:
        cfg["data"]["exclude_keys"] = a.exclude_keys
    cfg["train"]["init_checkpoint"] = str(ckpt)
    cfg["train"]["amp"] = (
        "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "none"
    )
    cfg["validation"]["root"] = str(a.real_root)
    for ov in a.set:
        k, v = ov.split("=", 1)
        set_dotted(cfg, k, _scalar(v))
    n_val = len(ShardDataset(cfg["validation"]["root"], "val"))
    cfg["validation"]["samples"] = min(int(cfg["validation"]["samples"]), n_val)
    if cfg["validation"]["samples"] < 1:
        raise SystemExit("no real validation rows")
    # every training label must fit the decoder
    for root, split in ((a.synthetic_root, "train"), (a.real_root, "train")):
        ds = ShardDataset(root, split)
        for i in range(len(ds)):
            label = ds.meta(i)["smiles"]
            if len(tok.encode(label)) - 1 > max_len:
                raise SystemExit(f"label longer than max_len={max_len} in {root}:{split}: {label}")
    cfg_path = out / "finetune.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    summary = {
        "config": str(cfg_path),
        "vocab": str(vocab_path),
        "vocab_size": len(tok),
        "vocab_base": n_base,
        "vocab_added_labels": n_labels,
        "vocab_added_observed": len(added),
        "added_tokens": added,
        "n_synthetic_labels_scanned": n_synth,
        "n_real_train_labels_scanned": n_real,
        "n_real_val_available": n_val,
        "validation_samples": cfg["validation"]["samples"],
        "amp": cfg["train"]["amp"],
        "lr": cfg["train"]["lr"],
        "steps": cfg["train"]["steps"],
        "batch_size": cfg["train"]["batch_size"],
        "real_fraction": cfg["data"]["real_fraction"],
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
