"""Export a fine-tuned checkpoint as a Hugging Face model folder (see ``ouroboros/hf.py``).

    python scripts/export_hf_release.py --ckpt /content/drive/.../C-selected.pt --arm C \
        --out-dir dist/hf/C --report runs/v2/C-selected_report.json \
        --profiling runs/v2/profiling/infer_C.json

Writes ``model.safetensors`` (full state dict, escnn filter caches dropped), ``encoder.safetensors``
(``encoder.*`` with the prefix stripped), ``config.json``, ``vocab.json``, ``README.md`` (from
``docs/hf_model_card_template.md``), ``metrics.json`` (if --report / --profiling) and
``checksums.txt`` (SHA256 of every other file). Without the safetensors package the weights are
written as ``pytorch_model.bin`` / ``encoder.bin`` (tensors only, ``weights_only`` loadable).

Before writing checksums the folder is reloaded with ``ouroboros.hf.load_pretrained`` and its
logits on a random image are required to equal the checkpoint's. Training checkpoints contain
pickled RNG state: only export files you trust (``weights_only`` is tried first).
Nothing is uploaded; see ``scripts/upload_hf.py``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pickle
import re
import shutil
from pathlib import Path

import torch

from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.hf import (
    CHECKSUMS_FILE,
    CONFIG_FILE,
    FORMAT,
    VOCAB_FILE,
    encoder_state,
    inference_state,
    load_pretrained,
    save_weights,
    sha256,
)
from ouroboros.model import build_model, count_params, load_model_state

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "hf_model_card_template.md"
DEFAULT_REPO_ID = "yaniguan/ouroboros-ocsr-encoder"
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
SUBSET_ORDER = ("ACS", "UOB", "USPTO")


def load_source(path: Path) -> dict:
    """Training checkpoint or ``ouroboros-inference-v1`` export (both: model/config/vocab/step)."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:  # training states pickle RNG objects: trusted input only
        state = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("model", "config", "vocab") if k not in state]
    if missing:
        raise ValueError(f"{path}: not an Ouroboros checkpoint (missing {missing})")
    return state


def symmetry_group(enc_cfg: dict, encoder) -> str:
    n = getattr(getattr(encoder, "c", None), "N", None)  # every escnn encoder config has N
    if enc_cfg["type"] == "baseline" or n is None:
        return "none: ordinary CNN with absolute token positions (not rotation-equivariant)"
    group = f"C{n} rotations only, no reflections"
    if enc_cfg["type"] == "canonicalize":
        group += " (C_N-equivariant orientation net + canonicalization)"
    return group


def _clean(value):
    """JSON-safe copy: NaN/inf -> null, absolute paths -> file names (no machine paths)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and os.path.isabs(value):
        return Path(value).name
    return value


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.2%}"


def eval_table(report: dict | None) -> str:
    if report is None:
        return "_No evaluation report supplied (`--report`); results to be added._"

    def order(g):  # per kind: ACS, UOB, USPTO, other subsets, then the overall row
        s = g["subset"]
        rank = SUBSET_ORDER.index(s) if s in SUBSET_ORDER else len(SUBSET_ORDER)
        return (str(g.get("kind")), s == "ALL", rank, s)

    lines = [
        "| Subset | n | Exact match (95% CI) | Exact, stereo ignored | InChIKey | Valid | Tanimoto |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in sorted(report["groups"], key=order):
        name = "**all**" if g["subset"] == "ALL" else g["subset"]
        if g.get("kind"):
            name += f" ({g['kind']})"
        ci = g.get("exact_ci95")
        exact = _pct(g.get("exact")) + (f" ({_pct(ci[0])}–{_pct(ci[1])})" if ci else "")
        n_exact = g.get("exact")
        count = (
            f" = {round(n_exact * g['n'])}/{g['n']}" if n_exact is not None and g.get("n") else ""
        )
        tani = g.get("tanimoto")
        tani = "–" if tani is None else f"{tani:.3f}"
        lines.append(
            f"| {name} | {g.get('n', '–')} | {exact}{count} | {_pct(g.get('exact_flat'))} | "
            f"{_pct(g.get('inchikey'))} | {_pct(g.get('valid'))} | {tani} |"
        )
    return "\n".join(lines)


def eval_notes(report: dict | None, decoding: str) -> str:
    if report is None:
        return ""
    notes = [f"Decoding: {decoding}. Scored rows: {report.get('n_scored')} "
             f"(gold labels that do not standardize: {report['gold_invalid']['n']})."]
    if report.get("footnote"):
        notes.append(f"_{report['footnote']}_")
    return "\n\n".join(notes)


def profiling_table(profiling: dict | None) -> str:
    variants = (profiling or {}).get("variants")
    if not variants:
        return (
            "_No inference profile supplied (`--profiling`, output of scripts/profile_infer.py)._"
            if profiling is None
            else "_See `metrics.json` (`profiling`)._"
        )
    lines = [
        f"Device: `{profiling.get('device', '?')}`, batch-level wall time, "
        f"{next(iter(variants.values())).get('samples', '?')} images.",
        "",
        "| Decoding | encoder ms/img | decode ms/img | total ms/img | images/s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    def num(x, digits: int = 1) -> str:
        return "–" if x is None else f"{x:.{digits}f}"

    for name, v in variants.items():
        ms = v.get("ms_per_image") or {}
        lines.append(
            f"| {name} | {num(ms.get('encoder'))} | {num(ms.get('decode'))} | "
            f"{num(ms.get('total'))} | {num(v.get('images_per_s'), 2)} |"
        )
    return "\n".join(lines)


def training_summary(cfg: dict) -> str:
    t, d, v = cfg.get("train"), cfg.get("data", {}), cfg.get("validation") or {}
    if not t:
        return "_Not recorded (inference export without the training configuration)._"
    init = t.get("init_checkpoint")
    items = [
        ("initialized from", Path(init).name if init else None),
        ("steps (maximum)", t.get("steps")),
        ("batch size", t.get("batch_size")),
        ("learning rate", t.get("lr")),
        ("warmup steps", t.get("warmup")),
        ("weight decay", t.get("weight_decay")),
        ("label smoothing", t.get("label_smoothing")),
        ("rotation augmentation", t.get("rotation_aug")),
        ("mixed precision", t.get("amp")),
        ("real fraction per batch", d.get("real_fraction")),
        ("seed", cfg.get("seed")),
        ("validation images", v.get("samples")),
        ("validation every (steps)", v.get("every")),
        ("early-stopping patience", v.get("patience")),
    ]
    return "\n".join(f"- {k}: `{val}`" for k, val in items if val is not None)


def render_card(template: str, values: dict) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    if unfilled := sorted(set(PLACEHOLDER.findall(text))):
        raise ValueError(f"model card template has unfilled placeholders: {unfilled}")
    return text


def write_checksums(folder: Path) -> dict[str, str]:
    sums = {
        p.name: sha256(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != CHECKSUMS_FILE
    }
    (folder / CHECKSUMS_FILE).write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    return sums


@torch.no_grad()
def _logits(model, image_size: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(1, 1, image_size, image_size, generator=g)
    prefix = torch.tensor([[model.tokenizer.bos_id, 4, 5, 6]]) % len(model.tokenizer)
    enc = model.encoder(images)
    return model.decoder(prefix, enc.tokens, enc.padding_mask)


def export(
    ckpt: Path,
    out_dir: Path,
    arm: str = "C",
    report: Path | None = None,
    profiling: Path | None = None,
    template: Path = TEMPLATE,
    repo_id: str = DEFAULT_REPO_ID,
    model_name: str | None = None,
    decoding: str = "greedy",
    author: str = "yaniguan",
    overwrite: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty (use --overwrite to replace it)")
        try:  # only ever delete a folder this script wrote
            previous = json.loads((out_dir / CONFIG_FILE).read_text()).get("format")
        except (OSError, ValueError):
            previous = None
        if previous != FORMAT:
            raise FileExistsError(f"{out_dir} is not an {FORMAT} export; refusing to delete it")
        shutil.rmtree(out_dir)
    state = load_source(Path(ckpt))
    cfg = state["config"]
    model_cfg = {"model": cfg["model"], "data": {"image_size": cfg["data"]["image_size"]}}
    tok = SmilesTokenizer(state["vocab"])
    model = build_model(model_cfg, tok)
    load_model_state(model, state["model"])  # the weights must match the recorded architecture
    model.eval()
    weights = inference_state(state["model"])
    enc_cfg = cfg["model"]["encoder"]

    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = save_weights(weights, out_dir, "model")
    encoder_file = save_weights(encoder_state(weights), out_dir, "encoder")
    config = {
        "format": FORMAT,
        "architecture": "OCSRModel",
        "arm": arm,
        "model": cfg["model"],
        "data": model_cfg["data"],
        "vocab_size": len(tok),
        "encoder_type": enc_cfg["type"],
        "symmetry_group": symmetry_group(enc_cfg, model.encoder),
        "n_params": count_params(model),
        "n_params_encoder": count_params(model.encoder),
        "weights_file": model_file.name,
        "encoder_weights_file": encoder_file.name,
        "encoder_key_prefix_stripped": "encoder.",
        "step": state.get("step"),
        "source_checkpoint_sha256": sha256(Path(ckpt)),
        "input": "grayscale ink = 1 - gray/255, white background = 0, shape [B, 1, S, S]",
    }
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / VOCAB_FILE).write_text(json.dumps(tok.itos, indent=0) + "\n")

    report_d = json.loads(Path(report).read_text()) if report else None
    profiling_d = json.loads(Path(profiling).read_text()) if profiling else None
    if report_d is not None or profiling_d is not None:
        metrics = {"format": FORMAT, "decoding": decoding,
                   "report": _clean(report_d), "profiling": _clean(profiling_d)}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, allow_nan=False) + "\n")

    card = render_card(
        Path(template).read_text(),
        {
            "model_name": model_name or f"Ouroboros OCSR, arm {arm} ({enc_cfg['type']} encoder)",
            "arm": arm,
            "repo_id": repo_id,
            "encoder_type": enc_cfg["type"],
            "symmetry_group": config["symmetry_group"],
            "image_size": config["data"]["image_size"],
            "vocab_size": config["vocab_size"],
            "n_params_total": f"{config['n_params']:,}",
            "n_params_encoder": f"{config['n_params_encoder']:,}",
            "weights_file": model_file.name,
            "encoder_weights_file": encoder_file.name,
            "step": config["step"],
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "training_summary": training_summary(cfg),
            "eval_table": eval_table(_clean(report_d)),
            "eval_notes": eval_notes(report_d, decoding),
            "profiling_table": profiling_table(_clean(profiling_d)),
            "author": author,
            "date": datetime.date.today().isoformat(),
        },
    )
    (out_dir / "README.md").write_text(card)

    reloaded = load_pretrained(out_dir, export_encoder=False, verify=False)
    size = config["data"]["image_size"]
    if not torch.equal(_logits(model, size), _logits(reloaded, size)):
        raise ValueError("reloaded export does not reproduce the checkpoint's logits")
    sums = write_checksums(out_dir)
    return {"out_dir": str(out_dir), "files": sums, "config": config}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="trusted checkpoint, e.g. C-selected.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arm", default="C")
    ap.add_argument("--report", type=Path, help="eval_predictions.py --json report")
    ap.add_argument("--profiling", type=Path, help="profile_infer.py output (infer_<name>.json)")
    ap.add_argument("--decoding", default="greedy",
                    help="how the --report predictions were decoded, e.g. 'beam 5, valid rerank'")
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="used in the card's usage example")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--author", default="yaniguan", help="citation author")
    ap.add_argument("--overwrite", action="store_true", help="replace a non-empty --out-dir")
    a = ap.parse_args(argv)
    result = export(
        a.ckpt, a.out_dir, arm=a.arm, report=a.report, profiling=a.profiling,
        template=a.template, repo_id=a.repo_id, model_name=a.model_name, decoding=a.decoding,
        author=a.author, overwrite=a.overwrite,
    )
    for name, digest in result["files"].items():
        print(f"{digest}  {name}")
    print(f"exported to {result['out_dir']} ({result['config']['symmetry_group']})")
    return result


if __name__ == "__main__":
    main()
