"""Export trusted training checkpoints, or fetch checksum-verified release weights.

python -m ouroboros.weights fetch --arm C
python -m ouroboros.weights fetch --arm C --pure-torch
python -m ouroboros.weights info
python -m ouroboros.weights export --checkpoint RUN/ckpt/best-step-00010000.pt --arm C --pure-torch
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


def has_escnn() -> bool:
    """Return True when escnn is importable in the current Python environment."""
    if "escnn" in sys.modules:
        return sys.modules["escnn"] is not None
    return importlib.util.find_spec("escnn") is not None


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Prefer ``ouroboros-C-pure.pt`` when ``ouroboros-C.pt`` is requested without ``escnn``."""
    p = Path(path)
    if p.name == "ouroboros-C.pt" and (not p.is_file() or not has_escnn()):
        pure = p.with_name("ouroboros-C-pure.pt")
        if pure.is_file() and (not p.is_file() or not has_escnn()):
            return pure
    return p


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_checkpoint(
    checkpoint: Path, output: Path, arm: str, *, pure_torch: bool = False
) -> dict:
    """Input must be trusted: training states include Python/NumPy RNG objects."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model_cfg = {
        "data": {"image_size": cfg["data"]["image_size"]},
        "model": copy.deepcopy(cfg["model"]),
    }
    weights = state["model"]
    if pure_torch:
        from ouroboros.decode.tokenizer import SmilesTokenizer
        from ouroboros.hf import EXPORTED_ENCODER_TYPES, inference_state
        from ouroboros.model import build_model, load_model_state

        tok = SmilesTokenizer(state["vocab"])
        built = build_model(model_cfg, tok)
        load_model_state(built, weights)
        built.eval()
        enc_type = model_cfg["model"]["encoder"]["type"]
        if enc_type in EXPORTED_ENCODER_TYPES and hasattr(built.encoder, "export"):
            built.encoder = built.encoder.export()
            model_cfg["model"]["encoder"]["type"] = EXPORTED_ENCODER_TYPES[enc_type]
        weights = inference_state(built.state_dict())

    # Keep only inference configuration: no machine paths, optimizer, RNG or training data.
    inference = {
        "format": "ouroboros-inference-v1",
        "arm": arm,
        "model": weights,
        "config": model_cfg,
        "vocab": state["vocab"],
        "step": state["step"],
        "source_checkpoint_sha256": digest(checkpoint),
    }
    if pure_torch:
        inference["pure_torch"] = True
    # Config/vocabulary should be plain JSON, and every weight a tensor.
    json.dumps({k: v for k, v in inference.items() if k != "model"})
    if not all(isinstance(value, torch.Tensor) for value in inference["model"].values()):
        raise ValueError("Unexpected non-tensor model state")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(inference, output)
    restored = torch.load(output, map_location="cpu", weights_only=True)
    for key, value in inference["model"].items():
        if not torch.equal(value, restored["model"][key]):
            raise ValueError(f"Export changed weight {key}")
    return {
        "arm": arm,
        "filename": output.name,
        "bytes": output.stat().st_size,
        "sha256": digest(output),
        "step": state["step"],
        "source_checkpoint_sha256": inference["source_checkpoint_sha256"],
    }


def resolve_weight_entry(manifest: dict, arm: str, *, mode: str = "auto") -> dict:
    """Select the manifest entry for ``arm`` under ``mode`` ('auto', 'pure-torch', 'escnn')."""
    if arm == "C-pure":
        return manifest["weights"]["C"]["pure_torch"]
    entry = manifest["weights"][arm]
    if arm == "C" and "pure_torch" in entry:
        if mode == "pure-torch":
            return entry["pure_torch"]
        if mode == "auto" and not has_escnn():
            return entry["pure_torch"]
    return entry


def fetch(entry: dict, directory: Path) -> Path:
    filename = entry["filename"]
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("Invalid weight filename")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.exists():
        if target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
            return target
        raise ValueError(
            f"Existing file has a different checksum: {target}; move it before retrying"
        )
    url = entry["url"]
    if not url.startswith("https://github.com/yaniguan/ouroboros-ocsr/releases/download/"):
        raise ValueError("Weight URL is outside the project's release assets")
    fd, name = tempfile.mkstemp(prefix=filename + ".", suffix=".part", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > entry["bytes"]:
                    raise ValueError("Download exceeds the expected weight size")
                output.write(chunk)
        if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
            raise ValueError("Weight download checksum mismatch; no checkpoint installed")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest() -> dict:
    return json.loads(Path(__file__).with_name("weights.json").read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    download = sub.add_parser("fetch", help="download published weights and verify SHA256")
    download.add_argument("--arm", choices=("A", "B", "C", "C-pure", "all"), default="C")
    download.add_argument("--out", type=Path, default=Path("checkpoints"))
    fmt = download.add_mutually_exclusive_group()
    fmt.add_argument(
        "--pure-torch",
        action="store_true",
        help="prefer zero-escnn pure-PyTorch inference weights when available",
    )
    fmt.add_argument(
        "--escnn",
        action="store_true",
        help="require raw escnn steerable basis weights (for fine-tuning)",
    )
    sub.add_parser("info", help="show published release weight domain, licenses, and checksums")
    export = sub.add_parser("export", help="export a TRUSTED training checkpoint for inference")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    export.add_argument("--out", type=Path, default=Path("dist/weights"))
    export.add_argument(
        "--pure-torch",
        action="store_true",
        help="expand steerable filters into pure-PyTorch Conv2d weights (zero escnn at inference)",
    )
    args = ap.parse_args(argv)
    if args.command == "export":
        suffix = "-pure" if args.pure_torch and args.arm in ("C", "D") else ""
        out_file = args.out / f"ouroboros-{args.arm}{suffix}.pt"
        if args.pure_torch and (args.out / f"ouroboros-{args.arm}.pt") != out_file:
            # Keep backward compatibility with callers checking ouroboros-<arm>.pt when requested
            pass
        result = export_checkpoint(
            args.checkpoint,
            args.out / f"ouroboros-{args.arm}.pt",
            args.arm,
            pure_torch=args.pure_torch,
        )
        (args.out / f"{args.arm}-receipt.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    elif args.command == "info":
        manifest = load_manifest()
        info = {
            "version": manifest["version"],
            "training_domain": manifest.get("training_domain"),
            "license": manifest.get("license"),
            "escnn_installed": has_escnn(),
            "note": manifest.get("real_benchmark_note"),
            "arms": {
                arm: {
                    "filename": e["filename"],
                    "bytes": e["bytes"],
                    "sha256": e["sha256"],
                    "requires_escnn": e.get("requires_escnn", False),
                    "pure_torch_filename": e.get("pure_torch", {}).get("filename"),
                }
                for arm, e in manifest["weights"].items()
            },
        }
        print(json.dumps(info, indent=2))
    else:
        manifest = load_manifest()
        mode = "pure-torch" if args.pure_torch else ("escnn" if args.escnn else "auto")
        arms = ("A", "B", "C") if args.arm == "all" else (args.arm,)
        print(
            f"[ouroboros.weights] Release {manifest['version']} · "
            f"{manifest.get('training_domain', 'synthetic pilot')} · "
            f"License: {manifest.get('license', 'MIT')}",
            file=sys.stderr,
        )
        if note := manifest.get("real_benchmark_note"):
            print(f"[ouroboros.weights] Note: {note}", file=sys.stderr)
        for arm in arms:
            entry = resolve_weight_entry(manifest, arm, mode=mode)
            if arm == "C" and mode == "auto" and not has_escnn():
                print(
                    "[ouroboros.weights] escnn not installed -> fetching pure-PyTorch "
                    f"{entry['filename']} (zero gfortran/escnn needed).",
                    file=sys.stderr,
                )
            print(fetch(entry, args.out))


if __name__ == "__main__":
    main()
