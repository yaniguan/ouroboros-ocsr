"""Export trusted training checkpoints, or fetch checksum-verified release weights.

python -m ouroboros.weights fetch --arm C
python -m ouroboros.weights export --checkpoint RUN/ckpt/best-step-00010000.pt --arm C
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_checkpoint(checkpoint: Path, output: Path, arm: str) -> dict:
    """Input must be trusted: training states include Python/NumPy RNG objects."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    # Keep only inference configuration: no machine paths, optimizer, RNG or training data.
    inference = {
        "format": "ouroboros-inference-v1",
        "arm": arm,
        "model": state["model"],
        "config": {"data": {"image_size": cfg["data"]["image_size"]}, "model": cfg["model"]},
        "vocab": state["vocab"],
        "step": state["step"],
        "source_checkpoint_sha256": digest(checkpoint),
    }
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    download = sub.add_parser("fetch", help="download published weights and verify SHA256")
    download.add_argument("--arm", choices=("A", "B", "C", "all"), default="C")
    download.add_argument("--out", type=Path, default=Path("checkpoints"))
    export = sub.add_parser("export", help="export a TRUSTED training checkpoint for inference")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--arm", choices=("A", "B", "C"), required=True)
    export.add_argument("--out", type=Path, default=Path("dist/weights"))
    args = ap.parse_args(argv)
    if args.command == "export":
        result = export_checkpoint(args.checkpoint, args.out / f"ouroboros-{args.arm}.pt", args.arm)
        (args.out / f"{args.arm}-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        manifest = json.loads(Path(__file__).with_name("weights.json").read_text())
        arms = ("A", "B", "C") if args.arm == "all" else (args.arm,)
        for arm in arms:
            print(fetch(manifest["weights"][arm], args.out))


if __name__ == "__main__":
    main()
