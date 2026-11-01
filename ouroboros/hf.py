"""Hugging Face model folders: write the weights of a checkpoint and rebuild the model from them.

Folder layout (written by ``scripts/export_hf_release.py``)::

    model.safetensors     full OCSRModel state dict (pytorch_model.bin if safetensors is missing)
    encoder.safetensors   encoder weights only, ``encoder.`` prefix stripped (encoder.bin fallback)
    config.json           {"model": ..., "data": {"image_size": ...}, vocab_size, encoder_type, ...}
    vocab.json            tokenizer vocabulary (list of tokens, index = token id)
    README.md, metrics.json, checksums.txt

escnn's cached filters (``*.filter``, ``*.expanded_bias``) are derived from the weights and never
stored, exactly as in ``ouroboros.model.load_model_state``. Both weight formats load without
executing pickled code (safetensors, or ``torch.load(weights_only=True)``).

    from ouroboros.hf import load_pretrained
    model = load_pretrained("dist/hf/C")          # eval mode, steerable encoder exported
    smiles = model.generate(images)               # images: [B, 1, S, S] ink in [0, 1]
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ouroboros.model import _ESCNN_CACHE_SUFFIXES, OCSRModel, build_model, load_model_state
from ouroboros.weights import digest as sha256

FORMAT = "ouroboros-hf-v1"
CONFIG_FILE = "config.json"
VOCAB_FILE = "vocab.json"
CHECKSUMS_FILE = "checksums.txt"
WEIGHT_FILES = {"model": ("model.safetensors", "pytorch_model.bin"),
                "encoder": ("encoder.safetensors", "encoder.bin")}
ENCODER_PREFIX = "encoder."


def _safetensors():
    """The ``safetensors.torch`` module, or None if the package is not installed."""
    try:
        import safetensors.torch
    except ImportError:
        return None
    return safetensors.torch


def inference_state(state: dict) -> dict[str, torch.Tensor]:
    """Weights without escnn filter caches, as independent contiguous CPU tensors.

    Cloning also unties shared storage (tied token/output embeddings), which safetensors refuses.
    """
    out = {}
    for key, value in state.items():
        if key.endswith(_ESCNN_CACHE_SUFFIXES):
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Unexpected non-tensor model state: {key}")
        out[key] = value.detach().cpu().clone().contiguous()
    return out


def encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k[len(ENCODER_PREFIX):]: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}


def save_weights(state: dict[str, torch.Tensor], folder: Path, kind: str) -> Path:
    """Write ``state`` as ``<kind>.safetensors`` (or the .bin fallback); return the file."""
    st = _safetensors()
    safe_name, bin_name = WEIGHT_FILES[kind]
    if st is not None:
        path = Path(folder) / safe_name
        st.save_file(state, str(path), metadata={"format": "pt", "ouroboros": FORMAT})
    else:
        path = Path(folder) / bin_name
        torch.save(state, path)
    restored = read_weights(path)
    if restored.keys() != state.keys() or not all(
        torch.equal(restored[k], v) for k, v in state.items()
    ):
        raise ValueError(f"Weights changed when written to {path.name}")
    return path


def read_weights(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        st = _safetensors()
        if st is None:
            raise ImportError(f"{path.name} needs the safetensors package (pip install safetensors)")
        return st.load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def weights_file(folder: Path, kind: str = "model") -> Path:
    for name in WEIGHT_FILES[kind]:
        if (Path(folder) / name).is_file():
            return Path(folder) / name
    raise FileNotFoundError(f"no {' or '.join(WEIGHT_FILES[kind])} in {folder}")


def read_checksums(folder: Path) -> dict[str, str]:
    """``checksums.txt`` (sha256sum format: ``<hex>  <filename>``) as {filename: hex}."""
    out = {}
    for line in (Path(folder) / CHECKSUMS_FILE).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def verify_checksums(folder: Path, names: list[str] | None = None) -> None:
    """Raise if a listed file (default: all) is missing or differs from ``checksums.txt``."""
    sums = read_checksums(folder)
    for name in names if names is not None else sorted(sums):
        if name not in sums:
            raise ValueError(f"{name} is not listed in {CHECKSUMS_FILE}")
        path = Path(folder) / name
        if not path.is_file() or sha256(path) != sums[name]:
            raise ValueError(f"checksum mismatch for {name} in {folder}")


def load_pretrained(
    path: str | Path,
    device: str | torch.device = "cpu",
    export_encoder: bool = True,
    verify: bool = True,
) -> OCSRModel:
    """Rebuild an ``OCSRModel`` (eval mode) from an exported Hugging Face model folder.

    ``export_encoder``: replace a steerable encoder by its pure-PyTorch inference copy, as
    ``scripts/predict_real.py`` does (identical outputs, no escnn work per forward). Keep it
    False to fine-tune. ``verify``: check SHA256 of config, vocabulary and weights against
    ``checksums.txt`` when that file is present.
    """
    from ouroboros.decode.tokenizer import SmilesTokenizer

    folder = Path(path)
    weights = weights_file(folder)
    if verify and (folder / CHECKSUMS_FILE).is_file():
        verify_checksums(folder, [CONFIG_FILE, VOCAB_FILE, weights.name])
    config = json.loads((folder / CONFIG_FILE).read_text())
    if config.get("format") != FORMAT:
        raise ValueError(f"{folder / CONFIG_FILE}: expected format {FORMAT!r}")
    tok = SmilesTokenizer(json.loads((folder / VOCAB_FILE).read_text()))
    if len(tok) != config["vocab_size"]:
        raise ValueError(f"vocab.json has {len(tok)} tokens, config says {config['vocab_size']}")
    model = build_model({"model": config["model"], "data": config["data"]}, tok)
    load_model_state(model, read_weights(weights))
    model.to(device).eval()
    if export_encoder and hasattr(model.encoder, "export"):
        model.encoder = model.encoder.export().to(device)
    return model
