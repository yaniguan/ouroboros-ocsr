"""Persistent MACE jobs keyed by actual weights, numerical settings and implementation."""

import hashlib

import torch


def calculator_weights_sha256(calc) -> str:
    digest = hashlib.sha256()
    for i, model in enumerate(calc.models):
        digest.update(str(i).encode())
        for name, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
