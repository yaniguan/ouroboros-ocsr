"""YAML config loading with dotted-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _scalar(v: str) -> Any:
    """YAML scalar, but also accept '5e-4'-style floats (YAML 1.1 reads them as strings)."""
    out = yaml.safe_load(v)
    if isinstance(out, str):
        try:
            return float(out)
        except ValueError:
            return out
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load YAML; a top-level ``base:`` key (path relative to this file) is merged first.

    ``overrides`` are ``"a.b.c=value"`` strings; values are parsed as YAML scalars.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_update(load_config(path.parent / base), cfg)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        set_dotted(cfg, k, _scalar(v))
    return cfg
