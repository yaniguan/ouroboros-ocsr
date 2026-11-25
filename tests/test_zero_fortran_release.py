"""Tests for zero-Fortran pure-PyTorch inference packaging and weight provenance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from ouroboros.demo.rotation import RotationLab
from ouroboros.weights import (
    has_escnn,
    load_manifest,
    resolve_checkpoint_path,
    resolve_weight_entry,
)
from ouroboros.weights import (
    main as weights_main,
)

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_inference_is_pure_wheels_without_escnn_or_fortran():
    lines = [
        ln.strip()
        for ln in (ROOT / "requirements-inference.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    joined = "\n".join(lines).lower()
    assert "torch==" in joined
    assert "rdkit==" in joined
    for forbidden in ("escnn", "py3nj", "lie-learn", "lie_learn", "gfortran", "git+"):
        assert forbidden not in joined


def test_weights_manifest_has_pure_torch_c_and_license_provenance():
    manifest = load_manifest()
    assert manifest["license"] == "MIT"
    assert "synthetic" in manifest["training_domain"].lower()
    assert "CC-BY-NC-4.0" in manifest["real_benchmark_note"]
    c_entry = manifest["weights"]["C"]
    assert c_entry["requires_escnn"] is True
    pure = c_entry["pure_torch"]
    assert pure["filename"] == "ouroboros-C-pure.pt"
    assert pure["requires_escnn"] is False
    assert len(pure["sha256"]) == 64 and int(pure["sha256"], 16) > 0
    assert pure["bytes"] > 100_000_000
    assert pure["source_checkpoint_sha256"] == c_entry["sha256"]


def test_resolve_weight_entry_and_checkpoint_path_when_escnn_missing(tmp_path, monkeypatch):
    manifest = load_manifest()
    monkeypatch.setitem(sys.modules, "escnn", None)
    assert has_escnn() is False

    # Auto mode selects ouroboros-C-pure.pt when escnn is missing
    auto_c = resolve_weight_entry(manifest, "C", mode="auto")
    assert auto_c["filename"] == "ouroboros-C-pure.pt"
    # Explicit --escnn mode still returns raw ouroboros-C.pt for training callers
    escnn_c = resolve_weight_entry(manifest, "C", mode="escnn")
    assert escnn_c["filename"] == "ouroboros-C.pt"

    pure_file = tmp_path / "ouroboros-C-pure.pt"
    pure_file.write_bytes(b"stub")
    assert resolve_checkpoint_path(tmp_path / "ouroboros-C.pt") == pure_file


def test_weights_info_cli_prints_domain_and_license(capsys):
    weights_main(["info"])
    info = json.loads(capsys.readouterr().out)
    assert info["version"] == "0.1.0-alpha"
    assert info["license"] == "MIT"
    assert info["arms"]["C"]["pure_torch_filename"] == "ouroboros-C-pure.pt"


def test_prepare_inference_succeeds_without_escnn(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "prepare_inference", ROOT / "scripts" / "prepare_inference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name in ("escnn", "lie_learn") else True,
    )
    assert mod.main([]) == 0
    assert "Pure-PyTorch inference dependencies ready" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="escnn"):
        mod.main(["--require-escnn"])


def test_rotation_lab_random_runs_with_exact_c4_equivariance_without_escnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "escnn", None)
    lab = RotationLab.random(order=4)
    assert not any(type(m).__module__.startswith("escnn") for m in lab.model.modules())
    r90 = lab.evaluate("CCO", angle=90)
    assert r90["on_group"] is True and r90["on_grid"] is True
    assert r90["feature_error"] < 1e-5
    assert r90["logit_error"] < 1e-5
    with pytest.raises(ImportError, match="order=4"):
        RotationLab.random(order=8)


def test_exported_pure_c_checkpoint_if_present_locally(monkeypatch):
    pure_ckpt = ROOT / "checkpoints" / "ouroboros-C-pure.pt"
    if not pure_ckpt.is_file():
        pytest.skip("local ouroboros-C-pure.pt not downloaded")
    monkeypatch.setitem(sys.modules, "escnn", None)
    state = torch.load(pure_ckpt, map_location="cpu", weights_only=True)
    assert state["pure_torch"] is True
    # Requesting ouroboros-C.pt without escnn transparently loads ouroboros-C-pure.pt
    lab = RotationLab.from_checkpoint(ROOT / "checkpoints" / "ouroboros-C.pt")
    r90 = lab.evaluate("CC(=O)Oc1ccccc1C(=O)O", angle=90)
    assert r90["rotated_prediction"]["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert r90["feature_error"] < 1e-5


def test_ui_engine_finds_pure_c_checkpoint_when_only_pure_is_downloaded(tmp_path, monkeypatch):
    from ouroboros.ui import backend

    monkeypatch.setattr(backend, "CKPT_DIR", tmp_path)
    (tmp_path / "ouroboros-A.pt").write_bytes(b"stub")
    pure_file = tmp_path / "ouroboros-C-pure.pt"
    pure_file.write_bytes(b"stub")
    found = backend.WorkbenchEngine(device="cpu").available_checkpoints()
    assert found == {"A": str(tmp_path / "ouroboros-A.pt"), "C": str(pure_file)}
