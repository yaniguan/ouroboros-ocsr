import json
from types import SimpleNamespace

import pytest
import torch

from ouroboros.eval.evaluate import run_eval
from ouroboros.geometry.cache import calculator_weights_sha256
from ouroboros.provenance import JsonCache


class Dataset:
    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 3

    def __getitem__(self, i):
        self.reads += 1
        return {"image": torch.full((1, 4, 4), i / 3)}

    def meta(self, i):
        return {"smiles": "CCCCC"}

    def identity(self):
        return {"fixture": "three-pentanes"}


def test_evaluation_recovers_completed_batches_and_rejects_changed_identity(tmp_path):
    calls = []

    def interrupted(images):
        calls.append(len(images))
        if len(calls) == 2:
            raise RuntimeError("disconnect")
        return ["CCCCC"] * len(images)

    kwargs = dict(
        sets={"synth": (Dataset(), "rendered"), "real": (Dataset(), "real")},
        out_dir=tmp_path,
        batch_size=2,
        angles=[0, 90],
        n_boot=10,
        identity={"checkpoint": "one"},
    )
    with pytest.raises(RuntimeError, match="disconnect"):
        run_eval(interrupted, **kwargs)
    assert len(list((tmp_path / "batches").glob("*.json"))) == 1
    assert sum(ds.reads for ds, _ in kwargs["sets"].values()) == 2
    calls.clear()

    def resumed(images):
        calls.append(len(images))
        return ["CCCCC"] * len(images)

    summary = run_eval(resumed, **kwargs)
    assert sum(ds.reads for ds, _ in kwargs["sets"].values()) == 8
    assert len(calls) == 7  # 2 sets x 2 angles x 2 batches, one already completed
    assert len(summary["entries"]) == 4
    assert all(e["exact"] == 1 for e in summary["entries"])
    original = (tmp_path / "predictions.jsonl").read_bytes()
    assert json.dumps(run_eval(resumed, **kwargs), sort_keys=True) == json.dumps(
        summary, sort_keys=True
    )
    assert len(calls) == 7
    assert sum(ds.reads for ds, _ in kwargs["sets"].values()) == 8
    assert (tmp_path / "predictions.jsonl").read_bytes() == original
    with pytest.raises(ValueError, match="identity changed"):
        run_eval(resumed, **{**kwargs, "identity": {"checkpoint": "two"}})


def test_cache_rejects_corruption_and_weight_identity_changes(tmp_path):
    cache = JsonCache(tmp_path)
    cache.get({"job": 1}, lambda: {"energy": -1})
    file = next(tmp_path.glob("*.json"))
    record = json.loads(file.read_text())
    record["result"]["energy"] = -100
    file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Corrupt cache"):
        cache.get({"job": 1}, lambda: pytest.fail("must not recompute corrupt result"))
    calc = SimpleNamespace(models=[torch.nn.Linear(1, 1)])
    before = calculator_weights_sha256(calc)
    with torch.no_grad():
        calc.models[0].weight.add_(1)
    assert calculator_weights_sha256(calc) != before


def test_energy_cli_persists_search_and_mirror_jobs(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import numpy as np

    from ouroboros.geometry.conformers import GeometryResult, Relaxed, embed

    spec = importlib.util.spec_from_file_location(
        "energy_cli", Path("scripts/error_propagation.py")
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    calc = SimpleNamespace(models=[torch.nn.Linear(1, 1)], device="cpu")
    monkeypatch.setattr(cli, "mace_calculator", lambda model: calc)
    searches, mirrors = [], []

    def search(smi, **kwargs):
        searches.append(smi)
        mol, _ = embed(smi, n_conf=1)
        return GeometryResult(
            smi,
            "ok",
            energy=-1.0,
            converged=True,
            symbols=[a.GetSymbol() for a in mol.GetAtoms()],
            positions=mol.GetConformer().GetPositions(),
        )

    def relax(atoms, calculator):
        mirrors.append(True)
        return Relaxed(-1.0, np.zeros_like(atoms.positions), atoms.positions, True, 1, 0.01, 0.0)

    monkeypatch.setattr(cli, "lowest_energy_conformer", search)
    monkeypatch.setattr(cli, "relax", relax)
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("pred\tref\nC[C@@H](N)C(=O)O\tC[C@H](N)C(=O)O\nCCN\tCCO\n")
    out = tmp_path / "out"
    args = ["--pairs-tsv", str(pairs), "--out", str(out)]
    cli.main(args)
    first = (out / "pairs.jsonl").read_bytes()
    records = [json.loads(line) for line in first.splitlines()]
    assert records[0]["dE"] == 0 and records[0]["converged"]
    assert records[0]["pred_energy_method"] == "mirrored truth geometry"
    assert not records[1]["isomer"] and np.isnan(records[1]["dE"])
    counts = len(searches), len(mirrors)
    cli.main(args)
    assert (len(searches), len(mirrors)) == counts
    assert (out / "pairs.jsonl").read_bytes() == first
