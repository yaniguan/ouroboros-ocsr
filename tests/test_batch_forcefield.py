"""Parallel ETKDG embedding and batched force-field evaluation tests."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from ouroboros.geometry.batch import embed_many, evaluate_single_points
from ouroboros.geometry.conformers import embed_conformers, to_atoms

ROOT = Path(__file__).resolve().parents[1]


def _load_ff():
    spec = importlib.util.spec_from_file_location(
        "image_to_forcefield", ROOT / "scripts" / "image_to_forcefield.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BatchHarmonicCalculator(Calculator):
    """Analytic harmonic calculator supporting both single-point ASE and calculate_batch."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, fail_atom_count: int | None = None):
        super().__init__()
        self.fail_atom_count = fail_atom_count
        self.batch_sizes: list[int] = []
        self.single_calls: int = 0

    def _eval_pos(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        if self.fail_atom_count is not None and len(pos) == self.fail_atom_count:
            raise RuntimeError("bad_geom")
        return float(0.5 * np.sum(pos * pos)), -pos.copy()

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.single_calls += 1
        energy, forces = self._eval_pos(self.atoms.positions)
        self.results = {"energy": energy, "forces": forces}

    def calculate_batch(self, atoms_chunk):
        self.batch_sizes.append(len(atoms_chunk))
        return [self._eval_pos(a.positions) for a in atoms_chunk]


def test_embed_many_matches_sequential_coordinates_and_failures():
    items = [
        ("CCO", 2, 0),
        ("c1ccccc1O", 2, 3),
        ("C[C@H](N)C(=O)O", 2, 7),
        ("not-a-smiles", 1, 0),
    ]
    seq = embed_many(items, workers=1)
    par = embed_many(items, workers=3)
    assert len(seq) == len(par) == len(items)
    for (m1, r1, _), (m2, r2, _) in zip(seq, par, strict=True):
        assert r1 == r2
        if m1 is None:
            assert m2 is None
            continue
        assert m1.GetNumConformers() == m2.GetNumConformers()
        for c1, c2 in zip(m1.GetConformers(), m2.GetConformers(), strict=True):
            np.testing.assert_array_equal(c1.GetPositions(), c2.GetPositions())

    assert embed_many([], workers=2) == []
    with pytest.raises(ValueError, match="workers"):
        embed_many(items, workers=0)
    with pytest.raises(ValueError, match="workers"):
        embed_many(items, workers=True)


def test_evaluate_single_points_batching_and_error_isolation():
    mol_eth = embed_conformers("CCO", n_conf=3, seed=1)
    mol_ben = embed_conformers("c1ccccc1", n_conf=2, seed=2)
    structures = [to_atoms(mol_eth, c.GetId()) for c in mol_eth.GetConformers()] + [
        to_atoms(mol_ben, c.GetId()) for c in mol_ben.GetConformers()
    ]

    seq_calc = BatchHarmonicCalculator()
    seq = evaluate_single_points(structures, seq_calc, batch_size=1)
    assert seq_calc.batch_sizes == [] and seq_calc.single_calls == 5

    bat_calc = BatchHarmonicCalculator()
    bat = evaluate_single_points(structures, bat_calc, batch_size=2)
    assert bat_calc.batch_sizes == [2, 2] and bat_calc.single_calls == 1
    for a, b in zip(seq, bat, strict=True):
        assert a.error is None and b.error is None
        assert a.energy == pytest.approx(b.energy)
        assert a.fmax == pytest.approx(b.fmax)
        np.testing.assert_allclose(a.forces, b.forces)

    # Inject a failing structure (4 atoms) in the middle of a batch: neighbours must still succeed
    poison = Atoms("H4", positions=np.zeros((4, 3)))
    mixed = [structures[0], poison, structures[1], structures[2]]
    iso_calc = BatchHarmonicCalculator(fail_atom_count=4)
    res = evaluate_single_points(mixed, iso_calc, batch_size=3)
    assert res[0].error is None and res[0].energy == pytest.approx(seq[0].energy)
    assert res[1].error == "mace_failed: bad_geom" and res[1].energy is None
    assert res[2].error is None and res[2].energy == pytest.approx(seq[1].energy)
    assert res[3].error is None and res[3].energy == pytest.approx(seq[2].energy)

    with pytest.raises(ValueError, match="batch_size"):
        evaluate_single_points(structures, bat_calc, batch_size=0)


def test_mace_batch_adapter_committee_and_units():
    pytest.importorskip("mace")
    from mace.tools import AtomicNumberTable

    class _ToyMACEModel(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))

        def forward(self, data: dict, **kwargs):
            pos = data["positions"].detach().requires_grad_(True)
            atom_e = 0.5 * self.scale * (pos * pos).sum(dim=-1)
            n_graphs = int(data["ptr"].numel() - 1)
            energy = torch.zeros(n_graphs, dtype=pos.dtype, device=pos.device)
            energy.scatter_add_(0, data["batch"], atom_e)
            (grad,) = torch.autograd.grad(energy.sum(), pos)
            return {"energy": energy, "forces": -grad}

    fake_mace_calc = SimpleNamespace(
        models=[_ToyMACEModel(1.0), _ToyMACEModel(3.0)],  # committee mean scale = 2.0
        z_table=AtomicNumberTable([1, 6, 7, 8]),
        r_max=4.0,
        default_dtype="float64",
        device="cpu",
        energy_units_to_eV=2.0,
        length_units_to_A=0.5,
        info_keys={},
        arrays_keys={},
        charges_key="Qs",
        head=None,
        available_heads=None,
    )
    mol = embed_conformers("CCO", n_conf=3, seed=4)
    atoms_list = [to_atoms(mol, c.GetId()) for c in mol.GetConformers()]
    with torch.no_grad():  # must still compute autograd forces inside _eval_mace_batch
        res = evaluate_single_points(atoms_list, fake_mace_calc, batch_size=3)
    assert len(res) == 3
    for atoms, r in zip(atoms_list, res, strict=True):
        pos = atoms.positions
        # committee scale 2.0 * energy_units_to_eV 2.0 => E = 2.0 * sum(pos^2)
        # force scale = 2.0 / 0.5 = 4.0; -d/dx (0.5 * 2.0 * x^2) * 4.0 = -8.0 * x
        assert r.error is None
        assert r.energy == pytest.approx(float(2.0 * np.sum(pos * pos)), rel=1e-6)
        np.testing.assert_allclose(r.forces, -8.0 * pos, atol=1e-6)


def test_cli_batch_forcefield_exact_parity_with_streaming(tmp_path, monkeypatch):
    ff = _load_ff()
    csv_path = tmp_path / "preds.csv"
    rows = [
        ("1", "CCO"),
        ("2", "OCC"),  # intra-chunk duplicate InChIKey of row 1
        ("3", "C1CC"),  # invalid SMILES
        ("4", "*c1ccccc1"),  # pseudo-atom
        ("5", "C[C@H](N)C(=O)O"),
        ("6", "c1ccccc1O"),
        ("7", "CC(=O)O"),
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gold", "pred", "subset"])
        w.writeheader()
        for rid, smi in rows:
            w.writerow({"id": rid, "gold": smi, "pred": smi, "subset": "USPTO"})

    seq_calc = BatchHarmonicCalculator()
    monkeypatch.setattr(ff, "load_calculator", lambda *a: (seq_calc, "harmonic"))
    seq_out = tmp_path / "seq.extxyz"
    seq_sum = ff.main(
        [
            "--predictions",
            str(csv_path),
            "--out",
            str(seq_out),
            "--n-conf",
            "2",
            "--displaced-samples",
            "1",
            "--dedupe-inchikey",
            "--only-valid",
        ]
    )

    bat_calc = BatchHarmonicCalculator()
    monkeypatch.setattr(ff, "load_calculator", lambda *a: (bat_calc, "harmonic"))
    bat_out = tmp_path / "bat.extxyz"
    bat_sum = ff.main(
        [
            "--predictions",
            str(csv_path),
            "--out",
            str(bat_out),
            "--n-conf",
            "2",
            "--displaced-samples",
            "1",
            "--dedupe-inchikey",
            "--only-valid",
            "--calc-batch-size",
            "4",
            "--embed-workers",
            "3",
        ]
    )

    assert bat_calc.batch_sizes  # used batched calculator calls
    assert bat_sum["frames_written"] == seq_sum["frames_written"] > 0
    assert bat_sum["rows"] == seq_sum["rows"]

    seq_manifest = seq_out.with_suffix(".manifest.jsonl").read_text().splitlines()
    bat_manifest = bat_out.with_suffix(".manifest.jsonl").read_text().splitlines()
    seq_recs = [json.loads(ln) for ln in seq_manifest]
    bat_recs = [json.loads(ln) for ln in bat_manifest]
    assert seq_recs == bat_recs

    seq_frames = read(seq_out, index=":", format="extxyz")
    bat_frames = read(bat_out, index=":", format="extxyz")
    assert len(seq_frames) == len(bat_frames)
    for fa, fb in zip(seq_frames, bat_frames, strict=True):
        assert fa.get_chemical_symbols() == fb.get_chemical_symbols()
        assert fa.info == fb.info
        np.testing.assert_allclose(fa.positions, fb.positions, atol=1e-7)
        np.testing.assert_allclose(fa.arrays["mace_forces"], fb.arrays["mace_forces"], atol=1e-7)


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--calc-batch-size", "0"],
        ["--embed-workers", "0"],
        ["--calc-batch-size", "4", "--relax"],
        ["--embed-workers", "2", "--relax"],
    ],
)
def test_cli_batch_argument_validation(tmp_path, bad_args):
    ff = _load_ff()
    with pytest.raises(SystemExit):
        ff.main(["--predictions", "unused.csv", "--out", str(tmp_path / "x.extxyz"), *bad_args])
