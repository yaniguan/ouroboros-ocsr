"""Regression tests for Issue #120: extxyz backslashes, UI dataset export, and conformer guards."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read as ase_read

from ouroboros.geometry.conformers import embed, embed_smiles, lowest_energy_conformer
from ouroboros.ui.backend import WorkbenchEngine
from scripts import image_to_forcefield, split_forcefield


def test_windows_paths_and_z_alkene_backslashes_roundtrip_extxyz_and_split(tmp_path: Path) -> None:
    preds = tmp_path / "preds.csv"
    with open(preds, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "pred", "image_path"])
        w.writeheader()
        w.writerow({"id": "cis_alkene", "pred": r"F/C=C\F", "image_path": r"C:\figs\sub\mol_1.png"})
        w.writerow({"id": "trans_alkene", "pred": "F/C=C/F", "image_path": r"figs\mol_2.png"})

    out_xyz = tmp_path / "ff.extxyz"
    summary = image_to_forcefield.main(
        [
            "--predictions",
            str(preds),
            "--out",
            str(out_xyz),
            "--mace-model",
            "none",
            "--n-conf",
            "1",
        ]
    )
    assert summary["frames_written"] == 2

    frames = ase_read(out_xyz, index=":", format="extxyz")
    by_id = {str(fr.info["source_id"]): fr for fr in frames}
    assert "\\" in by_id["cis_alkene"].info["smiles"]
    assert by_id["cis_alkene"].info["source_image"] == "C:/figs/sub/mol_1.png"
    assert by_id["trans_alkene"].info["source_image"] == "figs/mol_2.png"

    split_dir = tmp_path / "splits"
    split_res = split_forcefield.main(
        [
            "--frames",
            str(out_xyz),
            "--manifest",
            str(tmp_path / "ff.manifest.jsonl"),
            "--out-dir",
            str(split_dir),
        ]
    )
    assert sum(split_res["frames"].values()) == 2
    # Stereoisomers (cis vs trans) must land in the same split with intact backslash SMILES
    non_empty = [
        n for n in ("train", "val", "test") if (split_dir / f"{n}.extxyz").stat().st_size > 0
    ]
    assert len(non_empty) == 1
    split_frames = ase_read(split_dir / f"{non_empty[0]}.extxyz", index=":", format="extxyz")
    split_smiles = {str(fr.info["source_id"]): fr.info["smiles"] for fr in split_frames}
    assert "\\" in split_smiles["cis_alkene"]


def test_ui_forcefield_builder_exports_split_cleanly(tmp_path: Path) -> None:
    eng = WorkbenchEngine(device="cpu")
    res = eng.run_forcefield_builder(
        {
            "smiles_list": [r"F/C=C\F", "CCO"],
            "potential": "mmff94s",
            "n_conf": 1,
            "displaced_samples": 0,
            "expand_aliases": True,
            "dedupe_inchikey": True,
        }
    )
    assert res["accepted_molecules"] == 2
    xyz_path = tmp_path / "ui.extxyz"
    man_path = tmp_path / "ui.manifest.jsonl"
    xyz_path.write_text(res["extxyz_preview"], encoding="utf-8")
    man_path.write_text(res["manifest_jsonl"], encoding="utf-8")

    out_dir = tmp_path / "ui_splits"
    summary = split_forcefield.main(
        ["--frames", str(xyz_path), "--manifest", str(man_path), "--out-dir", str(out_dir)]
    )
    assert sum(summary["frames"].values()) == 2


def test_conformers_reject_dummy_and_multifragment_and_embed_smiles() -> None:
    assert embed("*c1ccccc1") == (None, "pseudo_atoms")
    assert embed("CCO.O") == (None, "multi_fragment")
    res = lowest_energy_conformer("*c1ccccc1", n_conf=1)
    assert res.status == "pseudo_atoms"

    emb = embed_smiles("C[C@H](O)C(=O)O", num_confs=2, seed=0)
    assert emb is not None
    assert emb.stereo_ok is True
    assert emb.positions.shape[1] == 3
    assert len(emb.atomic_numbers) == emb.positions.shape[0]
    assert len(emb.all_positions) == 2
    assert embed_smiles("*C") is None


def test_batched_alias_dedupe_and_salt_alias_expansion(tmp_path: Path) -> None:
    preds = tmp_path / "alias_dups.csv"
    with open(preds, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "pred"])
        w.writeheader()
        w.writerow({"id": "1", "pred": "[OMe]c1ccccc1"})
        w.writerow({"id": "2", "pred": "COc1ccccc1"})
        w.writerow({"id": "3", "pred": "CC(=O)[O-].[Na+]"})
        w.writerow({"id": "4", "pred": "CC[OMe].[Na+]"})

    out_xyz = tmp_path / "batch.extxyz"
    summary = image_to_forcefield.main(
        [
            "--predictions",
            str(preds),
            "--out",
            str(out_xyz),
            "--mace-model",
            "none",
            "--n-conf",
            "1",
            "--expand-aliases",
            "--dedupe-inchikey",
            "--embed-workers",
            "2",
        ]
    )
    assert summary["frames_written"] == 3
    records = [
        json.loads(line)
        for line in (tmp_path / "batch.manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {r["id"]: r for r in records}
    assert by_id["1"]["status"] == "ok"
    assert by_id["2"]["status"] == "duplicate_inchikey"
    assert by_id["3"]["status"] == "ok"
    assert by_id["4"]["status"] == "ok" and by_id["4"]["smiles"] == "CCOC"
