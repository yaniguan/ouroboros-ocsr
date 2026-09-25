"""Real-data adapter, leakage check and the no-data pre-commit guard (Am1-A)."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rdkit import Chem

from ouroboros.data.loader import ShardDataset
from ouroboros.data.real import fit_to_canvas, ingest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stand_in_manifest(tmp_path: Path) -> Path:
    """A synthetic stand-in for a real manifest: odd image sizes/modes, salts, bad rows."""
    (tmp_path / "img").mkdir()
    rows = [
        ("a.png", "CC(=O)Oc1ccccc1C(=O)O", "test", (300, 120), "L"),
        ("b.jpg", "C[C@H](N)C(=O)O.Cl", "test", (90, 200), "RGB"),  # salt -> parent
        ("c.png", "CC(=O)[O-].[Na+]", "train", (50, 50), "RGBA"),
        ("d.png", "C1CC(", "test", (64, 64), "L"),  # unparsable label
        ("missing.png", "CCO", "test", None, None),  # image does not exist
        ("e.png", "", "val", (64, 64), "L"),  # empty label
    ]
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "smiles", "source", "split", "id"])
        for i, (name, smi, split, size, mode) in enumerate(rows):
            if size is not None:
                arr = np.full((size[1], size[0]), 255, np.uint8)
                arr[size[1] // 3 : size[1] // 2, 5:-5] = 0
                img = Image.fromarray(arr, "L").convert(mode)
                img.save(tmp_path / "img" / name)
            w.writerow([f"img/{name}", smi, "STANDIN", split, f"id{i}"])
    return tmp_path / "manifest.csv"


def test_fit_to_canvas_inscribed():
    img = fit_to_canvas(Image.new("L", (400, 100), 0), size=128)
    a = np.asarray(img)
    assert a.shape == (128, 128)
    yy, xx = np.mgrid[:128, :128]
    outside = (xx + 0.5 - 64) ** 2 + (yy + 0.5 - 64) ** 2 > 65**2
    assert (a[outside] == 255).all() and (a == 0).any()


def test_ingest_roundtrip_and_failure_log(tmp_path):
    out = tmp_path / "shards"
    stats = ingest(_stand_in_manifest(tmp_path), out, size=96, shard_size=2)
    assert stats["n_rows"] == 6
    assert stats["written"] == {"train": 1, "test": 2}
    assert stats["n_failures"] == 3
    assert stats["failure_reasons"] == {
        "label_unparsable": 1,
        "image_unreadable:FileNotFoundError": 1,
        "label_empty": 1,
    }
    fails = [json.loads(line) for line in (out / "ingest_failures.jsonl").read_text().splitlines()]
    assert {f["id"] for f in fails} == {"id3", "id4", "id5"}  # every failed row is logged
    ds = ShardDataset(out, "test")
    assert len(ds) == 2
    metas = [ds.meta(i) for i in range(2)]
    assert metas[1]["smiles"] == Chem.CanonSmiles("C[C@H](N)C(=O)O")  # salt stripped, stereo kept
    assert metas[1]["raw_smiles"] == "C[C@H](N)C(=O)O.Cl" and metas[1]["real"] == 1
    assert ds[0]["image"].shape == (1, 96, 96)
    assert ShardDataset(out, "train").meta(0)["smiles"] == "CC(=O)O"


def test_leakage_check_detects_overlap(tmp_path, capsys):
    mod = _load_script("check_leakage")
    ingest(_stand_in_manifest(tmp_path), tmp_path / "real", size=64)
    clean = tmp_path / "clean.tsv"
    clean.write_text("idx\tsmiles\n0\tCCCCO\n1\tc1ccccc1\n")
    leaky = tmp_path / "leaky.tsv"
    leaky.write_text("idx\tsmiles\n0\tCCCCO\n1\tOC(=O)c1ccccc1OC(C)=O\n")  # aspirin, reordered
    ev = f"standin={tmp_path / 'real'}:test"
    assert mod.main(["--eval", ev, "--train", f"clean={clean}"]) == 0
    assert mod.main(["--eval", ev, "--train", f"clean={clean}", f"leaky={leaky}"]) == 1
    printed = capsys.readouterr().out
    assert "leaky" in printed
    dump = tmp_path / "keys.txt"
    mod.main(["--eval", ev, "--dump-eval-keys", str(dump)])
    assert len(dump.read_text().split()) == 2


def test_exclusion_at_load_time(tmp_path):
    out = tmp_path / "shards"
    ingest(_stand_in_manifest(tmp_path), out, size=64)
    aspirin14 = Chem.MolToInchiKey(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))[:14]
    ds = ShardDataset(out, "test", exclude_key14={aspirin14})
    assert len(ds) == 1 and ds.n_excluded == 1


def test_no_data_guard_rules():
    mod = _load_script("check_no_data")
    ok = [
        ("ouroboros/data/real.py", 10_000),
        ("benchmarks/data/contact_sheet.png", 2_000_000),
        ("README.md", 100),
    ]
    assert mod.violations(ok) == []
    bad = [
        ("data/full/shards/train-000000.tar", 10),
        ("real/uspto/test-000000.tar", 10),
        ("notebooks/x.tar", 10),
        ("tests/fixtures/doc.png", 10),
        ("benchmarks/real_examples/fig.png", 10),
        ("configs/pool.tsv.gz", 10),
        ("ouroboros/big.py", 6 * 1024 * 1024),
    ]
    assert len(mod.violations(bad)) == len(bad)
