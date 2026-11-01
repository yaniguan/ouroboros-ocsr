"""Real-data fine-tune path: PMC preparation, leakage gating, real mixing and step-0 validation."""

import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from ouroboros.data.real import ingest
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.eval.standardize import leakage_key, standardize
from ouroboros.train.trainer import Trainer, load_checkpoint


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png(shade=255, size=(40, 24)):
    blob = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(blob, format="PNG")
    return blob.getvalue()


def _parquet(path, smiles):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "image": [{"bytes": _png(), "path": None} for _ in smiles],
            "smiles": smiles,
            "n_agree_open": [2] * len(smiles),
            "pmcid": [f"PMC{i}" for i in range(len(smiles))],
            "figure_id": ["fig"] * len(smiles),
            "crop_index": list(range(len(smiles))),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


TEST_MOLECULE = "C[C@H](N)C(=O)O"  # held-out "test" molecule


@pytest.fixture
def pmc(tmp_path):
    keys = tmp_path / "eval_keys.txt"
    keys.write_text(leakage_key(standardize(TEST_MOLECULE)) + "\n")
    long_label = "C" * 140
    _parquet(
        tmp_path / "hf/data/train-00000-of-00001.parquet",
        ["CCO", "CCCO", "C[C@@H](N)C(=O)O", long_label, "c1ccccc1O", "CCN"],
    )
    _parquet(
        tmp_path / "hf/data/validation-00000-of-00001.parquet",
        ["CCO", "CC(=O)O", "NC(C)C(=O)O", "CCCCN", "OCCO"],
    )
    stats = _script("prepare_pmc").prepare(
        _script("prepare_pmc").find_parquets([str(tmp_path / "hf")]),
        tmp_path / "manifest",
        {k[:14] for k in keys.read_text().split()},
        max_len=128,
    )
    return tmp_path, keys, stats


def test_prepare_pmc_drops_test_molecules_and_overlong(pmc):
    tmp_path, _, stats = pmc
    # train: the enantiomer of the test molecule (same connectivity block) and the overlong label
    assert stats["dropped"]["train"] == {"eval_overlap": 1, "label_too_long": 1}
    assert stats["kept"]["train"] == 4
    # val: the stereo-free test molecule is removed; CCO is also in train (molecule-disjoint val)
    assert stats["dropped"]["val"] == {"eval_overlap": 1, "val_in_train": 1}
    assert stats["kept"]["val"] == 3
    rows = (tmp_path / "manifest/manifest.csv").read_text().splitlines()
    assert len(rows) == 1 + 4 + 3
    assert not any("C[C@@H](N)" in r or "NC(C)C(=O)O" in r for r in rows)
    tokens = json.loads((tmp_path / "manifest/train_tokens.json").read_text())
    assert tokens["C"] > 0


def test_leakage_check_connectivity_gate(pmc, tmp_path):
    _, keys, _ = pmc
    leak = _script("check_leakage")
    ingest(tmp_path / "manifest/manifest.csv", tmp_path / "real", size=32)
    # a test set containing the held-out molecule
    test_manifest = tmp_path / "test.csv"
    (tmp_path / "t.png").write_bytes(_png())
    test_manifest.write_text(f"image,smiles,source,split,id\nt.png,{TEST_MOLECULE},T,test,0\n")
    ingest(test_manifest, tmp_path / "test_shards", size=32)
    clean = [
        "--eval", f"t={tmp_path / 'test_shards'}:test",
        "--train", f"tr={tmp_path / 'real'}:train", f"va={tmp_path / 'real'}:val",
        "--fail-on-connectivity", "--json", str(tmp_path / "leak.json"),
    ]
    assert leak.main(clean) == 0
    assert all(r["overlap_connectivity"] == 0 for r in json.loads((tmp_path / "leak.json").read_text()))
    # an enantiomer in training: full InChIKey differs, connectivity block matches
    bad_manifest = tmp_path / "bad.csv"
    bad_manifest.write_text("image,smiles,source,split,id\nt.png,C[C@@H](N)C(=O)O,T,train,0\n")
    (tmp_path / "t.png").write_bytes(_png())
    ingest(bad_manifest, tmp_path / "bad_shards", size=32)
    stereo = ["--eval", f"t={tmp_path / 'test_shards'}:test",
              "--train", f"b={tmp_path / 'bad_shards'}:train"]
    assert leak.main(stereo) == 0  # default gate: full InChIKey only
    assert leak.main([*stereo, "--fail-on-connectivity"]) == 1


def _tiny_config(tmp_path, real_root, labels):
    synth = tmp_path / "synth"
    synth.mkdir()
    import tarfile

    blob = io.BytesIO()
    Image.new("L", (32, 32), 255).save(blob, format="PNG")
    with tarfile.open(synth / "train-000000.tar", "w") as tar:
        for i, label in enumerate(labels):
            for ext, data in (("png", blob.getvalue()), ("json", json.dumps({"smiles": label}).encode())):
                item = tarfile.TarInfo(f"s_{i}.{ext}")
                item.size = len(data)
                tar.addfile(item, io.BytesIO(data))
    vocab = tmp_path / "vocab.json"
    SmilesTokenizer.build(labels + ["CCCO", "c1ccccc1O", "CCN", "CC(=O)O", "CCCCN", "OCCO"]).save(vocab)
    return {
        "seed": 3,
        "data": {
            "synthetic_root": str(synth),
            "synthetic_max_samples": len(labels),
            "real_roots": [str(real_root)],
            "real_fraction": 0.5,
            "image_size": 32,
            "vocab": str(vocab),
            "num_workers": 0,
        },
        "model": {
            "encoder": {"type": "baseline", "widths": [4, 4, 8, 8], "blocks": [1, 1, 1, 1],
                        "mixer_layers": 1, "mixer_heads": 2, "mixer_ff": 32},
            "decoder": {"d_model": 16, "n_layers": 2, "n_heads": 2, "d_ff": 32, "max_len": 32,
                        "dropout": 0.1},
        },
        "train": {"device": "cpu", "amp": "none", "steps": 4, "batch_size": 4, "warmup": 1,
                  "lr": 1e-3, "ckpt_every": 2, "log_every": 1},
        "validation": {"enabled": True, "at_start": True, "keep_only_best": True, "root": str(real_root), "samples": 3,
                       "batch_size": 2, "every": 2, "patience": 5, "min_delta": 0.0},
    }


def test_real_mix_with_real_validation_at_start(pmc, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("OUROBOROS_NUM_WORKERS", "0")
    real = tmp_path / "real_pmc"
    ingest(tmp_path / "manifest/manifest.csv", real, size=32)
    cfg = _tiny_config(tmp_path, real, ["CCCC", "CCCCC", "CCCCCC", "CCOC"])
    trainer = Trainer(cfg, tmp_path / "run")
    assert len(trainer.real) == 4 and len(trainer.validation) == 3
    n_synth = len(trainer.synth)
    it = iter(trainer.sampler)
    first = [next(it) for _ in range(16)]
    assert sum(i >= n_synth for i in first) == 8  # exactly real_fraction * 16
    trainer.sampler.set_position(0)
    trainer.fit()
    log = [json.loads(x) for x in (tmp_path / "run/log.jsonl").read_text().splitlines()]
    assert log[0]["event"] == "start" and log[0]["n_real"] == 4 and log[0]["n_validation"] == 3
    vals = [r for r in log if r.get("event") == "validation"]
    assert [r["step"] for r in vals] == [0, 2, 4]
    state = load_checkpoint(tmp_path / "run/ckpt/last.pt")
    best = state["early_stop"]["best_step"]
    assert (tmp_path / "run/ckpt" / f"best-step-{best:08d}.pt").is_file()
    assert len(list((tmp_path / "run/ckpt").glob("best-step-*.pt"))) == 1  # keep_only_best
    assert vals[0]["early_stop"]["best_step"] == 0  # step 0 = the initial weights were scored
    # resuming a finished run neither re-validates step 0 nor duplicates log lines
    Trainer(cfg, tmp_path / "run").fit()
    again = (tmp_path / "run/log.jsonl").read_text().splitlines()
    assert len(again) == len(log)
