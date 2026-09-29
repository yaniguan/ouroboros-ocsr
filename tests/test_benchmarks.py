import json

import pytest
import torch

from ouroboros.data import build
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.train.benchmark import capacity, resume_comparison, validate_device


@pytest.fixture
def config(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    tok = SmilesTokenizer.build(
        r["smiles"] for r in build.read_manifest(root / "manifests/train.tsv")
    )
    vocab = tmp_path / "vocab.json"
    tok.save(vocab)
    return {
        "seed": 3,
        "data": {
            "synthetic_root": str(root / "shards"),
            "real_roots": [],
            "real_fraction": 0.0,
            "image_size": 64,
            "vocab": str(vocab),
            "num_workers": 0,
        },
        "model": {
            "encoder": {
                "type": "baseline",
                "widths": [8, 16, 16, 32],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 2,
                "mixer_ff": 64,
            },
            "decoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64, "max_len": 96},
        },
        "train": {
            "device": "cpu",
            "amp": "none",
            "batch_size": 4,
            "steps": 10,
            "lr": 1e-3,
            "warmup": 1,
            "grad_clip": 1.0,
            "rotation_aug": True,
            "log_every": 1,
            "ckpt_every": 2,
        },
    }


def test_current_model_capacity_smoke(config, tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "capacity.json"
    result = capacity(
        config, path, steps=2, warmup=1, batch=2, sequence_length=8, device="cpu", precision="fp32"
    )
    assert result["cpu_smoke_only"] and result["peak_allocated_bytes"] is None
    assert result["warm_images_per_second"] > 0 and len(result["losses"]) == 2
    assert json.loads(path.read_text())["parameters"] == result["parameters"]
    with pytest.raises(FileExistsError):
        capacity(config, path, device="cpu", precision="fp32")


def test_current_trainer_resume_smoke(config, tmp_path):
    torch.set_num_threads(1)
    result = resume_comparison(
        config,
        tmp_path / "resume",
        checkpoint_step=2,
        compared_steps=3,
        kill_after=1,
        tolerance=0,
        device="cpu",
        precision="fp32",
    )
    assert result["passed"] and result["max_relative_loss_difference"] == 0
    assert result["compared_steps"] == 3 and result["cpu_smoke_only"]


def test_capacity_never_labels_cpu_as_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        validate_device("cuda", "bf16")
    with pytest.raises(ValueError, match="CPU smoke"):
        validate_device("cpu", "bf16")
