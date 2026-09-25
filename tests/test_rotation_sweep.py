"""Rotation-sweep evaluation on a dummy model that is exactly C4-invariant and nothing more."""

import hashlib
import json

import torch

from ouroboros.data.loader import ShardDataset
from ouroboros.eval.evaluate import SWEEP_ANGLES, angle_flags, on_grid, run_eval


class DummyC4:
    """Looks up the SMILES of a C4-canonicalized image: exact under 90-deg rotations only."""

    def __init__(self, ds):
        self.table = {self.key(ds[i]["image"]): ds[i]["smiles"] for i in range(len(ds))}

    @staticmethod
    def key(img: torch.Tensor) -> str:
        q = (img * 255).round().to(torch.uint8)
        orbit = [torch.rot90(q, k, dims=(-2, -1)).numpy().tobytes() for k in range(4)]
        return hashlib.sha1(min(orbit)).hexdigest()

    def __call__(self, images):
        return [self.table.get(self.key(x), "") for x in images]


def test_on_grid_flags():
    assert on_grid(90, 4) and on_grid(45, 8) and not on_grid(45, 4) and on_grid(22.5, 16)
    assert angle_flags(135) == {
        "pixel_exact": False,
        "on_grid_C4": False,
        "on_grid_C8": True,
        "on_grid_C16": True,
    }
    assert len(SWEEP_ANGLES) == 24 and SWEEP_ANGLES[1] == 15


def test_sweep_separates_on_and_off_grid_and_kinds(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    ds = ShardDataset(root / "shards", "test")
    model = DummyC4(ds)
    summary = run_eval(
        model,
        {"synthetic_test": (ds, "rendered"), "pretend_real": (ds, "real")},
        tmp_path,
        angles=SWEEP_ANGLES,
        batch_size=8,
        n_boot=200,
    )
    ent = summary["entries"]
    assert len(ent) == 2 * 24
    assert {e["kind"] for e in ent} == {"rendered", "real"}
    for e in ent:
        assert e["n"] == len(ds)
        if e["pixel_exact"]:
            assert e["exact"] == 1.0 and e["exact_ci95"] == [1.0, 1.0]
        else:
            assert e["exact"] == 0.0 and e["invalid_rate"] == 1.0
    assert summary["footnote"] and not summary["parity_verified"]
    lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 2 * 24 * len(ds) and "angle" in json.loads(lines[0])
