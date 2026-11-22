"""Regression tests for script, pre-commit guard, HPC bounds, and doc link fixes."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from scripts import bench_infer, check_no_data, eval_diagnostics, memory_estimate

ROOT = Path(__file__).resolve().parents[1]


def test_eval_diagnostics_wilson_empty_and_predict_device_resize(tmp_path: Path) -> None:
    assert eval_diagnostics.wilson(0, 0) == [None, None]
    lo, hi = eval_diagnostics.wilson(5, 10)
    assert lo is not None and hi is not None and 0.0 < lo < hi < 1.0

    img_path = tmp_path / "mol.png"
    Image.new("RGB", (96, 64), color=(255, 255, 255)).save(img_path)

    seen: list[tuple[tuple[int, ...], torch.device]] = []

    class DummyModel:
        def generate(self, x: torch.Tensor) -> list[str]:
            seen.append((tuple(x.shape), x.device))
            return ["CCO" for _ in range(x.shape[0])]

    lab = SimpleNamespace(size=32, device=torch.device("cpu"), model=DummyModel())
    preds = eval_diagnostics.predict(lab, [img_path], batch=4)
    assert preds == ["CCO"]
    assert seen == [((1, 1, 32, 32), torch.device("cpu"))]


def test_bench_infer_windows_and_empty_split_helpers() -> None:
    name = bench_infer.cpu_name()
    assert isinstance(name, str) and len(name) > 0
    assert bench_infer.q([]) == {"p50": None, "p95": None, "max": None}
    assert bench_infer.selective([], 0.9) == {"coverage": 0.0, "precision": None}


def test_memory_estimate_imports_cross_platform() -> None:
    assert callable(memory_estimate.saved_bytes)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def loss(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            return (self.lin(x).sum() + ids.float().sum()) ** 2

    m = TinyModel()
    used = memory_estimate.saved_bytes(m, torch.ones(2, 4), torch.ones(2, 3, dtype=torch.long))
    assert used > 0


def test_check_no_data_allows_profiling_charts_and_blocks_leaks() -> None:
    clean = [
        ("benchmarks/rotation/plot.png", 1024),
        ("site/assets/fig.png", 2048),
        ("docs/profiling/charts/throughput.png", 4096),
    ]
    assert check_no_data.violations(clean) == []

    bad = [
        ("docs/profiling/charts/real_sample.png", 1024),
        ("docs/random_plot.png", 1024),
        ("checkpoints/model.pt", 1024),
    ]
    probs = check_no_data.violations(bad)
    assert len(probs) == 3


@pytest.mark.parametrize("bad_task", ["0", "37"])
def test_hpc_v1_task_rejects_out_of_range_task_id(bad_task: str, tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "hpc" / "v1_task.py"),
            "--task",
            bad_task,
            "--original",
            str(tmp_path),
            "--v0",
            str(tmp_path),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "--task must be in 1..36" in proc.stderr


def test_markdown_local_links_resolve() -> None:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    md_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    missing: list[str] = []
    for md in md_files:
        text = md.read_text(encoding="utf-8")
        # Strip fenced code blocks so example markdown snippets do not false-trigger.
        stripped = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for raw_target in link_re.findall(stripped):
            target = raw_target.strip().split()[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            rel_path = target.split("#", 1)[0]
            if not rel_path:
                continue
            resolved = (md.parent / rel_path).resolve()
            if not resolved.exists():
                missing.append(f"{md.relative_to(ROOT)} -> {raw_target}")
    assert missing == [], f"Broken markdown links: {missing}"
