"""Disjoint, resumable GPU geometry benchmark on the fixed first 1,000 test rows."""

import argparse
import dataclasses
import importlib.metadata
import os
from pathlib import Path

import numpy as np
import torch

from ouroboros.data.build import read_manifest
from ouroboros.geometry.conformers import lowest_energy_conformer, mace_calculator
from ouroboros.provenance import JsonCache, atomic_json, file_sha256, implementation_identity


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--task", type=int, required=True, help="0..19, 50 molecules each")
    a = ap.parse_args()
    if not 0 <= a.task < 20:
        ap.error("task must be 0..19")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Requires a scheduled CUDA GPU")
    torch.set_num_threads(2)
    manifest = a.root / "data/full/manifests/test.tsv"
    model = a.root / "models/MACE-OFF23_medium.model"
    rows = read_manifest(manifest)
    if len(rows) < 1000:
        raise ValueError("Need at least 1,000 test molecules")
    out = a.root / "geometry" / f"part-{a.task:02d}"
    settings = {
        "manifest_sha256": file_sha256(manifest),
        "model_sha256": file_sha256(model),
        "model": "MACE-OFF23 medium",
        "dtype": "float64",
        "n_conf": 10,
        "seed": 0,
        "fmax": 0.05,
        "steps": 500,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "rdkit", "mace-torch")},
        "implementation": implementation_identity(Path("ouroboros/geometry")),
    }
    cache = JsonCache(out / "molecules")
    calc = mace_calculator(str(model), device="cuda", dtype="float64")
    results = []
    for index in range(a.task * 50, (a.task + 1) * 50):
        row = rows[index]

        def compute(row=row):
            result = lowest_energy_conformer(
                row["smiles"], n_conf=10, seed=0, calc=calc, fmax=0.05, steps=500
            )
            return plain(dataclasses.asdict(result))

        result = cache.get({**settings, "index": index, "row": row}, compute)
        results.append({"index": index, "result": result})
        print(index, result["status"], result["seconds"], flush=True)
        atomic_json(out / "progress.json", {"task": a.task, "completed": len(results)})
    atomic_json(
        out / "results.json",
        {
            "settings": settings,
            "job_id": os.environ["SLURM_JOB_ID"],
            "gpu": torch.cuda.get_device_name(),
            "denominator": 50,
            "lowest_converged": sum(r["result"]["converged"] for r in results),
            "all_10_converged": sum(
                r["result"]["all_converged"] and r["result"]["n_conf"] == 10 for r in results
            ),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
