"""Parallel ETKDG conformer embedding and batched single-point force-field evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from ouroboros.geometry.conformers import embed


@dataclass
class SinglePointResult:
    energy: float | None = None
    forces: np.ndarray | None = None
    fmax: float | None = None
    error: str | None = None


def embed_many(
    items: Sequence[tuple[str, int, int]],
    workers: int = 1,
    embed_fn: Callable = embed,
) -> list[tuple[Chem.Mol | None, str, list[dict]]]:
    """Embed ``(smiles, n_conf, seed)`` items in input order using up to ``workers`` threads.

    RDKit's C++ ``EmbedMultipleConfs`` releases the GIL and uses a private ``ETKDGv3`` parameter
    object seeded per molecule, so multi-threaded execution produces identical conformers to
    sequential calls.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not items:
        return []

    def _one(item: tuple[str, int, int]) -> tuple[Chem.Mol | None, str, list[dict]]:
        smiles, n_conf, seed = item
        attempts: list[dict] = []
        mol, reason = embed_fn(smiles, n_conf, seed, diagnostics=attempts)
        return mol, reason, attempts

    if workers == 1 or len(items) == 1:
        return [_one(it) for it in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(_one, items))


def _finish(energy: float, forces: np.ndarray) -> SinglePointResult:
    arr = np.asarray(forces, dtype=float)
    fmax = float(np.linalg.norm(arr, axis=1).max()) if len(arr) else 0.0
    return SinglePointResult(energy=float(energy), forces=arr, fmax=fmax, error=None)


def _eval_one(atoms, calc) -> SinglePointResult:
    try:
        work = atoms.copy()
        work.calc = calc
        forces = work.get_forces()
        energy = float(work.get_potential_energy())
        return _finish(energy, forces)
    except (RuntimeError, ValueError) as exc:
        return SinglePointResult(error=f"mace_failed: {exc}")


def _is_mace_calculator(calc) -> bool:
    if getattr(calc, "use_compile", False):
        return False
    if getattr(calc, "pad_num_atoms", 0) > 0 or getattr(calc, "pad_num_edges", 0) > 0:
        return False
    required = ("models", "z_table", "r_max", "default_dtype", "device")
    return all(hasattr(calc, attr) for attr in required) and bool(calc.models)


def _eval_mace_batch(chunk: Sequence, calc) -> list[SinglePointResult]:
    import torch
    from mace import data as mace_data
    from mace.tools import torch_geometric, torch_tools

    arrays_keys = dict(getattr(calc, "arrays_keys", None) or {})
    charges_key = getattr(calc, "charges_key", "Qs")
    if charges_key:
        arrays_keys[charges_key] = "charges"
    keyspec = mace_data.KeySpecification(
        info_keys=getattr(calc, "info_keys", None),
        arrays_keys=arrays_keys,
    )
    head = getattr(calc, "head", None)
    available_heads = getattr(calc, "available_heads", None)
    with torch_tools.default_dtype(calc.default_dtype):
        graphs = [
            mace_data.AtomicData.from_config(
                mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=head),
                z_table=calc.z_table,
                cutoff=calc.r_max,
                heads=available_heads,
            )
            for atoms in chunk
        ]
    batch_base = torch_geometric.Batch.from_data_list(graphs).to(calc.device)

    n_models = len(calc.models)
    e_scale = float(getattr(calc, "energy_units_to_eV", 1.0))
    f_scale = e_scale / float(getattr(calc, "length_units_to_A", 1.0))
    energy_sum = None
    forces_sum = None

    for model in calc.models:
        batch = batch_base.clone()
        model_dtype = next(model.parameters()).dtype
        for key in batch.keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=model_dtype)
        with torch.enable_grad():
            out = model(
                batch.to_dict(),
                compute_stress=False,
                training=False,
                compute_edge_forces=False,
                compute_atomic_stresses=False,
            )
        e_cur = out["energy"].detach().reshape(-1).to(dtype=torch.float64, device="cpu") * e_scale
        f_cur = out["forces"].detach().to(dtype=torch.float64, device="cpu") * f_scale
        energy_sum = e_cur if energy_sum is None else energy_sum + e_cur
        forces_sum = f_cur if forces_sum is None else forces_sum + f_cur

    energies = (energy_sum / n_models).numpy()
    forces_all = (forces_sum / n_models).numpy()

    results: list[SinglePointResult] = []
    offset = 0
    for idx, atoms in enumerate(chunk):
        n_atoms = len(atoms)
        results.append(_finish(float(energies[idx]), forces_all[offset : offset + n_atoms]))
        offset += n_atoms
    return results


def evaluate_single_points(
    atoms_list: Sequence,
    calc,
    batch_size: int = 1,
) -> list[SinglePointResult]:
    """Evaluate single-point energies and forces for ``atoms_list`` in chunks of ``batch_size``.

    Supports:
    * calculators implementing ``calc.calculate_batch(atoms_chunk) -> Sequence[(energy, forces)]``;
    * ``MACECalculator`` instances (``mace-torch`` single or committee models) via multi-graph
      ``torch_geometric.Batch`` forward passes;
    * any standard ASE ``Calculator`` via sequential fallback.

    If a multi-structure batch raises ``RuntimeError`` or ``ValueError``, that chunk automatically
    falls back to per-structure evaluation so a single bad conformer never invalidates neighbours.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not atoms_list:
        return []
    if calc is None:
        return [SinglePointResult() for _ in atoms_list]
    if batch_size == 1 or len(atoms_list) == 1:
        return [_eval_one(atoms, calc) for atoms in atoms_list]

    custom_batch = getattr(calc, "calculate_batch", None)
    use_mace = not callable(custom_batch) and _is_mace_calculator(calc)

    out: list[SinglePointResult] = []
    for start in range(0, len(atoms_list), batch_size):
        chunk = atoms_list[start : start + batch_size]
        if len(chunk) == 1:
            out.append(_eval_one(chunk[0], calc))
            continue
        if callable(custom_batch):
            try:
                pairs = custom_batch([a.copy() for a in chunk])
                if len(pairs) != len(chunk):
                    raise ValueError("calculate_batch returned wrong number of results")
                out.extend(_finish(e, f) for e, f in pairs)
                continue
            except (RuntimeError, ValueError):
                out.extend(_eval_one(a, calc) for a in chunk)
                continue
        if use_mace:
            try:
                out.extend(_eval_mace_batch(chunk, calc))
                continue
            except (RuntimeError, ValueError):
                out.extend(_eval_one(a, calc) for a in chunk)
                continue
        out.extend(_eval_one(a, calc) for a in chunk)
    return out
