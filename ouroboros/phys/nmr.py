"""Topological O(3) symmetry counting + Boltzmann 3D 13C/1H NMR & Karplus 3J_HH inversion.

Combines zero-cost exact achiral symmetry signal counting (homotopic + enantiotopic orbits),
Boltzmann-weighted 3D through-space steric/gamma-gauche chemical shift prediction, Karplus
vicinal 3J_HH dihedral couplings, and exact monotone dynamic-programming peak matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from ouroboros.eval.standardize import standardize
from ouroboros.phys.epimers import expand_superatoms
from ouroboros.phys.strain import StrainResult, evaluate_3d_strain


@dataclass(frozen=True)
class H1Multiplet:
    ppm: float
    integration: int = 1
    multiplicity: str = "m"
    j_hz: tuple[float, ...] = ()


@dataclass
class NMRPrediction:
    c13_sym_count: int
    dept_counts: dict[str, int]
    c13_shifts_ppm: list[float]
    h1_shifts_ppm: list[float]
    vicinal_j_hh_hz: list[float]
    h1_orbit_j_pairs: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class NMRMatchResult:
    valid: bool
    c13_sym_count: int | None = None
    dept_counts: dict[str, int] = field(default_factory=dict)
    c13_mae_ppm: float | None = None
    c13_scaled_mae_ppm: float | None = None
    c13_max_err_ppm: float | None = None
    h1_mae_ppm: float | None = None
    j_hh_mae_hz: float | None = None
    signal_count_penalty: float = 0.0
    dept_penalty: float = 0.0
    c13_loss: float = 0.0
    h1_j_loss: float = 0.0
    loss: float = 0.0
    prediction: NMRPrediction | None = None


class NMRForwardPredictor(Protocol):
    def predict(self, mol_no_h: Chem.Mol, strain: StrainResult | None) -> NMRPrediction: ...


def _invert_tetrahedral_inplace(mol: Chem.Mol) -> Chem.Mol:
    mm = Chem.Mol(mol)
    for atom in mm.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(mm, force=True, cleanIt=True)
    return mm


def _perm_sign(mapped: Sequence[int], orig: Sequence[int]) -> int:
    """Return +1 for even permutation, -1 for odd permutation, or 0 if not a self-bijection."""
    pos = {val: idx for idx, val in enumerate(orig)}
    if any(x not in pos for x in mapped):
        return 0
    arr = [pos[x] for x in mapped]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return 1 if (inv % 2) == 0 else -1


def _local_tetrahedral_det_ok(mol_no_h: Chem.Mol, match: Sequence[int], expected_det: int) -> bool:
    """Enforce O(3) local Jacobian sign (+1 for SO(3), -1 for improper reflection) at fixed sp3
    tetrahedral centers so diastereotopic geminal/isopropyl groups are never falsely merged."""
    for p in range(mol_no_h.GetNumAtoms()):
        if match[p] != p:
            continue
        atom = mol_no_h.GetAtomWithIdx(p)
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        if atom.GetAtomicNum() not in (6, 14):
            continue
        deg = atom.GetDegree()
        n_h = atom.GetTotalNumHs()
        if not (deg == 4 or (deg == 3 and n_h == 1)):
            continue
        nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
        mapped_nbrs = [match[x] for x in nbrs]
        sign = _perm_sign(mapped_nbrs, nbrs)
        if sign != 0 and sign != expected_det:
            return False
    return True


def _achiral_symmetry_orbits(mol_no_h: Chem.Mol) -> list[int]:
    """Group heavy atoms that are homotopic (SO(3)) or enantiotopic in achiral solution NMR."""
    n = mol_no_h.GetNumAtoms()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for match in mol_no_h.GetSubstructMatches(
        mol_no_h, uniquify=False, useChirality=True, maxMatches=256
    ):
        if _local_tetrahedral_det_ok(mol_no_h, match, expected_det=1):
            for i, j in enumerate(match):
                union(i, j)

    mm = _invert_tetrahedral_inplace(mol_no_h)
    if Chem.MolToSmiles(mol_no_h, isomericSmiles=True) == Chem.MolToSmiles(mm, isomericSmiles=True):
        for match in mol_no_h.GetSubstructMatches(
            mm, uniquify=False, useChirality=True, maxMatches=256
        ):
            if _local_tetrahedral_det_ok(mol_no_h, match, expected_det=-1):
                for i, j in enumerate(match):
                    union(i, j)

    return [find(i) for i in range(n)]


def optimal_1d_match_stats(
    pred: Sequence[float], obs: Sequence[float]
) -> tuple[float, float] | None:
    """Exact minimum-L1 injective matching via monotone DP; returns (mae, max_abs_err)."""
    if not pred or not obs:
        return None
    a = np.sort(np.asarray(pred, dtype=float))
    b = np.sort(np.asarray(obs, dtype=float))
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    dp = np.full((m + 1, n + 1), np.inf, dtype=float)
    take = np.zeros((m + 1, n + 1), dtype=bool)
    dp[0, :] = 0.0
    for i in range(1, m + 1):
        slack = n - (m - i)
        for j in range(i, slack + 1):
            cost = abs(a[i - 1] - b[j - 1])
            cand_take = dp[i - 1, j - 1] + cost
            if cand_take <= dp[i, j - 1]:
                dp[i, j] = cand_take
                take[i, j] = True
            else:
                dp[i, j] = dp[i, j - 1]

    max_err = 0.0
    ci, cj = m, n
    while ci > 0 and cj > 0:
        if take[ci, cj]:
            err = float(abs(a[ci - 1] - b[cj - 1]))
            if err > max_err:
                max_err = err
            ci -= 1
            cj -= 1
        else:
            cj -= 1
    return float(dp[m, n] / m), max_err


def optimal_1d_match_mae(pred: Sequence[float], obs: Sequence[float]) -> float | None:
    """Exact minimum-L1 injective matching MAE between sorted 1D shift lists."""
    stats = optimal_1d_match_stats(pred, obs)
    return stats[0] if stats is not None else None


_ALPHA_INC_C13 = {
    7: 26.5,
    8: 44.5,
    9: 68.0,
    14: -4.0,
    15: 12.0,
    16: 18.0,
    17: 28.0,
    35: 19.0,
    53: -6.0,
}
_BETA_INC_C13 = {6: 7.2, 7: 6.5, 8: 6.8, 9: 5.0, 16: 6.0, 17: 7.5, 35: 8.0}


def _base_c13_shift(mol: Chem.Mol, idx: int) -> float:
    atom = mol.GetAtomWithIdx(idx)
    hyb = atom.GetHybridization()
    nbrs = list(atom.GetNeighbors())
    z_nbrs = [nb.GetAtomicNum() for nb in nbrs]

    if atom.GetIsAromatic():
        base = 128.0
        for nb in nbrs:
            z = nb.GetAtomicNum()
            if z == 8:
                base += 27.0
            elif z == 7:
                base += 18.0
            elif z == 9:
                base += 34.0
            elif z == 6 and not nb.GetIsAromatic():
                base += 11.0
        return base

    if hyb == Chem.HybridizationType.SP2:
        has_dbl_o = any(
            b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(atom).GetAtomicNum() == 8
            for b in atom.GetBonds()
        )
        if has_dbl_o:
            n_het = sum(1 for z in z_nbrs if z in (7, 8))
            return 172.0 if n_het >= 2 else 212.0
        if len(nbrs) == 1:
            return 108.0  # terminal exocyclic/vinylidene =CH2
        return (
            124.0
            + 9.5 * sum(1 for z in z_nbrs if z == 6)
            + sum(8.0 for z in z_nbrs if z in (7, 8, 9, 17))
        )

    if hyb == Chem.HybridizationType.SP:
        return 118.0 if 7 in z_nbrs else 81.0

    # sp3 carbon: calibrated Grant-Paul / Dalling-Grant increments
    deg_c = sum(1 for z in z_nbrs if z == 6)
    base = 6.8 + 8.6 * deg_c
    for nb in nbrs:
        z = nb.GetAtomicNum()
        base += _ALPHA_INC_C13.get(z, 0.0)
        if z == 6 and nb.GetDegree() >= 3:
            base -= 1.8 * (nb.GetDegree() - 2)  # branching steric correction
        for b_nb in nb.GetNeighbors():
            if b_nb.GetIdx() != idx:
                base += 0.34 * _BETA_INC_C13.get(b_nb.GetAtomicNum(), 0.0)

    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings() if idx in r]
    if 3 in ring_sizes:
        base += 20.0 if (8 in z_nbrs or 7 in z_nbrs) else -13.5
    elif 4 in ring_sizes and len(ring_sizes) >= 2:
        base += 6.5  # strained BCP / BCH / cubane / pinane bridgehead shift
    elif 5 in ring_sizes and len(ring_sizes) >= 2:
        base += 5.5  # norbornane / bicyclo[2.2.1]heptane bridge enhancement
    return base


def _through_space_c13_corrections(mol_no_h: Chem.Mol, strain: StrainResult | None) -> np.ndarray:
    """Boltzmann-weighted 3D Grant-Cheney-Roberts gamma-gauche & steric shielding shifts."""
    n = mol_no_h.GetNumAtoms()
    corr = np.zeros(n, dtype=float)
    if strain is None or not strain.embed_ok or strain.mol_3d is None:
        return corr
    dmat_top = Chem.GetDistanceMatrix(mol_no_h)
    z_heavy = [mol_no_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)]
    alpha_carbons = [
        [nb.GetIdx() for nb in mol_no_h.GetAtomWithIdx(j).GetNeighbors() if nb.GetAtomicNum() == 6]
        for j in range(n)
    ]
    mol_3d = strain.mol_3d
    for conf, w in zip(mol_3d.GetConformers(), strain.boltzmann_weights, strict=True):
        pos = conf.GetPositions()[:n]
        conf_shift = np.zeros(n, dtype=float)
        for i in range(n):
            if z_heavy[i] != 6:
                continue
            for j in range(n):
                top_d = dmat_top[i, j]
                if top_d < 3:
                    continue
                r = float(np.linalg.norm(pos[i] - pos[j]))
                if top_d == 3 and r < 3.35:
                    # Gamma-gauche upfield shielding (Roberts 1970: ~ -4.5 to -5.5 ppm at ~2.95 A,
                    # ~ 0 ppm for antiperiplanar at ~3.85 A); also shields C_alpha bearing axial j.
                    gauche = -4.6 * np.exp(-max(0.0, r - 2.85) / 0.32)
                    conf_shift[i] += gauche
                    if z_heavy[j] != 6:
                        for c_alpha in alpha_carbons[j]:
                            conf_shift[c_alpha] += 0.52 * gauche
                elif top_d >= 4 and r < 3.25:
                    # Syn-pentane / delta compression (e.g. syn-C7-Me vs exo-2-OH in isoborneol)
                    delta_comp = -3.4 * np.exp(-max(0.0, r - 2.65) / 0.35)
                    conf_shift[i] += delta_comp
        corr += w * conf_shift
    return corr


def _predict_h1_and_karplus(
    mol_no_h: Chem.Mol, strain: StrainResult | None, orbits: Sequence[int]
) -> tuple[list[float], list[float]]:
    """Predict unique 1H group shifts and Boltzmann-weighted vicinal 3J_HH Karplus couplings."""
    h_shifts_by_orbit: dict[int, list[float]] = {}
    for i in range(mol_no_h.GetNumAtoms()):
        atom = mol_no_h.GetAtomWithIdx(i)
        n_h = atom.GetTotalNumHs()
        if n_h == 0:
            continue
        z = atom.GetAtomicNum()
        if z != 6:
            base = 4.8 if z == 8 else 6.2
        elif atom.GetIsAromatic():
            base = 7.35
        elif atom.GetHybridization() == Chem.HybridizationType.SP2:
            has_co = any(nb.GetAtomicNum() == 8 for nb in atom.GetNeighbors())
            base = 9.65 if has_co else 5.65
        else:
            deg_c = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6)
            base = 0.90 + 0.36 * deg_c
            for nb in atom.GetNeighbors():
                nz = nb.GetAtomicNum()
                if nz == 8:
                    base += 2.35
                elif nz == 7:
                    base += 1.65
                elif nz in (9, 17, 35):
                    base += 2.15
                elif nb.GetIsAromatic():
                    base += 1.25
                elif nb.GetHybridization() == Chem.HybridizationType.SP2:
                    base += 1.05
        h_shifts_by_orbit.setdefault(orbits[i], []).append(base)

    # 3D through-space heteroatom proximity deshielding on CH protons
    if strain is not None and strain.embed_ok and strain.mol_3d is not None:
        mol_3d = strain.mol_3d
        n_heavy = mol_no_h.GetNumAtoms()
        dmat = Chem.GetDistanceMatrix(mol_3d)
        h_3d_delta: dict[int, float] = {}
        for h_idx in range(n_heavy, mol_3d.GetNumAtoms()):
            h_atom = mol_3d.GetAtomWithIdx(h_idx)
            if h_atom.GetAtomicNum() != 1 or not h_atom.GetNeighbors():
                continue
            parent_idx = h_atom.GetNeighbors()[0].GetIdx()
            if parent_idx >= n_heavy:
                continue
            delta = 0.0
            for conf, w in zip(mol_3d.GetConformers(), strain.boltzmann_weights, strict=True):
                pos = conf.GetPositions()
                for j in range(n_heavy):
                    if dmat[h_idx, j] < 4:
                        continue
                    zj = mol_3d.GetAtomWithIdx(j).GetAtomicNum()
                    if zj in (7, 8, 9, 17):
                        r = float(np.linalg.norm(pos[h_idx] - pos[j]))
                        if r < 3.2:
                            delta += w * 0.65 * np.exp(-(r - 2.2) / 0.5)
            h_3d_delta[orbits[parent_idx]] = max(h_3d_delta.get(orbits[parent_idx], 0.0), delta)
        for orb, vals in h_shifts_by_orbit.items():
            add = h_3d_delta.get(orb, 0.0)
            h_shifts_by_orbit[orb] = [v + add for v in vals]

    orbit_h_mean = {orb: float(np.mean(v)) for orb, v in h_shifts_by_orbit.items()}
    h1_shifts = sorted(orbit_h_mean.values())

    # Boltzmann-weighted Karplus 3J_HH across vicinal H-C(sp3)-C(sp3)-H dihedrals
    j_couplings: list[float] = []
    orbit_max_j: dict[int, float] = {}
    if strain is not None and strain.embed_ok and strain.mol_3d is not None:
        mol_3d = strain.mol_3d
        seen_bonds: set[tuple[int, int]] = set()
        for bond in mol_no_h.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            au, av = mol_no_h.GetAtomWithIdx(u), mol_no_h.GetAtomWithIdx(v)
            if au.GetAtomicNum() != 6 or av.GetAtomicNum() != 6:
                continue
            if (
                au.GetHybridization() != Chem.HybridizationType.SP3
                or av.GetHybridization() != Chem.HybridizationType.SP3
            ):
                continue
            pair = (min(orbits[u], orbits[v]), max(orbits[u], orbits[v]))
            hu = [
                nb.GetIdx()
                for nb in mol_3d.GetAtomWithIdx(u).GetNeighbors()
                if nb.GetAtomicNum() == 1
            ]
            hv = [
                nb.GetIdx()
                for nb in mol_3d.GetAtomWithIdx(v).GetNeighbors()
                if nb.GetAtomicNum() == 1
            ]
            if not hu or not hv:
                continue
            elec = 0.45 * sum(
                1 for x in (au, av) for nb in x.GetNeighbors() if nb.GetAtomicNum() in (7, 8, 9, 17)
            )
            max_j_bond = 0.0
            for h1 in hu:
                for h2 in hv:
                    j_val = 0.0
                    for conf, w in zip(
                        mol_3d.GetConformers(), strain.boltzmann_weights, strict=True
                    ):
                        phi = rdMolTransforms.GetDihedralRad(conf, h1, u, v, h2)
                        c = float(np.cos(phi))
                        karplus = max(0.2, 7.76 * (c**2) - 1.10 * c + 1.40 - elec)
                        j_val += w * karplus
                    if j_val > max_j_bond:
                        max_j_bond = j_val
            orbit_max_j[orbits[u]] = max(orbit_max_j.get(orbits[u], 0.0), float(max_j_bond))
            orbit_max_j[orbits[v]] = max(orbit_max_j.get(orbits[v], 0.0), float(max_j_bond))
            if pair not in seen_bonds:
                seen_bonds.add(pair)
                j_couplings.append(round(float(max_j_bond), 3))

    j_couplings.sort()
    orbit_pairs = sorted(
        (round(h_ppm, 3), round(orbit_max_j.get(orb, 0.0), 3))
        for orb, h_ppm in orbit_h_mean.items()
    )
    return h1_shifts, j_couplings, orbit_pairs


class Boltzmann3DNMRPredictor:
    """Default CPU-fast 3D + topological NMR predictor (swap-compatible with CASCADE/GIAO)."""

    def predict(self, mol_no_h: Chem.Mol, strain: StrainResult | None = None) -> NMRPrediction:
        orbits = _achiral_symmetry_orbits(mol_no_h)
        corr_3d = _through_space_c13_corrections(mol_no_h, strain)

        dept = {"C": 0, "CH": 0, "CH2": 0, "CH3": 0}
        c_by_orbit: dict[int, list[float]] = {}
        orbit_n_h: dict[int, int] = {}
        for idx in range(mol_no_h.GetNumAtoms()):
            atom = mol_no_h.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6:
                continue
            orb = orbits[idx]
            shift = _base_c13_shift(mol_no_h, idx) + float(corr_3d[idx])
            c_by_orbit.setdefault(orb, []).append(shift)
            orbit_n_h[orb] = min(3, atom.GetTotalNumHs())

        dept_keys = ("C", "CH", "CH2", "CH3")
        for _orb, nh in orbit_n_h.items():
            dept[dept_keys[nh]] += 1

        c13_shifts = sorted(float(np.mean(vals)) for vals in c_by_orbit.values())
        h1_shifts, j_couplings, orbit_pairs = _predict_h1_and_karplus(mol_no_h, strain, orbits)
        return NMRPrediction(
            c13_sym_count=len(c13_shifts),
            dept_counts=dept,
            c13_shifts_ppm=c13_shifts,
            h1_shifts_ppm=h1_shifts,
            vicinal_j_hh_hz=j_couplings,
            h1_orbit_j_pairs=orbit_pairs,
        )


DEFAULT_PREDICTOR = Boltzmann3DNMRPredictor()


def predict_nmr(
    smiles: str,
    *,
    strain: StrainResult | None = None,
    n_conf: int = 8,
    seed: int = 0,
    predictor: NMRForwardPredictor | None = None,
) -> NMRPrediction:
    expanded = expand_superatoms(smiles)
    std = standardize(expanded)
    if not std.valid or std.mol is None:
        raise ValueError(f"cannot standardize SMILES for NMR: {smiles!r}")
    if strain is None:
        strain = evaluate_3d_strain(std.smiles, n_conf=n_conf, seed=seed)
    engine = predictor or DEFAULT_PREDICTOR
    return engine.predict(std.mol, strain)


def score_nmr(
    smiles: str | None,
    *,
    obs_c13_ppm: Sequence[float] = (),
    obs_dept_counts: dict[str, int] | None = None,
    obs_h1_multiplets: Sequence[H1Multiplet] = (),
    strain: StrainResult | None = None,
    predictor: NMRForwardPredictor | None = None,
    use_h1_and_j: bool = True,
) -> NMRMatchResult:
    """Compute unassigned 13C/1H/3J_HH match loss with asymmetric symmetry-count penalties."""
    expanded = expand_superatoms(smiles) if smiles else None
    std = standardize(expanded)
    if not std.valid or std.mol is None or any(a.GetAtomicNum() == 0 for a in std.mol.GetAtoms()):
        return NMRMatchResult(valid=False, c13_loss=30.0, loss=30.0)

    if strain is None:
        strain = evaluate_3d_strain(std.smiles)
    engine = predictor or DEFAULT_PREDICTOR
    pred = engine.predict(std.mol, strain)

    c13_loss = 0.0
    sig_pen = 0.0
    c13_mae: float | None = None
    c13_max: float | None = None
    if obs_c13_ppm:
        n_obs = len(obs_c13_ppm)
        n_sym = pred.c13_sym_count
        # Asymmetric penalty: observing MORE peaks than symmetry allows strongly falsifies a
        # meso/symmetric structure; observing 1-2 fewer peaks can occur via accidental overlap.
        if n_obs > n_sym:
            sig_pen = 3.5 * float(n_obs - n_sym)
        elif n_sym - n_obs > 2:
            sig_pen = 1.0 * float(n_sym - n_obs - 2)
        c13_stats = optimal_1d_match_stats(pred.c13_shifts_ppm, obs_c13_ppm)
        if c13_stats is not None:
            c13_mae, c13_max = c13_stats
            c13_loss += sig_pen + 0.65 * c13_mae + 0.35 * c13_max
        else:
            c13_loss += sig_pen

    dept_pen = 0.0
    if obs_dept_counts:
        for key in ("C", "CH", "CH2", "CH3"):
            if key in obs_dept_counts:
                dept_pen += 1.5 * abs(pred.dept_counts.get(key, 0) - int(obs_dept_counts[key]))
        c13_loss += dept_pen

    h1_j_loss = 0.0
    h1_mae: float | None = None
    j_mae: float | None = None
    if use_h1_and_j and obs_h1_multiplets:
        obs_h1_ppm = [m.ppm for m in obs_h1_multiplets]
        h1_mae = optimal_1d_match_mae(pred.h1_shifts_ppm, obs_h1_ppm)
        if h1_mae is not None:
            h1_j_loss += 1.4 * h1_mae

        # Orbit-coupled Karplus 3J_HH matching: match each multiplet's largest vicinal 3J (<14 Hz)
        # against the proton orbit in the matching chemical-shift window.
        j_errs: list[float] = []
        for mult in obs_h1_multiplets:
            vic_js = [j for j in mult.j_hz if 1.5 <= j < 14.0]
            if not vic_js:
                continue
            obs_j_max = max(vic_js)
            if pred.h1_orbit_j_pairs:
                coupled_orbits = [p for p in pred.h1_orbit_j_pairs if p[1] > 0.0]
                pool = coupled_orbits or pred.h1_orbit_j_pairs
                nearby = [p for p in pool if abs(p[0] - mult.ppm) <= 1.25]
                cands = nearby if nearby else [min(pool, key=lambda p: abs(p[0] - mult.ppm))]
                best_j_err = min(abs(p[1] - obs_j_max) for p in cands)
                j_errs.append(float(best_j_err))
            elif pred.vicinal_j_hh_hz:
                j_errs.append(float(min(abs(j - obs_j_max) for j in pred.vicinal_j_hh_hz)))
        if j_errs:
            j_mae = float(np.mean(j_errs))
            h1_j_loss += 0.55 * j_mae

    loss = c13_loss + h1_j_loss
    return NMRMatchResult(
        valid=True,
        c13_sym_count=pred.c13_sym_count,
        dept_counts=pred.dept_counts,
        c13_mae_ppm=c13_mae,
        c13_scaled_mae_ppm=c13_mae,
        c13_max_err_ppm=c13_max,
        h1_mae_ppm=h1_mae,
        j_hh_mae_hz=j_mae,
        signal_count_penalty=sig_pen,
        dept_penalty=dept_pen,
        c13_loss=float(c13_loss),
        h1_j_loss=float(h1_j_loss),
        loss=float(loss),
        prediction=pred,
    )
