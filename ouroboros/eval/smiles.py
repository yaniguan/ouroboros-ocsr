"""String-level SMILES utilities over the scoring rules of ``ouroboros.eval.standardize``.

These are conveniences for notebooks, scripts and the predictions harness; the scoring itself
stays in ``standardize`` / ``metrics`` so there is one definition of every metric. With the
defaults, each function agrees with the corresponding ``score_pair`` field:

  canonicalize(s)                  standardized canonical isomeric SMILES (None if invalid)
  canonicalize(s, stereo=False)    the same with all stereo removed
  is_valid(s)                      PairScore.valid
  exact_match(p, g)                PairScore.exact        (stereo-aware)
  exact_match(p, g, stereo=False)  PairScore.exact_flat   (stereo-agnostic)
  inchikey_match(p, g)             PairScore.inchikey     (standard InChIKey, stereo layers kept)
  tanimoto(p, g)                   PairScore.tanimoto     (Morgan r=2, 2048 bits, no chirality)
  batch_metrics(preds, golds)      metrics.aggregate over the pairs whose gold is valid

``standardize=False`` skips salt stripping / charge neutralization and only canonicalizes the
RDKit parse. An invalid gold never matches anything here; ``score_pair`` raises instead.
"""

from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ouroboros.eval import standardize as _std
from ouroboros.eval.metrics import PairScore, aggregate, score_pair

_GENERATORS: dict[tuple[int, int, bool], object] = {}


def _prep(smiles: str | None, standardize: bool) -> _std.Std:
    if standardize:
        return _std.standardize(smiles)
    mol = _std.parse(smiles)
    if mol is None:
        empty = smiles is None or not smiles.strip()
        return _std.Std(None, None, "empty" if empty else "unparsable")
    return _std.Std(mol, Chem.MolToSmiles(mol), "ok")


def is_valid(smiles: str | None, *, standardize: bool = True) -> bool:
    return _prep(smiles, standardize).valid


def canonicalize(
    smiles: str | None, *, stereo: bool = True, standardize: bool = True
) -> str | None:
    s = _prep(smiles, standardize)
    if not s.valid:
        return None
    return s.smiles if stereo else _std.flat_smiles(s)


def inchikey(
    smiles: str | None, *, connectivity_only: bool = False, standardize: bool = True
) -> str | None:
    """Standard InChIKey; ``connectivity_only`` keeps the first block (skeleton, no stereo)."""
    key = _std.standard_inchikey(_prep(smiles, standardize))
    return key[:14] if key and connectivity_only else key


def exact_match(
    pred: str | None, gold: str | None, *, stereo: bool = True, standardize: bool = True
) -> bool:
    p = canonicalize(pred, stereo=stereo, standardize=standardize)
    return p is not None and p == canonicalize(gold, stereo=stereo, standardize=standardize)


def inchikey_match(
    pred: str | None,
    gold: str | None,
    *,
    connectivity_only: bool = False,
    standardize: bool = True,
) -> bool:
    p = inchikey(pred, connectivity_only=connectivity_only, standardize=standardize)
    return p is not None and p == inchikey(
        gold, connectivity_only=connectivity_only, standardize=standardize
    )


def _morgan(radius: int, n_bits: int, chirality: bool):
    key = (radius, n_bits, chirality)
    if key not in _GENERATORS:
        _GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=chirality
        )
    return _GENERATORS[key]


def tanimoto(
    pred: str | None,
    gold: str | None,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    chirality: bool = False,
    standardize: bool = True,
) -> float:
    """Morgan-fingerprint Tanimoto similarity; 0.0 if either side is invalid."""
    p, g = _prep(pred, standardize), _prep(gold, standardize)
    if not (p.valid and g.valid):
        return 0.0
    gen = _morgan(radius, n_bits, chirality)
    return DataStructs.TanimotoSimilarity(gen.GetFingerprint(p.mol), gen.GetFingerprint(g.mol))


def score_batch(
    preds: Sequence[str | None], golds: Sequence[str | None]
) -> list[PairScore | None]:
    """``score_pair`` for every pair; None where the gold does not standardize."""
    if len(preds) != len(golds):
        raise ValueError(f"{len(preds)} predictions for {len(golds)} references")
    out: list[PairScore | None] = []
    for p, g in zip(preds, golds, strict=True):
        try:
            out.append(score_pair(p, g))
        except ValueError:  # raised only for an invalid reference
            out.append(None)
    return out


def batch_metrics(preds: Sequence[str | None], golds: Sequence[str | None]) -> dict:
    """``aggregate`` over the valid-gold pairs, plus ``n_gold_invalid`` (excluded pairs)."""
    scores = score_batch(preds, golds)
    kept = [s for s in scores if s is not None]
    return {**aggregate(kept), "n_gold_invalid": len(scores) - len(kept)}


__all__ = [
    "batch_metrics",
    "canonicalize",
    "exact_match",
    "inchikey",
    "inchikey_match",
    "is_valid",
    "score_batch",
    "tanimoto",
]
