"""Pure candidate selection over n-best lists: validity reranking and ensemble pooling.

Candidates are ``(smiles, score)`` pairs. ``choose`` takes the best-scoring RDKit-valid one
(falling back to the best overall). ``ensemble_rank`` pools the candidates of several sources
(models and/or test-time views), merges spellings of the same molecule by canonical SMILES and
scores each by the mean over sources of its length-normalized log probability.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ouroboros.decode.tokenizer import tokenize

Candidate = tuple[str, float]
NEG_INF = float("-inf")


def _is_valid(smiles: str) -> bool:
    from ouroboros.eval.smiles import is_valid

    return is_valid(smiles)


def _canonical(smiles: str) -> str | None:
    from ouroboros.eval.smiles import canonicalize

    return canonicalize(smiles)


def choose(
    candidates: Sequence[Candidate], valid_fn: Callable[[str], bool] | None = None
) -> Candidate:
    """Highest-scoring candidate accepted by ``valid_fn`` (default: RDKit ``is_valid``), else
    the highest-scoring candidate overall. Ties keep the input order."""
    if not candidates:
        raise ValueError("no candidates")
    valid_fn = valid_fn or _is_valid
    ranked = sorted(candidates, key=lambda c: c[1], reverse=True)
    return next((c for c in ranked if valid_fn(c[0])), ranked[0])


def normalized(smiles: str, logprob: float, length_penalty: float = 1.0) -> float:
    """``logprob / len ** length_penalty`` with ``len`` = token count + EOS (as in beam search).
    Untokenizable SMILES score -inf."""
    try:
        n = len(tokenize(smiles)) + 1
    except ValueError:
        return NEG_INF
    return logprob / n**length_penalty


def ensemble_rank(
    cands_per_model: Sequence[Sequence[Candidate]],
    rescored: dict[str, Sequence[float]],
    length_penalty: float = 1.0,
    canon_fn: Callable[[str], str | None] | None = None,
) -> list[Candidate]:
    """Pool the candidates of all sources, best first, one entry per molecule.

    ``rescored[smiles]`` holds the sum log probability of ``smiles`` under every source; a
    candidate missing from it falls back to the scores it received in ``cands_per_model``. Each
    raw string scores the mean over sources of its length-normalized log probability; spellings
    of one molecule (same canonical SMILES, else same raw string) are merged and represented by
    the best-scoring spelling.
    """
    canon_fn = canon_fn or _canonical
    own: dict[str, list[float]] = {}
    for cands in cands_per_model:
        for smiles, lp in cands:
            own.setdefault(smiles, []).append(lp)
    best: dict[str, Candidate] = {}
    for smiles, lps in own.items():
        lps = list(rescored.get(smiles) or lps)
        score = sum(normalized(smiles, lp, length_penalty) for lp in lps) / len(lps)
        key = canon_fn(smiles) or smiles
        if key not in best or score > best[key][1]:
            best[key] = (smiles, score)
    return sorted(best.values(), key=lambda c: c[1], reverse=True)


def ensemble_choose(
    cands_per_model: Sequence[Sequence[Candidate]],
    rescored: dict[str, Sequence[float]],
    length_penalty: float = 1.0,
    valid_fn: Callable[[str], bool] | None = None,
    prefer_valid: bool = True,
) -> Candidate:
    """Best pooled candidate (see ``ensemble_rank``), preferring RDKit-valid ones."""
    ranked = ensemble_rank(cands_per_model, rescored, length_penalty)
    return choose(ranked, valid_fn) if prefer_valid else ranked[0]
