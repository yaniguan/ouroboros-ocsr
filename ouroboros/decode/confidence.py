"""Check score gaps and agreement between predictions."""
import math
from ouroboros.eval.standardize import standardize


def confidence(pred, candidates, source_predictions=(), *, min_margin=None,
               max_disagreement=None):
    if min_margin is not None and (not math.isfinite(min_margin) or min_margin < 0):
        raise ValueError('min_margin must be finite and nonnegative')
    if max_disagreement is not None and not 0 <= max_disagreement <= 1:
        raise ValueError('max_disagreement must be in [0, 1]')
    selected = standardize(pred)
    scores = {}
    nonfinite = False
    for candidate in candidates:
        if isinstance(candidate, dict):
            smi = candidate['smiles']
            score = candidate.get('score')
        else:
            smi, score = candidate
        std = standardize(smi)
        if score is None or not math.isfinite(score):
            nonfinite = True
            continue
        if std.valid:
            scores[std.smiles] = max(score, scores.get(std.smiles, -math.inf))
    alternatives = [v for k, v in scores.items() if k != selected.smiles]
    margin = None
    if selected.smiles in scores and alternatives and not nonfinite:
        margin = scores[selected.smiles] - max(alternatives)
    sources = [standardize(s) for s in source_predictions]
    disagreement = None
    if sources:
        disagreement = sum(not s.valid or s.smiles != selected.smiles for s in sources) / len(sources)
    reasons = []
    if not selected.valid:
        reasons.append('invalid_prediction')
    if min_margin is not None:
        if margin is None:
            reasons.append('unknown_margin')
        elif margin < min_margin:
            reasons.append('low_margin')
    if max_disagreement is not None:
        if disagreement is None:
            reasons.append('unknown_disagreement')
        elif disagreement > max_disagreement:
            reasons.append('source_disagreement')
    return dict(accepted=not reasons, rejection_reason=';'.join(reasons),
                margin=margin, disagreement=disagreement)


def selective_report(records, margins, max_disagreement=None):
    """Count accepted and correct predictions for each margin limit."""
    records = list(records)
    groups = sorted({r.get('subset', '') for r in records})

    def report(rows, threshold):
        accepted = []
        for row in rows:
            result = confidence(row['pred'], row['candidates'],
                                row.get('source_predictions', ()), min_margin=threshold,
                                max_disagreement=max_disagreement)
            if result['accepted']:
                accepted.append(row)
        correct = sum(standardize(r['pred']).smiles == standardize(r['gold']).smiles
                      for r in accepted)
        return dict(n=len(rows), accepted=len(accepted),
                    coverage=len(accepted)/len(rows) if rows else 0,
                    precision=correct/len(accepted) if accepted else None)
    for row in records:
        if not standardize(row['gold']).valid:
            raise ValueError('invalid reference')
    results = []
    for threshold in margins:
        overall = report(records, threshold)
        subsets = {}
        for group in groups:
            rows = [row for row in records if row.get('subset', '') == group]
            subsets[group] = report(rows, threshold)
        results.append(dict(min_margin=threshold, overall=overall,
                            subsets=subsets))
    return results
