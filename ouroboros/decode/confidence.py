"""Selective acceptance signals, not calibrated probabilities."""
import math
from ouroboros.eval.standardize import standardize


def confidence(pred, candidates, source_predictions=(), *, min_margin=None,
               max_disagreement=None):
    if min_margin is not None and (not math.isfinite(min_margin) or min_margin < 0):
        raise ValueError('min_margin must be finite and nonnegative')
    if max_disagreement is not None and not 0 <= max_disagreement <= 1:
        raise ValueError('max_disagreement must be in [0, 1]')
    selected = standardize(pred)
    scores, nonfinite = {}, False
    for candidate in candidates:
        smi, score = (candidate['smiles'], candidate.get('score')) if isinstance(candidate, dict) else candidate
        std = standardize(smi)
        if score is None or not math.isfinite(score):
            nonfinite = True
            continue
        if std.valid:
            scores[std.smiles] = max(score, scores.get(std.smiles, -math.inf))
    alternatives = [v for k, v in scores.items() if k != selected.smiles]
    margin = (scores[selected.smiles] - max(alternatives)
              if selected.smiles in scores and alternatives and not nonfinite else None)
    sources = [standardize(s) for s in source_predictions]
    disagreement = (sum(not s.valid or s.smiles != selected.smiles for s in sources) / len(sources)
                    if sources else None)
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
    """Evaluate a threshold grid on held-out labelled JSONL records."""
    records = list(records)
    groups = sorted({r.get('subset', '') for r in records})
    def report(rows, threshold):
        accepted = [r for r in rows if confidence(r['pred'], r['candidates'],
                    r.get('source_predictions', ()), min_margin=threshold,
                    max_disagreement=max_disagreement)['accepted']]
        correct = sum(standardize(r['pred']).smiles == standardize(r['gold']).smiles
                      for r in accepted)
        return dict(n=len(rows), accepted=len(accepted),
                    coverage=len(accepted)/len(rows) if rows else 0,
                    precision=correct/len(accepted) if accepted else None)
    for row in records:
        if not standardize(row['gold']).valid:
            raise ValueError('invalid reference')
    return [dict(min_margin=t, overall=report(records, t),
                 subsets={s: report([r for r in records if r.get('subset', '') == s], t)
                          for s in groups}) for t in margins]
