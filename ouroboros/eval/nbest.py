"""Oracle diagnostics over distinct standardized molecules in ranked JSONL records."""
from collections import defaultdict
from ouroboros.eval.standardize import standardize, flat_smiles


def diagnose(records, ks=(1, 3, 5)):
    ks = sorted(set(ks))
    if not ks or min(ks) < 1:
        raise ValueError('k must be positive')
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (row.get('subset', ''), str(row['id']))
        if key in seen:
            raise ValueError(f'duplicate subset/id: {key}')
        seen.add(key)
        gold = standardize(row['gold'])
        if not gold.valid:
            raise ValueError(f'invalid reference: {key}')
        unique, identities = [], set()
        for candidate in row['candidates']:
            std = standardize(candidate['smiles'])
            if std.valid and std.smiles not in identities:
                identities.add(std.smiles)
                unique.append(std)
        rank = next((i for i, s in enumerate(unique, 1) if s.smiles == gold.smiles), None)
        flat_rank = next((i for i, s in enumerate(unique, 1)
                          if flat_smiles(s) == flat_smiles(gold)), None)
        selected = standardize(row.get('pred'))
        detail = dict(subset=key[0], id=key[1], correct_rank=rank,
                      constitution_rank=flat_rank, distinct_candidates=len(unique),
                      selection_failure=rank is not None and selected.smiles != gold.smiles)
        groups[key[0]].append(detail)
    def aggregate(rows):
        return dict(n=len(rows), selection_failures=sum(r['selection_failure'] for r in rows),
                    oracle_exact={str(k): sum(r['correct_rank'] is not None and
                        r['correct_rank'] <= k for r in rows) / len(rows) for k in ks},
                    oracle_constitution={str(k): sum(r['constitution_rank'] is not None and
                        r['constitution_rank'] <= k for r in rows) / len(rows) for k in ks})
    details = [r for rows in groups.values() for r in rows]
    return dict(overall=aggregate(details) if details else {'n': 0},
                subsets={s: aggregate(rows) for s, rows in groups.items()}, rows=details)
