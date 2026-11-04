# OCSR predictions: A-selected-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 298 | 51.01% | [45.29%, 56.71%] | 52.68% | 54.70% | 95.30% | 0.7367 | 53.69% |
| ACS | 100 | 38.00% | [29.00%, 48.00%] | 41.00% | 38.00% | 96.00% | 0.6145 | 42.00% |
| UOB | 100 | 84.00% | [76.00%, 91.00%] | 84.00% | 93.00% | 100.00% | 0.9200 | 84.00% |
| USPTO | 98 | 30.61% | [21.43%, 39.80%] | 32.65% | 32.65% | 89.80% | 0.6742 | 34.69% |

Rows: 298; scored: 298; excluded because the gold SMILES does not standardize: 0.
CI: percentile bootstrap over items, 1000 resamples.

Metrics:

- `exact`: canonical SMILES equal after standardization, full stereochemistry
- `exact_flat`: canonical SMILES equal with all stereo removed (stereo-agnostic)
- `inchikey`: standard InChIKeys equal (stereo layers kept; tautomers may match)
- `valid`: prediction parses and standardizes in RDKit (blank = invalid)
- `tanimoto`: mean Morgan r=2 2048-bit Tanimoto, chirality ignored, 0 for invalid
- `tanimoto_1`: fraction with Tanimoto 1.0 (an enantiomer or stereo error still counts)

_Scoring parity with arXiv:2608.09100 is unverified (see TASK.md Am1-C)._
