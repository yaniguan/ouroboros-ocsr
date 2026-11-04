# OCSR predictions: AC-ensemble-beam5-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 298 | 52.35% | [46.64%, 58.05%] | 54.36% | 55.70% | 100.00% | 0.7724 | 54.70% |
| ACS | 100 | 37.00% | [28.00%, 47.00%] | 40.00% | 37.00% | 100.00% | 0.6449 | 40.00% |
| UOB | 100 | 85.00% | [77.00%, 92.00%] | 85.00% | 94.00% | 100.00% | 0.9249 | 85.00% |
| USPTO | 98 | 34.69% | [24.49%, 43.88%] | 37.76% | 35.71% | 100.00% | 0.7470 | 38.78% |

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
