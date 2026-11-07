# OCSR predictions: AC-ensemble-beam5-ttapad-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 298 | 54.03% | [48.32%, 59.73%] | 56.04% | 57.05% | 100.00% | 0.7828 | 56.71% |
| ACS | 100 | 40.00% | [31.00%, 51.00%] | 43.00% | 40.00% | 100.00% | 0.6733 | 45.00% |
| UOB | 100 | 84.00% | [76.00%, 91.00%] | 84.00% | 93.00% | 100.00% | 0.9195 | 84.00% |
| USPTO | 98 | 37.76% | [27.55%, 45.92%] | 40.82% | 37.76% | 100.00% | 0.7549 | 40.82% |

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
