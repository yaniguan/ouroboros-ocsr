# OCSR predictions: C-selected-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 298 | 37.58% | [31.88%, 43.29%] | 39.26% | 40.60% | 90.60% | 0.6434 | 39.26% |
| ACS | 100 | 20.00% | [12.00%, 28.00%] | 24.00% | 21.00% | 91.00% | 0.5069 | 24.00% |
| UOB | 100 | 74.00% | [65.00%, 83.00%] | 74.00% | 80.00% | 100.00% | 0.8775 | 74.00% |
| USPTO | 98 | 18.37% | [11.22%, 26.53%] | 19.39% | 20.41% | 80.61% | 0.5437 | 19.39% |

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
