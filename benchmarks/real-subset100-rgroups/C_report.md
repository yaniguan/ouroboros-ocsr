# OCSR predictions: C_preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 298 | 16.78% | [12.42%, 21.14%] | 16.78% | 18.12% | 79.19% | 0.4274 | 16.78% |
| ACS | 100 | 1.00% | [0.00%, 3.00%] | 1.00% | 1.00% | 71.00% | 0.1993 | 1.00% |
| UOB | 100 | 39.00% | [29.00%, 49.00%] | 39.00% | 43.00% | 96.00% | 0.6872 | 39.00% |
| USPTO | 98 | 10.20% | [5.10%, 16.33%] | 10.20% | 10.20% | 70.41% | 0.3951 | 10.20% |

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
