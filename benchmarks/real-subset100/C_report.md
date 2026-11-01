# OCSR predictions: C_preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 278 | 17.99% | [13.67%, 22.30%] | 17.99% | 19.42% | 80.94% | 0.4507 | 17.99% |
| ACS | 80 | 1.25% | [0.00%, 3.75%] | 1.25% | 1.25% | 75.00% | 0.2233 | 1.25% |
| UOB | 100 | 39.00% | [29.00%, 49.00%] | 39.00% | 43.00% | 96.00% | 0.6872 | 39.00% |
| USPTO | 98 | 10.20% | [5.10%, 16.33%] | 10.20% | 10.20% | 70.41% | 0.3951 | 10.20% |

Rows: 278; scored: 278; excluded because the gold SMILES does not standardize: 0.
CI: percentile bootstrap over items, 1000 resamples.

Metrics:

- `exact`: canonical SMILES equal after standardization, full stereochemistry
- `exact_flat`: canonical SMILES equal with all stereo removed (stereo-agnostic)
- `inchikey`: standard InChIKeys equal (stereo layers kept; tautomers may match)
- `valid`: prediction parses and standardizes in RDKit (blank = invalid)
- `tanimoto`: mean Morgan r=2 2048-bit Tanimoto, chirality ignored, 0 for invalid
- `tanimoto_1`: fraction with Tanimoto 1.0 (an enantiomer or stereo error still counts)

_Scoring parity with arXiv:2608.09100 is unverified (see TASK.md Am1-C)._
