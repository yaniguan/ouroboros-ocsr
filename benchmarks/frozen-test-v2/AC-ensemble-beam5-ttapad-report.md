# OCSR predictions: AC-ensemble-beam5-ttapad-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 300 | 46.67% | [40.67%, 51.67%] | 49.33% | 49.67% | 99.67% | 0.7609 | 50.00% |
| ACS | 100 | 30.00% | [21.00%, 39.00%] | 35.00% | 30.00% | 100.00% | 0.6388 | 36.00% |
| UOB | 100 | 85.00% | [78.00%, 91.00%] | 87.00% | 92.00% | 100.00% | 0.9437 | 87.00% |
| USPTO | 100 | 25.00% | [16.00%, 33.00%] | 26.00% | 27.00% | 99.00% | 0.7001 | 27.00% |
| has_stereo | 57 | 1.75% | [0.00%, 5.26%] | 14.04% | 1.75% | 98.25% | 0.5516 | 15.79% |
| no_stereo | 243 | 57.20% | [51.03%, 63.37%] | 57.61% | 60.91% | 100.00% | 0.8100 | 58.02% |

Rows: 300; scored: 300; excluded because the gold SMILES does not standardize: 0.
CI: percentile bootstrap over items, 1000 resamples.

Metrics:

- `exact`: canonical SMILES equal after standardization, full stereochemistry
- `exact_flat`: canonical SMILES equal with all stereo removed (stereo-agnostic)
- `inchikey`: standard InChIKeys equal (stereo layers kept; tautomers may match)
- `valid`: prediction parses and standardizes in RDKit (blank = invalid)
- `tanimoto`: mean Morgan r=2 2048-bit Tanimoto, chirality ignored, 0 for invalid
- `tanimoto_1`: fraction with Tanimoto 1.0 (an enantiomer or stereo error still counts)

_Scoring parity with arXiv:2608.09100 is unverified._
