# OCSR predictions: sanity_preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 72 | 62.50% | [51.39%, 73.61%] | 83.33% | 62.50% | 88.89% | 0.8551 | 83.33% |
| mirror | 24 | 45.83% | [29.17%, 66.67%] | 100.00% | 45.83% | 100.00% | 1.0000 | 100.00% |
| perturb | 24 | 41.67% | [20.83%, 62.50%] | 50.00% | 41.67% | 66.67% | 0.5654 | 50.00% |
| render | 24 | 100.00% | [100.00%, 100.00%] | 100.00% | 100.00% | 100.00% | 1.0000 | 100.00% |

Rows: 72; scored: 72; excluded because the gold SMILES does not standardize: 0.
CI: percentile bootstrap over items, 1000 resamples.

Metrics:

- `exact`: canonical SMILES equal after standardization, full stereochemistry
- `exact_flat`: canonical SMILES equal with all stereo removed (stereo-agnostic)
- `inchikey`: standard InChIKeys equal (stereo layers kept; tautomers may match)
- `valid`: prediction parses and standardizes in RDKit (blank = invalid)
- `tanimoto`: mean Morgan r=2 2048-bit Tanimoto, chirality ignored, 0 for invalid
- `tanimoto_1`: fraction with Tanimoto 1.0 (an enantiomer or stereo error still counts)

_Scoring parity with arXiv:2608.09100 is unverified (see TASK.md Am1-C)._
