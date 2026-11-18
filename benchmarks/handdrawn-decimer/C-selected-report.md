# OCSR predictions: C-selected-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 200 | 2.00% | [0.50%, 4.50%] | 3.50% | 2.00% | 93.50% | 0.2439 | 3.50% |
| DECIMER_HDM | 200 | 2.00% | [0.50%, 4.50%] | 3.50% | 2.00% | 93.50% | 0.2439 | 3.50% |
| has_stereo | 46 | 2.17% | [0.00%, 6.52%] | 8.70% | 2.17% | 86.96% | 0.3576 | 8.70% |
| no_stereo | 154 | 1.95% | [0.00%, 4.55%] | 1.95% | 1.95% | 95.45% | 0.2099 | 1.95% |

Rows: 200; scored: 200; excluded because the gold SMILES does not standardize: 0.
CI: percentile bootstrap over items, 1000 resamples.

Metrics:

- `exact`: canonical SMILES equal after standardization, full stereochemistry
- `exact_flat`: canonical SMILES equal with all stereo removed (stereo-agnostic)
- `inchikey`: standard InChIKeys equal (stereo layers kept; tautomers may match)
- `valid`: prediction parses and standardizes in RDKit (blank = invalid)
- `tanimoto`: mean Morgan r=2 2048-bit Tanimoto, chirality ignored, 0 for invalid
- `tanimoto_1`: fraction with Tanimoto 1.0 (an enantiomer or stereo error still counts)

_Scoring parity with arXiv:2608.09100 is unverified._
