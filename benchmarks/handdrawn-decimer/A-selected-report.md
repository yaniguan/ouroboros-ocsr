# OCSR predictions: A-selected-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 200 | 3.00% | [1.00%, 5.50%] | 5.00% | 3.50% | 90.50% | 0.2383 | 5.00% |
| DECIMER_HDM | 200 | 3.00% | [1.00%, 5.50%] | 5.00% | 3.50% | 90.50% | 0.2383 | 5.00% |
| has_stereo | 46 | 2.17% | [0.00%, 6.52%] | 8.70% | 2.17% | 73.91% | 0.2960 | 8.70% |
| no_stereo | 154 | 3.25% | [0.65%, 5.86%] | 3.90% | 3.90% | 95.45% | 0.2211 | 3.90% |

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
