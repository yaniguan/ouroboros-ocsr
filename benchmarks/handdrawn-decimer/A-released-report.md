# OCSR predictions: A-released-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 200 | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 79.00% | 0.1293 | 0.00% |
| DECIMER_HDM | 200 | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 79.00% | 0.1293 | 0.00% |
| has_stereo | 46 | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 82.61% | 0.2121 | 0.00% |
| no_stereo | 154 | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 77.92% | 0.1045 | 0.00% |

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
