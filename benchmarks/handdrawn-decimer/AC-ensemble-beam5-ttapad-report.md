# OCSR predictions: AC-ensemble-beam5-ttapad-preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 200 | 4.50% | [2.00%, 7.50%] | 6.00% | 4.50% | 100.00% | 0.3116 | 6.00% |
| DECIMER_HDM | 200 | 4.50% | [2.00%, 7.50%] | 6.00% | 4.50% | 100.00% | 0.3116 | 6.00% |
| has_stereo | 46 | 4.35% | [0.00%, 10.87%] | 8.70% | 4.35% | 100.00% | 0.4539 | 8.70% |
| no_stereo | 154 | 4.55% | [1.30%, 7.79%] | 5.19% | 4.55% | 100.00% | 0.2691 | 5.19% |

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
