# OCSR predictions: A_preds.csv

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 298 | 18.12% | [13.76%, 22.82%] | 19.13% | 19.80% | 78.19% | 0.4260 | 19.13% |
| ACS | 100 | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 66.00% | 0.1647 | 0.00% |
| UOB | 100 | 41.00% | [31.00%, 52.00%] | 41.00% | 46.00% | 95.00% | 0.6618 | 41.00% |
| USPTO | 98 | 13.27% | [7.14%, 20.41%] | 16.33% | 13.27% | 73.47% | 0.4520 | 16.33% |

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
