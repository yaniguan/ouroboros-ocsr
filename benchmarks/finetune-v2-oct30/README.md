# Fine-tune v2 (Oct 30): real-image dev-test, 298 images

Reports scored from the original fine-tune v2 prediction CSVs on `subset100` (298 images: ACS 100,
UOB 100, USPTO 98; 49 `has_stereo`, 249 `no_stereo`), made by `colab/finetune_v2.ipynb` with
`scripts/predict_real.py` (decoding) and `scripts/eval_predictions.py` (scoring).

| File | Decoding | All (`n=298`) | `has_stereo` (`n=49`) | `no_stereo` (`n=249`) |
|---|---|---|---|---|
| `AC-ensemble-beam5-ttapad-report.md` | A + C ensemble, beam 5, TTA `pad` | 54.03% | 18.37% | 61.04% |
| `AC-ensemble-beam5-report.md` | A + C ensemble, beam 5, no TTA | 52.35% | 20.41% | 58.63% |
| `A-selected-report.md` | A alone, greedy | 51.01% | 16.33% | 57.83% |
| `C-selected-report.md` | C alone, greedy | 37.58% | 8.16% | 43.37% |

Equivalent command (fine-tuned checkpoints `A-selected.pt`, `C-selected.pt`; not on the GitHub release yet):

```bash
python scripts/predict_real.py --ckpt A-selected.pt --ckpt C-selected.pt --beam 5 --tta pad \
    --shards <real-test-shards> --out AC-ensemble-beam5-ttapad-preds.csv
python scripts/eval_predictions.py AC-ensemble-beam5-ttapad-preds.csv \
    --json AC-ensemble-beam5-ttapad-report.json --md AC-ensemble-beam5-ttapad-report.md
```
