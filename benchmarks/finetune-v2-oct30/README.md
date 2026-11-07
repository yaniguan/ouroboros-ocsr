# Fine-tune v2 (Oct 30): real-image test, 298 images

Reports copied unchanged from the Colab run output on Google Drive
(`ouroboros-ocsr_runs/finetune_v2_oct30/`), made by `colab/finetune_v2.ipynb` with
`scripts/predict_real.py` (decoding) and `scripts/eval_predictions.py` (scoring).

| File | Decoding | Exact match (298 images) |
|---|---|---|
| `AC-ensemble-beam5-ttapad-report.md` | A + C ensemble, beam 5, TTA `pad` | 54.03% |
| `AC-ensemble-beam5-report.md` | A + C ensemble, beam 5, no TTA | 52.35% |
| `A-selected-report.md` | A alone, greedy | 51.01% |
| `C-selected-report.md` | C alone, greedy | 37.58% |

Equivalent command (fine-tuned checkpoints `A-selected.pt`, `C-selected.pt`; not on the GitHub release yet):

```bash
python scripts/predict_real.py --ckpt A-selected.pt --ckpt C-selected.pt --beam 5 --tta pad \
    --shards <real-test-shards> --out AC-ensemble-beam5-ttapad-preds.csv
python scripts/eval_predictions.py AC-ensemble-beam5-ttapad-preds.csv --md AC-ensemble-beam5-ttapad-report.md
```
