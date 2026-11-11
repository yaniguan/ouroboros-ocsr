# ouroboros-ocsr Colab Results and Inventory Status

Inventory date: 2026-11-09 (America/Los_Angeles)

## Repository State

- Repository: `yaniguan/ouroboros-ocsr`
- `main` commit: `034e2dcc9cd68456b1c838fc1e5425ca9e2c1169`
- This note is in open pull request #27. It is not a claim that no pull request is open.

## What Is Known

- **Do not mix evaluations**: `docs/v0-results.md` is the v0.1.0-alpha synthetic evaluation. The `0.540` figure below is a different evaluation on 298 real images. Do not combine or replace those tables.
- **Pinned exact-match source**: `0.540` is the rounded headline for **54.03%** exact match on **ALL, 298 images**. It comes from the Oct 30 fine-tune v2 run: the A + C ensemble, beam 5, TTA pad, produced with `colab/finetune_v2.ipynb`. The README cites `benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md`, and that report is the source in this repository. A separate small official Drive metric file was not pinned in this note.
- **The headline may be a bit optimistic**: `colab/finetune_v2.ipynb` scores several decoding variants on this same 298-image test set and keeps the variant with the highest exact match. Of the four reports in `benchmarks/finetune-v2-oct30/`, 54.03% is the highest. This is not the synthetic protocol in `docs/v0-results.md`.
- **Different experiment**: `ouroboros-real-aug-results.zip` (~833 MB, modified 2026-10-30) is the archive written by `colab/finetune_real_aug.ipynb` for the real-aug run (`ouroboros-ocsr_runs/real_aug_oct29/`). It is not the Oct 30 fine-tune v2 result behind `0.540`.
- Oct 20 `pol_s2` results on Drive belong in the YLM docs, not here.

## What Is Still Missing

- **Separate Drive metric file**: The 54.03% number is already in `benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md`. A separate small official metric file on Google Drive was not pinned in this note.
- **Real-aug zip summary**: A lightweight summary of `ouroboros-real-aug-results.zip` is not in this repository. That zip is the real-aug / `finetune_real_aug` run, not the `0.540` fine-tune v2 result. Do not download it to fill this gap.

## Artifacts Not to Download or Push

- Do not download, commit, or upload `ouroboros-real-aug-results.zip` (~833 MB, modified 2026-10-30). It belongs to the real-aug run, not to the `0.540` result.
- Do not commit or upload `.pt` model checkpoints or large trace files.
