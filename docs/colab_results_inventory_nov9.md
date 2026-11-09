# ouroboros-ocsr Colab Results and Inventory Status

Inventory date: 2026-11-09 (America/Los_Angeles)

## Repository State

- Repository: `yaniguan/ouroboros-ocsr`
- `main` commit: `034e2dcc9cd68456b1c838fc1e5425ca9e2c1169`
- Open pull requests: none

## What Is Known

- **Do not mix evaluations**: The repository already has `docs/v0-results.md` for the v0.1.0-alpha synthetic evaluation. The `0.540` exact-match figure below comes from a different official evaluation. Do not combine or replace those tables.
- **Last known official exact-match metric**: `0.540` from earlier official evaluations (citation: official Drive metric file not yet pinned; exact file path is not yet confirmed from the official file).
- **Colab notebook**: "Ouroboros — A100 experiments" exists on Google Drive.
- **Large results archive on Drive**: `ouroboros-real-aug-results.zip` (~833 MB, modified 2026-10-30).
- **Disambiguation note**: The Oct 20 `results.json` files under `pol_s2` on Drive belong to the YLM polarizability track (`tensorqa_pol` / `matscibert`, seed 2), not `ouroboros-ocsr`.

## What Is Still Missing

- **Small official metric file on Google Drive**: The exact small JSON/text metric file on Drive that records the `0.540` exact-match evaluation (and any sub-metrics or per-split numbers) was not yet identified or pinned before this writeup. Any additional digits or secondary metrics are not yet confirmed from the official file.
- **Extracted lightweight summary from `ouroboros-real-aug-results.zip`**: Because `ouroboros-real-aug-results.zip` is ~833 MB, it must not be downloaded or pushed to GitHub. A lightweight summary file path on Drive is not yet confirmed from the official file.

## Artifacts Not to Download or Push

- Do not download, commit, or upload `ouroboros-real-aug-results.zip` (~833 MB, modified 2026-10-30).
- Do not commit or upload `.pt` model checkpoints or large trace files.
