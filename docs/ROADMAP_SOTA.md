# Roadmap: from the v0 pilot to a competitive OCSR model and an image-derived force-field set

Status 2026-10-30. Numbers below are the repository's own measurements; nothing here is a
claim of state-of-the-art. Published reference numbers go in the section reserved for them and
must carry their scoring protocol, because exact-match definitions differ between papers.

## Where we are (measured)

| Model | Keylab subset (298 real images), exact match incl. stereo | Source |
| --- | ---: | --- |
| Released A (baseline CNN) | 18.12% (54/298) | `docs/WORKLOG_oct29b.md` |
| Released B (A + rotation aug.) | 16.78% (50/298) | same |
| Released C (C8 steerable) | 16.78% (50/298) | same |
| Oct 29 synthetic-only fine-tune | A 8.39%, C 13.76% | `docs/WORKLOG_oct30.md` |
| Real-PMC fine-tune A (Oct 30, greedy) | 28.19% (84/298): ACS 7%, UOB 64%, USPTO 13.27% | Drive `real_pmc_oct30/summary.json` |
| Released A + beam 5 + validity rerank (CPU, this branch) | 20.13% (60/298): ACS 2%, UOB 46%, USPTO 12.24% | `docs/WORKLOG_oct30_v2.md` |
| Fine-tune v2 (all-epoch PMC, 20k steps, beam/ensemble) | _pending: `colab/finetune_v2.ipynb`_ | — |

Known failure modes on the real subset: ACS is near 0% (R-groups / abbreviations, which PMC
labels never contain), USPTO errors are dominated by long molecules (> 60 tokens), and stereo
accuracy on the stereo subset is very low. Scoring parity with arXiv:2608.09100 is unverified.

## Published SOTA (exact match, %)

<!-- Fill from the papers themselves; record protocol differences in the last column. -->

| Method | Year | USPTO | UOB | ACS | CLEF | JPO | Staker | Scoring notes (stereo? standardization? subset?) | Reference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MolNexTR | 2024 | 93.8 | 88.5 | 81.9 | 90.4 | 82.1 | 88.3 | full sets; tetrahedral stereo required, cis/trans ignored, R-groups -> `*` | arXiv:2403.03691 Table 2 (J. Cheminf. 2024) |
| MolScribe | 2023 | 92.6 | 87.9 | 71.9 | 88.9 | 76.2 | 86.9 | defines the protocol above | arXiv:2205.14311 Table 2 (JCIM 2023) |
| DECIMER 2.x | 2023 | 41.1 | 88.2 | 46.5 | 62.7 | 55.2 | 40.8 | as reported in MolScribe/MolNexTR | Nat. Commun. 14, 5045 (2023) |
| MolSeek-OCR (DeepSeek-OCR-2 FT) | 2026 | 61.0 | 72.6 | 29.9 | 63.3 | — | 50.5 | VLM | arXiv:2604.03476 Table 2 |

Hand-drawn (DECIMER Hand-drawn, 5,088 images): GTR-VL-Stage2 75.44% exact SMILES (arXiv:2506.07553
Table 2, HD-test split), DECIMER hand-drawn model 73.25% (J. Cheminf. 2024, 10.1186/s13321-024-00872-7);
MolScribe 12.70% (DECIMER.ai paper, Table 6).

### Gap table (ours = best measured, subset of 100/98/100 images, 95% CI ±~9 pts per source)

| Benchmark | SOTA | Ours (real-PMC A, greedy) | Gap | Most plausible target in 1–2 phases |
| --- | ---: | ---: | ---: | --- |
| UOB | 88.5 (MolNexTR) | 64.0 | −24.5 | **Primary SOTA target**: small, clean, no stereo/abbreviations; gap is recognition accuracy and data volume |
| USPTO | 93.8 (MolNexTR) | 13.3 | −80.5 | needs long-molecule data (>60 tokens: 0/35 correct) and stereo |
| ACS | 81.9 (MolNexTR) | 7.0 | −74.9 | needs R-group/abbreviation labels (0/20 on `*` molecules) |
| Hand-drawn | 75.4 (GTR-VL) | not measured | — | **Secondary target**: rotation/style robustness is where the C_N steerable encoder should help; needs RanDepict-style hand-drawn synthetic data |

Caveat: our numbers come from a seeded 100-per-source sample of Keylab/OCSR-Benchmarks and the
standardization in `ouroboros/eval/standardize.py` (parity unverified); a SOTA claim needs the full
benchmark sets with the MolScribe scoring script.

Comparison rule: a published number is only placed next to ours if the benchmark images are the
same set (not a sample) and the exact-match definition is the same, or the difference is stated.
Our 298-image Keylab subset is a seeded *sample*: its confidence intervals (± ~4–5 points) must be
shown with every comparison.

## Phase 0 — Release hygiene (this change)

- `scripts/export_hf_release.py` → HF folder with safetensors weights, encoder-only weights,
  config, vocabulary, generated model card, metrics and SHA256 checksums;
  `ouroboros.hf.load_pretrained` reproduces the checkpoint's logits exactly (tested).
- `scripts/upload_hf.py`: dry run by default; `--yes` to publish. Weights are CC-BY-NC-4.0
  (VERDICT-PMC fine-tuning data).
- **Exit criterion**: the v2-selected checkpoint(s) exported, card filled from the harness
  report, uploaded after a manual read of the card.

## Phase 1 — Close the synthetic-to-real gap (data, not architecture)

1. Finish fine-tune v2; report released vs selected per source with CIs, greedy and beam 5.
2. Full benchmark evaluation (all images of each Keylab benchmark, not the 100-per-source
   sample) once the run budget allows; keep the 298 subset as the fast regression set.
3. Real data: train on all PMC epochs that real-val selection supports; add other real
   sources only with compatible licences, each through the connectivity-block leakage gate.
4. Synthetic data aimed at the measured failures: R-group / abbreviation substitution with
   ACS-like fonts, long molecules (curriculum on token length), wedge/hash-heavy stereo set,
   document noise (JPEG, scan blur, line thickness, neighbouring text).
5. Scoring parity: per-sample agreement with the reference protocol (TASK.md Am1-C) so that
   the published table above becomes comparable.
- **Exit criterion**: selected model beats released A (18.12%) on the 298 subset with
  non-overlapping or clearly shifted CI, and on every source individually.

## Phase 2 — Model capacity and decoding

1. Raise decoder `max_len` (128 tokens truncates USPTO molecules) with matching positional
   capacity; measure accuracy vs token length.
2. Encoder scaling at matched compute: pretrained ViT/Swin baseline vs steerable C8/C16 at
   higher resolution (512 px); equivariant-attention arm D. Matched parameter counts, ≥ 3 seeds
   for any claimed difference.
3. Decoding: beam + validity rerank (implemented), A+C ensembles, TTA; calibrated
   confidence (sequence log-prob → reliability diagram) so low-confidence outputs can be routed
   to a human.
4. Rotation robustness reported separately (24 angles) — this is where C_N equivariance should
   pay off; upright accuracy alone does not test it.
- **Exit criterion**: per-source gains attributable to a single change (ablation table).

## Phase 3 — Chemistry the decoder currently gets wrong

1. Stereo: auxiliary wedge/hash bond detection head; stereo-aware loss weighting; report
   stereo-element accuracy (strict and given correct constitution).
2. R-groups / Markush: richer pseudo-atom vocabulary, attachment points, variable
   substituent tables (graph or extended-SMILES output instead of plain SMILES).
3. Optional graph-output head (atoms + bonds + coordinates) for rerank against the
   SMILES decoder.
- **Exit criterion**: ACS exact match no longer near zero; stereo-subset exact match reported
  with CI.

## Phase 4 — Image-derived force-field training set

`scripts/image_to_forcefield.py`: images / predictions → standardized SMILES → ETKDG
conformers (stereo enforced, re-checked from 3D) → MACE-OFF energies and forces → extxyz +
manifest, with InChIKey dedupe and unassigned-stereo flags.

1. Run over PMC / patent figure corpora; filter by OCSR confidence (Phase 2) and RDKit
   validity; track how many molecules are new w.r.t. existing force-field sets (InChIKey).
2. Diversity: several conformers per molecule, MD snapshots / normal-mode displacements
   around minima (relaxed minima alone have near-zero forces).
3. Label quality: MACE-OFF labels are a distillation target; relabel a stratified subset with
   DFT (the reference level of the target force field) and measure MACE-OFF vs DFT error on
   exactly these molecules.
4. Error propagation: OCSR mistakes change the molecule, not just the label —
   `scripts/error_propagation.py` already quantifies energy differences by error category;
   use it to set the confidence threshold for inclusion.
5. Train / fine-tune a force field on the union with established datasets; evaluate on
   held-out molecules and standard benchmarks, never on molecules seen in OCSR training eval sets.
- **Licensing**: MACE-OFF weights are ASL (academic); PMC-derived molecules inherit the source
  terms; record both in the dataset card.
- **Exit criterion**: a versioned dataset (frames, molecules, label method, DFT-checked
  subset error) and a force field whose held-out error is reported against a baseline trained
  without the image-derived data.

## Cross-cutting rules

- Every new training source passes `check_leakage.py --fail-on-connectivity` against all eval sets.
- Checkpoint selection on real validation only; test sets touched once per reported result.
- Every table: n, CI, decoding mode, scoring protocol, parity footnote while unverified.
