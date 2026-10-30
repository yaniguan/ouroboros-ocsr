# Worklog Oct 30 (v2): data audit, error analysis, decoding gains, finetune_v2 notebook

Branch `feat/finetune-v2-oct30`, stacked on `feat/real-finetune-oct30` (05886b6).
All numbers are on the held-out real test set `real/subset100` (298 images: ACS 100, UOB 100,
USPTO 98). The leakage gate (0 InChIKey, 0 connectivity overlap) is unchanged and enforced in the notebook.

## 1. HF dataset `yaniguan/ocsr-dataset` (VERDICT-PMC)
- 18,442 rows: train 17,249 (1,119 articles, 13,371 InChIKeys), validation 1,193 (125 articles, 1,025 InChIKeys); ~173 MB parquet; CC BY 12,421 / CC BY-NC 4,828 (train).
- Weak labels: n_agree 2/3/4 = 6,345 / 5,680 / 5,224 (train). n_agree_open=1 (relies on the VLM vote): 1,576 train rows. Engines most often agreeing: decimer|molnextr|molscribe|vlm (5,224).
- All labels parse in RDKit; 32% contain stereo, 7% charges, 2.8% multi-fragment, **0% R-groups / `*`**.
- Images are small: median 145x93 px (p10 69x49, p90 318x205). Molecules are small: median 18 heavy atoms (p90 32), SMILES p90 61 chars, 1.1% > 100 chars.
- Heavy duplication: max 46 rows per InChIKey in train.
- After prep (oct30 run): train 17,220 kept (16 too long, 12 eval overlap, 1 unparsable); val 963 kept (224 dropped as in train).
- **Unused last time:** every train row was eligible, but 3,000 steps x 16 x 0.6 real = 28.8k real samples = **~1.7 epochs**; validation used only 400 of 963 kept rows; synthetic capped at 10k. The run finished in 456 s of training — badly under-trained (val curve still rising at 2,250).

## 2. Error analysis (fine-tuned A, 84/298)
| source | exact | main failure (of wrong) |
|---|---|---|
| UOB | 64/100 | same atom count but wrong graph 21, too many/few atoms 13 |
| USPTO | 13/98 | too few atoms 42, invalid 21, same count 12, too many 7, stereo-only 3 |
| ACS | 7/100 | too many 33, too few 33, same count 14, invalid 11, stereo-only 2 |

- Length: USPTO gold <40 tokens 4/27, 40–60 9/36, **60–90 0/35**. Heavy atoms >30: 2/52 overall. PMC training molecules are short (p90 32 heavy atoms), so long patent molecules are out of distribution.
- R-groups: molecules with `*` 0/20 (all ACS); PMC labels contain none.
- Stereo: 2/49 molecules with stereo correct (ACS 0/26, USPTO 2/23).
- Resolution: ACS long side median 427 px; images < 300 px 1/26 correct. Shards place the image inside the inscribed circle of the 384 canvas (rotation-safe), so only ~272 px of the side carries content.
- Ensemble ceiling: A or C correct on 93/298 (oracle); "use C when A is invalid" gives 85/298 post hoc.

## 3. Decoding changes, measured on CPU (released checkpoints in `checkpoints/`; fine-tuned weights are 173 MB on Drive and could not be fetched to the box)
| variant | ALL | ACS | UOB | USPTO | valid |
|---|---|---|---|---|---|
| released A greedy (baseline) | 18.12% (54) | 0% | 41% | 13.27% | 78.2% |
| released C greedy | 16.78% (50) | 1% | 39% | 10.20% | 79.2% |
| A beam 5 + validity rerank | 20.13% (60) | 2% | 46% | 12.24% | 93.6% |
| A beam 5 + rerank + TTA pad | 23.49% (70) | 1% | 56% | 13.27% | 96.6% |
| A+C ensemble beam 5 + rerank | 24.50% (73) | 3% | 50% | 20.41% | 97.7% |
| A+C ensemble beam 5 + TTA pad | not measured (CPU box overloaded; aborted at 16/298) — runs in the notebook | | | | |

These are applied to the fine-tuned checkpoints in `colab/finetune_v2.ipynb`.

## 4. Training recipe v2 (`colab/finetune_v2.ipynb`)
20k steps, batch 32, real_fraction 0.7, lr 3e-5, warmup 500, label smoothing 0.05, 50k real-style synthetic
(R-group p=0.5), validation on all 963 PMC val rows every 1k steps (patience 6), arms A and C,
final eval greedy / beam5 / beam5+TTA / A+C ensemble; profiling to `profiling/`.
