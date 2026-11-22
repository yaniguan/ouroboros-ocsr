# Lab-scale plan: Ouroboros as the "eyes" of the lab agent

Status: plan. Nothing in the Plan section has been run. Numbers below come from files in this repo or from the shared cost doc.

## Role in the lab agent

The lab agent has three parts. [YLM](https://github.com/yaniguan/YLM) is the brain: it reasons over protocols. WignerFlow is the physics sense. Ouroboros is the eyes. It reads molecule drawings in papers, patents, labels and notebooks and turns them into structures (SMILES / molfile) that the brain can use. Each structure has a status, so the brain knows when not to trust it.

Analogy: a robot that only drills moves is an athlete who only drills dunks; giving it eyes that read the chemistry in every step is the first step to also knowing the "training science".

## What exists today

| Fact | Source |
|---|---|
| Image -> SMILES with stereo. Three encoder arms (A CNN, B CNN + rotation aug, C C8 steerable), one Transformer decoder | [README.md](../README.md), [MODEL_CARD.md](../MODEL_CARD.md) |
| Size: A 43.2M params, C 32.4M params | [MODEL_CARD.md](../MODEL_CARD.md) |
| Real dev-test (298 figures, ACS/UOB/USPTO): A + C ensemble, beam 5 + TTA = 54.0% exact match (optimistic: set was reused) | [AC-ensemble-beam5-ttapad-report.md](../benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md) |
| Single arm, greedy: A 51.0%, C 37.6% | [A-selected-report.md](../benchmarks/finetune-v2-oct30/A-selected-report.md), [C-selected-report.md](../benchmarks/finetune-v2-oct30/C-selected-report.md) |
| Weak spot: stereo figures (A + C ensemble 18.4% with TTA, 20.4% without, on the 49 `has_stereo` figures) | [README.md](../README.md) |
| Rotated synthetic drawings: C matches rotation augmentation (65.8% vs 65.7%, 24 angles) | [v0-results.md](v0-results.md) |
| Held-out release test `frozen-test-v2` (300 figures), score once per release | [DATA.md](../DATA.md), [frozen-test-v2/](../benchmarks/frozen-test-v2/) |
| Confidence gate (score margin, model disagreement) and selective report | [confidence.py](../ouroboros/decode/confidence.py), [eval_selective.py](../scripts/eval_selective.py) |
| One scorer: RDKit standardization, exact match, InChIKey | [standardize.py](../ouroboros/eval/standardize.py), [eval_predictions.py](../scripts/eval_predictions.py) |
| Leakage gate on InChIKey / connectivity | [check_leakage.py](../scripts/check_leakage.py) |
| CPU speed, encoder only (4-core VM, fp32): A-type CNN 15.9 img/s; C8 encoder 10.8 img/s | [profile.json](../benchmarks/encoders/profile.json) |
| C8 encoder exports to plain PyTorch with zero token error | [export.json](../benchmarks/encoders/export.json) |
| GPU training, arm A: 2.86 GB peak memory, ~439 img/s (Colab) | [docs/profiling/README.md](profiling/README.md) |
| YLM already reads Ouroboros outputs | [YLM ouroboros_bridge.py](https://github.com/yaniguan/YLM/blob/main/src/toolkit/ouroboros_bridge.py) |

New in this change (scaffold, no results):

- [lab/molecule_context.py](../lab/molecule_context.py): predictions CSV -> molecule context JSONL + a short prompt block for the reasoner. Status `ok / review / rejected / invalid`. The `gold` column is never copied.
- [lab/edge_plan.py](../lab/edge_plan.py) + [lab/configs/edge_distill.json](../lab/configs/edge_distill.json): checks the distillation config and prints the plan with `--dry-run`. It refuses test sets for selection.
- Tests: [test_lab_molecule_context.py](../tests/test_lab_molecule_context.py), [test_lab_edge_plan.py](../tests/test_lab_edge_plan.py).

## Lab tasks it serves

| Task | Input | Output | Benchmark |
|---|---|---|---|
| Read a structure from a paper or patent figure | figure image | standardized SMILES, InChIKey, molfile, status | Keylab/OCSR-Benchmarks (dev-test `subset100`, `frozen-test-v2`) |
| Molecule context for protocol QA | figures in a protocol or paper | context JSONL / prompt block for YLM | LAB-Bench ProtocolQA, end-to-end through [labbench-posttrain](https://github.com/yaniguan/labbench-posttrain) |
| Step debugging: right reagent? | reagent drawing or label | InChIKey to compare with the protocol step | BioProBench error correction (via labbench-posttrain); OCSR part: no label set yet |
| Read notebook sketches | photo of a hand-drawn structure | SMILES, or `review` | DECIMER Hand-drawn 200-image subset: 4.5% exact for v2 A+C beam 5 ([handdrawn-decimer/](../benchmarks/handdrawn-decimer/)) |

OCSR is only one input in the LAB-Bench and BioProBench rows. Those scores measure the whole agent, not Ouroboros alone.

## Plan

| | Goal | Data | Method | Metric | Done when |
|---|---|---|---|---|---|
| M1 | Molecule context reaches the reasoner | predictions CSV from `predict_real.py` | `lab/molecule_context.py`; YLM reads the JSONL | schema tests pass; YLM test reads the file | YLM prompt uses only `ok` rows |
| M2 | Trustworthy "I don't know" | PMC validation split (never test) | pick gate limits with `eval_selective.py` | precision of accepted rows vs coverage, on validation | limits fixed on validation; `frozen-test-v2` scored once |
| M3 | Distill teacher into one fast student | PMC train + teacher pseudo-labels on unlabeled figures, leakage-gated | teacher = A + C, beam 5, TTA; student = one arm, greedy | validation exact match vs teacher; img/s | gap to teacher is TBD and agreed before the run |
| M4 | Run next to the robot | same validation set | int8 / fp16 quantization; edge export | exact-match drop vs fp32; latency per image on device | numbers recorded on a Jetson Orin-class box |

Progress on M1–M4:

- **M1 (Ouroboros side done; YLM import open in #77):** `ouroboros.molecule_context/v1` schema frozen, `read_context_jsonl()` rejects `gold`, and `prompt_block()` emits SMILES only on `status=ok`.
- **M2 (completed in #79):** Gate limits (`min_margin=0.05`, `max_disagreement=0.0`) locked on `PMC val` (`92.19%` precision at `19.94%` coverage) and scored once on `frozen-test-v2` (`97.78%` precision, `88/90`, `30.00%` coverage; [benchmarks/selective-gate/](../benchmarks/selective-gate/)).
- **M3 (tooling ready; GPU student training open in #80):** `predict_real.py --unlabeled`, `molecule_context --manifest-out --source-shards`, and `prepare_real_finetune.py` wired in `lab/edge_plan.py`.
- **M4 (quantizer + CPU check landed; Jetson run open in #81):** `lab/quantize.py` measured `40.08%` (`fp32`) vs `39.88%` (`int8-dynamic`, `-0.21 pp`) on `PMC val` ([benchmarks/quantize-cpu/](../benchmarks/quantize-cpu/)).

## Distillation and edge path

```
teacher: A + C ensemble, beam 5, TTA pad   (2 models x 2 views, many decode steps)
   |  pseudo-labels on unlabeled figures, kept only if status = ok and leakage gate passes
student: one arm, greedy                   (1 model x 1 view)
   |  int8 dynamic (decoder Linear layers) or fp16
edge: Jetson Orin Nano Super, or CPU next to the robot
```

- Target size: student no bigger than arm A (43.2M params). A smaller decoder is TBD.
- Arm choice: A is stronger on upright figures; C is the choice if inputs are rotated (photos, labels). Decide on validation.
- Target hardware: Jetson Orin Nano Super ($399) first; Jetson AGX Orin 64GB ($3,499) only if needed.
- Optional later: DPO on n-best pairs (right vs wrong stereo), then GRPO with a code-checked reward (standardized exact match). Same SFT -> DPO -> GRPO recipe as YLM. Not run yet.
- Plan printout: `python -m lab.edge_plan --dry-run`.

## Cost

Full price table and buy-vs-rent math: [YLM docs/lab_cost_setup.md](https://github.com/yaniguan/YLM/blob/main/docs/lab_cost_setup.md). All prices are Nov 2026 estimates and move daily.

Single-GPU plan (one H200 or B200 kept busy, with this repo's calls batched into the same loop): [YLM docs/single_gpu_saturation.md](https://github.com/yaniguan/YLM/blob/main/docs/single_gpu_saturation.md).

- The OCSR student is small. Arm A training peaked at 2.86 GB GPU memory ([profiling](profiling/README.md)), so a rented L4 (from $0.44/hr) or RTX 4090 (~$0.34/hr) is enough.
- Teacher pseudo-labeling is many forward passes. Use short A100 bursts ($1.19-1.39/hr) or Colab Pro ($9.99/mo, ~18 A100-40GB hours).
- Do not buy a GPU box for this repo. A used 4090 (~$3k) equals ~8,800 rented 4090 hours at $0.34/hr.
- Edge: one Jetson Orin Nano Super ($399), bought only after M3 has a student worth testing.

## Risks

- Licenses: the fine-tuned v2 weights are CC-BY-NC-4.0 because of the PMC training data ([ROADMAP_SOTA.md](ROADMAP_SOTA.md)). Only the synthetic v0.1.0-alpha weights are MIT. A commercial product needs retraining on cleared data.
- Data rights: real journal and patent figures may not be redistributable ([DATA.md](../DATA.md)). BioProBench is CC-BY-NC-4.0 (non-commercial); LAB-Bench is CC-BY-SA-4.0 (share-alike).
- Domain gap: lab photos, labels and hand-drawn sketches are not measured. Real-figure accuracy may not transfer.
- Stereo is weak (18.4-20.4% for the A + C ensemble on stereo figures). A wrong enantiomer looks valid. Keep `review` / `rejected` rows out of the reasoner.
- Dev-test is optimistic and scoring parity with published work is unverified ([standardize.py](../ouroboros/eval/standardize.py)).

## Remaining steps

1. Import `read_context_jsonl` / `prompt_block` inside the `YLM` repository (M1, #77).
2. Agree on the target student-vs-teacher gap and run GPU student distillation (M3, #80).
3. Quantize the distilled student and measure latency/accuracy on a Jetson Orin Nano Super (M4, #81).
4. Record single-GPU `lab/grpo_verifier.py` throughput and peak VRAM alongside a sleeping rollout engine (#82).
