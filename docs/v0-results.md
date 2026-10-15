# v0.1.0-alpha measured results

Single seed (0), 200,000 synthetic training examples, 10,000 steps, bf16. Same data selection,
decoder and stopping rule for A/B/C. Best validation checkpoint selected before test; all three
selected step 10,000 (validation exact still rising at the cap, so all arms are under-trained).
Raw summaries, logs, configs and weight receipts: `benchmarks/v0.1.0-alpha/`.

| Arm | Encoder | Rotation aug | Params | Val exact (1k) | Test exact (10k, 0°) | 95% CI | Stereo-subset exact | 90° / 270° exact | 24-angle mean | 24-angle min |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| A | ResNet-style CNN | no | 43.2M | 66.8% | 68.69% | 67.84–69.54 | 39.3% | 7.6% / 7.3% | 25.4% | 7.3% |
| B | ResNet-style CNN | yes | 43.2M | 61.1% | 64.41% | 63.44–65.31 | 36.7% | 64.7% / 64.3% | 65.7% | 64.3% |
| C | C8 steerable CNN | no | 32.4M | 63.5% | 67.08% | 66.15–67.96 | 38.0% | 67.1% / 66.9% | 65.8% | 65.0% |

Rotation sweep: fixed 2,000-example test prefix, 24 angles (0–345°, 15° steps); the mean is an
equal-weight average over angles, not 48,000 independent molecules. A/C ran on Colab A100-40GB;
B ran on Hoffman2 (A6000 queue), so wall-clock numbers are not comparable across A/C and B.

## Reading the numbers (what is and is not supported)

- Rotation robustness: C matches B across angles (65.8% vs 65.7% mean) **without** augmentation and
  with 25% fewer parameters; A collapses off 0°/180°. The prior is a working substitute for
  augmentation on rendered, rotated inputs.
- At 0°, C is 2.7 points above B (non-overlapping per-set CIs), but 1.6 points below A. The
  invariance cost at 0° is real for B and small for C. Per-set CIs do not include seed variance,
  so differences of a few points are not established until multiple seeds are run.
- Stereo remains the weak point for every arm (stereo-subset exact 37–39%).
- Not established: real-document performance (H2/H3), C+ (equivariant plus augmentation),
  parameter-matched ablations, external-paper scoring parity.
