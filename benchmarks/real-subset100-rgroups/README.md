# Real subset100 after labelled R-group support

CPU reruns of unchanged v0.1.0-alpha checkpoints, greedy decoding, angle 0, batch 8, two threads. Re-ingestion recovered 20 of 22 previously invalid labels: 298/300 references are usable (USPTO 98, UOB 100, ACS 100).

| Arm | Exact | Valid | Prediction seconds |
|---|---:|---:|---:|
| A | 54/298 = 18.12% | 78.19% | 68.0 |
| B | 50/298 = 16.78% | 75.84% | 99.6 |
| C | 50/298 = 16.78% | 79.19% | 83.4 |

Correct counts are unchanged from the earlier 278-example denominator. These are not fine-tuned-model results. Reports contain source breakdowns and confidence intervals. Raw predictions stay in ignored `runs/real-subset100-rgroups/`.

Scoring parity with arXiv:2608.09100 remains unverified.

Commands, implementation notes, validation and outstanding Colab work: [worklog](../../docs/WORKLOG_oct29b.md). Synthetic examples and throughput: [augmentation artifacts](../augmentation-oct29/).
