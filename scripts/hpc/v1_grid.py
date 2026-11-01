"""v1 experiment grid: arm x real-label budget K x seed. Task ids are 1-based and stable.

Arms (one question per contrast; every arm shares trunk depth, decoder, optimizer rules, data,
selection rule and training budget):
  Babs  plain CNN, absolute 2-D position table, rotation augmentation   (the v0 "B")
  Brel  plain CNN, the same rotation-invariant distance mixer as C, rotation augmentation
  C     C8 steerable CNN + group pooling + distance mixer, no augmentation
Babs -> Brel isolates the position representation; Brel -> C isolates the equivariant convolutions
(against augmentation). K is the number of independent real labels in the training pool (nested,
article-sampled); every K > 0 run mixes real data at the same ratio (REAL_FRACTION) and the same
total step budget, so K changes label count, not exposure.

    python scripts/hpc/v1_grid.py            # print the grid
    python scripts/hpc/v1_grid.py --task 7   # print one task as JSON
"""

from __future__ import annotations

import argparse
import json

ARMS = ("Babs", "Brel", "C")
BUDGETS = (0, 128, 512, 2048)
SEEDS = (0, 1, 2)
REAL_FRACTION = 0.1

CONFIGS = {"Babs": "pilot-B-200k.yaml", "Brel": "pilot-Brel-200k.yaml", "C": "pilot-C-200k.yaml"}


def grid() -> list[dict]:
    tasks = []
    for seed in SEEDS:
        for k in BUDGETS:
            for arm in ARMS:
                tasks.append(
                    {
                        "id": len(tasks) + 1,
                        "arm": arm,
                        "budget": k,
                        "fraction": REAL_FRACTION if k else 0.0,
                        "seed": seed,
                        "config": CONFIGS[arm],
                        "run_id": f"{arm}_K{k}_s{seed}",
                    }
                )
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    tasks = grid()
    if a.task:
        print(json.dumps(tasks[a.task - 1]))
    else:
        for t in tasks:
            print(t["id"], t["run_id"])
        print(len(tasks), "tasks")


if __name__ == "__main__":
    main()
