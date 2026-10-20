"""v1 experiment grid: arm x real-data fraction x seed. Task ids are 1-based and stable.

Order is seed-major so the headline comparison (all arms, seed 0) finishes first. Seed-0
synthetic-only A/B/C already exist from v0 and are not rerun (their real-document evaluation is a
separate task).

    python scripts/hpc/v1_grid.py            # print the grid
    python scripts/hpc/v1_grid.py --task 7   # print one task as JSON
"""

from __future__ import annotations

import argparse
import json

ARMS = ("A", "B", "C", "Cplus")
FRACTIONS = (0.0, 0.1, 0.5)
SEEDS = (0, 1, 2)
DONE_IN_V0 = {("A", 0.0, 0), ("B", 0.0, 0), ("C", 0.0, 0)}

CONFIGS = {
    "A": "pilot-A-200k.yaml",
    "B": "pilot-B-200k.yaml",
    "C": "pilot-C-200k.yaml",
    "Cplus": "pilot-Cplus-200k.yaml",
}


def grid() -> list[dict]:
    tasks = []
    for seed in SEEDS:
        for fraction in FRACTIONS:
            for arm in ARMS:
                if (arm, fraction, seed) in DONE_IN_V0:
                    continue
                tasks.append(
                    {
                        "id": len(tasks) + 1,
                        "arm": arm,
                        "fraction": fraction,
                        "seed": seed,
                        "config": CONFIGS[arm],
                        "run_id": f"{arm}_n200k_f{fraction:g}_s{seed}",
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
