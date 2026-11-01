"""Score a predictions file (csv / tsv / jsonl with id, gold, pred[, subset][, kind]) and write a
JSON and Markdown report. See ``ouroboros/eval/harness.py`` for columns and rules.

    python scripts/eval_predictions.py preds.csv --json report.json --md report.md
    python scripts/eval_predictions.py runs/X/eval/predictions.jsonl --where angle=0
"""

from __future__ import annotations

from ouroboros.eval.harness import main

if __name__ == "__main__":
    raise SystemExit(main())
