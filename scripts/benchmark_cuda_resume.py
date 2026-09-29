"""Verify current Trainer resume on configured shards; writes reports/checkpoints to a new run."""

import argparse
import json

from ouroboros.train.benchmark import resume_comparison
from ouroboros.train.config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checkpoint-step", type=int, default=25)
    ap.add_argument("--compared-steps", type=int, default=100)
    ap.add_argument("--kill-after", type=int, default=3)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = ap.parse_args(argv)
    print(
        json.dumps(
            resume_comparison(
                load_config(args.config, args.set),
                args.output,
                args.checkpoint_step,
                args.compared_steps,
                args.kill_after,
                args.tolerance,
                args.device,
                args.precision,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
