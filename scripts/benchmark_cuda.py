"""Current-model capacity probe. CUDA/bf16 by default; CPU/fp32 for small smoke checks."""

import argparse
import json

from ouroboros.train.benchmark import capacity
from ouroboros.train.config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sequence-length", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = ap.parse_args(argv)
    print(
        json.dumps(
            capacity(
                load_config(args.config, args.set),
                args.output,
                args.steps,
                args.warmup,
                args.batch,
                args.sequence_length,
                args.device,
                args.precision,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
