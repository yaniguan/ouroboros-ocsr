"""``python -m ouroboros.train --config configs/x.yaml --out runs/x [--set a.b=v ...]``.

Re-running the same command after a disconnect resumes from ``{out}/ckpt/last.pt``.
"""

import argparse

from ouroboros.train.config import load_config
from ouroboros.train.trainer import Trainer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4")
    a = ap.parse_args(argv)
    Trainer(load_config(a.config, a.set), a.out).fit()


if __name__ == "__main__":
    main()
