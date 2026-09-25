"""Build the fixed decoder vocabulary (configs/vocab.json), shared by every arm.

Tokens = specials + every token of the synthetic 1M train manifest + a fixed list of tokens that
occur in real documents but not in the drug-like synthetic pool (so real-data labels do not map
to <unk> when real data is mixed in).
"""

import argparse
import csv
import json

from ouroboros.decode.tokenizer import SPECIALS, tokenize

EXTRA = (
    ["B", "b", "[B]", "[Si]", "[SiH]", "[SiH2]", "[SiH3]", "[Se]", "[se]", "[Te]", "[As]", "[Sn]"]
    + ["[Na+]", "[K+]", "[Li+]", "[Mg]", "[Ca]", "[Zn]", "[Fe]", "[Cu]", "[Pt]", "[Pd]", "[Al]"]
    + ["[H]", "[2H]", "[3H]", "[13C]", "[OH]", "[NH]", "[NH2]", "[CH]", "[CH2]", "[C]", "[c]"]
    + ["[NH+]", "[NH2+]", "[NH3+]", "[nH+]", "[O+]", "[o+]", "[OH+]", "[C-]", "[CH-]", "[Cl-]"]
    + ["[Br-]", "[I-]", "[I+]", "[Cl+]", "[P+]", "[PH]", "[S+]", "[SH]", "[se+]", "[B-]", "[BH-]"]
    + ["[N@+]", "[N@@+]", "[Si@]", "[Si@@]", "[C@@H2]", "[P@@+]", "[P@+]", "[S@+]", "[S@@+]"]
    + [str(i) for i in range(1, 10)]
    + [f"%{i}" for i in range(10, 20)]
    + [".", ":", "*", "~", "$", "+"]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/full/manifests/train.tsv")
    ap.add_argument("--out", default="configs/vocab.json")
    a = ap.parse_args()
    seen = set()
    with open(a.manifest, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            seen.update(tokenize(r["smiles"]))
    n_syn = len(seen)
    seen.update(EXTRA)
    vocab = SPECIALS + sorted(seen - set(SPECIALS))
    with open(a.out, "w") as f:
        json.dump(vocab, f, indent=0)
    print(f"{len(vocab)} tokens ({n_syn} from the synthetic manifest) -> {a.out}")


if __name__ == "__main__":
    main()
