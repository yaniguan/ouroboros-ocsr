"""Regex SMILES tokenizer (atom-level, stereo tokens @, @@, /, \\ kept as-is)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

# Standard atom-level SMILES regex (Schwaller et al. 2019), bracket atoms kept whole so
# chirality tags like [C@@H] are single tokens.
SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


def tokenize(smiles: str) -> list[str]:
    tokens = SMILES_REGEX.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"SMILES not fully tokenized: {smiles!r}")
    return tokens


class SmilesTokenizer:
    def __init__(self, vocab: list[str]):
        if vocab[: len(SPECIALS)] != SPECIALS:
            vocab = SPECIALS + [t for t in vocab if t not in SPECIALS]
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, smiles: Iterable[str]) -> SmilesTokenizer:
        seen: dict[str, None] = {}
        for s in smiles:
            for t in tokenize(s):
                seen.setdefault(t, None)
        return cls(SPECIALS + sorted(seen))

    pad_id = property(lambda self: self.stoi[PAD])
    bos_id = property(lambda self: self.stoi[BOS])
    eos_id = property(lambda self: self.stoi[EOS])
    unk_id = property(lambda self: self.stoi[UNK])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, smiles: str, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(smiles)]
        return [self.bos_id, *ids, self.eos_id] if add_special else ids

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, indent=0))

    @classmethod
    def load(cls, path: str | Path) -> SmilesTokenizer:
        return cls(json.loads(Path(path).read_text()))
