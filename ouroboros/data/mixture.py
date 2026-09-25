"""Synthetic/real mixture sampler with an exact per-batch real fraction and exact resume (Am1-B).

Indices refer to ``ConcatDataset([synthetic, real])``: ``i < n_synth`` is synthetic,
``n_synth + j`` is real sample ``j``.

Batch ``k`` (of size ``B``) contains ``q_k = floor((k+1)·B·f) - floor(k·B·f)`` real samples, so
the cumulative real count after ``k`` batches is exactly ``floor(k·B·f)`` (error diffusion: every
batch is within one sample of ``B·f``). The real slots inside a batch are placed at seeded random
positions; synthetic samples fill the remaining slots in the order of their own stream.

Each source has its own stream of seeded per-epoch permutations (``ResumableSampler``); the
synthetic stream uses the same seed as the synthetic-only pipeline, so with ``f = 0`` the
sampler yields exactly the synthetic-only order. Because the per-batch counts are closed-form,
``set_position(p)`` recovers both streams' positions without replaying the stream.
"""

from __future__ import annotations

import math
import random

from torch.utils.data import Sampler

from ouroboros.data.loader import ResumableSampler


class MixtureSampler(Sampler[int]):
    def __init__(
        self,
        n_synth: int,
        n_real: int,
        real_fraction: float,
        batch_size: int,
        seed: int = 0,
    ):
        if not 0.0 <= real_fraction <= 1.0:
            raise ValueError("real_fraction must be in [0, 1]")
        if real_fraction > 0 and n_real == 0:
            raise ValueError("real_fraction > 0 but the real dataset is empty")
        if real_fraction < 1 and n_synth == 0:
            raise ValueError("real_fraction < 1 but the synthetic dataset is empty")
        self.n_synth, self.n_real = n_synth, n_real
        self.f, self.B, self.seed = real_fraction, batch_size, seed
        self.position = 0

    # closed-form bookkeeping -------------------------------------------------------------
    def _real_before_batch(self, k: int) -> int:
        return math.floor(k * self.B * self.f + 1e-9)

    def _real_slots(self, k: int) -> set[int]:
        q = self._real_before_batch(k + 1) - self._real_before_batch(k)
        if q == 0:
            return set()
        if q == self.B:
            return set(range(self.B))
        return set(random.Random(f"{self.seed}-mix-{k}").sample(range(self.B), q))

    def _stream_positions(self, position: int) -> tuple[int, int]:
        """(#synthetic, #real) samples consumed before global ``position``."""
        k, off = divmod(position, self.B)
        n_real = self._real_before_batch(k) + sum(1 for s in self._real_slots(k) if s < off)
        return position - n_real, n_real

    def set_position(self, position: int) -> None:
        self.position = position

    # iteration ---------------------------------------------------------------------------
    def __iter__(self):
        n_s, n_r = self._stream_positions(self.position)
        synth = ResumableSampler(max(self.n_synth, 1), seed=self.seed)
        synth.set_position(n_s)
        real = ResumableSampler(max(self.n_real, 1), seed=self.seed + 7919)
        real.set_position(n_r)
        it_s, it_r = iter(synth), iter(real)
        pos = self.position
        while True:
            k, off = divmod(pos, self.B)
            slots = self._real_slots(k)
            for s in range(off, self.B):
                yield self.n_synth + next(it_r) if s in slots else next(it_s)
            pos = (k + 1) * self.B

    def __len__(self) -> int:  # nominal
        return self.n_synth + self.n_real


def is_real(index: int, n_synth: int) -> bool:
    return index >= n_synth
