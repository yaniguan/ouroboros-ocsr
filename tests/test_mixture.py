import itertools

import pytest

from ouroboros.data.loader import ResumableSampler
from ouroboros.data.mixture import MixtureSampler

PLACEHOLDER_FRACTIONS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]  # Am1-B placeholders (U2)


def take(sampler, n):
    return list(itertools.islice(iter(sampler), n))


@pytest.mark.parametrize("f", PLACEHOLDER_FRACTIONS + [0.095, 0.502, 1 / 3])
def test_fraction_over_10k_and_per_batch(f):
    n_synth, n_real, B = 5000, 700, 64
    idx = take(MixtureSampler(n_synth, n_real, f, B, seed=1), 10_000)
    real = [i >= n_synth for i in idx]
    assert abs(sum(real) / len(real) - f) <= 0.01
    for k in range(len(idx) // B):  # every full batch is within one sample of B*f
        assert abs(sum(real[k * B : (k + 1) * B]) - B * f) < 1.0 + 1e-9


def test_deterministic_and_seed_dependent():
    a = take(MixtureSampler(1000, 300, 0.25, 32, seed=5), 5000)
    b = take(MixtureSampler(1000, 300, 0.25, 32, seed=5), 5000)
    c = take(MixtureSampler(1000, 300, 0.25, 32, seed=6), 5000)
    assert a == b and a != c


def test_zero_fraction_reproduces_synthetic_only_order():
    mix = take(MixtureSampler(777, 50, 0.0, 32, seed=3), 3000)
    plain = take(ResumableSampler(777, seed=3), 3000)
    assert mix == plain


def test_full_real_fraction():
    idx = take(MixtureSampler(100, 40, 1.0, 16, seed=0), 400)
    assert all(i >= 100 for i in idx)
    assert sorted(i - 100 for i in idx[:40]) == list(range(40))  # real epochs are permutations


@pytest.mark.parametrize("pos", [0, 1, 31, 32, 33, 1000, 4097])
def test_resume_matches_uninterrupted(pos):
    full = take(MixtureSampler(900, 250, 0.1, 32, seed=2), 6000)
    s = MixtureSampler(900, 250, 0.1, 32, seed=2)
    s.set_position(pos)
    assert take(s, 6000 - pos) == full[pos:]


def test_bad_arguments():
    with pytest.raises(ValueError):
        MixtureSampler(10, 0, 0.1, 8)
    with pytest.raises(ValueError):
        MixtureSampler(0, 10, 0.5, 8)
    with pytest.raises(ValueError):
        MixtureSampler(10, 10, 1.5, 8)
