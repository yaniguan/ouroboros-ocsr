import numpy as np
import pytest
from PIL import Image, ImageDraw

from ouroboros.data.augment import handdrawn_augment


def _page():
    img = Image.new('L', (96, 96), 255)
    d = ImageDraw.Draw(img)
    d.line([(10, 48), (86, 48)], fill=0, width=2)
    d.line([(48, 10), (48, 86)], fill=0, width=2)
    return img


def test_deterministic_and_shape():
    x = handdrawn_augment(_page(), np.random.default_rng(3))
    y = handdrawn_augment(_page(), np.random.default_rng(3))
    z = handdrawn_augment(_page(), np.random.default_rng(4))
    assert x.size == (96, 96) and x.mode == 'L'
    assert np.array_equal(np.asarray(x), np.asarray(y))
    assert not np.array_equal(np.asarray(x), np.asarray(z))


def test_identity_when_disabled():
    p = _page()
    out = handdrawn_augment(p, np.random.default_rng(0), wobble=0, jitter=0, width_var=0,
                            ink_var=0, paper_noise=0, blur=0)
    assert np.array_equal(np.asarray(out), np.asarray(p))


def test_wobble_moves_lines_but_keeps_ink():
    p = _page()
    out = handdrawn_augment(p, np.random.default_rng(1), wobble=3, jitter=0.5, width_var=0,
                            ink_var=0, paper_noise=0, blur=0)
    a, b = np.asarray(p) < 128, np.asarray(out) < 128
    assert (a != b).sum() > 20
    assert 0.5 < b.sum() / a.sum() < 1.5


def test_ink_and_paper_ranges():
    out = np.asarray(handdrawn_augment(_page(), np.random.default_rng(2), ink_var=0.5, paper_noise=8))
    assert out.min() < 128 and np.median(out) > 200


@pytest.mark.parametrize('kw', [dict(wobble=-1), dict(width_var=2), dict(wobble_scale=1), dict(blur=-1)])
def test_invalid(kw):
    with pytest.raises(ValueError):
        handdrawn_augment(_page(), np.random.default_rng(), **kw)
