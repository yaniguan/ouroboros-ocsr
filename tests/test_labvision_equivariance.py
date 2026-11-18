import numpy as np
import pytest

from ouroboros.labvision.decode import decode, lattice_phase
from ouroboros.labvision.keypoints import kp_from_pixels
from ouroboros.labvision.render import render


@pytest.mark.parametrize("k", [6, 36])
@pytest.mark.parametrize("n", [1, 7, 133, 719])
def test_roll_shifts_phase_exactly(k, n):
    row = np.random.default_rng(0).random(720)
    period = 2 * np.pi / k
    shift = lattice_phase(np.roll(row, n), k) - lattice_phase(row, k)
    expected = n * 2 * np.pi / 720
    assert abs((shift - expected + period / 2) % period - period / 2) < 1e-9


@pytest.mark.parametrize("kind", ["rect", "circ"])
@pytest.mark.parametrize("alpha,delta", [(10.0, 37.0), (200.0, 95.0)])
def test_rotated_renders_agree(kind, alpha, delta):
    thetas = []
    for rot in (alpha, alpha + delta):
        scene = render(
            kind, np.random.default_rng(5), rot_deg=rot, persp=0.0, occlude=False, glare=False
        )
        dec = decode(scene.image, kp_from_pixels(scene.image, kind), scene.spec)
        assert not dec.ambiguous
        thetas.append(dec.theta0)
    assert abs((thetas[1] - thetas[0] - delta + 180) % 360 - 180) < 0.5
