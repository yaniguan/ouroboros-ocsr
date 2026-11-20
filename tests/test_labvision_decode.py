import numpy as np
import pytest

from ouroboros.labvision.decode import decode, locate_well, pixel_to_well
from ouroboros.labvision.geometry import apply_h
from ouroboros.labvision.keypoints import kp_from_gt, kp_from_pixels
from ouroboros.labvision.render import render
from ouroboros.labvision.spec import all_wells, default_order
from scripts.labvision_bench import pitch_px

CLEAN = dict(persp=0.0, occlude=False, glare=False)


def n_correct(scene, dec):
    order = default_order(scene.spec)
    gt = apply_h(scene.H, all_wells(scene.spec, order=order))
    return sum(pixel_to_well(dec, gt[j], scene.spec, order=order) == j + 1 for j in range(96))


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize("rot", [20.0, 133.0, 250.0])
def test_rect_rotation_and_flip_both_decode(rot):
    for r in (rot, rot + 180):
        scene = render("rect", np.random.default_rng(1), rot_deg=r, **CLEAN)
        dec = decode(scene.image, kp_from_pixels(scene.image, "rect"), scene.spec)
        assert not dec.ambiguous
        assert n_correct(scene, dec) == 96
        assert angle_diff(dec.theta0, r) < 1.0


@pytest.mark.parametrize("kind", ["rect", "circ"])
def test_hide_mark_sets_ambiguous(kind):
    scene = render(kind, np.random.default_rng(2), rot_deg=70, hide_mark=True, **CLEAN)
    dec = decode(scene.image, kp_from_pixels(scene.image, kind), scene.spec)
    assert dec.ambiguous


@pytest.mark.parametrize("kind", ["rect", "circ"])
def test_both_sources_perfect_at_zero_noise(kind):
    for seed in range(3):
        rng = np.random.default_rng(seed)
        scene = render(kind, rng, **CLEAN)
        for kp in (kp_from_pixels(scene.image, kind), kp_from_gt(scene, 0.0, rng)):
            dec = decode(scene.image, kp, scene.spec)
            assert not dec.ambiguous and not dec.mirrored
            assert n_correct(scene, dec) == 96
            order = default_order(scene.spec)
            err = locate_well(dec, 65, scene.spec, order=order) - apply_h(
                scene.H, all_wells(scene.spec, order=order)[64]
            )
            assert np.hypot(*err) <= 0.25 * pitch_px(scene)  # well-65 hit


@pytest.mark.parametrize("rot", [359.8, 0.0, 0.3])
def test_circ_wraps_at_zero(rot):
    scene = render("circ", np.random.default_rng(3), rot_deg=rot, **CLEAN)
    dec = decode(scene.image, kp_from_pixels(scene.image, "circ"), scene.spec)
    assert not dec.ambiguous
    assert angle_diff(dec.theta0, rot) < 0.5
    assert n_correct(scene, dec) == 96


def test_pixel_far_from_wells_is_none():
    scene = render("rect", np.random.default_rng(4), **CLEAN)
    dec = decode(scene.image, kp_from_pixels(scene.image, "rect"), scene.spec)
    order = default_order(scene.spec)
    between = apply_h(scene.H, all_wells(scene.spec, order=order)[0] + [4.5, 4.5])
    assert pixel_to_well(dec, between, scene.spec, order=order) is None
    assert pixel_to_well(dec, apply_h(scene.H, [-50.0, -50.0]), scene.spec, order=order) is None
