import numpy as np

from ouroboros.labvision.decode import decode, pixel_to_well
from ouroboros.labvision.geometry import apply_h
from ouroboros.labvision.keypoints import kp_from_pixels
from ouroboros.labvision.render import render
from ouroboros.labvision.spec import CircSpec, IndexOrder, index, well_xy

RING = IndexOrder.RING_MAJOR


def _flipped():
    return render(
        "circ",
        np.random.default_rng(6),
        rot_deg=75,
        persp=0.0,
        occlude=False,
        glare=False,
        mirror=True,
    )


def test_mirror_flagged_with_clockwise_spec():
    scene = _flipped()
    spec = CircSpec(clockwise=True)
    dec = decode(scene.image, kp_from_pixels(scene.image, "circ"), spec)
    assert dec.mirrored and dec.ambiguous
    # raw IDs of asymmetric wells come out mirrored: R4S04 <-> R4S32
    i65, i93 = index(spec, "R4S04", order=RING), index(spec, "R4S32", order=RING)
    gt65 = apply_h(scene.H, well_xy(scene.spec, i65, order=RING))
    assert pixel_to_well(dec, gt65, spec, order=RING) == i93


def test_mirror_decodes_with_counter_clockwise_spec():
    scene = _flipped()
    spec = CircSpec(clockwise=False)
    dec = decode(scene.image, kp_from_pixels(scene.image, "circ"), spec)
    assert not dec.mirrored and not dec.ambiguous
    for i in range(1, 97):
        gt = apply_h(scene.H, well_xy(scene.spec, i, order=RING))
        assert pixel_to_well(dec, gt, spec, order=RING) == i


def test_render_follows_counter_clockwise_spec():
    spec = CircSpec(clockwise=False)
    scene = render(
        "circ",
        np.random.default_rng(7),
        rot_deg=75,
        persp=0.0,
        occlude=False,
        glare=False,
        spec=spec,
    )
    dec = decode(scene.image, kp_from_pixels(scene.image, "circ"), spec)
    assert not dec.mirrored and not dec.ambiguous
    for i in range(1, 97):
        gt = apply_h(scene.H, well_xy(spec, i, order=RING))
        assert pixel_to_well(dec, gt, spec, order=RING) == i
