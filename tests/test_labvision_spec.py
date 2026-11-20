import pytest

from ouroboros.labvision.spec import CircSpec, IndexOrder, RectSpec, index, label

RECT, CIRC = RectSpec(), CircSpec()


def test_well_65_labels():
    assert label(RECT, 65, order=IndexOrder.ROW_MAJOR) == "F5"
    assert label(RECT, 65, order=IndexOrder.COL_MAJOR) == "A9"
    assert label(CIRC, 65, order=IndexOrder.RING_MAJOR) == "R4S04"


@pytest.mark.parametrize(
    "spec,order",
    [(RECT, IndexOrder.ROW_MAJOR), (RECT, IndexOrder.COL_MAJOR), (CIRC, IndexOrder.RING_MAJOR)],
)
def test_all_96_round_trip(spec, order):
    labels = [label(spec, i, order=order) for i in range(1, 97)]
    assert len(set(labels)) == 96
    assert [index(spec, s, order=order) for s in labels] == list(range(1, 97))


def test_bad_or_missing_order_raises():
    with pytest.raises(TypeError):
        label(RECT, 65)
    with pytest.raises(TypeError):
        label(RECT, 65, order="row_major")
    with pytest.raises(ValueError):
        label(RECT, 65, order=IndexOrder.RING_MAJOR)
    with pytest.raises(ValueError):
        label(CIRC, 65, order=IndexOrder.ROW_MAJOR)
    with pytest.raises(ValueError):
        label(RECT, 97, order=IndexOrder.ROW_MAJOR)
    with pytest.raises(ValueError):
        index(RECT, "I1", order=IndexOrder.ROW_MAJOR)
    with pytest.raises(ValueError):
        index(RECT, "A0", order=IndexOrder.ROW_MAJOR)
    with pytest.raises(ValueError):
        index(CIRC, "R0S06", order=IndexOrder.RING_MAJOR)
