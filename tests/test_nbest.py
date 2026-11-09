import pytest
from ouroboros.eval.nbest import diagnose


def test_distinct_stereo_and_selection():
    row = dict(id='1', subset='ACS', gold='N[C@@H](C)O', pred='CC', candidates=[
        {'smiles': s} for s in ['invalid', 'CC', 'C(C)', 'N[C@H](C)O', 'N[C@@H](C)O']])
    out = diagnose([row, row | {'subset': 'other'}], (1, 2, 3))
    assert out['rows'][0]['correct_rank'] == 3
    assert out['rows'][0]['constitution_rank'] == 2
    assert out['rows'][0]['selection_failure']
    assert list(out['overall']['oracle_exact'].values()) == [0, 0, 1]
    assert out['overall']['n'] == 2
    with pytest.raises(ValueError):
        diagnose([row, row])


def test_equivalent_and_invalid():
    out = diagnose([dict(id=1, gold='CCO', pred='OCC', candidates=[
        {'smiles': s} for s in ['OCC', 'CCO', 'bad']])])
    assert out['rows'][0]['distinct_candidates'] == 1
    assert not out['rows'][0]['selection_failure']
