import json
import pytest
from ouroboros.decode.confidence import confidence, selective_report


def test_equivalent_stereo_and_boundaries():
    c = [('CCO', -1), ('OCC', -1.2), ('CC', -2)]
    assert confidence('OCC', c, ['CCO', 'OCC'], min_margin=1, max_disagreement=0)['accepted']
    assert not confidence('OCC', c, min_margin=1.01)['accepted']
    result = confidence('N[C@H](C)O', [('N[C@H](C)O', -1), ('CC', -2)],
                        ['N[C@@H](C)O'], max_disagreement=0)
    assert result['rejection_reason'] == 'source_disagreement'


@pytest.mark.parametrize('c', [[('CC', 0)], [('CC', 0), ('C', float('nan'))],
                                [('CC', None), ('C', -1)]])
def test_unknown_margin(c):
    result = confidence('CC', c, min_margin=0)
    assert not result['accepted'] and result['margin'] is None


def test_report():
    rows = [dict(pred='CC', gold='CC', subset='ACS', candidates=[('CC', 0), ('C', -1)]),
            dict(pred='CC', gold='C', subset='ACS', candidates=[('CC', 0), ('C', -.1)])]
    out = selective_report(rows, [0, .5, 2])
    assert [r['overall']['coverage'] for r in out] == [1, .5, 0]
    assert out[1]['overall']['precision'] == 1
    assert out[2]['overall']['precision'] is None


def test_rejected_export_without_embedding(tmp_path, monkeypatch):
    from scripts import image_to_forcefield as ff
    path = tmp_path / 'pred.csv'
    path.write_text('id,pred,accepted,rejection_reason\n1,CC,False,low_margin\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('rejected row reached embedding')
    monkeypatch.setattr(ff, 'embed', forbidden)
    out = tmp_path / 'data.extxyz'
    summary = ff.main(['--predictions', str(path), '--out', str(out), '--mace-model', 'none',
                       '--respect-confidence'])
    assert summary['frames_written'] == 0
    rec = json.loads(out.with_suffix('.manifest.jsonl').read_text())
    assert rec['status'] == 'confidence_rejected' and rec['rejection_reason'] == 'low_margin'
    assert not out.exists()
