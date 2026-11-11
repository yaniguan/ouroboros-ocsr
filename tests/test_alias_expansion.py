import json
import random

import pytest
from rdkit import Chem

from ouroboros.data.rgroups import ABBREVIATIONS, expand_aliases, substitute
from ouroboros.labels import labelled_smiles


def canon(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


@pytest.mark.parametrize('alias', ABBREVIATIONS)
def test_round_trip(alias):
    original = Chem.MolFromSmiles('NCC' + ABBREVIATIONS[alias])
    collapsed = substitute(original, random.Random(0), label=alias)
    assert any(a.GetAtomicNum() == 0 for a in collapsed.GetAtoms())
    expanded = expand_aliases(labelled_smiles(collapsed))
    assert Chem.MolToSmiles(expanded) == Chem.MolToSmiles(original)


@pytest.mark.parametrize('source,expected', [
    ('CC[OMe]', 'CCOC'), ('N[Boc]', 'NC(=O)OC(C)(C)C'),
    ('CC[Ph]', 'CCc1ccccc1'), ('[Me]O[Me]', 'COC'),
    ('N[C@@H](C)CC[OMe]', 'N[C@@H](C)CCOC')])
def test_attachment_and_remote_stereo(source, expected):
    assert Chem.MolToSmiles(expand_aliases(source)) == canon(expected)


@pytest.mark.parametrize(
    'source', ['CC[R1]', 'CC*', 'CC[PG]', 'CC[Ar]', 'C[Me]C', 'C=[Me]', '[Ph]']
)
def test_reject_undefined_or_unsafe(source):
    with pytest.raises(ValueError):
        expand_aliases(source)


def test_export_opt_in(tmp_path):
    from scripts.image_to_forcefield import main
    path = tmp_path / 'p.csv'
    path.write_text('id,pred\n1,CC[OMe]\n2,CC[R]\n')
    out = tmp_path / 'ff.extxyz'
    result = main(['--predictions', str(path), '--out', str(out), '--expand-aliases',
                   '--mace-model', 'none', '--n-conf', '1'])
    rows = [json.loads(s) for s in out.with_suffix('.manifest.jsonl').read_text().splitlines()]
    assert result['frames_written'] == 1
    assert rows[0]['recognized_smiles'] != rows[0]['expanded_smiles']
    assert rows[0]['smiles'] == canon('CCOC')
    assert rows[1]['status'].startswith('unsupported_alias')
