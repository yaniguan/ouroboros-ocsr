import json
import pytest
from ase import Atoms
from ase.io import read, write
from ouroboros.geometry.splits import assign_split, split_dataset


def dataset():
    frames, rows = [], []
    for i, smiles in enumerate(['CCO', 'OCC', 'N[C@H](C)O', 'N[C@@H](C)O', 'CCC']):
        rows.append(dict(id=str(i), source_image=f'subset/{i}', smiles=smiles, n_frames=2,
                         conformers=[dict(conf_id=j, written=True) for j in range(2)]))
        for j in range(2):
            atoms = Atoms('H', positions=[[i, j, 0]])
            atoms.info.update(source_id=str(i), source_image=f'subset/{i}', conf_id=j,
                              mace_energy=float(i+j))
            frames.append(atoms)
    return frames, rows


def test_equivalence_stereo_and_order():
    frames, rows = dataset()
    result = split_dataset(frames, rows)
    reverse = split_dataset(list(reversed(frames)), list(reversed(rows)))
    assert assign_split('CCO') == assign_split('OCC')
    assert assign_split('N[C@H](C)O') == assign_split('N[C@@H](C)O')
    for split in ('train', 'val', 'test'):
        assert {r['id'] for r in result[split]['manifest']} == {r['id'] for r in reverse[split]['manifest']}
        assert len(result[split]['frames']) == 2*len(result[split]['manifest'])
    assert sum(len(x['frames']) for x in result.values()) == len(frames)
    assert 'split' not in frames[0].info


@pytest.mark.parametrize('smiles', [None, '', 'bad', '*C'])
def test_missing_identity(smiles):
    with pytest.raises(ValueError, match='identity'):
        assign_split(smiles)


def test_missing_and_ambiguous_associations():
    frames, rows = dataset()
    with pytest.raises(ValueError, match='missing frames'):
        split_dataset(frames[:-1], rows)
    with pytest.raises(ValueError, match='ambiguous'):
        split_dataset(frames, rows + rows[:1])
    with pytest.raises(ValueError, match='fractions'):
        assign_split('CC', [1, 1, 1])


def test_export_associations(tmp_path):
    from scripts.split_forcefield import main
    frames, rows = dataset()
    write(tmp_path / 'in.extxyz', frames, format='extxyz')
    (tmp_path / 'in.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    main(['--frames', str(tmp_path / 'in.extxyz'), '--manifest', str(tmp_path / 'in.jsonl'),
          '--out-dir', str(tmp_path / 'out')])
    for name in ('train', 'val', 'test'):
        path = tmp_path / 'out' / f'{name}.extxyz'
        data = read(path, index=':') if path.stat().st_size else []
        manifest = [json.loads(s) for s in path.with_suffix('.manifest.jsonl').read_text().splitlines()]
        for row in manifest:
            for old, new in zip(row['input_frame_indices'], row['output_frame_indices'], strict=True):
                assert str(data[new].info['source_id']) == row['id']
                assert data[new].info['mace_energy'] == frames[old].info['mace_energy']
