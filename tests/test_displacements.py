import json
import numpy as np
import pytest
from rdkit import Chem
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ouroboros.geometry.conformers import embed_conformers
from ouroboros.geometry.displacements import displaced_conformers, geometry_rejection


class HarmonicToy(Calculator):
    implemented_properties = ['energy', 'forces']
    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = self.atoms.positions
        self.results = dict(energy=float(.5*np.sum(x*x)), forces=-x.copy())


def test_seed_bounds_identity_and_parent_ids():
    mol = embed_conformers('CCO', n_conf=2)
    a, records = displaced_conformers(mol, 'CCO', 3, amplitude=.05, seed=8)
    b, again = displaced_conformers(mol, 'CCO', 3, amplitude=.05, seed=8)
    assert records == again and len(records) == 6
    assert len({(r['parent_conf_id'], r['sample_id']) for r in records}) == 6
    for r in records:
        assert r['conf_id'] is not None
        positions = a.GetConformer(r['conf_id']).GetPositions()
        np.testing.assert_array_equal(positions, b.GetConformer(r['conf_id']).GetPositions())
        delta = positions - mol.GetConformer(r['parent_conf_id']).GetPositions()
        assert np.linalg.norm(delta, axis=1).max() <= .05 + 1e-12
    zero, rows = displaced_conformers(mol, 'CCO', 1, amplitude=0)
    for r in rows:
        np.testing.assert_array_equal(zero.GetConformer(r['conf_id']).GetPositions(),
                                      mol.GetConformer(r['parent_conf_id']).GetPositions())
    assert mol.GetNumConformers() == 2


def test_clash_and_stereo_rejection():
    smiles = 'N[C@H](C)O'
    mol = embed_conformers(smiles, n_conf=1)
    bad = Chem.Mol(mol)
    conf = bad.GetConformer()
    for i, p in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, [-float(p[0]), float(p[1]), float(p[2])])
    assert geometry_rejection(bad, smiles, 0, .1) == 'stereo_mismatch'
    out, records = displaced_conformers(bad, smiles, 1, amplitude=0)
    assert records[0]['rejection_reason'] == 'stereo_mismatch'
    assert out.GetNumConformers() == 1
    out, records = displaced_conformers(mol, smiles, 2, min_distance=100)
    assert all(r['rejection_reason'] == 'minimum_distance' and r['conf_id'] is None for r in records)
    assert out.GetNumConformers() == 1


@pytest.mark.parametrize('kwargs', [dict(n_samples=-1), dict(n_samples=1, amplitude=float('nan')),
                                     dict(n_samples=1, min_distance=-1)])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        displaced_conformers(embed_conformers('C', n_conf=1), 'C', **kwargs)


def test_analytic_energy_force_export(tmp_path, monkeypatch):
    from scripts import image_to_forcefield as ff
    monkeypatch.setattr(ff, 'load_calculator', lambda *args: (HarmonicToy(), 'analytic-toy'))
    path = tmp_path / 'p.csv'
    path.write_text('id,pred\nparent,CCO\n')
    out = tmp_path / 'ff.extxyz'
    summary = ff.main(['--predictions', str(path), '--out', str(out), '--n-conf', '1',
                       '--displaced-samples', '2', '--max-displacement', '.03', '--only-valid'])
    assert summary['frames_written'] == 3
    frames = read(out, index=':')
    assert {f.info['sample_id'] for f in frames} == {0, 1, 2}
    for frame in frames:
        assert frame.info['parent_conf_id'] == 0
        assert frame.arrays['mace_forces'].shape == frame.positions.shape
        np.testing.assert_allclose(frame.arrays['mace_forces'], -frame.positions, atol=1e-7)
        assert frame.info['mace_energy'] == pytest.approx(.5*np.sum(frame.positions**2), abs=1e-6)
    manifest = json.loads(out.with_suffix('.manifest.jsonl').read_text())
    assert len(manifest['displacement_samples']) == 2
    assert [c['sample_id'] for c in manifest['conformers']] == [0, 1, 2]
    assert manifest['sampling_parameters']['amplitude'] == .03


def test_relax_displacement_conflict(tmp_path):
    from scripts.image_to_forcefield import main
    with pytest.raises(SystemExit):
        main(['--predictions', 'unused.csv', '--out', str(tmp_path / 'x.extxyz'),
              '--displaced-samples', '1', '--relax'])
