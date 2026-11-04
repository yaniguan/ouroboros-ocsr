from types import SimpleNamespace
import pytest
from rdkit import Chem
from ouroboros.geometry.eligibility import EligibilityPolicy, calculator_elements


def test_charge_radicals_and_neutral_nitro():
    policy = EligibilityPolicy()
    assert policy.reasons(Chem.MolFromSmiles('C[N+](C)(C)C')) == ['residual_charge']
    assert policy.reasons(Chem.MolFromSmiles('[CH3]')) == ['radicals']
    assert policy.reasons(Chem.MolFromSmiles('C[N+](=O)[O-]')) == []


@pytest.mark.parametrize('smiles', ['CC(O)N', 'CC=CC'])
def test_optional_unassigned_stereo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert EligibilityPolicy().reasons(mol) == []
    assert EligibilityPolicy(require_assigned_stereo=True).reasons(mol) == ['unassigned_stereo']


@pytest.mark.parametrize('smiles', ['C[C@H](O)N', 'C/C=C/C'])
def test_assigned_passes(smiles):
    assert EligibilityPolicy(require_assigned_stereo=True).reasons(Chem.MolFromSmiles(smiles)) == []


def test_capabilities_and_implicit_hydrogen():
    calc = SimpleNamespace(z_table=SimpleNamespace(zs=[6, 8]))
    assert calculator_elements(calc) == {'C', 'O'}
    assert EligibilityPolicy().reasons(Chem.MolFromSmiles('CO'), calc) == ['unsupported_elements:H']
    assert EligibilityPolicy().reasons(Chem.MolFromSmiles('CO'), object()) == ['unknown_calculator_elements']
    ensemble = SimpleNamespace(models=[SimpleNamespace(atomic_numbers=[1, 6, 8]),
                                       SimpleNamespace(atomic_numbers=[1, 6])])
    assert calculator_elements(ensemble) == {'H', 'C'}


def test_unsupported_skips_embedding_and_calculator(monkeypatch):
    from scripts import image_to_forcefield as ff
    from ase.calculators.calculator import Calculator
    class Toy(Calculator):
        supported_elements = {'H', 'C', 'O'}
        def calculate(self, *args, **kwargs):
            raise AssertionError('ineligible calculator call')
    def forbidden(*args, **kwargs):
        raise AssertionError('ineligible embedding')
    monkeypatch.setattr(ff, 'embed', forbidden)
    args = SimpleNamespace(relax=False, geometry_eligibility=True)
    frames, rec = ff.process(dict(id='1', source_image='x', smiles='CCl'), args, Toy(), 'toy', set())
    assert frames == [] and rec['status'] == 'geometry_ineligible'
    assert rec['eligibility_reasons'] == ['unsupported_elements:Cl']
