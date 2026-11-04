"""Optional molecule checks before creating geometry labels."""
from dataclasses import dataclass
from rdkit import Chem


def calculator_elements(calc):
    """Read the elements a calculator says it supports.

    Read MACE's atomic numbers or ASE calculator attributes.
    With multiple models, keep only elements they all support.
    """
    if calc is None:
        return None
    declared = getattr(calc, 'supported_elements', None)
    if declared is not None:
        return set(declared)
    table = getattr(calc, 'z_table', None)
    numbers = getattr(table, 'zs', None)
    if numbers is None:
        numbers = getattr(calc, 'atomic_numbers', None)
    pt = Chem.GetPeriodicTable()
    if numbers is not None:
        return {pt.GetElementSymbol(int(z)) for z in numbers}
    models = getattr(calc, 'models', ())
    supported = []
    for model in models:
        zs = getattr(model, 'atomic_numbers', None)
        if zs is None:
            return None
        supported.append({pt.GetElementSymbol(int(z)) for z in zs})
    if not supported:
        return None
    return set.intersection(*supported)


@dataclass(frozen=True)
class EligibilityPolicy:
    supported_elements: frozenset[str] | None = None
    require_assigned_stereo: bool = False
    allow_charge: bool = False
    allow_radicals: bool = False

    def reasons(self, mol, calc=None):
        """Check the standardized molecule before embedding or calculation."""
        if mol is None or not mol.GetNumAtoms():
            return ['invalid_molecule']
        reasons = []
        if self.supported_elements is not None:
            supported = set(self.supported_elements)
        else:
            supported = calculator_elements(calc)
        if supported is None and calc is not None:
            reasons.append('unknown_calculator_elements')
        elif supported is not None:
            required = {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()}
            unsupported = sorted(required - supported)
            if unsupported:
                reasons.append('unsupported_elements:' + ','.join(unsupported))
        if not self.allow_charge and sum(a.GetFormalCharge() for a in mol.GetAtoms()):
            reasons.append('residual_charge')
        if not self.allow_radicals and any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            reasons.append('radicals')
        if self.require_assigned_stereo and any(
                el.specified == Chem.StereoSpecified.Unspecified
                for el in Chem.FindPotentialStereo(Chem.Mol(mol))):
            reasons.append('unassigned_stereo')
        return reasons
