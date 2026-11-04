"""Deterministic connectivity groups for exported force-field frames, not scaffold splits."""
import hashlib
import math
from ouroboros.eval.standardize import standardize, flat_smiles

NAMES = ('train', 'val', 'test')


def connectivity_key(smiles):
    std = standardize(smiles)
    if not std.valid or any(a.GetAtomicNum() == 0 for a in std.mol.GetAtoms()):
        raise ValueError('missing or invalid molecular identity')
    return flat_smiles(std)


def assign_split(smiles, fractions=(.8, .1, .1), seed=0):
    if (len(fractions) != 3 or any(not math.isfinite(x) or x < 0 for x in fractions)
            or not math.isclose(sum(fractions), 1, abs_tol=1e-9)):
        raise ValueError('three nonnegative finite fractions must sum to one')
    key = connectivity_key(smiles)
    digest = hashlib.sha256(f'{seed}\0{key}'.encode()).digest()
    value = int.from_bytes(digest[:8], 'big') / 2**64
    return NAMES[0] if value < fractions[0] else NAMES[1] if value < sum(fractions[:2]) else NAMES[2]


def _key(source_image, source_id, conf_id):
    return str(source_image), str(source_id), int(conf_id)


def split_dataset(frames, records, fractions=(.8, .1, .1), seed=0):
    """Join frames to authoritative manifest identities and keep their associations.

    Every written conformer must occur exactly once. Ambiguous repeated source IDs
    within the same image are rejected, rather than silently joined. Skipped records
    remain in a separate manifest. Frame indices are zero-based in each output file.
    Input objects are not modified.
    """
    output = {name: {'frames': [], 'manifest': []} for name in (*NAMES, 'skipped')}
    expected, joined = {}, set()
    for row in records:
        if not row.get('n_frames', 0):
            output['skipped']['manifest'].append(dict(row))
            continue
        name = assign_split(row.get('smiles'), fractions, seed)
        rec = dict(row, split=name, connectivity_key=connectivity_key(row.get('smiles')),
                   input_frame_indices=[], output_frame_indices=[])
        written = [c for c in row.get('conformers', []) if c.get('written', True)]
        if len(written) != row['n_frames']:
            raise ValueError('manifest n_frames disagrees with written conformers')
        for conf in written:
            key = _key(row['source_image'], row['id'], conf['conf_id'])
            if key in expected:
                raise ValueError(f'ambiguous manifest association: {key}')
            expected[key] = (name, rec)
        output[name]['manifest'].append(rec)
    for index, atoms in enumerate(frames):
        try:
            key = _key(atoms.info['source_image'], atoms.info['source_id'], atoms.info['conf_id'])
        except KeyError as exc:
            raise ValueError('frame lacks manifest association') from exc
        if key not in expected or key in joined:
            raise ValueError(f'unmatched or duplicate frame: {key}')
        joined.add(key)
        name, rec = expected[key]
        rec['input_frame_indices'].append(index)
        rec['output_frame_indices'].append(len(output[name]['frames']))
        frame = atoms.copy()
        frame.info.update(split=name, connectivity_key=rec['connectivity_key'])
        output[name]['frames'].append(frame)
    if joined != set(expected):
        raise ValueError('manifest references missing frames')
    return output
