import random

import numpy as np
import pytest
import torch
from PIL import Image
from rdkit import Chem

from ouroboros.data.augment import scan_augment
from ouroboros.data.build import render_shards
from ouroboros.data.loader import ShardDataset
from ouroboros.data.render import RenderStyle, render
from ouroboros.data.rgroups import ABBREVIATIONS, substitute
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.eval.metrics import score_pair
from ouroboros.eval.smiles import canonicalize, exact_match, inchikey
from ouroboros.labels import LABELS, labelled_smiles, parse_labelled
from ouroboros.model import build_model, initialize_finetune


@pytest.mark.parametrize('label', LABELS)
def test_label_roundtrip_and_scoring(label):
    s = f'CC[{label}]'
    canonical = canonicalize(s)
    assert canonical and f'[{label}]' in canonical
    assert canonicalize(canonical) == canonical
    assert exact_match(s, f'[{label}]CC')
    assert not exact_match(s, 'CC*')
    assert inchikey(s) is None
    assert not score_pair(s, s).inchikey  # unavailable keys must never compare equal


def test_labels_remain_distinct_and_real_atoms_are_preserved():
    assert not exact_match('C[R1]', 'C[R2]')
    assert not exact_match('C[R]', 'C[X]')
    assert canonicalize('CR1') == canonicalize('C[R1]')
    assert canonicalize('COC') == 'COC'
    assert parse_labelled('[Ru]').GetAtomWithIdx(0).GetAtomicNum() == 44
    assert parse_labelled('[Xe]').GetAtomWithIdx(0).GetAtomicNum() == 54
    assert canonicalize('[C2H2O4]') is None
    assert canonicalize('[(].[)]') is None
    assert exact_match('CC*', '*CC')


def test_vocab_extension_preserves_ids_and_persists(tmp_path):
    tok = SmilesTokenizer.build(['CCO'])
    prefix = tok.itos.copy()
    old_ids = tok.encode('CCO')
    tok.add_label_tokens()
    assert tok.itos[:len(prefix)] == prefix
    assert tok.encode('CCO') == old_ids
    assert tok.unk_id not in tok.encode('C[R4]')
    assert tok.decode(tok.encode('CR4')) == 'C[R4]'
    tok.save(tmp_path / 'vocab.json')
    assert SmilesTokenizer.load(tmp_path / 'vocab.json').itos == tok.itos
    assert tok.add_label_tokens() == []


@pytest.mark.parametrize('label,fragment', ABBREVIATIONS.items())
def test_abbreviation_substitution_and_drawing(label, fragment):
    mol = Chem.MolFromSmiles('NCC' + fragment)
    out = substitute(mol, random.Random(3), label)
    text = labelled_smiles(out)
    assert f'[{label}]' in text
    picture = render(out, RenderStyle(), size=128)
    assert canonicalize(picture.label, standardize=False) == canonicalize(text, standardize=False)
    drawn = Chem.MolFromMolBlock(picture.molblock)
    assert any(a.GetAtomicNum() == 0 for a in drawn.GetAtoms())
    assert np.asarray(picture.image).min() < 128


def test_generic_and_chiral_core():
    mol = Chem.MolFromSmiles('CCCCCC[C@H](O)CC')
    out = substitute(mol, random.Random(1), 'R1')
    assert '[R1]' in labelled_smiles(out)
    assert '@' in labelled_smiles(out)
    assert labelled_smiles(substitute(Chem.MolFromSmiles('c1ccccc1'), random.Random(1))) == 'c1ccccc1'


def test_real_style_deterministic_and_morphology():
    a = np.full((64, 64), 255, dtype='uint8')
    a[20:44, 30:34] = 0
    image = Image.fromarray(a)
    x = scan_augment(image, np.random.default_rng(4), morphology=1, threshold=180)
    y = scan_augment(image, np.random.default_rng(4), morphology=1, threshold=180)
    assert np.array_equal(x, y)
    assert set(np.unique(x)) <= {0, 255}
    for sign in (-1, 1):
        z = scan_augment(image, np.random.default_rng(1), affine=0, scan_noise=0, morphology=sign)
        assert (255 - np.asarray(z)).sum() * sign > (255 - a).sum() * sign
    with pytest.raises(ValueError):
        scan_augment(image, np.random.default_rng(), affine=1)


def test_augmented_builder_loader_and_identity(tmp_path):
    rows = [dict(idx=i, smiles='CCCOC', key14=str(i), stereo=0) for i in range(8)]
    stats = render_shards(rows, 'train', tmp_path, size=96, workers=1,
                          rgroup_probability=1, real_style=True)
    assert stats['samples_written'] == 8 and not stats['drops']
    ds = ShardDataset(tmp_path, 'train')
    assert all('[' in ds.meta(i)['smiles'] for i in range(len(ds)))
    assert ds.meta(0)['style']['real_style']
    assert torch.equal(ds[0]['image'], ds[0]['image'])
    with pytest.raises(ValueError, match='identity changed'):
        render_shards(rows, 'train', tmp_path, size=96, workers=1)


def test_finetune_expands_only_embedding_rows():
    cfg = {'data': {'image_size': 64}, 'model': {
        'encoder': {'type': 'baseline', 'widths': [8, 8, 8, 8], 'blocks': [1]*4,
                    'mixer_layers': 1, 'mixer_heads': 2, 'mixer_ff': 32},
        'decoder': {'d_model': 16, 'n_layers': 1, 'n_heads': 2, 'd_ff': 32}}}
    tok = SmilesTokenizer.build(['CCO'])
    old = build_model(cfg, tok)
    state = {'vocab': tok.itos.copy(), 'model': old.state_dict()}
    tok.add_label_tokens()
    new = build_model(cfg, tok)
    fresh = new.decoder.tok.weight.detach().clone()
    initialize_finetune(new, state)
    n = len(state['vocab'])
    assert torch.equal(new.decoder.tok.weight[:n], old.decoder.tok.weight)
    assert torch.equal(new.decoder.tok.weight[n:], fresh[n:])
    assert new.decoder.head.weight is new.decoder.tok.weight
    for key, value in old.state_dict().items():
        if key not in ('decoder.tok.weight', 'decoder.head.weight'):
            assert torch.equal(new.state_dict()[key], value)
    state['vocab'] = list(reversed(state['vocab']))
    with pytest.raises(ValueError, match='prefix'):
        initialize_finetune(new, state)


def test_labelled_leakage_identity():
    from ouroboros.eval.standardize import leakage_key, standardize
    assert leakage_key(standardize('CC[R1]')) == leakage_key(standardize('[R1]CC'))
    assert leakage_key(standardize('CC[R1]')) != leakage_key(standardize('CC[R2]'))
    assert leakage_key(standardize('CC[R1]')).startswith('L')
    assert leakage_key(standardize('CCO')) == inchikey('CCO')


def test_builder_counts_overlength_labels(tmp_path):
    rows = [dict(idx=0, smiles='CCCCCC', key14='x', stereo=0)]
    stats = render_shards(rows, 'train', tmp_path, size=64, workers=1, max_tokens=3)
    assert stats['samples_written'] == 0
    assert stats['drops'] == {'label_too_long': 1}


def test_bare_and_cx_aliases_preserve_labels():
    assert exact_match('c1ccccc1R1', 'c1ccccc1[R1]')
    assert exact_match('CCMe', 'CC[Me]')
    assert exact_match('*CC |$R1;;$|', '[R1]CC')
    assert canonicalize('*CC |$unsupported;;$|') is None
