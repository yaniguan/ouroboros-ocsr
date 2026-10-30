"""CPU smoke coverage and deterministic profiling summary checks."""
import json

import pytest
import torch
import yaml

from ouroboros.data import build
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.train.trainer import Trainer
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


profile_infer = _load("profile_infer")
profile_train = _load("profile_train")


def test_phase_summary():
    result = profile_train.summarise_phase_times({'forward': [0.001, 0.003, 0.002], 'empty': []})
    assert result['forward'] == pytest.approx(dict(mean_ms=2, median_ms=2, p90_ms=2.8))
    assert result['empty'] is None


def test_gpu_csv():
    text = ('timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]\n'
            '2026/10/30 12:00:00, 20, 10, 123, 50\n'
            'timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]\n'
            '2026/10/30 12:00:01, 80 %, 30, 456, [N/A]\n'
            'partial\n2026/10/30, [N/A], [Not Supported], 1, 1\n')
    assert profile_train.parse_nvidia_smi_csv(text) == {
        'gpu': {'mean': 50, 'max': 80}, 'memory': {'mean': 20, 'max': 30}}
    assert profile_train.parse_nvidia_smi_csv('') is None
    assert profile_train.parse_nvidia_smi_csv('timestamp, [N/A], [N/A]\n') is None


def test_sampler_missing(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(profile_train.subprocess, 'Popen', missing)
    assert profile_train.start_nvidia_smi(tmp_path / 'gpu.csv') is None
    profile_train.stop_nvidia_smi(None)


def test_cpu_profiles(tiny_dataset, tmp_path, monkeypatch):
    # Same tiny baseline architecture and real shard fixture used by test_train.
    root, _ = tiny_dataset
    vocab = tmp_path / 'vocab.json'
    rows = build.read_manifest(root / 'manifests/train.tsv')
    SmilesTokenizer.build(row['smiles'] for row in rows).save(vocab)
    cfg = {
        'seed': 3,
        'data': {'synthetic_root': str(root / 'shards'), 'vocab': str(vocab),
                 'image_size': 64, 'num_workers': 0, 'real_roots': [], 'real_fraction': 0.0},
        'model': {
            'encoder': {'type': 'baseline', 'widths': [8, 16, 16, 32],
                        'blocks': [1, 1, 1, 1], 'mixer_layers': 1,
                        'mixer_heads': 2, 'mixer_ff': 64},
            'decoder': {'d_model': 32, 'n_layers': 1, 'n_heads': 2,
                        'd_ff': 64, 'max_len': 96}},
        'train': {'device': 'cpu', 'batch_size': 2, 'steps': 10, 'lr': 1e-3,
                  'warmup': 1, 'amp': 'none', 'grad_clip': 1.0, 'rotation_aug': True}}
    config = tmp_path / 'finetune.yaml'
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('OUROBOROS_NUM_WORKERS', '0')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out = tmp_path / 'profiling'
        train = profile_train.main(['--config', str(config), '--out-dir', str(out),
                                    '--name', 'tiny', '--steps', '2', '--warmup', '1',
                                    '--trace-steps', '1'])
        assert train['device'] == 'cpu'
        assert train['peak_memory'] is None and train['gpu_util'] is None
        assert train['img_per_s'] > 0 and train['tokens_per_s'] > 0
        assert set(train['phases']) == {'data_loading', 'host_to_device', 'forward', 'backward', 'optimizer'}
        assert (out / 'trace_tiny.json').is_file()
        assert (out / 'keyavg_tiny.txt').read_text()
        assert json.loads((out / 'train_tiny.json').read_text()) == train
        assert not (tmp_path / 'ckpt').exists()
        trainer = Trainer(cfg, tmp_path / 'checkpoint-source')
        trainer.save()
        ckpt = str(trainer.ckpt_path)
        # Repeat checkpoints exercises the ensemble rescoring path too.
        infer = profile_infer.main(['--ckpt', ckpt, '--ckpt', ckpt,
                                    '--shards', str(root / 'shards'), '--split', 'test',
                                    '--beam', '2', '--batch-size', '1', '--max-samples', '1',
                                    '--name', 'tiny', '--out-dir', str(out)])
        assert set(infer['variants']) == {'beam1', 'beam2'}
        for variant in infer['variants'].values():
            assert variant['samples'] == 1 and variant['batches'] == 1
            assert variant['images_per_s'] > 0
            assert variant['ms_per_image']['encoder'] > 0
            assert variant['peak_memory'] is None
        assert json.loads((out / 'infer_tiny.json').read_text()) == infer
    finally:
        torch.set_num_threads(previous_threads)
