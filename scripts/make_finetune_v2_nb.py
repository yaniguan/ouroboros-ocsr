"""Generate colab/finetune_v2.ipynb (real PMC fine-tune, real-val selection, real test eval).

    python scripts/make_finetune_v2_nb.py
"""

from pathlib import Path

import nbformat as nbf

BRANCH = "feat/finetune-v2-oct30"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(f"""
# Fine-tune v2: Ouroboros A/C on real PMC, select by real validation

Branch `{BRANCH}`. **Runtime → Change runtime type → A100 GPU, then Runtime → Run all.** No manual steps.

Colab: https://colab.research.google.com/github/yaniguan/ouroboros-ocsr/blob/feat/finetune-v2-oct30/colab/finetune_v2.ipynb?authuser=1

Changes vs `real_pmc_oct30`: that run saw only ~1.7 epochs of the 17,220 PMC rows
(3000 steps × 16 × 0.6). V2 uses 20,000 steps, batch 32, 70% real, LR 3e-5,
500 warmup steps and label smoothing 0.05, with all kept PMC validation rows (963,
clipped to availability). USPTO failures were dominated by long molecules (>60 tokens);
ACS failures by R-groups `*`, which PMC labels never contain. A larger 50k synthetic
ZINC250k pool with R-group substitution p=0.5 supplies those missing labels.

What it does:
1. mounts Drive, clones the branch, installs the Python 3.11 training stack, fetches released A/C weights;
2. fetches the held-out real test set `real/subset100` (Keylab/OCSR-Benchmarks USPTO/UOB/ACS,
   100 each, seed 0 → 298 scoreable images; same command as Oct 29) and dumps its molecule keys;
3. downloads the public HF dataset `yaniguan/ocsr-dataset` (VERDICT-PMC, pinned to the resolved
   commit), removes every row whose InChIKey connectivity block matches a test molecule, drops
   overlong labels, makes the PMC validation split molecule-disjoint from PMC train, ingests both;
4. builds 50k real-style synthetic training images (test molecules excluded);
5. **leakage gate**: `check_leakage.py --fail-on-connectivity` over test vs PMC train / PMC val /
   synthetic train — the notebook stops on any overlap;
6. fine-tunes each arm with `real_fraction=0.7` real rows per batch, no rotation augmentation,
   bf16, validating exact match on all 963 kept PMC val images every 1000 steps (the released weights
   are scored at step 0, so selection can fall back to them); early stopping, best checkpoint kept;
7. evaluates released and selected checkpoints greedily on the 298 real test images,
   then selected checkpoints with beam 5 (default validity rerank), and an A+C beam-5
   ensemble; writes predictions, JSON/Markdown reports, and `summary.json`.
8. profiles training phases, GPU utilization and greedy/beam inference, saving traces and
   throughput/memory summaries under `profiling/` on Drive.

Everything goes to `/content/drive/MyDrive/ouroboros-ocsr_runs/finetune_v2_oct30/`. Re-running after a
disconnect restores data from Drive and resumes training from `ckpt/last.pt` (exact resume).
Change `RUN_NAME` when changing data or hyperparameters. Target: beat released A = 18.12% (54/298).

**Runtime estimate (A100):** previous measured A throughput was 277 img/s at batch 16.
20,000 steps × 32 = 640k images: allow ~40–50 min per arm, plus validation
(21 checks × 963 images) ~10 min. Building 50k synthetic images takes ~25–40 min
(cached on Drive after the first run); install + data ~15 min; evaluations ~10 min;
profiling adds ~5 min.
Total ~2.5–3 h for A+C, or ~1.5–2 h for A only: set `ARMS=['A']` to save time.
Actual throughput is logged in `log.jsonl` (`img_per_s`); early stopping can shorten training.
""")

code(f"""
import os, sys, json, shutil, subprocess, time, hashlib, urllib.request
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
RUN_NAME = 'finetune_v2_oct30'
BRANCH = '{BRANCH}'
REPO = Path('/content/ouroboros-ocsr')
DRIVE_ROOT = Path('/content/drive/MyDrive/ouroboros-ocsr_runs') / RUN_NAME
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# ---- experiment knobs -------------------------------------------------------------------------
ARMS = ['A', 'C']                        # A first; C runs after A finishes
SWEEP_LR = {{'A': [3e-5], 'C': [3e-5]}}  # e.g. {{'A': [1e-5, 2e-5, 3e-5]}}; best by REAL val exact
STEPS, BATCH, REAL_FRACTION = 20000, 32, 0.7
WARMUP, LABEL_SMOOTHING = 500, 0.05
N_VALIDATION, VAL_EVERY, PATIENCE = 963, 1000, 6
N_SYNTH = 50000
PMC_REPO = 'yaniguan/ocsr-dataset'
EXPECTED_TEST_MANIFEST_SHA256 = '957f0042825dab214aad2463de2974d7b4f23a1fcbf5d52ab78be2526cc3fc61'
RELEASED_TARGET = {{'A': (54, 298), 'C': (50, 298)}}  # Oct 29 CPU re-eval of the released models

DATA = Path('/content/data')
PY = '/content/venv-real-pmc/bin/python'
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  MPLBACKEND='Agg', MPLCONFIGDIR='/tmp/mpl-real-pmc', PYTHONUNBUFFERED='1')

def shell(args, cwd=None, check=True, log=None, tail=80):
    \"\"\"Run a command, streaming its merged stdout+stderr live. The full output goes to ``log``
    (default ``DRIVE_ROOT/logs/<cmd>-<time>.log``, kept only on failure; an explicit ``log`` is
    appended to and always kept);
    on a nonzero exit the last ``tail`` lines are printed again and the log path is shown.\"\"\"
    import collections
    args = [str(x) for x in args]
    keep = log is not None
    if log is None:
        stem = Path(args[2] if len(args) > 2 and args[1] == '-m' else args[1] if len(args) > 1
                    else args[0]).name.replace('.py', '')
        log = DRIVE_ROOT / 'logs' / f"{{stem.lstrip('-')}}-{{time.strftime('%Y%m%d-%H%M%S')}}-{{time.time_ns() % 10**9:09d}}.log"
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    last = collections.deque(maxlen=tail)
    with open(log, 'a' if keep else 'w') as stream:  # explicit logs accumulate across sessions
        stream.write(time.strftime('[%Y-%m-%dT%H:%M:%S] $ ') + ' '.join(args) + '\\n')
        proc = subprocess.Popen(args, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        for line in proc.stdout:
            print(line, end='')
            stream.write(line)
            last.append(line)
        proc.wait()
        stream.write(f'\\n[exit status {{proc.returncode}}]\\n')
    if proc.returncode != 0:
        print(f'\\n===== FAILED (exit {{proc.returncode}}): last {{len(last)}} lines; full log: {{log}} =====')
        print(''.join(last), end='')
        print('=====')
        if check:
            raise subprocess.CalledProcessError(proc.returncode, args)
    elif not keep:
        log.unlink(missing_ok=True)
    return subprocess.CompletedProcess(args, proc.returncode)

def run(*args, check=True, log=None):
    return shell([PY, *args], cwd=REPO, check=check, log=log)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def restore_or_build(local, cached, build):
    \"\"\"Copy a finished artifact back from Drive (byte-identical shards keep exact resume
    valid), otherwise build it locally and cache it on Drive.\"\"\"
    local, cached = Path(local), Path(cached)
    if (cached / 'DONE').exists():
        if not (local / 'DONE').exists():
            shutil.rmtree(local, ignore_errors=True)
            shutil.copytree(cached, local)
        print('restored', local, 'from', cached)
        return
    shutil.rmtree(local, ignore_errors=True)
    build()
    (local / 'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S') + '\\n')
    shutil.rmtree(cached, ignore_errors=True)
    shutil.copytree(local, cached)

T_START = time.perf_counter()
TIMINGS = {{}}
shell(['nvidia-smi'])
if not REPO.exists():
    shell(['git', 'clone', 'https://github.com/yaniguan/ouroboros-ocsr.git', REPO])
shell(['git', 'fetch', 'origin', BRANCH], cwd=REPO)
shell(['git', 'checkout', '-B', BRANCH, f'origin/{{BRANCH}}'], cwd=REPO)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
(DRIVE_ROOT / 'source_revision.txt').write_text(revision + '\\n')
print('Source:', revision)
""")

md("""
## Install (Python 3.11 venv, pinned requirements) and fetch released weights
""")

code("""
t0 = time.perf_counter()
if not shutil.which('gfortran'):
    shell(['apt-get', '-qq', 'update'])
    shell(['apt-get', '-qq', 'install', '-y', 'gfortran'])
shell([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
uv = shutil.which('uv')
assert uv, 'uv installation did not put uv on PATH'
if not Path(PY).exists():
    shell([uv, 'venv', '--python', '3.11', '--python-preference', 'only-managed',
           str(Path(PY).parent.parent)])
shell([uv, 'pip', 'install', '--python', PY, '-r', REPO / 'requirements.txt'])
shell([uv, 'pip', 'install', '--python', PY, '--no-deps', '-e', REPO])
shell([uv, 'pip', 'install', '--python', PY, 'pyarrow'])
run('-c', "import torch, rdkit; print(torch.__version__, rdkit.__version__); "
          "assert torch.cuda.is_available(), 'Select a GPU runtime'; "
          "print(torch.cuda.get_device_name(), 'bf16:', torch.cuda.is_bf16_supported())")
with (DRIVE_ROOT / 'environment.txt').open('w') as stream:
    subprocess.run([uv, 'pip', 'freeze', '--python', PY], stdout=stream, check=True)
os.environ['OUROBOROS_NUM_WORKERS'] = '0'   # tests only
run('-m', 'pytest', '-q', 'tests/test_real_finetune.py', 'tests/test_real.py', 'tests/test_mixture.py')
del os.environ['OUROBOROS_NUM_WORKERS']
for arm in ARMS:
    run('-m', 'ouroboros.weights', 'fetch', '--arm', arm)
TIMINGS['install_s'] = round(time.perf_counter() - t0, 1)
""")

md("""
## Held-out real test set (identical 298 as Oct 29) and its exclusion keys

Same command/seed as `colab/finetune_real_aug.ipynb` and the local Oct 29 evaluation. The manifest
SHA256 is compared with the Oct 29 value; a mismatch means HF `main` changed and the 18.12%
reference is not directly comparable (released vs fine-tuned below are still on the same images).
""")

code("""
t0 = time.perf_counter()
def build_test():
    run('scripts/fetch_real_subset.py', '--sets', 'USPTO', 'UOB', 'ACS', '--n', '100',
        '--seed', '0', '--out', 'real_data/subset100')
    run('scripts/ingest_real.py', '--manifest', 'real_data/subset100/manifest.csv',
        '--out', DATA / 'real/subset100')
    shutil.copy2(REPO / 'real_data/subset100/manifest.csv', DATA / 'real/subset100/manifest.csv')
restore_or_build(DATA / 'real/subset100', DRIVE_ROOT / 'cache/subset100', build_test)
TEST = DATA / 'real/subset100'
test_stats = json.loads((TEST / 'ingest_stats.json').read_text())
test_sha = sha256(TEST / 'manifest.csv')
assert test_stats['written'].get('test') == 298, test_stats
if test_sha != EXPECTED_TEST_MANIFEST_SHA256:
    print('WARNING: test manifest differs from Oct 29:', test_sha)
run('scripts/check_leakage.py', '--eval', f'subset100={TEST}:test',
    '--dump-eval-keys', '/content/eval_keys.txt')
shutil.copy2('/content/eval_keys.txt', DRIVE_ROOT / 'eval_keys.txt')
TIMINGS['test_set_s'] = round(time.perf_counter() - t0, 1)
""")

md("""
## Real training/validation data: VERDICT-PMC (HF `yaniguan/ocsr-dataset`, public, CC-BY-NC)

Downloads the parquet files at the resolved dataset commit (recorded), then
`scripts/prepare_pmc.py` (test-molecule exclusion by connectivity block, decoder-length filter,
molecule-disjoint val, seeded val order) and `scripts/ingest_real.py` (same standardization as
the scoring harness). Nothing here is ever committed.
""")

code("""
t0 = time.perf_counter()
def build_pmc():
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{PMC_REPO}'))
    rev = info['sha']
    tree = json.load(urllib.request.urlopen(
        f'https://huggingface.co/api/datasets/{PMC_REPO}/tree/{rev}/data'))
    files = sorted(x['path'] for x in tree if x['path'].endswith('.parquet'))
    hf = Path('/content/hf_pmc')
    hf.mkdir(parents=True, exist_ok=True)
    receipt = {'repo': PMC_REPO, 'revision': rev, 'files': {}}
    for f in files:
        dst = hf / Path(f).name
        if not dst.exists():
            urllib.request.urlretrieve(
                f'https://huggingface.co/datasets/{PMC_REPO}/resolve/{rev}/{f}', dst)
        receipt['files'][f] = sha256(dst)
    print(json.dumps(receipt, indent=1))
    run('scripts/prepare_pmc.py', '--parquet', hf, '--out', '/content/pmc_prep',
        '--exclude-keys', '/content/eval_keys.txt', '--max-len', '128')
    run('scripts/ingest_real.py', '--manifest', '/content/pmc_prep/manifest.csv',
        '--out', DATA / 'real/pmc')
    shutil.copy2('/content/pmc_prep/prepare_stats.json', DATA / 'real/pmc/prepare_stats.json')
    (DATA / 'real/pmc/source_receipt.json').write_text(json.dumps(receipt, indent=1))
restore_or_build(DATA / 'real/pmc', DRIVE_ROOT / 'cache/pmc', build_pmc)
PMC = DATA / 'real/pmc'
pmc_prepare = json.loads((PMC / 'prepare_stats.json').read_text())
pmc_ingest = json.loads((PMC / 'ingest_stats.json').read_text())
print(json.dumps({'prepare': pmc_prepare, 'ingest': pmc_ingest}, indent=1))
assert pmc_ingest['written'].get('train', 0) > 0 and pmc_ingest['written'].get('val', 0) > 0
TIMINGS['pmc_s'] = round(time.perf_counter() - t0, 1)
""")

md("""
## Real-style synthetic shards (mixed in at 30% to limit forgetting)

Same recipe as Oct 29 (ZINC250k, R-group/abbreviation substitution p=0.5, real-style
degradation, max 128 tokens), test molecules excluded before composition.
""")

code("""
t0 = time.perf_counter()
def build_synth():
    run('scripts/build_dataset.py', '--out', DATA / 'synth-aug', '--sources', 'zinc250k',
        '--cache', '/content/data_cache', '--train-size', str(N_SYNTH), '--val-size', '300',
        '--test-size', '200', '--workers', str(os.cpu_count() or 2), '--seed', '29',
        '--exclude-keys', '/content/eval_keys.txt', '--rgroup-probability', '0.5',
        '--real-style', '--max-tokens', '128')
restore_or_build(DATA / 'synth-aug', DRIVE_ROOT / 'cache/synth-aug', build_synth)
SYNTH = DATA / 'synth-aug/shards'
TIMINGS['synthetic_s'] = round(time.perf_counter() - t0, 1)
""")

md("""
## Leakage gate (stops the notebook on any overlap)

Full InChIKey **and** connectivity-block overlap (stereoisomers included) between the 298 test
images and every set used for training or checkpoint selection must be zero.
""")

code("""
leak = run('scripts/check_leakage.py', '--eval', f'subset100={TEST}:test',
           '--train', f'pmc_train={PMC}:train', f'pmc_val={PMC}:val', f'synth_train={SYNTH}:train',
           '--fail-on-connectivity', '--json', DRIVE_ROOT / 'leakage.json', check=False)
LEAKAGE = json.loads((DRIVE_ROOT / 'leakage.json').read_text())
if leak.returncode != 0 or any(r['overlap_inchikey'] or r['overlap_connectivity'] for r in LEAKAGE):
    raise SystemExit('LEAKAGE between the real test set and training/selection data: stopping')
print('leakage gate passed')
""")

md("""
## Configs: append-only vocabulary per arm (+ PMC train tokens), released architecture

`scripts/prepare_real_finetune.py` keeps every released token ID, appends alias tokens and the
tokens of synthetic + **PMC train** labels, copies `model` from the checkpoint, sets bf16 on
A100, and clips `validation.samples` to the available real val rows. A quick CUDA capacity probe
at the chosen batch falls back to batch 16 if it fails.
""")

code("""
RUNS = {}
for arm in ARMS:
    batch = BATCH
    for lr in SWEEP_LR[arm]:
        name = f'{arm}-lr{lr:g}'
        out = DRIVE_ROOT / 'runs' / name
        args = ['scripts/prepare_real_finetune.py', '--ckpt', f'checkpoints/ouroboros-{arm}.pt',
                '--config', 'configs/finetune-real-pmc.yaml', '--synthetic-root', SYNTH,
                '--real-root', PMC, '--exclude-keys', '/content/eval_keys.txt', '--out-dir', out]
        sets = [f'train.lr={lr}', f'train.steps={STEPS}', f'train.batch_size={batch}',
                f'train.warmup={WARMUP}', f'train.label_smoothing={LABEL_SMOOTHING}',
                f'data.real_fraction={REAL_FRACTION}', f'data.synthetic_max_samples={N_SYNTH}',
                f'validation.samples={N_VALIDATION}', f'validation.every={VAL_EVERY}',
                f'validation.patience={PATIENCE}', f'data.num_workers={min(8, os.cpu_count() or 2)}']
        if not (out / 'finetune.yaml').exists():
            run(*args, '--set', *sets)
            (out / 'capacity.json').unlink(missing_ok=True)
            probe = run('scripts/benchmark_cuda.py', '--config', out / 'finetune.yaml',
                        '--output', out / 'capacity.json', '--steps', '6', '--warmup', '2',
                        '--batch', str(batch), '--sequence-length', '127', '--precision', 'bf16',
                        check=False)
            if probe.returncode != 0:
                print(f'{name}: capacity probe failed at batch {batch}; using batch 16')
                batch = 16
                sets[2] = 'train.batch_size=16'
                run(*args, '--set', *sets)
        RUNS[name] = {'arm': arm, 'lr': lr, 'dir': out}
print({k: str(v['dir']) for k, v in RUNS.items()})
""")

md("""
## Training profiling probes (saved to Drive, skipped when already complete)

Each arm uses its first prepared configuration. Timings exclude warmup and profiler overhead;
Chrome traces, operator tables and GPU samples are saved alongside the JSON measurements.
""")

code("""
PROFILING = DRIVE_ROOT / 'profiling'
PROFILING.mkdir(parents=True, exist_ok=True)
from scripts.profile_train import start_nvidia_smi, stop_nvidia_smi, parse_nvidia_smi_csv
for arm in ARMS:
    r = next(r for r in RUNS.values() if r['arm'] == arm)
    if not (PROFILING / f'train_{arm}.json').exists():
        run('scripts/profile_train.py', '--config', r['dir'] / 'finetune.yaml',
            '--out-dir', PROFILING, '--name', arm)
""")

md("""
## Fine-tune (resumable) — real-val exact selects the checkpoint

Training state lives on Drive (`runs/<name>/train/ckpt/last.pt`, `best-step-*.pt`, `log.jsonl`).
Re-running this cell after a disconnect resumes exactly; finished runs return immediately.
""")

code("""
DONE_CHECK = r\"\"\"
import sys, torch, yaml
from pathlib import Path
d = Path(sys.argv[1])
last = d / 'train/ckpt/last.pt'
if not last.is_file():
    sys.exit(1)
st = torch.load(last, map_location='cpu', weights_only=False)
steps = yaml.safe_load((d / 'finetune.yaml').read_text())['train']['steps']
es = st.get('early_stop') or {}
sys.exit(0 if st['step'] >= steps or es.get('stopped') else 1)
\"\"\"

def train_done(run_dir):
    \"\"\"True if train/ckpt/last.pt reached train.steps or early-stopped (finished run).\"\"\"
    if not (run_dir / 'train/ckpt/last.pt').is_file():
        return False
    return subprocess.run([PY, '-c', DONE_CHECK, str(run_dir)], cwd=REPO).returncode == 0

for name, r in RUNS.items():
    if train_done(r['dir']):
        print(name, 'already finished (train/ckpt/last.pt complete): skipping')
        continue
    t0 = time.perf_counter()
    # stdout+stderr of the trainer are streamed here and saved to <run>/train_stderr.log on Drive
    # nvidia-smi samples append across resumed sessions.
    gpu_sampler = start_nvidia_smi(PROFILING / f'nvidia_smi_train_{name}.csv')
    try:
        run('-m', 'ouroboros.train', '--config', r['dir'] / 'finetune.yaml', '--out', r['dir'] / 'train',
            log=r['dir'] / 'train_stderr.log')
    finally:
        stop_nvidia_smi(gpu_sampler)
    wall = time.perf_counter() - t0
    timing_file = r['dir'] / 'train_wall_s.json'
    prev = json.loads(timing_file.read_text()) if timing_file.exists() else []
    timing_file.write_text(json.dumps(prev + [round(wall, 1)]))
    print(name, 'training wall seconds (this session):', round(wall, 1))
""")

md("""
## Select per arm by real validation, evaluate released vs fine-tuned on the 298 test images
""")

code(r'''
select_src = r"""
import json, sys, torch
from pathlib import Path
run = Path(sys.argv[1])
st = torch.load(run / 'train/ckpt/last.pt', map_location='cpu', weights_only=False)
es = st['early_stop']
best = run / 'train/ckpt' / f"best-step-{es['best_step']:08d}.pt"
b = torch.load(best, map_location='cpu', weights_only=False)
slim = {k: b[k] for k in ('config', 'vocab', 'model', 'step')}
torch.save(slim, run / 'selected.pt')
log = [json.loads(l) for l in (run / 'train/log.jsonl').read_text().splitlines()]
vals = [r for r in log if r.get('event') == 'validation']
out = {'best_step': es['best_step'], 'best_val_exact': es['best_score'],
       'stopped_early': es['stopped'], 'last_step': st['step'],
       'val_exact_step0': next((r['exact'] for r in vals if r['step'] == 0), None),
       'val_curve': [[r['step'], r['exact']] for r in vals], 'n_val': vals[0]['count'] if vals else None,
       'start': next((r for r in log if r.get('event') == 'start'), None),
       'img_per_s_median': sorted(r['img_per_s'] for r in log if 'img_per_s' in r)[
           len([r for r in log if 'img_per_s' in r]) // 2] if any('img_per_s' in r for r in log) else None}
(run / 'selection.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out))
"""
Path('/content/select_ckpt.py').write_text(select_src)

def evaluate(ckpts, prefix, beam=1):
    report = Path(f'{prefix}-report.json')
    if report.exists():
        print('evaluation already finished:', report)
        return json.loads(report.read_text())
    t0 = time.perf_counter()
    preds = Path(f'{prefix}-preds.csv')
    args = ['scripts/predict_real.py']
    for ckpt in ckpts:
        args.extend(['--ckpt', ckpt])
    args.extend(['--beam', str(beam), '--shards', TEST, '--out', preds, '--batch-size', '32'])
    if beam > 1:
        args.extend(['--nbest-out', f'{prefix}-nbest.jsonl'])
    run(*args)
    run('scripts/eval_predictions.py', preds, '--json', report, '--md', f'{prefix}-report.md')
    TIMINGS[Path(prefix).name + '_eval_s'] = round(time.perf_counter() - t0, 1)
    rep = json.loads(report.read_text())
    return rep

def overall(rep):
    o = next(g for g in rep['groups'] if g['subset'] == 'ALL')
    out = {k: o.get(k) for k in ('n', 'exact', 'exact_flat', 'valid', 'tanimoto', 'exact_ci95')}
    out['per_source_exact'] = {g['subset']: g['exact'] for g in rep['groups'] if g['subset'] != 'ALL'}
    return out

SUMMARY = {'run_name': RUN_NAME, 'branch': BRANCH, 'revision': revision,
           'test_manifest_sha256': test_sha,
           'test_manifest_matches_oct29': test_sha == EXPECTED_TEST_MANIFEST_SHA256,
           'n_test': test_stats['written'].get('test'),
           'pmc': {'prepare': pmc_prepare, 'ingest_written': pmc_ingest['written']},
           'leakage': LEAKAGE, 'arms': {}, 'timings_s': TIMINGS,
           'recipe': {'ARMS': ARMS, 'SWEEP_LR': SWEEP_LR, 'STEPS': STEPS, 'BATCH': BATCH,
                      'REAL_FRACTION': REAL_FRACTION, 'WARMUP': WARMUP,
                      'LABEL_SMOOTHING': LABEL_SMOOTHING, 'N_SYNTH': N_SYNTH,
                      'N_VALIDATION': N_VALIDATION, 'VAL_EVERY': VAL_EVERY,
                      'PATIENCE': PATIENCE, 'BATCH_FALLBACK': 16},
           'decoding': {}}

def record_decoding(variant, rep):
    metrics = overall(rep)
    SUMMARY['decoding'][variant] = {
        k: metrics[k] for k in ('exact', 'exact_flat', 'valid', 'per_source_exact', 'n')}
    SUMMARY['best_variant'] = max(SUMMARY['decoding'],
                                  key=lambda v: SUMMARY['decoding'][v]['exact'])
    (DRIVE_ROOT / 'summary.json').write_text(json.dumps(SUMMARY, indent=1, default=str))
for name, r in RUNS.items():
    run('/content/select_ckpt.py', r['dir'])
    r['selection'] = json.loads((r['dir'] / 'selection.json').read_text())
for arm in ARMS:
    cands = [(n, r) for n, r in RUNS.items() if r['arm'] == arm]
    name, best = max(cands, key=lambda nr: (nr[1]['selection']['best_val_exact'], -nr[1]['lr']))
    shutil.copy2(best['dir'] / 'selected.pt', DRIVE_ROOT / f'{arm}-selected.pt')
    shutil.copy2(best['dir'] / 'finetune.yaml', DRIVE_ROOT / f'{arm}-finetune.yaml')
    shutil.copy2(best['dir'] / 'train/log.jsonl', DRIVE_ROOT / f'{arm}-log.jsonl')
    released = evaluate([REPO / f'checkpoints/ouroboros-{arm}.pt'], DRIVE_ROOT / f'{arm}-released')
    tuned = evaluate([DRIVE_ROOT / f'{arm}-selected.pt'], DRIVE_ROOT / f'{arm}-selected')
    record_decoding(f'{arm}-released', released)
    record_decoding(f'{arm}-selected', tuned)
    variant = f'{arm}-selected-beam5'
    record_decoding(variant, evaluate([DRIVE_ROOT / f'{arm}-selected.pt'],
                                     DRIVE_ROOT / variant, beam=5))
    rel, tun = overall(released), overall(tuned)
    ref_k, ref_n = RELEASED_TARGET[arm]
    SUMMARY['arms'][arm] = {
        'selected_run': name, 'lr': best['lr'], 'selection': best['selection'],
        'sweep': {n: r['selection']['best_val_exact'] for n, r in cands},
        'released_test': rel, 'finetuned_test': tun,
        'released_correct': round(rel['exact'] * rel['n']), 'finetuned_correct': round(tun['exact'] * tun['n']),
        'oct29_released_reference': f'{ref_k}/{ref_n}',
        'beats_released_A_18.12pct': tun['exact'] > 54 / 298,
        'beats_own_released_here': tun['exact'] > rel['exact'],
        'train_wall_s': (json.loads((best['dir'] / 'train_wall_s.json').read_text())
                         if (best['dir'] / 'train_wall_s.json').exists() else []),
    }
    (DRIVE_ROOT / 'summary.json').write_text(json.dumps(SUMMARY, indent=1, default=str))
if 'A' in ARMS and 'C' in ARMS:
    variant = 'AC-ensemble-beam5'
    record_decoding(variant, evaluate([DRIVE_ROOT / 'A-selected.pt', DRIVE_ROOT / 'C-selected.pt'],
                                     DRIVE_ROOT / variant, beam=5))
SUMMARY['timings_s']['total_notebook_s_this_session'] = round(time.perf_counter() - T_START, 1)
(DRIVE_ROOT / 'summary.json').write_text(json.dumps(SUMMARY, indent=1, default=str))
for arm, s in SUMMARY['arms'].items():
    print(f"{arm}: released {s['released_correct']}/{s['released_test']['n']} "
          f"({100*s['released_test']['exact']:.2f}%) -> fine-tuned {s['finetuned_correct']}/"
          f"{s['finetuned_test']['n']} ({100*s['finetuned_test']['exact']:.2f}%); "
          f"best real-val exact {s['selection']['best_val_exact']:.4f} at step {s['selection']['best_step']} "
          f"(step-0 {s['selection']['val_exact_step0']})")
print(f"{'Variant':<24} {'Exact':>8} {'Flat':>8} {'Valid':>8} {'N':>5}")
for variant, metrics in SUMMARY['decoding'].items():
    print(f"{variant:<24} {metrics['exact']:8.2%} {metrics['exact_flat']:8.2%} "
          f"{metrics['valid']:8.2%} {metrics['n']:5d}")
print('Best variant:', SUMMARY['best_variant'])
print('Saved to', DRIVE_ROOT)
''')

md("""
## Inference profiling and combined profiling report

Released, selected and ensemble variants are measured greedily and with beam 5. Encoder timing
includes every encoding call made during ensemble rescoring; decode includes candidate ranking.
""")

code("""
variants = {}
for arm in ARMS:
    variants[f'{arm}-released'] = [REPO / f'checkpoints/ouroboros-{arm}.pt']
    variants[f'{arm}-selected'] = [DRIVE_ROOT / f'{arm}-selected.pt']
if 'A' in ARMS and 'C' in ARMS:
    variants['AC-ensemble'] = [DRIVE_ROOT / 'A-selected.pt', DRIVE_ROOT / 'C-selected.pt']
for name, ckpts in variants.items():
    if (PROFILING / f'infer_{name}.json').exists():
        continue
    args = ['scripts/profile_infer.py', '--shards', TEST, '--split', 'test', '--beam', '5',
            '--batch-size', '32', '--max-samples', '100', '--name', name, '--out-dir', PROFILING]
    for ckpt in ckpts:
        args.extend(['--ckpt', ckpt])
    run(*args)
""")

code("""
import statistics
profiling_summary = {'arms': {}, 'training_runs': {}, 'inference': {}}
for name, r in RUNS.items():
    csv_path = PROFILING / f'nvidia_smi_train_{name}.csv'
    log_path = r['dir'] / 'train/log.jsonl'
    records = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    rates = [rec['img_per_s'] for rec in records if 'img_per_s' in rec]
    profiling_summary['training_runs'][name] = {
        'img_per_s_median': statistics.median(rates) if rates else None,
        'gpu_util': parse_nvidia_smi_csv(csv_path.read_text()) if csv_path.exists() else None}
for arm in ARMS:
    probe = json.loads((PROFILING / f'train_{arm}.json').read_text())
    selected_run = SUMMARY['arms'][arm]['selected_run']
    profiling_summary['arms'][arm] = {
        'probe': probe, 'full_train': profiling_summary['training_runs'][selected_run]}
for name in variants:
    profiling_summary['inference'][name] = json.loads((PROFILING / f'infer_{name}.json').read_text())
(PROFILING / 'summary.json').write_text(json.dumps(profiling_summary, indent=2))
SUMMARY['profiling'] = profiling_summary
SUMMARY['timings_s']['total_notebook_s_this_session'] = round(time.perf_counter() - T_START, 1)
(DRIVE_ROOT / 'summary.json').write_text(json.dumps(SUMMARY, indent=1, default=str))
print('Profiling report (profiling/summary.json):')
print(json.dumps(SUMMARY['profiling'], indent=2))
print('Best variant:', SUMMARY['best_variant'])
print('Saved to', DRIVE_ROOT)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"name": "finetune_v2.ipynb", "provenance": [], "gpuType": "A100",
              "machine_shape": "hm"},
    "accelerator": "GPU",
}
out = Path(__file__).resolve().parent.parent / "colab" / "finetune_v2.ipynb"
nbf.write(nb, out)
print("wrote", out)
