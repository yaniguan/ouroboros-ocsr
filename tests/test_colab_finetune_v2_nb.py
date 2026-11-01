"""Generate the v2 notebook in isolation; validate its format and experiment contract."""

import ast
import json
import subprocess
import sys
import time
import runpy
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    root = tmp_path_factory.mktemp("finetune_v2")
    (root / "scripts").mkdir()
    (root / "colab").mkdir()
    generator = root / "scripts/make_finetune_v2_nb.py"
    generator.write_text((ROOT / "scripts/make_finetune_v2_nb.py").read_text())
    runpy.run_path(str(generator))
    nb = json.loads((root / "colab/finetune_v2.ipynb").read_text())
    nbformat.validate(nbformat.from_dict(nb))
    return nb


def _codes(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_compile(notebook):
    for src in _codes(notebook):
        ast.parse(src)


def test_run_helper_saves_tail_on_failure(notebook, tmp_path, capsys):
    first = _codes(notebook)[0]
    src = first[first.index("def shell") : first.index("def sha256")]
    g = {"Path": Path, "subprocess": subprocess, "time": time, "DRIVE_ROOT": tmp_path,
         "PY": sys.executable, "REPO": Path.cwd()}
    exec(src, g)
    code = "import sys; print('to-stdout'); print('to-stderr', file=sys.stderr); sys.exit(3)"
    with pytest.raises(subprocess.CalledProcessError):
        g["run"]("-c", code, log=tmp_path / "train_stderr.log")
    out = capsys.readouterr().out
    assert "FAILED (exit 3)" in out and "to-stderr" in out.split("FAILED")[1]
    saved = (tmp_path / "train_stderr.log").read_text()
    assert "to-stdout" in saved and "to-stderr" in saved and "[exit status 3]" in saved
    assert g["run"]("-c", code, check=False).returncode == 3  # default log kept on failure
    assert len(list((tmp_path / "logs").glob("*.log"))) == 1
    g["run"]("-c", "print('ok')")  # default log removed on success
    assert len(list((tmp_path / "logs").glob("*.log"))) == 1


def test_train_cell_skips_finished_runs_and_logs(notebook):
    train = next(c for c in _codes(notebook) if "ouroboros.train" in c and "DONE_CHECK" in c)
    assert "if train_done(r['dir'])" in train
    assert "log=r['dir'] / 'train_stderr.log'" in train


def test_recipe_and_existing_override_keys(notebook):
    from ouroboros.train.config import load_config

    codes = _codes(notebook)
    first = ast.parse(codes[0])
    knobs = {}
    names = {"RUN_NAME", "BRANCH", "ARMS", "SWEEP_LR", "STEPS", "BATCH", "REAL_FRACTION",
             "WARMUP", "LABEL_SMOOTHING", "N_VALIDATION", "VAL_EVERY", "PATIENCE", "N_SYNTH"}
    for node in first.body:
        if isinstance(node, ast.Assign):
            targets = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if targets & names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "knobs", "exec"), knobs)
    assert {k: knobs[k] for k in names} == {
        "RUN_NAME": "finetune_v2_oct30", "BRANCH": "feat/finetune-v2-oct30",
        "ARMS": ["A", "C"], "SWEEP_LR": {"A": [3e-5], "C": [3e-5]},
        "STEPS": 20000, "BATCH": 32, "REAL_FRACTION": 0.7, "WARMUP": 500,
        "LABEL_SMOOTHING": 0.05, "N_VALIDATION": 963, "VAL_EVERY": 1000,
        "PATIENCE": 6, "N_SYNTH": 50000,
    }
    assert "MyDrive/ouroboros-ocsr_runs') / RUN_NAME" in codes[0]
    config = next(c for c in codes if "sets = [" in c)
    sets_node = next(n for n in ast.walk(ast.parse(config))
                     if isinstance(n, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == "sets" for t in n.targets))
    keys = set()
    for value in sets_node.value.elts:
        key = value.values[0].value.split("=", 1)[0]
        keys.add(key)
        cfg = load_config(ROOT / "configs/finetune-real-pmc.yaml")
        for part in key.split("."):
            assert part in cfg, key
            cfg = cfg[part]
    assert {"train.warmup", "train.label_smoothing", "train.lr", "train.steps",
            "train.batch_size", "data.real_fraction", "data.synthetic_max_samples",
            "validation.samples", "validation.every", "validation.patience"} <= keys
    assert "sets[2] = 'train.batch_size=16'" in config


def test_data_gate_and_resume_preserved(notebook):
    source = "\n".join(_codes(notebook))
    original = (ROOT / "scripts/make_real_pmc_nb.py").read_text()
    gate = next(c for c in _codes(notebook) if "leak = run(" in c)
    assert gate.strip() in original  # retain the original executable gate verbatim
    assert "--fail-on-connectivity" in gate
    assert "r['overlap_inchikey'] or r['overlap_connectivity']" in gate
    assert "raise SystemExit" in gate
    assert "restore_or_build" in source and "train/ckpt/last.pt" in source
    assert "'--sets', 'USPTO', 'UOB', 'ACS', '--n', '100'" in source
    assert "'--seed', '0', '--out', 'real_data/subset100'" in source
    assert "957f0042825dab214aad2463de2974d7b4f23a1fcbf5d52ab78be2526cc3fc61" in source
    assert "test_sha != EXPECTED_TEST_MANIFEST_SHA256" in source
    assert "'--sources', 'zinc250k'" in source and "moses" not in source
    assert "'--rgroup-probability', '0.5'" in source


def test_evaluation_commands_and_report_resume(notebook, tmp_path):
    source = next(c for c in _codes(notebook) if "def evaluate(" in c)
    calls = []
    report = {"groups": [{"subset": "ALL", "n": 298, "exact": 0.2}]}

    def run(*args):
        calls.append(args)
        if args[0] == "scripts/eval_predictions.py":
            Path(args[args.index("--json") + 1]).write_text(json.dumps(report))

    env = {"Path": Path, "json": json, "time": time, "run": run,
           "TEST": tmp_path / "test", "TIMINGS": {}}
    tree = ast.parse(source)
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    exec(compile(ast.Module(body=[evaluate], type_ignores=[]), "evaluate", "exec"), env)
    for variant, ckpts, beam in [("A-released", ["released.pt"], 1),
                                  ("A-selected-beam5", ["A-selected.pt"], 5),
                                  ("AC-ensemble-beam5", ["A-selected.pt", "C-selected.pt"], 5)]:
        prefix = tmp_path / variant
        assert env["evaluate"](ckpts, prefix, beam) == report
        predict, score = calls[-2:]
        assert predict[0] == "scripts/predict_real.py"
        assert predict.count("--ckpt") == len(ckpts)
        assert predict[predict.index("--beam") + 1] == str(beam)
        assert predict[predict.index("--out") + 1] == Path(f"{prefix}-preds.csv")
        assert ("--nbest-out" in predict) == (beam > 1)
        assert score[-1] == f"{prefix}-report.md"
        before = len(calls)
        assert env["evaluate"](ckpts, prefix, beam) == report
        assert len(calls) == before
    assert "for arm in ARMS:" in source
    assert "if 'A' in ARMS and 'C' in ARMS:" in source
    assert "'summary.json'" in source and "'recipe':" in source and "'timings_s':" in source
    assert "SUMMARY['best_variant'] = max" in source
    assert "('exact', 'exact_flat', 'valid', 'per_source_exact', 'n')" in source


def test_profiling_contract(notebook):
    codes = _codes(notebook)
    source = '\n'.join(codes)
    probe = next(i for i, c in enumerate(codes) if "run('scripts/profile_train.py'" in c)
    train = next(i for i, c in enumerate(codes) if 'DONE_CHECK' in c)
    config = next(i for i, c in enumerate(codes) if 'RUNS = {}' in c)
    infer = next(i for i, c in enumerate(codes) if "args = ['scripts/profile_infer.py'" in c)
    evaluation = next(i for i, c in enumerate(codes) if 'def evaluate(' in c)
    assert config < probe < train < evaluation < infer
    assert "DRIVE_ROOT / 'profiling'" in source
    assert "f'train_{arm}.json').exists()" in source
    assert "f'infer_{name}.json').exists()" in source
    assert 'nvidia-smi' in source
    assert "f'nvidia_smi_train_{name}.csv'" in source
    training = codes[train]
    assert 'finally:\n        stop_nvidia_smi(gpu_sampler)' in training
    assert "'--beam', '5'" in codes[infer]
    assert "variants['AC-ensemble']" in codes[infer]
    assert "SUMMARY['profiling'] = profiling_summary" in source
    assert "(PROFILING / 'summary.json').write_text" in source
    assert 'statistics.median(rates)' in source
    assert "print(json.dumps(SUMMARY['profiling']" in source
