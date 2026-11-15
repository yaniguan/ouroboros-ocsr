"""Generate the v2 notebook in isolation; validate its format, sync, and experiment contract."""

import ast
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _cell_pairs(nb: dict) -> list[tuple[str, str]]:
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    root = tmp_path_factory.mktemp("finetune_v2")
    (root / "scripts").mkdir()
    (root / "colab").mkdir()
    generator = root / "scripts/make_finetune_v2_nb.py"
    src = (ROOT / "scripts/make_finetune_v2_nb.py").read_text(encoding="utf-8")
    generator.write_text(src, encoding="utf-8")
    runpy.run_path(str(generator))
    nb = json.loads((root / "colab/finetune_v2.ipynb").read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(nb))
    return nb


def _codes(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_v2_notebook_matches_generator(notebook):
    committed = json.loads((ROOT / "colab/finetune_v2.ipynb").read_text(encoding="utf-8"))
    assert _cell_pairs(notebook) == _cell_pairs(committed)
    header = "".join(notebook["cells"][0]["source"])
    assert "authuser=" not in header
    assert "Branch `main`" in header
    assert "/blob/main/colab/finetune_v2.ipynb" in header


def test_committed_real_pmc_notebook_matches_generator(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "colab").mkdir()
    gen = tmp_path / "scripts/make_real_pmc_nb.py"
    src = (ROOT / "scripts/make_real_pmc_nb.py").read_text(encoding="utf-8")
    gen.write_text(src, encoding="utf-8")
    runpy.run_path(str(gen))
    built = json.loads((tmp_path / "colab/finetune_real_pmc.ipynb").read_text(encoding="utf-8"))
    committed = json.loads((ROOT / "colab/finetune_real_pmc.ipynb").read_text(encoding="utf-8"))
    assert _cell_pairs(built) == _cell_pairs(committed)


def test_no_authuser_in_repo():
    for path in (ROOT / "colab").glob("*.ipynb"):
        assert "authuser=" not in path.read_text(encoding="utf-8"), path
    for path in (ROOT / "scripts").glob("*.py"):
        if path.name.startswith("make_"):
            assert "authuser=" not in path.read_text(encoding="utf-8"), path


def test_cells_compile(notebook):
    for src in _codes(notebook):
        ast.parse(src)


def test_run_helper_saves_tail_on_failure(notebook, tmp_path, capsys):
    first = _codes(notebook)[0]
    src = first[first.index("def shell") : first.index("def sha256")]
    g = {
        "Path": Path,
        "subprocess": subprocess,
        "time": time,
        "DRIVE_ROOT": tmp_path,
        "PY": sys.executable,
        "REPO": Path.cwd(),
    }
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
    names = {
        "RUN_NAME",
        "BRANCH",
        "ARMS",
        "SWEEP_LR",
        "STEPS",
        "BATCH",
        "REAL_FRACTION",
        "WARMUP",
        "LABEL_SMOOTHING",
        "N_VALIDATION",
        "VAL_EVERY",
        "PATIENCE",
        "N_SYNTH",
    }
    for node in first.body:
        if isinstance(node, ast.Assign):
            targets = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if targets & names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "knobs", "exec"), knobs)
    assert {k: knobs[k] for k in names} == {
        "RUN_NAME": "finetune_v2_oct30",
        "BRANCH": "main",
        "ARMS": ["A", "C"],
        "SWEEP_LR": {"A": [3e-5], "C": [3e-5]},
        "STEPS": 20000,
        "BATCH": 32,
        "REAL_FRACTION": 0.7,
        "WARMUP": 500,
        "LABEL_SMOOTHING": 0.05,
        "N_VALIDATION": 963,
        "VAL_EVERY": 1000,
        "PATIENCE": 6,
        "N_SYNTH": 50000,
    }
    assert "MyDrive/ouroboros-ocsr_runs') / RUN_NAME" in codes[0]
    config = next(c for c in codes if "sets = [" in c)
    sets_node = next(
        n
        for n in ast.walk(ast.parse(config))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "sets" for t in n.targets)
    )
    keys = set()
    for value in sets_node.value.elts:
        key = value.values[0].value.split("=", 1)[0]
        keys.add(key)
        cfg = load_config(ROOT / "configs/finetune-real-pmc.yaml")
        for part in key.split("."):
            assert part in cfg, key
            cfg = cfg[part]
    assert {
        "train.warmup",
        "train.label_smoothing",
        "train.lr",
        "train.steps",
        "train.batch_size",
        "data.real_fraction",
        "data.synthetic_max_samples",
        "validation.samples",
        "validation.every",
        "validation.patience",
    } <= keys
    assert "sets[2] = 'train.batch_size=16'" in config


def test_data_gate_and_resume_preserved(notebook):
    source = "\n".join(_codes(notebook))
    original = (ROOT / "scripts/make_real_pmc_nb.py").read_text(encoding="utf-8")
    gate = next(c for c in _codes(notebook) if "leak = run(" in c)
    gate_body = "\n".join(
        line for line in gate.splitlines() if not line.startswith("stage(")
    ).strip()
    assert gate_body in original
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

    env = {
        "Path": Path,
        "json": json,
        "time": time,
        "run": run,
        "TEST": tmp_path / "test",
        "TIMINGS": {},
    }
    tree = ast.parse(source)
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    exec(compile(ast.Module(body=[evaluate], type_ignores=[]), "evaluate", "exec"), env)
    for variant, ckpts, beam in [
        ("A-released", ["released.pt"], 1),
        ("A-selected-beam5", ["A-selected.pt"], 5),
        ("AC-ensemble-beam5", ["A-selected.pt", "C-selected.pt"], 5),
    ]:
        prefix = tmp_path / variant
        assert env["evaluate"](ckpts, prefix, beam=beam) == report
        predict, score = calls[-2:]
        assert predict[0] == "scripts/predict_real.py"
        assert predict.count("--ckpt") == len(ckpts)
        assert predict[predict.index("--beam") + 1] == str(beam)
        assert predict[predict.index("--split") + 1] == "test"
        assert predict[predict.index("--out") + 1] == Path(f"{prefix}-preds.csv")
        assert ("--nbest-out" in predict) == (beam > 1)
        assert score[-1] == f"{prefix}-report.md"
        before = len(calls)
        assert env["evaluate"](ckpts, prefix, beam=beam) == report
        assert len(calls) == before


def test_decoding_selected_on_val_and_test_scored_once(notebook, tmp_path):
    import shutil

    source = next(c for c in _codes(notebook) if "def evaluate(" in c)
    drive_root = tmp_path / "drive"
    repo_root = tmp_path / "repo"
    pmc_root = tmp_path / "pmc"
    test_root = tmp_path / "test"
    drive_root.mkdir()
    (repo_root / "checkpoints").mkdir(parents=True)

    runs = {}
    for arm, best_val in [("A", 0.51), ("C", 0.38)]:
        run_dir = drive_root / "runs" / f"{arm}-lr3e-05"
        (run_dir / "train").mkdir(parents=True)
        (run_dir / "selected.pt").write_bytes(b"ckpt")
        (run_dir / "finetune.yaml").write_text("seed: 0\n", encoding="utf-8")
        (run_dir / "train/log.jsonl").write_text("", encoding="utf-8")
        (run_dir / "selection.json").write_text(
            json.dumps({"best_step": 9000, "best_val_exact": best_val, "val_exact_step0": 0.18}),
            encoding="utf-8",
        )
        runs[f"{arm}-lr3e-05"] = {"arm": arm, "lr": 3e-5, "dir": run_dir}

    val_scores = {
        "A-released": 0.18,
        "A-selected": 0.50,
        "A-selected-beam5": 0.52,
        "A-selected-beam5-ttapad": 0.51,
        "C-released": 0.16,
        "C-selected": 0.37,
        "C-selected-beam5": 0.39,
        "C-selected-beam5-ttapad": 0.38,
        "AC-ensemble-beam5": 0.56,
        "AC-ensemble-beam5-ttapad": 0.54,
    }
    predict_calls = []

    def run(*args):
        if str(args[0]).endswith("select_ckpt.py"):
            return
        if args[0] == "scripts/predict_real.py":
            shards = Path(args[args.index("--shards") + 1])
            split = args[args.index("--split") + 1]
            out_csv = Path(args[args.index("--out") + 1])
            predict_calls.append((shards, split, out_csv.name))
            return
        if args[0] == "scripts/eval_predictions.py":
            report_path = Path(args[args.index("--json") + 1])
            variant = report_path.name.removesuffix("-report.json")
            is_val = report_path.parent.name == "val_decoding"
            exact = val_scores[variant] if is_val else 0.5235
            n = 963 if is_val else 298
            rep = {
                "groups": [
                    {
                        "subset": "ALL",
                        "n": n,
                        "exact": exact,
                        "exact_flat": exact + 0.02,
                        "valid": 0.99,
                        "tanimoto": 0.85,
                        "exact_ci95": [exact - 0.05, exact + 0.05],
                    },
                    {"subset": "ACS", "n": 100, "exact": exact},
                ]
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(rep), encoding="utf-8")

    env = {
        "Path": Path,
        "json": json,
        "time": time,
        "shutil": shutil,
        "stage": lambda *a, **kw: None,
        "run": run,
        "DRIVE_ROOT": drive_root,
        "REPO": repo_root,
        "PMC": pmc_root,
        "TEST": test_root,
        "TIMINGS": {},
        "RUN_NAME": "finetune_v2_oct30",
        "BRANCH": "main",
        "revision": "abc123",
        "test_sha": "957f0042825dab214aad2463de2974d7b4f23a1fcbf5d52ab78be2526cc3fc61",
        "EXPECTED_TEST_MANIFEST_SHA256": (
            "957f0042825dab214aad2463de2974d7b4f23a1fcbf5d52ab78be2526cc3fc61"
        ),
        "test_stats": {"written": {"test": 298}},
        "pmc_prepare": {},
        "pmc_ingest": {"written": {"train": 17220, "val": 963}},
        "LEAKAGE": [],
        "ARMS": ["A", "C"],
        "SWEEP_LR": {"A": [3e-5], "C": [3e-5]},
        "STEPS": 20000,
        "BATCH": 32,
        "REAL_FRACTION": 0.7,
        "WARMUP": 500,
        "LABEL_SMOOTHING": 0.05,
        "N_SYNTH": 50000,
        "N_VALIDATION": 963,
        "VAL_EVERY": 1000,
        "PATIENCE": 6,
        "RUNS": runs,
        "RELEASED_TARGET": {"A": (54, 298), "C": (50, 298)},
        "T_START": time.perf_counter(),
    }
    exec(source, env)

    val_calls = [c for c in predict_calls if c[1] == "val"]
    test_calls = [c for c in predict_calls if c[1] == "test"]
    assert len(val_calls) == 10
    assert all(c[0] == pmc_root for c in val_calls)
    assert len(test_calls) == 1
    assert test_calls[0] == (test_root, "test", "AC-ensemble-beam5-preds.csv")

    saved = json.loads((drive_root / "summary.json").read_text(encoding="utf-8"))
    assert saved["best_variant"] == "AC-ensemble-beam5"
    assert set(saved["val_decoding"]) == set(val_scores)
    assert saved["val_decoding"]["AC-ensemble-beam5"]["exact"] == 0.56
    assert saved["test"]["variant"] == "AC-ensemble-beam5"
    assert saved["test"]["n"] == 298
    assert saved["test"]["exact"] == 0.5235


def test_profiling_contract(notebook):
    codes = _codes(notebook)
    source = "\n".join(codes)
    probe = next(i for i, c in enumerate(codes) if "--out-dir', PROFILING, '--name', arm" in c)
    train = next(i for i, c in enumerate(codes) if "DONE_CHECK" in c)
    config = next(i for i, c in enumerate(codes) if "RUNS = {}" in c)
    infer = next(i for i, c in enumerate(codes) if "args = ['scripts/profile_infer.py'" in c)
    evaluation = next(i for i, c in enumerate(codes) if "def evaluate(" in c)
    assert config < train < probe < evaluation < infer
    assert "DRIVE_ROOT / 'profiling'" in source
    assert "f'train_{arm}.json').exists()" in source
    assert "f'infer_{name}.json').exists()" in source
    assert "nvidia-smi" in source
    assert "f'nvidia_smi_train_{name}.csv'" in source
    training = codes[train]
    assert "finally:\n        stop_nvidia_smi(gpu_sampler)" in training
    assert "'--beam', '5'" in codes[infer]
    assert "variants['AC-ensemble']" in codes[infer]
    assert "SUMMARY['profiling'] = profiling_summary" in source
    assert "(PROFILING / 'summary.json').write_text" in source
    assert "statistics.median(rates)" in source
    assert "print(json.dumps(SUMMARY['profiling']" in source
