"""Fast offline tests for lab/edge_plan.py (distillation plan, --dry-run only)."""

import copy
import json
import re
from pathlib import Path

import pytest

from lab import edge_plan as ep

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg():
    return ep.load_config(ep.DEFAULT_CONFIG)


def test_default_config_is_valid(cfg):
    assert ep.validate(cfg) is cfg


def test_existing_steps_point_at_real_repo_code(cfg):
    steps = ep.plan(cfg)
    assert [s["step"] for s in steps] == list(range(1, len(steps) + 1))
    for s in steps:
        assert s["status"] in ("exists", "todo")
        if s["status"] == "todo":
            assert s["todo"]
            continue
        for script in re.findall(r"python (scripts/\S+\.py)", s["cmd"]):
            assert (REPO / script).is_file(), script
        for mod in re.findall(r"python -m (\S+)", s["cmd"]):
            assert (REPO / mod.replace(".", "/")).exists(), mod
    assert (REPO / cfg["student"]["config"]).is_file()


@pytest.mark.parametrize("key,split", [("selection", "test"), ("pseudo_labels", "train")])
def test_test_sets_are_refused(cfg, key, split):
    bad = copy.deepcopy(cfg)
    bad[key] = {**bad[key], "shards": "real/frozen_test_v2", "split": split}
    with pytest.raises(ep.PlanError, match=key):
        ep.validate(bad)


@pytest.mark.parametrize("path,value,msg", [
    (("student", "arm"), "Z", "student.arm"),
    (("teacher", "checkpoints"), [], "teacher.checkpoints"),
    (("teacher", "tta"), ["rotate"], "teacher.tta"),
    (("pseudo_labels", "max_disagreement"), 2, "max_disagreement"),
    (("pseudo_labels", "min_margin"), -1, "min_margin"),
    (("quantize", "method"), "int2", "quantize.method"),
    (("target", "device"), "toaster", "target.device"),
])
def test_bad_values_are_refused(cfg, path, value, msg):
    bad = copy.deepcopy(cfg)
    bad[path[0]][path[1]] = value
    with pytest.raises(ep.PlanError, match=re.escape(msg)):
        ep.validate(bad)


def test_cli_dry_run_json(capsys):
    assert ep.main(["--dry-run", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["name"] == "ocsr-edge-v0"
    # a plan carries no results
    assert not any(k in json.dumps(out) for k in ("exact_match", "accuracy"))


def test_cli_without_dry_run_does_nothing(capsys):
    assert ep.main([]) == 2
    assert "Only --dry-run" in capsys.readouterr().err


def test_cli_reports_config_errors(tmp_path, cfg, capsys):
    cfg["student"]["arm"] = "Z"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(cfg))
    assert ep.main([str(p), "--dry-run"]) == 1
    assert "student.arm" in capsys.readouterr().err
