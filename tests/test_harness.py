"""Predictions harness: input formats, per-subset breakdown, kind separation, reports."""

import csv
import json
from pathlib import Path

import pytest

from ouroboros.eval import harness
from ouroboros.eval.smiles import tanimoto

ALA, D_ALA = "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"
ROWS = [
    {"id": "1", "gold": ALA, "pred": ALA, "subset": "USPTO"},  # exact
    {"id": "2", "gold": ALA, "pred": D_ALA, "subset": "USPTO"},  # enantiomer
    {"id": "3", "gold": "CCO", "pred": "CCN", "subset": "UOB"},  # wrong constitution
    {"id": "4", "gold": "CCO", "pred": "", "subset": "UOB"},  # blank prediction
    {"id": "5", "gold": "c1ccccc1", "pred": "C1=CC=CC=C1", "subset": "DECIMER"},  # Kekulé form
    {"id": "6", "gold": "C1CC(", "pred": "CCO", "subset": "DECIMER"},  # invalid gold
]


def _write_csv(path, rows, delimiter=","):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)
    return path


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _groups(report):
    return {(g["subset"], g["kind"]): g for g in report["groups"]}


def test_cli_csv_numbers(tmp_path, capsys):
    src = _write_csv(tmp_path / "p.csv", ROWS)
    out_json, out_md, per = tmp_path / "r.json", tmp_path / "r.md", tmp_path / "s.jsonl"
    rc = harness.main(
        [str(src), "--json", str(out_json), "--md", str(out_md), "--per-sample", str(per)]
    )
    assert rc == 0
    rep = json.loads(out_json.read_text())
    assert rep["n_rows"] == 6 and rep["n_scored"] == 5
    assert rep["gold_invalid"] == {"n": 1, "ids": ["6"]}
    assert rep["columns"]["subset"] == "subset" and rep["columns"]["kind"] is None
    g = _groups(rep)
    assert list(g) == [
        ("ALL", None),
        ("DECIMER", None),
        ("UOB", None),
        ("USPTO", None),
        ("has_stereo", None),
        ("no_stereo", None),
    ]
    a = g[("ALL", None)]
    assert a["n"] == 5 and a["n_rows"] == 6 and a["n_gold_invalid"] == 1
    assert a["exact"] == pytest.approx(0.4)
    assert a["exact_flat"] == pytest.approx(0.6)
    assert a["inchikey"] == pytest.approx(0.4)
    assert a["valid"] == pytest.approx(0.8)
    assert a["tanimoto_1"] == pytest.approx(0.6)
    assert a["tanimoto"] == pytest.approx((3 + tanimoto("CCN", "CCO")) / 5)
    assert a["n_pred_blank"] == 1
    lo, hi = a["exact_ci95"]
    assert 0.0 <= lo <= 0.4 <= hi <= 1.0
    u = g[("USPTO", None)]
    assert (u["n"], u["exact"], u["exact_flat"], u["inchikey"], u["tanimoto_1"]) == (
        2, 0.5, 1.0, 0.5, 1.0
    )
    assert u["stereo_elem_acc_strict"] == 0.5
    o = g[("UOB", None)]
    assert (o["n"], o["exact"], o["valid"], o["tanimoto_1"]) == (2, 0.0, 0.5, 0.0)
    d = g[("DECIMER", None)]
    assert (d["n"], d["exact"], d["n_rows"], d["n_gold_invalid"]) == (1, 1.0, 2, 1)

    md = out_md.read_text()
    assert capsys.readouterr().out == md
    assert "| **ALL** | 5 | 40.00% |" in md and "| USPTO | 2 | 50.00% |" in md
    assert "Tanimoto@1.0" in md and "excluded because the gold" in md and "ids: 6" in md
    assert rep["footnote"] is None or rep["footnote"] in md

    per_rows = [json.loads(x) for x in per.read_text().splitlines()]
    assert len(per_rows) == 6 and per_rows[5]["gold_invalid"] and "exact" not in per_rows[5]
    assert per_rows[1]["exact_flat"] and not per_rows[1]["exact"]


def test_jsonl_tsv_and_workers_agree(tmp_path):
    reports = []
    for path, extra in [
        (_write_csv(tmp_path / "p.csv", ROWS), []),
        (_write_csv(tmp_path / "p.tsv", ROWS, delimiter="\t"), []),
        (_write_jsonl(tmp_path / "p.jsonl", ROWS), ["--workers", "2"]),
    ]:
        out = tmp_path / f"{path.suffix[1:]}.json"
        harness.main([str(path), "--json", str(out), "--quiet", *extra])
        rep = json.loads(out.read_text())
        reports.append(json.dumps(rep["groups"], sort_keys=True))  # NaN-safe comparison
    assert reports[0] == reports[1] == reports[2]


def test_run_eval_log_aliases_where_and_kinds(tmp_path):
    """The run_eval predictions.jsonl schema: set/kind/index/angle/ref/pred."""
    rows = []
    for angle in (0.0, 90.0):
        rows += [
            {"set": "rendered_test", "kind": "rendered", "index": 0, "angle": angle,
             "ref": ALA, "pred": ALA if angle == 0.0 else D_ALA},
            {"set": "rendered_test", "kind": "rendered", "index": 1, "angle": angle,
             "ref": "CCO", "pred": "CCO"},
            {"set": "uspto_real", "kind": "real", "index": 0, "angle": angle,
             "ref": "CCO", "pred": None},
        ]
    src = _write_jsonl(tmp_path / "predictions.jsonl", rows)
    with pytest.raises(ValueError, match="duplicate id"):
        harness.main([str(src), "--quiet"])
    out = tmp_path / "r.json"
    harness.main([str(src), "--where", "angle=0", "--json", str(out), "--quiet"])
    rep = json.loads(out.read_text())
    assert rep["columns"] == {
        "id": "index", "gold": "ref", "pred": "pred", "subset": "set", "kind": "kind"
    }
    g = _groups(rep)
    # one overall row per kind, never a row pooling rendered and real
    assert ("ALL", None) not in g and rep["pooled_across_kinds"] is False
    assert g[("ALL", "rendered")]["n"] == 2 and g[("ALL", "rendered")]["exact"] == 1.0
    assert g[("ALL", "real")]["n"] == 1 and g[("ALL", "real")]["valid"] == 0.0
    harness.main([str(src), "--where", "angle=90.0", "--json", str(out), "--quiet"])
    g = _groups(json.loads(out.read_text()))
    assert g[("rendered_test", "rendered")]["exact"] == 0.5


def test_column_overrides_and_errors(tmp_path):
    rows = [{"name": "a", "truth": "CCO", "guess": "OCC", "src": "X"}]
    src = _write_csv(tmp_path / "p.csv", rows)
    with pytest.raises(ValueError, match="no gold / pred column"):
        harness.main([str(src), "--quiet"])
    out = tmp_path / "r.json"
    harness.main([
        str(src), "--id-col", "name", "--gold-col", "truth", "--pred-col", "guess",
        "--subset-col", "src", "--json", str(out), "--quiet",
    ])
    g = _groups(json.loads(out.read_text()))
    assert g[("X", None)]["exact"] == 1.0
    with pytest.raises(ValueError, match="not found"):
        harness.main([str(src), "--gold-col", "nope", "--pred-col", "guess", "--quiet"])
    with pytest.raises(ValueError, match="unsupported"):
        harness.main([str(tmp_path / "p.txt"), "--quiet"])
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"gold": "CCO", "pred": "CCO"}\nnot json\n')
    with pytest.raises(ValueError, match=":2:"):
        harness.read_records(bad)
    with pytest.raises(ValueError, match="no records"):
        harness.main([str(_write_csv(tmp_path / "e.csv", ROWS)), "--where", "subset=none"])
    with pytest.raises(ValueError, match="COLUMN=VALUE"):
        harness.parse_where(["angle"])


def test_without_id_or_subset_columns(tmp_path):
    src = _write_jsonl(tmp_path / "p.jsonl", [{"gold": "CCO", "pred": "CCO"}] * 3)
    out = tmp_path / "r.json"
    harness.main([str(src), "--json", str(out), "--quiet"])
    rep = json.loads(out.read_text())
    assert [(g["subset"], g["n"], g.get("exact")) for g in rep["groups"]] == [
        ("ALL", 3, 1.0),
        ("has_stereo", 0, None),
        ("no_stereo", 3, 1.0),
    ]


def test_stereo_subset_split(tmp_path):
    rows = [
        # Tetrahedral stereo after salt stripping -> has_stereo, exact
        {"id": "1", "gold": "C[C@H](N)C(=O)[O-].[Na+]", "pred": ALA, "subset": "ACS"},
        # Wrong enantiomer -> has_stereo, exact=False, exact_flat=True
        {"id": "2", "gold": ALA, "pred": D_ALA, "subset": "ACS"},
        # Double-bond E/Z stereo -> has_stereo, exact=False, exact_flat=True
        {"id": "3", "gold": "C/C=C/C", "pred": "C/C=C\\C", "subset": "USPTO"},
        # Double-bond E/Z stereo -> has_stereo, exact=True
        {"id": "4", "gold": "C/C=C/C", "pred": "C/C=C/C", "subset": "USPTO"},
        # Achiral -> no_stereo, exact=True
        {"id": "5", "gold": "CCO", "pred": "OCC", "subset": "UOB"},
        # Achiral -> no_stereo, exact=False
        {"id": "6", "gold": "c1ccccc1", "pred": "CCO", "subset": "UOB"},
    ]
    src = _write_csv(tmp_path / "stereo.csv", rows)
    out_json, out_md = tmp_path / "stereo.json", tmp_path / "stereo.md"
    assert harness.main([str(src), "--json", str(out_json), "--md", str(out_md), "--quiet"]) == 0
    g = _groups(json.loads(out_json.read_text()))
    hs = g[("has_stereo", None)]
    ns = g[("no_stereo", None)]
    assert hs["n"] == 4
    assert hs["exact"] == pytest.approx(0.5)
    assert hs["exact_flat"] == pytest.approx(1.0)
    lo, hi = hs["exact_ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert ns["n"] == 2
    assert ns["exact"] == pytest.approx(0.5)
    assert ns["exact_flat"] == pytest.approx(0.5)
    assert len(ns["exact_ci95"]) == 2
    md = out_md.read_text()
    assert "| has_stereo | 4 | 50.00% |" in md
    assert "| no_stereo | 2 | 50.00% |" in md


def test_all_gold_invalid_group_renders(tmp_path):
    rows = [{"id": "1", "gold": "C1CC(", "pred": "CCO", "subset": "S"}]
    src = _write_csv(tmp_path / "p.csv", rows)
    md = tmp_path / "r.md"
    harness.main([str(src), "--md", str(md), "--quiet"])
    assert "| S | 0 | – |" in md.read_text()


def test_committed_v2_devtest_reports_include_stereo_rows():
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "finetune-v2-oct30"
    names = [
        "AC-ensemble-beam5-ttapad",
        "AC-ensemble-beam5",
        "A-selected",
        "C-selected",
    ]
    for name in names:
        md = (root / f"{name}-report.md").read_text(encoding="utf-8")
        assert "| has_stereo | 49 |" in md
        assert "| no_stereo | 249 |" in md
        rep = json.loads((root / f"{name}-report.json").read_text(encoding="utf-8"))
        assert rep["input"] == f"{name}-preds.csv"
        groups = {g["subset"]: g for g in rep["groups"]}
        assert groups["has_stereo"]["n"] == 49
        assert groups["no_stereo"]["n"] == 249
        assert len(groups["has_stereo"]["exact_ci95"]) == 2
        assert len(groups["no_stereo"]["exact_ci95"]) == 2


def test_empty_gold_refused_unless_skipped(tmp_path):
    rows = [
        {"id": "1", "gold": "CCO", "pred": "CCO", "subset": "S"},
        {"id": "2", "gold": "", "pred": "CCN", "subset": "S"},
        {"id": "3", "gold": " ", "pred": "", "subset": "S"},
    ]
    src = _write_csv(tmp_path / "p.csv", rows)
    with pytest.raises(ValueError, match="empty gold.*--skip-unlabeled"):
        harness.main([str(src), "--quiet"])

    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    rc = harness.main([str(src), "--skip-unlabeled", "--json", str(out_json),
                       "--md", str(out_md), "--quiet"])
    assert rc == 0
    rep = json.loads(out_json.read_text())
    assert rep["unlabeled"] == {"n": 2, "ids": ["2", "3"]}
    assert rep["n_rows"] == 1 and rep["n_scored"] == 1
    assert rep["gold_invalid"]["n"] == 0
    assert _groups(rep)[("ALL", None)]["exact"] == 1.0
    assert "Skipped 2 unlabeled row(s)" in out_md.read_text()

    only_unlabeled = _write_csv(tmp_path / "u.csv", rows[1:])
    with pytest.raises(ValueError, match="no labeled rows"):
        harness.main([str(only_unlabeled), "--skip-unlabeled", "--quiet"])
