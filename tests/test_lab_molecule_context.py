"""Fast offline tests for lab/molecule_context.py (OCSR output -> reasoner context)."""

import json

import pytest

from lab import molecule_context as mc
from ouroboros.eval.standardize import standardize

FIXTURE = "tests/fixtures/lab_predictions.csv"


@pytest.fixture(scope="module")
def ctx():
    return {c.id: c for c in mc.read_predictions(FIXTURE, with_molfile=True)}


def test_status_per_row(ctx):
    assert {k: c.status for k, c in ctx.items()} == {
        "aspirin": "ok",
        "acetate_salt": "ok",
        "alanine_flat": "review",
        "l_alanine": "ok",
        "r_group": "review",
        "broken": "invalid",
        "empty": "invalid",
        "low_margin": "rejected",
    }
    assert [k for k, c in ctx.items() if c.usable] == ["aspirin", "acetate_salt", "l_alanine"]


def test_ok_row_fields(ctx):
    a = ctx["aspirin"]
    assert a.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert a.formula == "C9H8O4"
    assert a.heavy_atoms == 13
    assert a.gate == {"accepted": True, "rejection_reason": "", "margin": 0.8,
                      "disagreement": 0.0}
    assert "V2000" in a.molfile and "M  END" in a.molfile


def test_salt_is_standardized_like_the_scorer(ctx):
    assert ctx["acetate_salt"].smiles == standardize("CC(=O)O").smiles


def test_stereo_and_pseudo_atoms(ctx):
    assert ctx["l_alanine"].stereo_specified == 1 and ctx["l_alanine"].stereo_unassigned == 0
    assert ctx["alanine_flat"].reasons == ["unassigned_stereo"]
    assert ctx["r_group"].pseudo_atoms == ["R1"]
    assert ctx["r_group"].reasons == ["pseudo_atoms"]


def test_invalid_and_rejected_reasons(ctx):
    assert ctx["broken"].reasons == ["unparsable"] and ctx["broken"].smiles is None
    assert ctx["empty"].reasons == ["empty"]
    assert ctx["low_margin"].reasons == ["low_margin"]
    assert not ctx["low_margin"].usable


def test_gold_never_leaks(ctx):
    for c in ctx.values():
        record = json.loads(c.to_json())
        assert "gold" not in record
        assert record["schema"] == mc.SCHEMA
    # the low_margin row has gold CCN; it must not appear anywhere in the outputs
    text = mc.prompt_block(list(ctx.values())) + "".join(c.to_json() for c in ctx.values())
    assert "CCN" not in text


def test_prompt_block_only_trusts_ok(ctx):
    lines = mc.prompt_block(list(ctx.values())).splitlines()
    by_id = {ln.split(":")[0][2:]: ln for ln in lines[1:]}
    assert "SMILES" in by_id["aspirin"]
    assert "confirm with a human" in by_id["r_group"]
    for k in ("broken", "empty", "low_margin"):
        assert "structure unknown" in by_id[k] and "SMILES" not in by_id[k]


def test_ungated_csv_has_empty_gate(tmp_path):
    p = tmp_path / "preds.csv"
    p.write_text("id,gold,pred,subset\nx,CCO,CCO,UOB\n")
    (c,) = mc.read_predictions(p)
    assert c.status == "ok" and c.gate == {} and c.subset == "UOB" and c.molfile is None


def test_cli_writes_jsonl(tmp_path, capsys):
    out = tmp_path / "ctx.jsonl"
    assert mc.main([FIXTURE, "--out", str(out), "--prompt"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 8
    summary = json.loads(capsys.readouterr().err.splitlines()[0])
    assert summary == {"rows": 8, "ok": 3, "review": 2, "rejected": 1, "invalid": 2}
