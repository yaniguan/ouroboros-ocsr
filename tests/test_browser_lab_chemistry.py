"""Regression tests for Issue #123: Browser 3D Lab graph-level aliases and aromatic Kekulization."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ouroboros.data.rgroups import expand_aliases

ROOT = Path(__file__).resolve().parents[1]


def _rdkit_formula(smiles: str) -> str:
    if "[" in smiles and any(
        tag in smiles for tag in ("[OMe]", "[Boc]", "[Ph]", "[Me]", "[Fmoc]", "[Cbz]", "[Ts]")
    ):
        mol = expand_aliases(smiles)
    else:
        mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return rdMolDescriptors.CalcMolFormula(mol)


def test_browser_lab_aliases_and_aromatics_match_rdkit() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")

    valid_molecules = [
        "[OMe]c1ccc(C(=O)O)cc1",  # ACS-0071: p-anisic acid (not 4-hydroxymethylbenzoic acid)
        "[Boc]NCCc1ccccc1",  # prefix carbamate (not formate ester at methyl)
        "c1ccc([Ph])cc1",  # biphenyl (no ring-1 collision)
        "[Me]O[Me]",  # dimethyl ether
        "[Fmoc]NCC",  # tricyclic fluorenylmethoxycarbonyl
        "[Cbz]NCC",  # benzyloxycarbonyl
        "[Ts]NCC",  # tosyl sulfonamide
        "Cn1cccc1",  # 1-methylpyrrole (tertiary pyrrolic n)
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine (purine dione + bridgehead c + tertiary n)
        "Cc1cc(F)ccc1-c1nc(N[C@H](C)CO)nc2c1ccc(=O)n2C",  # USPTO-0512 biaryl pyridopyrimidinone
        "c1ccc2[nH]ccc2c1",  # indole
        "c1ccc2ccccc2c1",  # naphthalene
        "c1ccsc1",  # thiophene
    ]
    invalid_syntax = [
        "[R1]c1ccccc1",
        "*c1ccccc1",
        "C=[Me]",
        "C[Me]C",
        "[Ph]",
        "c1cccc1",  # non-Kekulizable 5-carbon aromatic radical cycle
    ]

    js_driver = """
    const fs = require('fs');
    const vm = require('vm');
    const src = fs.readFileSync('site/lab/lab.js', 'utf8');
    const noop = () => ({
      addEventListener(){},
      classList:{ toggle(){}, remove(){} },
      style:{},
      querySelectorAll(){ return []; },
    });
    const ctx = {
      window: { addEventListener(){} },
      document: {
        readyState: 'loading',
        documentElement: { dataset: {} },
        addEventListener(){},
        getElementById: noop,
        querySelector: noop,
        querySelectorAll(){ return []; },
      },
      localStorage: { getItem(){ return null; }, setItem(){} },
      fetch: async () => ({ ok: false, status: 404 }),
      requestAnimationFrame(){},
      URL: { createObjectURL(){}, revokeObjectURL(){} },
      Blob: class {},
      console,
    };
    const exposed = src.replace(
      'function runCustomClientSmiles(rawSmiles) {',
      'globalThis.__runCustom = runCustomClientSmiles; function runCustomClientSmiles(rawSmiles) {'
    );
    vm.runInNewContext(exposed, ctx);
    const inputs = JSON.parse(process.argv[1]);
    const out = { valid: {}, invalid: {}, tbs: ctx.__runCustom('[TBS]OCC') };
    for (const s of inputs.valid) {
      const r = ctx.__runCustom(s);
      out.valid[s] = {
        overall: r.overall_status,
        formula: r.formula,
        g1: r.gates[0].status,
        nAtoms: r.structure_3d ? r.structure_3d.atoms.length : 0,
      };
    }
    for (const s of inputs.invalid) {
      const r = ctx.__runCustom(s);
      out.invalid[s] = { overall: r.overall_status, g1: r.gates[0].status };
    }
    console.log(JSON.stringify(out));
    """

    payload = json.dumps({"valid": valid_molecules, "invalid": invalid_syntax})
    out = json.loads(
        subprocess.check_output(
            [node, "-e", js_driver, payload], cwd=str(ROOT), text=True, encoding="utf-8"
        )
    )

    for smi in valid_molecules:
        expected_formula = _rdkit_formula(smi)
        got = out["valid"][smi]
        assert got["g1"] == "pass", f"{smi} failed Gate 1: {got}"
        assert got["overall"] == "pass", f"{smi} failed overall: {got}"
        assert got["formula"].split()[0] == expected_formula, (
            f"{smi}: got {got['formula']!r}, expected {expected_formula!r}"
        )

    for smi in invalid_syntax:
        got = out["invalid"][smi]
        assert got["g1"] == "fail" and got["overall"] == "fail", f"{smi} should fail: {got}"

    # [TBS]OCC expands cleanly (Gate 1 pass, C8H20OSi) and triggers Gate 2 unsupported-element Si
    tbs = out["tbs"]
    assert tbs["gates"][0]["status"] == "pass"
    assert tbs["formula"].split()[0] == "C8H20OSi"
    assert tbs["gates"][1]["status"] == "fail"
