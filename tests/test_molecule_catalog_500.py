"""Tests for Issue #128: zero fail presets, 500-molecule catalog, and PubChem 3D."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ouroboros.eval.standardize import standard_inchikey, standardize
from ouroboros.ui.compliance import PRESET_MOLECULES
from scripts.check_no_data import violations

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PUBCHEM_3D_SDF = """7303
  -OEChem-11222622253D

 10 10  0     0  0  0  0  0  0999 V2000
   -0.0181    1.1195   -0.1121 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0183   -1.1196    0.1115 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.9822   -0.0001    0.0004 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3529    0.7276    0.1723 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3532   -0.7274   -0.1718 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7603   -0.0001   -0.0003 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0630    1.3095   -0.4176 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5641    0.8754    1.2351 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5649   -0.8753   -1.2345 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0631   -1.3092    0.4184 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  4  1  0  0  0  0
  1  6  1  0  0  0  0
  2  5  1  0  0  0  0
  2  6  1  0  0  0  0
  3  6  2  0  0  0  0
  4  5  1  0  0  0  0
  4  7  1  0  0  0  0
  4  8  1  0  0  0  0
  5  9  1  0  0  0  0
  5 10  1  0  0  0  0
M  END
$$$$
"""


def test_featured_presets_and_ui_dropdowns_have_zero_fail_or_warn_entries() -> None:
    presets = json.loads(
        (ROOT / "site" / "assets" / "lab-presets.json").read_text(encoding="utf-8")
    )
    comp = presets["compliance"]
    assert len(comp) >= 15
    for item in comp:
        label = item["label"].lower()
        assert "(fail)" not in label and "(warn)" not in label, item["label"]
        assert item["clash"] is False
        assert item["result"]["overall_status"] == "pass", item["key"]
        assert all(g["status"] == "pass" for g in item["result"]["gates"]), item["key"]

    for group_name, entries in PRESET_MOLECULES.items():
        assert "breaker" not in group_name.lower()
        for entry in entries:
            low = entry["name"].lower()
            assert "(fail" not in low and "(warn" not in low

    ui_html = (ROOT / "ouroboros" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "(fail)" not in ui_html and "(warn)" not in ui_html


def test_500_molecule_bio_and_electrolyte_catalog_integrity() -> None:
    cat_path = ROOT / "site" / "assets" / "molecule-catalog-500.json"
    assert cat_path.is_file()
    assert violations([("site/assets/molecule-catalog-500.json", cat_path.stat().st_size)]) == []

    data = json.loads(cat_path.read_text(encoding="utf-8"))
    assert data["schema"] == "ouroboros.molecule_catalog_3d/v1"
    mols = data["molecules"]
    assert len(mols) == 500
    assert data["counts"] == {"total": 500, "bio": 260, "electrolyte": 240}
    assert sum(1 for m in mols if m["domain"] == "bio") == 260
    assert sum(1 for m in mols if m["domain"] == "electrolyte") == 240

    seen_ik: set[str] = set()
    for m in mols:
        std = standardize(m["smiles"])
        assert std.valid, m["name"]
        ik = standard_inchikey(std)
        assert ik is not None and ik not in seen_ik, f"Duplicate InChIKey on {m['name']}"
        seen_ik.add(ik)
        n_atoms = len(m["sym"])
        assert n_atoms >= 3
        assert len(m["xyz"]) == n_atoms and len(m["frc"]) == n_atoms
        assert m["dmin"] >= 0.75
        assert m["bdev"] <= 0.30
        assert m["fmax"] <= 0.05
        se3, so3, netf, grad = m["eq"]
        assert se3 < 1e-4 and so3 < 1e-4 and netf < 1e-3 and grad < 2e-3


def test_browser_lab_hydrates_all_500_and_parses_pubchem_3d_sdf() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")

    js_driver = """
    const fs = require('fs');
    const vm = require('vm');
    const src = fs.readFileSync('site/lab/lab.js', 'utf8');
    const cat = JSON.parse(fs.readFileSync('site/assets/molecule-catalog-500.json', 'utf8'));
    const noop = () => ({
      addEventListener(){},
      classList:{ toggle(){}, remove(){}, add(){} },
      style:{},
      querySelectorAll(){ return []; },
      closest(){ return null; },
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
    const exposed = src
      .replace(
        'function hydrateCatalogEntry(item) {',
        'globalThis.__hydrate = hydrateCatalogEntry; function hydrateCatalogEntry(item) {'
      )
      .replace(
        'function parsePubChem3dSdf(sdfText, meta) {',
        'globalThis.__parseSdf = parsePubChem3dSdf; function parsePubChem3dSdf(sdfText, meta) {'
      );
    vm.runInNewContext(exposed, ctx);

    let passCount = 0;
    for (const item of cat.molecules) {
      const r = ctx.__hydrate(item);
      const allPass = r.gates.length === 10 && r.gates.every(g => g.status === 'pass');
      if (r.overall_status === 'pass' && allPass) {
        passCount++;
      }
    }
    const sdfRes = ctx.__parseSdf(process.argv[1], {
      cid: 7303,
      smiles: 'C1COC(=O)O1',
      formula: 'C3H4O3',
      mw: 88.06,
      sourceLabel: 'PubChem 3D (CID 7303)',
    });
    console.log(JSON.stringify({
      passCount,
      sdfStatus: sdfRes.overall_status,
      sdfAtoms: sdfRes.structure_3d.atoms.length,
      sdfBonds: sdfRes.structure_3d.bonds.length,
      sdfDepict: sdfRes.depiction_2d,
      sdfSe3: sdfRes.equivariance_metrics.se3_energy_abs_err_ev,
    }));
    """
    out = json.loads(
        subprocess.check_output(
            [node, "-e", js_driver, SAMPLE_PUBCHEM_3D_SDF],
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
        )
    )
    assert out["passCount"] == 500
    assert out["sdfStatus"] == "pass"
    assert out["sdfAtoms"] == 10 and out["sdfBonds"] == 10
    assert "7303/PNG" in out["sdfDepict"]
    assert out["sdfSe3"] < 1e-9
