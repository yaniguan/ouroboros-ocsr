"""Tests for the static project website (site/) and README visual assets."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_project_site_assets_and_links_exist():
    assert (SITE / ".nojekyll").is_file()
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for required in ("styles.css", "stage.js", "charts.js", "widgets.js", "app.js"):
        assert (SITE / required).is_file()
        assert required in html

    # Verify all relative src= and data-image= targets exist inside site/
    for attr in re.findall(r'(?:src|data-image)="([^"]+)"', html):
        if attr.startswith(("http://", "https://", "data:")):
            continue
        assert (SITE / attr).is_file(), f"Missing site asset: {attr}"

    # Verify README <picture> SVG assets exist
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for rel in re.findall(r'(?:src|srcset)="(assets/[^"]+)"', readme):
        assert (ROOT / rel).is_file(), f"Missing README asset: {rel}"


def test_benchmark_data_matches_receipts():
    mod = _load_script("build_site_data")
    committed = json.loads((SITE / "assets" / "benchmark-data.json").read_text(encoding="utf-8"))
    assert committed == mod.build_data()
    assert len(committed["rotation_sweep"]["angles"]) == 24
    assert len(committed["p1_chemveritas"]["ablations"]) == 7
    assert len(committed["frozen_vs_dev"]["groups"]) == 6


def test_browser_lab_assets_complete():
    assert (SITE / "lab" / "index.html").is_file()
    assert (SITE / "lab" / "lab.js").is_file()
    presets = json.loads((SITE / "assets" / "lab-presets.json").read_text(encoding="utf-8"))
    assert len(presets["compliance"]) >= 14
    assert set(presets["equivariance2d"]) == {"steerable_c8", "equiv_attn_c4", "baseline_cnn"}
    assert set(presets["propagation"]) == {"menthol", "camphor"}


def test_site_and_ui_plain_english_no_cjk():
    cjk = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
    targets = [
        *SITE.rglob("*.html"),
        *SITE.rglob("*.css"),
        *SITE.rglob("*.js"),
        *SITE.rglob("*.svg"),
        ROOT / "ouroboros/ui/index.html",
        ROOT / "README.md",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert not cjk.search(text), f"Non-English CJK characters found in {path}"


def test_ui_html_metric_bindings_and_browser_lab_smiles_parser():
    import shutil
    import subprocess

    ui_html = (ROOT / "ouroboros" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "stereo_elem_acc_strict" in ui_html
    assert "agg.stereo_accuracy" not in ui_html
    assert 'id="ocsrRef" class="mono" value=""' in ui_html

    node = shutil.which("node")
    if node is None:
        return
    js_driver = """
    const fs = require('fs');
    const vm = require('vm');
    const src = fs.readFileSync('site/lab/lab.js', 'utf8');
    // Extract runCustomClientSmiles + helpers inside a minimal DOM stub
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
    const bad = ctx.__runCustom('Invalid Molecule!');
    const furan = ctx.__runCustom('c1ccoc1');
    const lactic = ctx.__runCustom('CC(O)C(=O)O');
    const ion = ctx.__runCustom('[NH4+]');
    const rad = ctx.__runCustom('[CH2]C');
    console.log(JSON.stringify({
      bad: bad.overall_status,
      badG1: bad.gates[0].status,
      furan: furan.overall_status,
      furanFormula: furan.formula,
      furanGates: furan.gates.length,
      lacticG3: lactic.gates[2].status,
      ionG2: ion.gates[1].status,
      radG2: rad.gates[1].status,
    }));
    """
    out = json.loads(
        subprocess.check_output([node, "-e", js_driver], cwd=str(ROOT), text=True, encoding="utf-8")
    )
    assert out["bad"] == "fail" and out["badG1"] == "fail"
    assert out["furan"] == "pass" and "C4H4O" in out["furanFormula"] and out["furanGates"] == 10
    assert out["lacticG3"] == "warn"
    assert out["ionG2"] == "fail"
    assert out["radG2"] == "fail"
