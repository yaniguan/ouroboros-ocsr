"""CPU inference API (ouroboros/infer.py). Fast tests use a small untrained model; the release
checkpoint test runs only when checkpoints/ouroboros-A.pt is present."""
import io
import time

import pytest
from PIL import Image

from ouroboros.infer import DEFAULT_CKPT, InputError, OCSRPredictor
from ouroboros.ui.backend import _build_small_probe_model

KEYS = {"smiles", "canonical_smiles", "valid", "confidence", "low_confidence", "reasons", "model",
        "decode", "timing_ms"}


@pytest.fixture(scope="module")
def pred():
    model, cfg = _build_small_probe_model("baseline", size=128)
    return OCSRPredictor(model=model, image_size=cfg["data"]["image_size"], int8=False)


def drawing(mode="L", fmt="PNG"):
    img = Image.new(mode, (200, 120), "white")
    for x in range(40, 160):
        img.putpixel((x, 60), (0, 0, 0, 255) if mode == "RGBA" else 0)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return img, buf.getvalue()


def test_bytes_path_and_pil_agree(pred, tmp_path):
    img, data = drawing()
    path = tmp_path / "x.png"
    path.write_bytes(data)
    out = [pred.predict(x) for x in (data, path, str(path), img)]
    assert set(out[0]) == KEYS
    assert len({o["smiles"] for o in out}) == 1
    assert pred.predict(data)["confidence"] == out[0]["confidence"]  # deterministic


def test_int8_decoder_runs():
    model, cfg = _build_small_probe_model("baseline", size=128)
    res = OCSRPredictor(model=model, image_size=128, int8=True).predict(drawing()[1])
    assert set(res) == KEYS


def test_rgba_and_jpeg_inputs(pred):
    for mode, fmt in (("RGBA", "PNG"), ("L", "JPEG"), ("RGB", "JPEG")):
        res = pred.predict(drawing(mode, fmt)[1])
        assert 0.0 <= res["confidence"] <= 1.0


def test_low_confidence_flag_matches_reasons(pred):
    res = pred.predict(drawing()[1])
    assert res["low_confidence"] == bool(res["reasons"])
    if not res["valid"]:
        assert "invalid_smiles" in res["reasons"] and res["canonical_smiles"] is None


@pytest.mark.parametrize("bad", [b"", b"not an image", 123])
def test_bad_inputs_are_rejected(pred, bad):
    with pytest.raises(InputError):
        pred.predict(bad)


def test_oversized_bytes_are_rejected(pred, monkeypatch):
    monkeypatch.setattr("ouroboros.infer.MAX_BYTES", 10)
    with pytest.raises(InputError, match="larger"):
        pred.predict(drawing()[1])


def test_fastapi_app(pred):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ouroboros.infer import create_app

    client = TestClient(create_app(pred))
    assert client.post("/predict", content=drawing()[1]).json().keys() == KEYS
    assert client.post("/predict", content=b"junk").status_code == 422


@pytest.mark.skipif(not DEFAULT_CKPT.exists(), reason="release weights not downloaded")
def test_release_checkpoint_reads_a_clean_drawing_quickly():
    from rdkit import Chem

    from ouroboros.data.render import RenderStyle, render

    p = OCSRPredictor(DEFAULT_CKPT)
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    img = render(mol, RenderStyle(), apply_postprocess=False).image
    t = time.perf_counter()
    res = p.predict(img)
    assert time.perf_counter() - t < 5.0  # loose bound for shared runners; see bench_infer.py
    assert res["valid"]
