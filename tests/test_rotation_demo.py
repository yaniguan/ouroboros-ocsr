"""Exercise actual rotation, checkpoint restoration, scoring and the notebook API contract."""

import importlib.util
import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ouroboros.demo.__main__ import LabHandler, LabServer
from ouroboros.demo.rotation import DEFAULT_SMILES, RotationLab, tiny_config
from ouroboros.model import build_model


@pytest.fixture(scope="module")
def lab():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield RotationLab.random(4)
    torch.set_num_threads(threads)


def test_rotation_action_and_off_group(lab):
    for angle in (0, 90, 180, 270, 360):
        result = lab.evaluate(angle=angle)
        assert result["feature_error"] < 1e-4
        assert result["logit_error"] < 1e-4
        assert result["on_group"] and result["on_grid"]
        assert result["original_prediction"] is None
        assert not result["has_checkpoint"]
    rotated = lab.evaluate(angle=90)
    assert rotated["original_image"] != rotated["rotated_image"]
    off_group = lab.evaluate(angle=30)
    assert off_group["feature_error"] is None
    assert off_group["feature_maps"] is None
    assert not off_group["on_group"]
    assert off_group["logit_error"] > 1e-4
    assert "outside C4" in off_group["note"]


def test_interpolated_group_angle():
    result = RotationLab.random(8).evaluate(angle=45)
    assert result["on_group"] and not result["on_grid"]
    assert result["feature_error"] is not None
    assert "interpolation" in result["note"]


def test_reference_cache_and_invalid_input(lab):
    lab.evaluate(DEFAULT_SMILES)
    before = lab._reference
    lab.evaluate(DEFAULT_SMILES, 90)
    assert lab._reference is before
    with pytest.raises(ValueError, match="valid, non-empty"):
        lab.evaluate("not a smiles")
    assert lab._reference is before
    result = lab.evaluate("CCO", 180)
    assert result["reference"] == "CCO"
    assert lab._reference is not before
    with pytest.raises(ValueError, match="finite"):
        lab.evaluate(angle=float("nan"))


def test_checkpoint_uses_embedded_config_and_vocab(tmp_path, lab):
    # Real state_dict restoration + real greedy decoding, with no external dataset or vocab.
    cfg = tiny_config(4)
    cfg["data"]["vocab"] = "/missing/training/vocab.json"
    cfg["model"]["decoder"]["max_len"] = 8
    model = build_model(cfg, lab.model.tokenizer).eval()
    path = tmp_path / "demo.pt"
    torch.save(
        {"model": model.state_dict(), "config": cfg, "vocab": model.tokenizer.itos, "step": 7}, path
    )
    restored = RotationLab.from_checkpoint(path)
    result = restored.evaluate(angle=90)
    assert result["has_checkpoint"]
    assert "step 7" in result["label"]
    assert result["feature_error"] < 1e-4
    assert result["original_prediction"]["smiles"] == model.generate(restored._reference["x"])[0]
    assert result["rotated_prediction"] is not None
    bad = tmp_path / "weights-only.pt"
    torch.save(model.state_dict(), bad)
    with pytest.raises(ValueError, match="model, config and vocab"):
        RotationLab.from_checkpoint(bad)


def test_stereo_scoring_and_invalid_predictions(lab):
    correct = lab._prediction("C[C@@H](O)C(=O)O", DEFAULT_SMILES)
    wrong_stereo = lab._prediction("C[C@H](O)C(=O)O", DEFAULT_SMILES)
    invalid = lab._prediction("", DEFAULT_SMILES)
    assert correct["score"]["exact"] and correct["image"]
    assert not wrong_stereo["score"]["exact"]
    assert wrong_stereo["score"]["exact_flat"]
    assert wrong_stereo["score"]["stereo_correct"] == 0
    assert not invalid["score"]["valid"] and invalid["image"] is None


def test_local_api_recovers_from_errors(lab):
    try:
        server = LabServer(0, lab)
    except PermissionError:
        pytest.skip("This sandbox forbids loopback listeners; in-memory HTTP tests still run.")
    with server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def post(route, payload):
            request = urllib.request.Request(
                server.url + route,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            return urllib.request.urlopen(request, timeout=30)

        try:
            with urllib.request.urlopen(server.url, timeout=10) as response:
                assert b"Turn the image" in response.read()
            with post("evaluate", {"angle": 90}) as response:
                assert json.load(response)["feature_error"] < 1e-4
            with pytest.raises(urllib.error.HTTPError) as error:
                post("model", {"checkpoint": "/missing/checkpoint.pt"})
            assert error.value.code == 400
            assert server.lab is lab
            with post("evaluate", {"angle": 30}) as response:
                assert json.load(response)["feature_error"] is None
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(server.url.replace(server.prefix, "/"), timeout=10)
            assert error.value.code == 404
        finally:
            server.shutdown()
            thread.join(timeout=10)


def test_http_handler_without_network(lab):
    """Run the real HTTP parser/handler and real model even in no-listen sandboxes."""
    server = SimpleNamespace(prefix="/test-session/", lab=lab, lock=threading.Lock())

    class Connection:
        def __init__(self, request):
            self.incoming = io.BytesIO(request)
            self.outgoing = io.BytesIO()

        def makefile(self, *_args):
            return self.incoming

        def sendall(self, data):
            self.outgoing.write(data)

    def request(path, data=None, content_type="application/json"):
        body = json.dumps(data).encode() if data is not None else b""
        method = "POST" if data is not None else "GET"
        header = (
            f"{method} {path} HTTP/1.0\r\nHost: localhost\r\n"
            f"Content-Type: {content_type}\r\nContent-Length: {len(body)}\r\n\r\n"
        )
        connection = Connection(header.encode() + body)
        LabHandler(connection, ("127.0.0.1", 1234), server)
        headers, response = connection.outgoing.getvalue().split(b"\r\n\r\n", 1)
        return int(headers.split()[1]), response

    status, response = request("/test-session/evaluate", {"angle": 90})
    assert status == 200 and json.loads(response)["feature_error"] < 1e-4
    status, _ = request("/test-session/model", {"checkpoint": "/missing.pt"})
    assert status == 400 and server.lab is lab
    assert request("/test-session/evaluate", {"smiles": "bad"})[0] == 400
    assert request("/test-session/evaluate", {"angle": 0}, "text/plain")[0] == 415
    assert request("/wrong-session/evaluate", {"angle": 0})[0] == 404
    status, response = request("/test-session/evaluate", {"angle": 30})
    assert status == 200 and json.loads(response)["feature_error"] is None
    assert request("/test-session/health")[0] == 200
    status, response = request("/test-session/")
    assert status == 200 and b"Turn the image" in response


def test_notebook_is_reproducible_and_has_no_saved_outputs():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "demo_notebook", root / "scripts/make_rotation_demo_nb.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = json.loads((root / "notebooks/05_rotation_demo.ipynb").read_text())
    assert notebook == module.notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), "rotation_demo_notebook", "exec")
