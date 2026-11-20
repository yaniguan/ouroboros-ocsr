"""Fast offline tests for lab/grpo_verifier.py (batched OCSR verifier & GRPO reward hook)."""

from __future__ import annotations

import io

from PIL import Image

from lab import grpo_verifier as gv
from ouroboros.ui.backend import _build_small_probe_model


def _sample_png(shade: int = 240) -> bytes:
    img = Image.new("L", (96, 96), shade)
    for x in range(20, 76):
        img.putpixel((x, 48), 10)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_batched_verifier_preserves_order_and_flags_bad_rows():
    model, cfg = _build_small_probe_model("baseline", size=64)
    verify = gv.make_verifier(
        model=model,
        image_size=cfg["data"]["image_size"],
        threshold=0.99,
        max_batch_size=2,
    )
    inputs = [_sample_png(240), b"not-an-image", _sample_png(230)]
    out = verify(inputs)
    assert len(out) == 3
    assert [r["id"] for r in out] == ["img_0000", "img_0001", "img_0002"]
    assert out[1]["status"] == "invalid"
    assert out[1]["flagged"] is True
    assert any("input_error" in reason for reason in out[1]["reasons"])
    # High threshold flags untrusted reads rather than dropping them
    assert out[0]["flagged"] is True
    assert out[2]["flagged"] is True


def test_reward_batch_and_benchmark_helper():
    model, cfg = _build_small_probe_model("baseline", size=64)
    verifier = gv.OCSRVerifier(
        model=model,
        image_size=cfg["data"]["image_size"],
        threshold=0.0,
        max_batch_size=2,
    )
    # Inject a deterministic clean decode by monkeypatching verifier.verify_batch
    verifier.verify_batch = lambda imgs, **kw: [  # type: ignore[method-assign]
        {
            "id": "a",
            "smiles": "CCO",
            "pred_raw": "CCO",
            "confidence": 0.99,
            "status": "ok",
            "flagged": False,
            "usable": True,
            "reasons": [],
        },
        {
            "id": "b",
            "smiles": None,
            "pred_raw": "",
            "confidence": 0.2,
            "status": "rejected",
            "flagged": True,
            "usable": False,
            "reasons": ["low_token_probability"],
        },
    ]
    rewards = verifier.reward_batch([_sample_png(), _sample_png()], ["OCC", "CCO"])
    assert rewards[0]["matched"] is True and rewards[0]["reward"] == 1.0
    assert rewards[1]["abstained"] is True and rewards[1]["reward"] == 0.0

    # Benchmark helper runs on real model
    v2 = gv.OCSRVerifier(model=model, image_size=cfg["data"]["image_size"], max_batch_size=4)
    bench = gv.benchmark_batch_sizes(v2, batch_sizes=[1, 2], repeats=1)
    assert [b["batch_size"] for b in bench] == [1, 2]
    assert all(b["img_per_s"] > 0 for b in bench)
