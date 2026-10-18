"""Numerical backend for the rotation notebook and its browser UI.

Feature equivariance is measured on *all* channels using escnn's group action, including
the orientation-channel permutation. Off-group rotations deliberately have no such metric.
The first decoder step is a separate rotation-sensitivity probe, not an accuracy score.
"""

from __future__ import annotations

import base64
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem

from ouroboros.data.loader import rotate_images
from ouroboros.data.render import RenderStyle, encode_png, render
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.eval.metrics import score_pair
from ouroboros.eval.standardize import PARITY_FOOTNOTE, parse
from ouroboros.model import OCSRModel, build_model, load_model_state

EXAMPLES = {
    "Lactic acid (one stereocenter)": "C[C@@H](O)C(=O)O",
    "Alanine (one stereocenter)": "N[C@@H](C)C(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "E-2-butene (double-bond stereo)": "C/C=C/C",
}
DEFAULT_SMILES = next(iter(EXAMPLES.values()))


def png_url(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-12))


def tiny_config(order: int = 4) -> dict:
    if order not in (4, 8, 16):
        raise ValueError("Choose C4, C8 or C16.")
    return {
        "data": {"image_size": 128},
        "model": {
            "encoder": {
                "type": "steerable",
                "N": order,
                "fields": [4, 4, 8, 8],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 4,
                "mixer_ff": 128,
                "dropout": 0.0,
            },
            "decoder": {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "d_ff": 128,
                "max_len": 128,
                "dropout": 0.0,
            },
        },
    }


def _heatmap(values: torch.Tensor, scale: float) -> str:
    a = (values.detach().cpu().numpy() / max(scale, 1e-12)).clip(0, 1)
    # White -> teal. The error image uses the feature amplitude, NOT its own maximum.
    rgb = 255 - a[..., None] * np.array([237, 137, 143])
    image = Image.fromarray(np.rint(rgb).astype(np.uint8)).resize((192, 192), Image.NEAREST)
    return png_url(image)


class RotationLab:
    def __init__(self, model: OCSRModel, config: dict, label: str, checkpoint: bool):
        self.model = model.eval()
        self.config = config
        self.label = label
        self.has_checkpoint = checkpoint
        self.size = int(config["data"]["image_size"])
        enc = config["model"]["encoder"]
        self.order = (
            int(enc.get("N", 8)) if enc["type"] in ("steerable", "equiv_attention") else None
        )
        self.device = next(model.parameters()).device
        self._source = None
        self._reference = None

    @classmethod
    def random(cls, order: int = 4) -> RotationLab:
        cfg = tiny_config(order)
        # Preserve the notebook's random state. No training or labels are used to set weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = build_model(cfg, SmilesTokenizer.build(EXAMPLES.values()))
        return cls(model, cfg, f"Random C{order} · seed 0 · CPU", False)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> RotationLab:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Checkpoint not found: {path}")
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Choose CPU.")
        # Training checkpoints include Python/NumPy RNG state: only load trusted local files.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or not {"model", "config", "vocab"} <= state.keys():
            raise ValueError("Use a training checkpoint containing model, config and vocab.")
        tok = SmilesTokenizer(state["vocab"])
        if tok.itos != state["vocab"]:
            raise ValueError("Checkpoint vocabulary has an invalid special-token order.")
        model = build_model(state["config"], tok)
        load_model_state(model, state["model"])
        model.to(device).eval()
        label = f"{path.name} · step {state.get('step', 'unknown')} · {device.upper()}"
        return cls(model, state["config"], label, True)

    def _prediction(self, smiles: str, reference: str) -> dict:
        mol = parse(smiles)
        return {
            "smiles": smiles,
            "image": png_url(render(mol, RenderStyle(), size=320).image) if mol else None,
            "score": asdict(score_pair(smiles, reference)),
        }

    @torch.inference_mode()
    def _prepare(self, smiles: str) -> dict:
        if self._source == smiles:
            return self._reference
        mol = parse(smiles)
        if mol is None:
            raise ValueError("Enter a valid, non-empty reference SMILES.")
        drawing = render(mol, RenderStyle(), size=self.size)
        ink = 1.0 - torch.from_numpy(np.array(drawing.image, dtype=np.float32)) / 255.0
        x = ink[None, None].to(self.device)
        enc = self.model.encoder
        tokens = enc(x).tokens
        ids = torch.tensor([[self.model.tokenizer.bos_id]], device=self.device)
        ref = {
            "x": x,
            "label": drawing.label,
            "image": png_url(drawing.image),
            "truth_image": png_url(render(mol, RenderStyle(), size=320).image),
            "features": enc.features(x) if self.order else None,
            "ids": ids,
            "logits": self.model.decoder(ids, tokens),
            "prediction": self._prediction(self.model.generate(x)[0], drawing.label)
            if self.has_checkpoint
            else None,
        }
        self._source, self._reference = smiles, ref
        return ref

    @torch.inference_mode()
    def evaluate(self, smiles: str = DEFAULT_SMILES, angle: float = 0.0) -> dict:
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError("Angle must be finite.")
        angle %= 360.0
        ref = self._prepare(smiles.strip())
        rotated = rotate_images(ref["x"], torch.tensor([angle]))
        enc = self.model.encoder
        logits = self.model.decoder(ref["ids"], enc(rotated).tokens)
        group_step = angle * self.order / 360.0 if self.order else None
        on_group = group_step is not None and abs(group_step - round(group_step)) < 1e-7
        on_grid = abs(angle / 90 - round(angle / 90)) < 1e-7
        error, maps = None, None
        if on_group:
            element = enc.gspace.fibergroup.element(round(group_step) % self.order)
            expected = ref["features"].transform(element).tensor
            actual = enc.features(rotated).tensor
            error = relative_error(actual, expected)
            # Visualization averages magnitudes across channels; the metric above does not.
            expected_map = expected[0].abs().mean(0)
            actual_map = actual[0].abs().mean(0)
            diff_map = (actual - expected)[0].abs().mean(0)
            scale = float(torch.maximum(expected_map.max(), actual_map.max()))
            maps = {
                "expected": _heatmap(expected_map, scale),
                "actual": _heatmap(actual_map, scale),
                "difference": _heatmap(diff_map, scale),
            }
        if not self.order:
            note = "This encoder has no C_N feature action; equivariance error is not defined."
        elif not on_group:
            note = (
                f"{angle:g}° is outside C{self.order}. No group-equivariance error is defined; "
                "the decoder change below measures rotation sensitivity only."
            )
        elif on_grid:
            note = "Pixel-exact rotation. Numerical errors should be small in float32 eval mode."
        else:
            note = "Group angle with pixel interpolation. Approximate equivariance; no exact bound."
        prediction = None
        stable = None
        if self.has_checkpoint:
            prediction = (
                ref["prediction"]
                if angle == 0
                else self._prediction(self.model.generate(rotated)[0], ref["label"])
            )
            original_mol = parse(ref["prediction"]["smiles"])
            rotated_mol = parse(prediction["smiles"])
            if original_mol is not None and rotated_mol is not None:
                stable = Chem.MolToSmiles(original_mol) == Chem.MolToSmiles(rotated_mol)
        pixels = np.rint(255 * (1 - rotated[0, 0].cpu().numpy())).clip(0, 255).astype(np.uint8)
        return {
            "angle": angle,
            "label": self.label,
            "order": self.order,
            "has_checkpoint": self.has_checkpoint,
            "image_size": self.size,
            "reference": ref["label"],
            "original_image": ref["image"],
            "rotated_image": png_url(Image.fromarray(pixels)),
            "truth_image": ref["truth_image"],
            "on_group": on_group,
            "on_grid": on_grid,
            "note": note,
            "feature_error": error,
            "feature_maps": maps,
            "logit_error": relative_error(logits, ref["logits"]),
            "original_prediction": ref["prediction"],
            "rotated_prediction": prediction,
            "prediction_stable": stable,
            "scoring_note": PARITY_FOOTNOTE,
        }
