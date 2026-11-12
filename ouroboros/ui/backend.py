"""Unified Backend Engine for the Ouroboros Interactive Workbench & Physics Compliance UI.

Powers all six interactive workbenches:
  1. Visual OCSR & Document Perturbation Studio (A/B/C/AC/ABC, beam, TTA, confidence gate)
  2. 2D C_N Steerable Equivariance & Chiral Reflection Audit (steerable, equiv_attention,
     canonicalized, baseline, plus pretrained rotation invariance & mirror parity)
  3. 3D Equivariant Molecule Generation & 10-Point Physical Compliance Inspector
  4. OCSR Error Propagation & Thermodynamic Disambiguation Lab
  5. Force-Field Dataset Builder, Non-Equilibrium Sampler & Scaffold Splitter
  6. Live Batch Benchmark, Selective Abstention & Repository Receipt Explorer
"""

from __future__ import annotations

import base64
import io
import json
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from rdkit import Chem

from ouroboros.data.augment import handdrawn_augment, scan_augment
from ouroboros.data.loader import rotate_images
from ouroboros.data.real import fit_to_canvas
from ouroboros.data.render import RenderStyle, encode_png, render
from ouroboros.data.rgroups import ABBREVIATIONS, expand_aliases, substitute
from ouroboros.decode.confidence import confidence, selective_report
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.demo.rotation import EXAMPLES, _heatmap, relative_error
from ouroboros.eval.metrics import aggregate, bootstrap_ci, score_pair
from ouroboros.eval.standardize import standard_inchikey, standardize
from ouroboros.geometry.conformers import embed, mirror_positions, relax
from ouroboros.geometry.displacements import displaced_conformers
from ouroboros.geometry.eligibility import EligibilityPolicy
from ouroboros.geometry.propagation import categorize, mirror_smiles, same_formula
from ouroboros.geometry.splits import assign_split, connectivity_key
from ouroboros.labels import labelled_smiles, parse_labelled
from ouroboros.model import OCSRModel, build_model, load_model_state
from ouroboros.ui.compliance import (
    EV_TO_KCAL,
    MACE_ELEMENTS,
    PRESET_MOLECULES,
    audit_3d_physical_compliance,
    get_3d_calculator,
    png_data_url,
)
from scripts.predict_real import predict_batch

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"
BENCH_DIR = ROOT / "benchmarks"


def _pil_to_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(img)).decode("ascii")


def _decode_data_url_image(data_url: str) -> Image.Image:
    raw = data_url.strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    blob = base64.b64decode(raw)
    with Image.open(io.BytesIO(blob)) as im:
        return im.convert("L")


def _build_arch_probe_model(arch: str, order: int = 8, size: int = 128) -> tuple[OCSRModel, dict]:
    # Ensure encoder registries are imported
    import ouroboros.encoder.baseline  # noqa: F401
    import ouroboros.encoder.canonicalize  # noqa: F401
    import ouroboros.encoder.equiv_attention  # noqa: F401
    import ouroboros.encoder.steerable  # noqa: F401

    if arch == "steerable":
        enc_cfg: dict[str, Any] = {
            "type": "steerable",
            "N": order,
            "fields": [4, 4, 8, 8],
            "blocks": [1, 1, 1, 1],
            "mixer_layers": 1,
            "mixer_heads": 4,
            "mixer_ff": 128,
            "dropout": 0.0,
        }
    elif arch == "equiv_attention":
        enc_cfg = {
            "type": "equiv_attention",
            "N": order,
            "fields": [4, 4, 8, 8],
            "blocks": [1, 1, 1, 1],
            "attn_layers": 1,
            "attn_heads": 4,
            "attn_ff": 128,
            "dropout": 0.0,
        }
    elif arch == "canonicalized":
        enc_cfg = {
            "type": "canonicalized",
            "N": order,
            "orient_fields": [4, 6, 8],
            "downsample": 2,
            "base": {
                "widths": [16, 32, 64, 64],
                "blocks": [1, 1, 1, 1],
                "mixer_layers": 1,
                "mixer_heads": 4,
                "mixer_ff": 128,
                "dropout": 0.0,
            },
        }
    elif arch == "baseline":
        enc_cfg = {
            "type": "baseline",
            "widths": [16, 32, 64, 64],
            "blocks": [1, 1, 1, 1],
            "mixer_layers": 1,
            "mixer_heads": 4,
            "mixer_ff": 128,
            "dropout": 0.0,
        }
    else:
        raise ValueError(f"Unsupported encoder architecture: {arch!r}")

    cfg = {
        "data": {"image_size": size},
        "model": {
            "encoder": enc_cfg,
            "decoder": {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "d_ff": 128,
                "max_len": 96,
                "dropout": 0.0,
            },
        },
    }
    tok = SmilesTokenizer.build(EXAMPLES.values())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_model(cfg, tok)
        model.train()
        # Populate running BatchNorm statistics deterministically so eval mode is non-trivial
        with torch.no_grad():
            for _ in range(2):
                model.encoder(torch.rand(2, 1, size, size))
        model.eval()
    return model, cfg


class WorkbenchEngine:
    """Thread-safe state and caching layer for all Ouroboros UI tabs."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.lock = threading.Lock()
        self._ckpt_exported: dict[str, tuple[OCSRModel, dict]] = {}
        self._ckpt_escnn: dict[str, tuple[OCSRModel, dict]] = {}
        self._arch_probes: dict[tuple[str, int], tuple[OCSRModel, dict]] = {}

    def available_checkpoints(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for key, fname in (
            ("A", "ouroboros-A.pt"),
            ("B", "ouroboros-B.pt"),
            ("C", "ouroboros-C.pt"),
        ):
            p = CKPT_DIR / fname
            if p.is_file():
                found[key] = str(p)
        return found

    def _get_ckpt_model(
        self, key_or_path: str, *, export_fast: bool = True
    ) -> tuple[OCSRModel, dict]:
        avail = self.available_checkpoints()
        resolved = Path(avail[key_or_path]) if key_or_path in avail else Path(key_or_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        cache = self._ckpt_exported if export_fast else self._ckpt_escnn
        ck = str(resolved)
        if ck not in cache:
            state = torch.load(resolved, map_location="cpu", weights_only=False)
            tok = SmilesTokenizer(state["vocab"])
            model = build_model(state["config"], tok)
            load_model_state(model, state["model"])
            model.to(self.device).eval()
            if export_fast and hasattr(model.encoder, "export"):
                model.encoder = model.encoder.export().to(self.device)
            cache[ck] = (model, state["config"])
        return cache[ck]

    def _get_arch_probe(self, arch: str, order: int) -> tuple[OCSRModel, dict]:
        key = (arch, int(order))
        if key not in self._arch_probes:
            self._arch_probes[key] = _build_arch_probe_model(arch, int(order), size=128)
        return self._arch_probes[key]

    def status(self) -> dict[str, Any]:
        ckpts = self.available_checkpoints()
        model_choices = []
        if "C" in ckpts:
            model_choices.append({"id": "C", "label": "Ouroboros-C (Pretrained C8 Equivariant)"})
        if "A" in ckpts and "C" in ckpts:
            model_choices.append({"id": "AC", "label": "Ouroboros-AC (Pretrained A+C Ensemble)"})
        if {"A", "B", "C"} <= ckpts.keys():
            model_choices.append({"id": "ABC", "label": "Ouroboros-ABC (Full 3-Arm Ensemble)"})
        if "B" in ckpts:
            model_choices.append({"id": "B", "label": "Ouroboros-B (Rotation-Augmented ResNet)"})
        if "A" in ckpts:
            model_choices.append({"id": "A", "label": "Ouroboros-A (Baseline ResNet)"})
        model_choices.append(
            {"id": "demo_c8", "label": "Untrained C8 Steerable Demo (Fast Fallback)"}
        )

        bench_files = sorted(
            str(p.relative_to(BENCH_DIR)).replace("\\", "/")
            for p in BENCH_DIR.rglob("*.json")
            if p.stat().st_size < 250_000
        )[:40]

        return {
            "ready": True,
            "device": str(self.device),
            "checkpoints_found": list(ckpts.keys()),
            "ocsr_models": model_choices,
            "encoders_2d": [
                {"id": "steerable", "label": "Arm C: C_N Steerable CNN + GroupPool + RelDist"},
                {"id": "equiv_attention", "label": "Arm D: C_N Steerable + Group Equivariant Attn"},
                {"id": "canonicalized", "label": "Arm E: Learned C_N Canonicalizer + ResNet"},
                {"id": "baseline", "label": "Arm A: Standard Non-Equivariant ResNet"},
            ],
            "potentials_3d": [
                {"id": "mmff94s", "label": "RDKit MMFF94s (Instant Analytical Conservative)"},
                {"id": "mace-small", "label": "MACE-OFF23-small (E(3) Equivariant MPNN)"},
                {"id": "mace-medium", "label": "MACE-OFF23-medium (E(3) Equivariant MPNN)"},
                {"id": "mace-large", "label": "MACE-OFF23-large (E(3) Equivariant MPNN)"},
                {"id": "ani2x", "label": "TorchANI ANI-2x (8-Model Neural Ensemble)"},
            ],
            "presets": PRESET_MOLECULES,
            "abbreviations": ABBREVIATIONS,
            "repo_benchmarks": bench_files,
        }

    # ------------------------------------------------------------------ Tab 1: Visual OCSR Studio
    def run_ocsr(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        model_choice = str(payload.get("model", "AC"))
        beam = max(1, min(8, int(payload.get("beam", 4))))
        tta_list = [t for t in payload.get("tta", []) if t in ("pad", "sharpen")]
        min_margin = float(payload.get("min_margin", 0.20))
        max_disagree = payload.get("max_disagreement")
        if max_disagree is not None:
            max_disagree = float(max_disagree)
        angle_deg = float(payload.get("angle_deg", 0.0)) % 360.0
        mirror_flip = bool(payload.get("mirror_flip", False))
        augment_mode = str(payload.get("augment_mode", "none"))
        rgroup_label = str(payload.get("rgroup_label", "")).strip() or None
        expand_abbrev = bool(payload.get("expand_aliases", True))
        ink_crop = bool(payload.get("ink_crop", False))
        ref_smiles = str(payload.get("reference_smiles", "")).strip()

        # Resolve model(s)
        ckpts = self.available_checkpoints()
        models: list[OCSRModel] = []
        img_size = 384
        model_label = model_choice
        if model_choice in ("A", "B", "C", "AC", "ABC") and all(c in ckpts for c in model_choice):
            for ch in model_choice:
                m, cfg = self._get_ckpt_model(ch, export_fast=True)
                models.append(m)
                img_size = int(cfg["data"]["image_size"])
            model_label = f"Ouroboros-{model_choice} (Pretrained)"
        elif payload.get("custom_checkpoint"):
            m, cfg = self._get_ckpt_model(str(payload["custom_checkpoint"]), export_fast=True)
            models = [m]
            img_size = int(cfg["data"]["image_size"])
            model_label = f"Custom ({Path(str(payload['custom_checkpoint'])).name})"
        else:
            m, cfg = self._get_arch_probe("steerable", 8)
            models = [m]
            img_size = int(cfg["data"]["image_size"])
            model_label = "Untrained C8 Demo Fallback"

        # Build or decode input image
        if payload.get("image_data_url"):
            raw_pil = _decode_data_url_image(str(payload["image_data_url"]))
            base_pil = fit_to_canvas(raw_pil, img_size, ink_crop=ink_crop)
        else:
            src_smi = str(payload.get("smiles") or ref_smiles or "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1")
            mol = Chem.MolFromSmiles(src_smi)
            if mol is None:
                raise ValueError(f"Invalid input SMILES: {src_smi!r}")
            if rgroup_label:
                mol = substitute(mol, random.Random(0), label=rgroup_label)
                ref_smiles = labelled_smiles(mol)
            elif not ref_smiles:
                ref_smiles = src_smi
            base_pil = render(mol, RenderStyle(), size=img_size, seed=0).image

        proc_pil = base_pil.copy()
        rng = np.random.default_rng(int(payload.get("seed", 0)))
        if augment_mode == "scan":
            proc_pil = scan_augment(proc_pil, rng, affine=0.03, scan_noise=5.0, morphology=1)
        elif augment_mode == "handdrawn":
            proc_pil = handdrawn_augment(proc_pil, rng, wobble=1.8, paper_noise=7.0)
        if mirror_flip:
            proc_pil = ImageOps.mirror(proc_pil)

        ink_np = 1.0 - np.asarray(proc_pil, dtype=np.float32) / 255.0
        x = torch.from_numpy(ink_np)[None, None].to(self.device)
        if abs(angle_deg) > 1e-7:
            x = rotate_images(x, torch.tensor([angle_deg], device=self.device))
            rot_arr = (
                np.rint(255.0 * (1.0 - x[0, 0].detach().cpu().numpy()))
                .clip(0, 255)
                .astype(np.uint8)
            )
            proc_pil = Image.fromarray(rot_arr, mode="L")

        diag_sources: list[list[str]] = []
        with torch.inference_mode():
            [(pred_smi, ranked)] = predict_batch(
                models, x, beam=beam, tta=tta_list, prefer_valid=True, diagnostics=diag_sources
            )

        src_preds = diag_sources[0] if diag_sources else []
        conf_gate = confidence(
            pred_smi,
            ranked,
            src_preds,
            min_margin=min_margin,
            max_disagreement=max_disagree,
        )

        expanded_smi = None
        pred_labelled = parse_labelled(pred_smi) if expand_abbrev else None
        if expand_abbrev and (
            "*" in pred_smi
            or (
                pred_labelled is not None
                and any(a.GetAtomicNum() == 0 for a in pred_labelled.GetAtoms())
            )
        ):
            try:
                expanded_smi = Chem.MolToSmiles(expand_aliases(pred_smi))
            except ValueError:
                expanded_smi = None

        ref_valid = bool(ref_smiles and standardize(ref_smiles).valid)
        candidates_out = []
        for rank_idx, (cand_smi, cand_score) in enumerate(ranked[:beam], start=1):
            std_c = standardize(cand_smi)
            cand_row: dict[str, Any] = {
                "rank": rank_idx,
                "smiles": cand_smi,
                "standardized_smiles": std_c.smiles if std_c.valid else None,
                "score": round(float(cand_score), 4),
                "valid": std_c.valid,
                "depiction": png_data_url(std_c.mol, size=220) if std_c.valid else None,
            }
            if ref_valid:
                ps = asdict(score_pair(cand_smi, ref_smiles))
                cand_row["score_vs_ref"] = ps
                cand_row["category_vs_ref"] = categorize(cand_smi, ref_smiles)
            candidates_out.append(cand_row)

        eval_summary = None
        if ref_valid:
            eval_target = expanded_smi or pred_smi
            eval_summary = {
                "reference_smiles": standardize(ref_smiles).smiles,
                "reference_depiction": png_data_url(ref_smiles, size=260),
                "category": categorize(eval_target, ref_smiles),
                "pair_score": asdict(score_pair(eval_target, ref_smiles)),
            }

        std_best = standardize(expanded_smi or pred_smi)
        return {
            "model_label": model_label,
            "beam": beam,
            "tta": tta_list,
            "angle_deg": angle_deg,
            "mirror_flip": mirror_flip,
            "augment_mode": augment_mode,
            "source_image": _pil_to_data_url(base_pil),
            "processed_image": _pil_to_data_url(proc_pil),
            "predicted_smiles": pred_smi,
            "expanded_smiles": expanded_smi,
            "standardized_smiles": std_best.smiles if std_best.valid else None,
            "predicted_depiction": png_data_url(std_best.mol, size=260) if std_best.valid else None,
            "confidence_gate": conf_gate,
            "source_predictions": src_preds,
            "candidates": candidates_out,
            "evaluation": eval_summary,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    # ------------------------------------------------- Tab 2: 2D C_N Equivariance & Chirality Audit
    def run_2d_equivariance(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        smiles = str(payload.get("smiles") or "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1").strip()
        mode = str(payload.get("mode", "arch"))  # "arch" | "checkpoint"
        arch = str(payload.get("arch", "steerable"))
        order = int(payload.get("order", 8))
        ckpt_id = str(payload.get("checkpoint_id", "C"))
        focus_angle = float(payload.get("angle_deg", 90.0)) % 360.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES for 2D equivariance audit: {smiles!r}")

        has_ckpt = False
        if mode == "checkpoint" and ckpt_id in self.available_checkpoints():
            model, cfg = self._get_ckpt_model(ckpt_id, export_fast=False)
            has_ckpt = True
            label = f"Pretrained Ouroboros-{ckpt_id} ({cfg['model']['encoder']['type']})"
        else:
            model, cfg = self._get_arch_probe(arch, order)
            label = f"Probe {arch} (C{order} if equivariant, size={cfg['data']['image_size']})"

        size = int(cfg["data"]["image_size"])
        enc = model.encoder
        enc_cfg = cfg["model"]["encoder"]
        enc_type = enc_cfg["type"]
        enc_order = (
            int(enc_cfg.get("N", order)) if enc_type in ("steerable", "equiv_attention") else None
        )

        # Locate escnn trunk for exact feature equivariance measurement
        escnn_enc = enc if hasattr(enc, "features") and hasattr(enc, "gspace") else None
        if escnn_enc is None and hasattr(enc, "trunk") and hasattr(enc.trunk, "features"):
            escnn_enc = enc.trunk

        drawing = render(mol, RenderStyle(), size=size, seed=0)
        ink = 1.0 - torch.from_numpy(np.asarray(drawing.image, dtype=np.float32)) / 255.0
        x = ink[None, None].to(self.device)
        bos = torch.tensor([[model.tokenizer.bos_id]], device=self.device)

        with torch.inference_mode():
            tok_ref = enc(x).tokens
            logits_ref = model.decoder(bos, tok_ref)
            feat_ref = escnn_enc.features(x) if escnn_enc is not None else None
            pred_0 = model.generate(x)[0] if has_ckpt else None

            # Evaluate focus_angle with heatmaps
            x_focus = rotate_images(x, torch.tensor([focus_angle], device=self.device))
            tok_focus = enc(x_focus).tokens
            logits_focus = model.decoder(bos, tok_focus)
            group_step = (focus_angle * enc_order / 360.0) if enc_order else None
            on_group = group_step is not None and abs(group_step - round(group_step)) < 1e-7
            focus_feat_err = None
            focus_maps = None
            if on_group and escnn_enc is not None and feat_ref is not None:
                elem = escnn_enc.gspace.fibergroup.element(round(group_step) % enc_order)
                exp_t = feat_ref.transform(elem).tensor
                act_t = escnn_enc.features(x_focus).tensor
                focus_feat_err = relative_error(act_t, exp_t)
                exp_m = exp_t[0].abs().mean(0)
                act_m = act_t[0].abs().mean(0)
                dif_m = (act_t - exp_t)[0].abs().mean(0)
                sc = float(torch.maximum(exp_m.max(), act_m.max()))
                focus_maps = {
                    "expected": _heatmap(exp_m, sc),
                    "actual": _heatmap(act_m, sc),
                    "difference": _heatmap(dif_m, sc),
                }

            # Multi-angle sweep: 90-deg grid + 45-deg off-grid + 30-deg off-group
            sweep_angles = sorted(
                {0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0, round(focus_angle, 2)}
            )
            h_tokens = tok_ref.shape[1]
            side = int(round(h_tokens**0.5))
            is_square_grid = side * side == h_tokens
            idx_grid = (
                torch.arange(h_tokens, device=self.device).view(1, side, side)
                if is_square_grid
                else None
            )

            sweep_rows: list[dict[str, Any]] = []
            for ang in sweep_angles:
                xr = rotate_images(x, torch.tensor([ang], device=self.device))
                tr = enc(xr).tokens
                lr = model.decoder(bos, tr)
                gs = (ang * enc_order / 360.0) if enc_order else None
                og = gs is not None and abs(gs - round(gs)) < 1e-7
                grid90 = abs(ang / 90.0 - round(ang / 90.0)) < 1e-7

                f_err = None
                if og and escnn_enc is not None and feat_ref is not None:
                    el = escnn_enc.gspace.fibergroup.element(round(gs) % enc_order)
                    f_err = relative_error(
                        escnn_enc.features(xr).tensor, feat_ref.transform(el).tensor
                    )

                perm_err = None
                if grid90 and idx_grid is not None:
                    k90 = int(round(ang / 90.0)) % 4
                    perm = torch.rot90(idx_grid, k90, dims=(-2, -1)).reshape(-1)
                    perm_err = relative_error(tr, tok_ref[:, perm])

                mean_tok_err = relative_error(tr.mean(dim=1), tok_ref.mean(dim=1))
                logit_err = relative_error(lr, logits_ref)

                pred_ang = None
                same_as_0 = None
                if has_ckpt:
                    pred_ang = pred_0 if ang == 0.0 else model.generate(xr)[0]
                    s0, sa = standardize(pred_0), standardize(pred_ang)
                    if s0.valid and sa.valid:
                        same_as_0 = s0.smiles == sa.smiles

                sweep_rows.append(
                    {
                        "angle_deg": ang,
                        "on_group": og,
                        "on_grid90": grid90,
                        "feature_rel_error": f_err,
                        "token_perm_rel_error": perm_err,
                        "mean_token_rel_error": mean_tok_err,
                        "logit_rel_error": logit_err,
                        "predicted_smiles": pred_ang,
                        "invariant_to_0deg": same_as_0,
                    }
                )

            # Mirror Reflection Non-Invariance Probe (x -> flip horizontal)
            x_mirror = torch.flip(x, dims=[-1])
            tok_mirror = enc(x_mirror).tokens
            mirror_mean_tok_diff = relative_error(tok_mirror.mean(dim=1), tok_ref.mean(dim=1))
            pred_mirror = model.generate(x_mirror)[0] if has_ckpt else None
            mirror_pred_relation = None
            if has_ckpt and pred_0 and pred_mirror and standardize(pred_0).valid:
                mirror_pred_relation = categorize(pred_mirror, pred_0)

        rot_pixels = (
            np.rint(255.0 * (1.0 - x_focus[0, 0].detach().cpu().numpy()))
            .clip(0, 255)
            .astype(np.uint8)
        )
        mir_pixels = (
            np.rint(255.0 * (1.0 - x_mirror[0, 0].detach().cpu().numpy()))
            .clip(0, 255)
            .astype(np.uint8)
        )

        # Compliance summary for 2D symmetry rules
        grid90_errs = [
            r["feature_rel_error"]
            for r in sweep_rows
            if r["on_grid90"] and r["feature_rel_error"] is not None
        ]
        exact_90_ok = bool(grid90_errs and max(grid90_errs) < 1e-4)
        reflection_separated = mirror_mean_tok_diff > 1e-3

        return {
            "label": label,
            "encoder_type": enc_type,
            "group_order": enc_order,
            "has_checkpoint": has_ckpt,
            "reference_smiles": drawing.label,
            "focus_angle_deg": focus_angle,
            "original_image": _pil_to_data_url(drawing.image),
            "rotated_image": _pil_to_data_url(Image.fromarray(rot_pixels, mode="L")),
            "mirrored_image": _pil_to_data_url(Image.fromarray(mir_pixels, mode="L")),
            "focus_feature_error": focus_feat_err,
            "focus_logit_error": relative_error(logits_focus, logits_ref),
            "feature_maps": focus_maps,
            "sweep": sweep_rows,
            "mirror_probe": {
                "mean_token_rel_diff": mirror_mean_tok_diff,
                "reflection_non_invariant_ok": reflection_separated,
                "upright_prediction": pred_0,
                "mirrored_prediction": pred_mirror,
                "prediction_relation": mirror_pred_relation,
                "explanation": (
                    "Ouroboros uses rot2dOnR2 (SO(2)/C_N rotations only, never D_N reflections) "
                    "because mirroring a wedge/hash drawing inverts 3D tetrahedral chirality."
                ),
            },
            "symmetry_verdict": {
                "exact_c4_grid_equivariant": exact_90_ok if enc_order else None,
                "max_grid90_feature_err": max(grid90_errs) if grid90_errs else None,
                "chirality_preserving_no_reflection": reflection_separated,
            },
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    # ------------------------------------------ Tab 3: 3D Generation & 10-Point Physics Compliance
    def run_3d_compliance(self, payload: dict[str, Any]) -> dict[str, Any]:
        return audit_3d_physical_compliance(
            str(payload.get("smiles", "")),
            potential=str(payload.get("potential", "mmff94s")),
            n_conf=max(1, min(12, int(payload.get("n_conf", 4)))),
            do_relax=bool(payload.get("relax", True)),
            fmax=float(payload.get("fmax", 0.05)),
            steps=max(10, min(500, int(payload.get("steps", 120)))),
            expand_abbrev=bool(payload.get("expand_aliases", True)),
            perturb_amplitude_a=max(0.0, min(0.8, float(payload.get("perturb_amplitude_a", 0.0)))),
            inject_steric_clash=bool(payload.get("inject_steric_clash", False)),
            seed=int(payload.get("seed", 0)),
            device=str(self.device),
        )

    # ------------------------------------ Tab 4: Error Propagation & Thermodynamic Disambiguation
    def run_error_propagation(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        truth_raw = str(payload.get("truth_smiles") or "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1").strip()
        potential = str(payload.get("potential", "mmff94s"))
        n_conf = max(1, min(12, int(payload.get("n_conf", 6))))
        do_relax = bool(payload.get("relax", True))

        std_t = standardize(truth_raw)
        if not std_t.valid:
            raise ValueError(f"Truth SMILES is invalid: {truth_raw!r}")

        cand_list = [
            c.strip()
            for c in payload.get("candidate_smiles", [])
            if isinstance(c, str) and c.strip()
        ]
        if not cand_list:
            # Auto-build illuminating stereochemical & constitutional variants!
            cand_list.append(std_t.smiles)
            enan = mirror_smiles(std_t.smiles)
            if Chem.CanonSmiles(enan) != std_t.smiles:
                cand_list.append(enan)
            # Flip a single stereocenter if molecule has >=2 tetrahedral centers -> diastereomer!
            mol_d = Chem.MolFromSmiles(std_t.smiles)
            chiral_atoms = [
                a
                for a in mol_d.GetAtoms()
                if a.GetChiralTag()
                in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            ]
            if len(chiral_atoms) >= 2:
                a0 = chiral_atoms[0]
                a0.SetChiralTag(
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                    if a0.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
                    else Chem.ChiralType.CHI_TETRAHEDRAL_CW
                )
                cand_list.append(Chem.MolToSmiles(mol_d))
            elif "/" in std_t.smiles or "\\" in std_t.smiles:
                flipped_db = (
                    std_t.smiles.replace("/", "__SLASH__")
                    .replace("\\", "/")
                    .replace("__SLASH__", "\\")
                )
                if standardize(flipped_db).valid:
                    cand_list.append(flipped_db)

        truth_audit = audit_3d_physical_compliance(
            std_t.smiles,
            potential=potential,
            n_conf=n_conf,
            do_relax=do_relax,
            steps=100,
            device=str(self.device),
        )
        e_truth_ev = truth_audit["structure_3d"]["energy_ev"] if truth_audit.get("ok") else None

        rows: list[dict[str, Any]] = []
        for idx, csmi in enumerate(cand_list, start=1):
            cat = categorize(csmi, std_t.smiles)
            iso_formula = same_formula(csmi, std_t.smiles) if cat != "invalid" else False
            ps = asdict(score_pair(csmi, std_t.smiles))
            energy_method = "independent_conformer_search"
            if cat == "correct" and truth_audit.get("ok"):
                c_audit = truth_audit
                e_cand_ev = e_truth_ev
                fmax_cand = truth_audit["structure_3d"]["fmax_ev_a"]
                energy_method = "truth_reference"
            elif cat == "enantiomer" and truth_audit.get("ok"):
                from ase import Atoms

                std_c = standardize(csmi)
                t_atoms = truth_audit["structure_3d"]["atoms"]
                syms = [a["symbol"] for a in t_atoms]
                pos_t = np.array([[a["x"], a["y"], a["z_coord"]] for a in t_atoms], dtype=float)
                pos_m = mirror_positions(pos_t)
                mol_tpl = Chem.AddHs(Chem.MolFromSmiles(std_t.smiles))
                calc_m, _, _ = get_3d_calculator(potential, mol_tpl, device=str(self.device))
                at_m = Atoms(symbols=syms, positions=pos_m)
                if do_relax:
                    rm = relax(at_m, calc_m, steps=100)
                    e_cand_ev = round(float(rm.energy), 6)
                    fmax_cand = round(float(rm.fmax), 5)
                else:
                    at_m.calc = calc_m
                    e_cand_ev = round(float(at_m.get_potential_energy()), 6)
                    fmax_cand = round(float(np.linalg.norm(at_m.get_forces(), axis=1).max()), 5)
                c_audit = {
                    "ok": True,
                    "standardized_smiles": std_c.smiles,
                    "depiction_2d": png_data_url(std_c.mol),
                }
                energy_method = "mirrored_truth_geometry"
            else:
                c_audit = (
                    audit_3d_physical_compliance(
                        csmi,
                        potential=potential,
                        n_conf=n_conf,
                        do_relax=do_relax,
                        steps=100,
                        device=str(self.device),
                    )
                    if cat != "invalid"
                    else {"ok": False}
                )
                e_cand_ev = c_audit["structure_3d"]["energy_ev"] if c_audit.get("ok") else None
                fmax_cand = c_audit.get("structure_3d", {}).get("fmax_ev_a")

            de_ev = (
                (e_cand_ev - e_truth_ev)
                if (iso_formula and e_cand_ev is not None and e_truth_ev is not None)
                else None
            )
            rows.append(
                {
                    "rank_input": idx,
                    "smiles": csmi,
                    "standardized_smiles": c_audit.get("standardized_smiles"),
                    "depiction": c_audit.get("depiction_2d"),
                    "category": cat,
                    "same_formula": iso_formula,
                    "pair_score": ps,
                    "compliance_ok": bool(c_audit.get("ok")),
                    "energy_method": energy_method,
                    "energy_ev": e_cand_ev,
                    "delta_e_ev": round(de_ev, 5) if de_ev is not None else None,
                    "delta_e_kcal": round(de_ev * EV_TO_KCAL, 3) if de_ev is not None else None,
                    "fmax_ev_a": fmax_cand,
                }
            )

        # Thermodynamic isomer reranking among formula-matched valid candidates
        isomer_energies = [
            r["energy_ev"] for r in rows if r["same_formula"] and r["energy_ev"] is not None
        ]
        if isomer_energies:
            emin = min(isomer_energies)
            for r in rows:
                if r["same_formula"] and r["energy_ev"] is not None:
                    r["rel_to_lowest_isomer_kcal"] = round((r["energy_ev"] - emin) * EV_TO_KCAL, 3)
                else:
                    r["rel_to_lowest_isomer_kcal"] = None

        return {
            "truth_smiles": std_t.smiles,
            "truth_depiction": truth_audit.get("depiction_2d"),
            "truth_energy_ev": e_truth_ev,
            "potential_label": truth_audit.get("potential_label", potential),
            "candidates": rows,
            "enantiomer_blindness_note": (
                "Enantiomers share identical scalar interatomic distances, so any "
                "E(3)/O(3)-invariant potential assigns exact dE = 0.000 kcal/mol; "
                "diastereomers and strained constitutional errors expose non-zero 3D strain!"
            ),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    # ------------------------------ Tab 5: Force-Field Dataset Builder & Non-Equilibrium Sampler
    def run_forcefield_builder(self, payload: dict[str, Any]) -> dict[str, Any]:
        from ase import Atoms
        from ase.io import write

        t0 = time.perf_counter()
        smiles_list = [
            s.strip() for s in payload.get("smiles_list", []) if isinstance(s, str) and s.strip()
        ]
        if not smiles_list:
            smiles_list = [
                "CNC(=O)C12CC(C1)C2",
                "N[C@@H](C)C(=O)O",
                "[Boc]NCc1ccc(F)cc1",
                "CC(=O)[O-].[Na+]",
            ]
        potential = str(payload.get("potential", "mmff94s"))
        n_conf = max(1, min(6, int(payload.get("n_conf", 2))))
        n_displaced = max(0, min(6, int(payload.get("displaced_samples", 2))))
        amp = max(0.01, min(0.5, float(payload.get("max_displacement", 0.10))))
        min_dist = max(0.4, min(1.5, float(payload.get("min_distance", 0.70))))
        expand_abbrev = bool(payload.get("expand_aliases", True))
        dedupe = bool(payload.get("dedupe_inchikey", True))

        seen_ik: set[str] = set()
        manifest_rows: list[dict[str, Any]] = []
        extxyz_frames: list[Atoms] = []

        policy = EligibilityPolicy(supported_elements=MACE_ELEMENTS)

        for idx, raw_smi in enumerate(smiles_list):
            rec: dict[str, Any] = {
                "id": f"mol_{idx:03d}",
                "input_smiles": raw_smi,
                "status": "ok",
                "n_frames": 0,
            }
            w_smi = raw_smi
            lbl_mol = parse_labelled(raw_smi) if expand_abbrev else None
            if expand_abbrev and (
                "*" in raw_smi
                or (lbl_mol is not None and any(a.GetAtomicNum() == 0 for a in lbl_mol.GetAtoms()))
            ):
                try:
                    w_smi = Chem.MolToSmiles(expand_aliases(raw_smi))
                    rec["expanded_smiles"] = w_smi
                except ValueError as exc:
                    rec["status"] = f"alias_rejected:{exc}"
                    manifest_rows.append(rec)
                    continue

            mol_raw = Chem.MolFromSmiles(w_smi)
            if mol_raw is None or len(Chem.GetMolFrags(mol_raw)) > 1:
                rec["status"] = "invalid_or_salt_mixture"
                manifest_rows.append(rec)
                continue

            std = standardize(w_smi)
            if not std.valid:
                rec["status"] = f"invalid:{std.reason}"
                manifest_rows.append(rec)
                continue

            reasons = policy.reasons(std.mol)
            if reasons:
                rec["status"] = "ineligible:" + ",".join(reasons)
                manifest_rows.append(rec)
                continue

            ik = standard_inchikey(std) or ""
            if dedupe and ik in seen_ik:
                rec["status"] = "duplicate_inchikey"
                manifest_rows.append(rec)
                continue

            mol_3d, emb_reason = embed(std.smiles, n_conf=n_conf, seed=idx)
            if mol_3d is None:
                rec["status"] = emb_reason
                manifest_rows.append(rec)
                continue

            disp_trials = []
            if n_displaced > 0:
                mol_3d, disp_trials = displaced_conformers(
                    mol_3d, std.smiles, n_displaced, amplitude=amp, seed=idx, min_distance=min_dist
                )

            calc, calc_name, _ = get_3d_calculator(potential, mol_3d, device=str(self.device))
            symbols = [a.GetSymbol() for a in mol_3d.GetAtoms()]
            frame_energies = []
            for conf in mol_3d.GetConformers():
                pos = np.asarray(conf.GetPositions(), dtype=np.float64)
                at = Atoms(symbols=symbols, positions=pos)
                at.calc = calc
                e_ev = float(at.get_potential_energy())
                f_ev = np.asarray(at.get_forces(), dtype=np.float64)
                at_out = Atoms(symbols=symbols, positions=pos)
                at_out.info.update(
                    smiles=std.smiles,
                    inchikey=ik,
                    source_id=rec["id"],
                    conf_id=int(conf.GetId()),
                    potential=calc_name,
                    mace_energy=e_ev,
                )
                at_out.arrays["mace_forces"] = f_ev
                extxyz_frames.append(at_out)
                frame_energies.append(e_ev)

            seen_ik.add(ik)
            rec.update(
                smiles=std.smiles,
                inchikey=ik,
                n_frames=len(frame_energies),
                displaced_accepted=sum(1 for t in disp_trials if t["conf_id"] is not None),
                displaced_rejected=sum(1 for t in disp_trials if t["conf_id"] is None),
                min_energy_ev=round(min(frame_energies), 5) if frame_energies else None,
            )
            manifest_rows.append(rec)

        # Connectivity-disjoint scaffold split preview
        for r in manifest_rows:
            if r.get("smiles"):
                r["connectivity_key"] = connectivity_key(r["smiles"])
                r["connectivity_split"] = assign_split(
                    r["smiles"], fractions=(0.6, 0.2, 0.2), seed=0
                )

        extxyz_buf = io.StringIO()
        if extxyz_frames:
            write(extxyz_buf, extxyz_frames, format="extxyz")
        manifest_jsonl = "\n".join(json.dumps(r) for r in manifest_rows) + "\n"

        return {
            "total_inputs": len(smiles_list),
            "accepted_molecules": sum(1 for r in manifest_rows if r["status"] == "ok"),
            "total_frames_written": len(extxyz_frames),
            "rows": manifest_rows,
            "extxyz_preview": extxyz_buf.getvalue(),
            "manifest_jsonl": manifest_jsonl,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    # ------------------------------------------ Tab 6: Batch Benchmark & Selective Evaluation
    def run_batch_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        mode = str(payload.get("mode", "live_ocsr"))
        records_for_selective: list[dict[str, Any]] = []
        per_item_rows: list[dict[str, Any]] = []

        if mode == "custom_pairs" and payload.get("pairs"):
            for idx, item in enumerate(payload["pairs"], start=1):
                gold = str(item.get("gold", "")).strip()
                pred = str(item.get("pred", "")).strip()
                margin = float(item.get("margin", 0.5))
                if not standardize(gold).valid:
                    continue
                ps = score_pair(pred, gold)
                per_item_rows.append(
                    {
                        "id": item.get("id", f"row_{idx}"),
                        "gold": gold,
                        "pred": pred,
                        "margin": round(margin, 4),
                        "category": categorize(pred, gold),
                        **asdict(ps),
                    }
                )
                records_for_selective.append(
                    {
                        "gold": gold,
                        "pred": pred,
                        "subset": str(item.get("subset", "custom")),
                        "candidates": [(pred, 0.0), ("C", -margin)],
                    }
                )
        else:
            model_choice = str(payload.get("model", "AC"))
            beam = max(1, min(6, int(payload.get("beam", 3))))
            perturbation = str(payload.get("perturbation", "clean"))
            suite_smiles = payload.get("smiles_list") or [
                "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
                "N[C@@H](C)C(=O)O",
                "CC(=O)Oc1ccccc1C(=O)O",
                "C/C=C/C(=O)O",
                "CNC(=O)C12CC(C1)C2",
                "COC(=O)C12COC(C1)C2",
            ]
            for idx, gsmi in enumerate(suite_smiles, start=1):
                res = self.run_ocsr(
                    {
                        "smiles": gsmi,
                        "reference_smiles": gsmi,
                        "model": model_choice,
                        "beam": beam,
                        "angle_deg": 90.0 if perturbation == "rot90" else 0.0,
                        "augment_mode": perturbation
                        if perturbation in ("scan", "handdrawn")
                        else "none",
                        "min_margin": 0.25,
                        "seed": idx,
                    }
                )
                pred = res["expanded_smiles"] or res["predicted_smiles"]
                ps = score_pair(pred, gsmi)
                cands = [(c["smiles"], c["score"]) for c in res["candidates"]]
                if len(cands) == 1:
                    cands.append(("C", cands[0][1] - 0.8))
                per_item_rows.append(
                    {
                        "id": f"eval_{idx:02d}",
                        "gold": standardize(gsmi).smiles,
                        "pred": pred,
                        "margin": res["confidence_gate"]["margin"],
                        "accepted_at_025": res["confidence_gate"]["accepted"],
                        "category": categorize(pred, gsmi),
                        **asdict(ps),
                    }
                )
                records_for_selective.append(
                    {
                        "gold": gsmi,
                        "pred": pred,
                        "subset": perturbation,
                        "candidates": cands,
                    }
                )

        pair_objs = [score_pair(r["pred"], r["gold"]) for r in per_item_rows]
        agg = aggregate(pair_objs)
        exact_ci = bootstrap_ci([1.0 if p.exact else 0.0 for p in pair_objs])
        flat_ci = bootstrap_ci([1.0 if p.exact_flat else 0.0 for p in pair_objs])
        tan_ci = bootstrap_ci([p.tanimoto for p in pair_objs])
        sel_curve = (
            selective_report(records_for_selective, [0.0, 0.1, 0.25, 0.5, 1.0])
            if records_for_selective
            else []
        )

        return {
            "n": len(per_item_rows),
            "aggregate": {
                k: (
                    round(float(v), 4)
                    if isinstance(v, float) and np.isfinite(v)
                    else (None if isinstance(v, float) else v)
                )
                for k, v in agg.items()
            },
            "bootstrap_95ci": {
                "exact": [round(x, 4) for x in exact_ci],
                "exact_flat": [round(x, 4) for x in flat_ci],
                "tanimoto": [round(x, 4) for x in tan_ci],
            },
            "selective_curve": sel_curve,
            "rows": per_item_rows,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    def load_repo_benchmark(self, rel_path: str) -> dict[str, Any]:
        clean = rel_path.strip().lstrip("/\\")
        target = (BENCH_DIR / clean).resolve()
        if not str(target).startswith(str(BENCH_DIR.resolve())) or not target.is_file():
            raise ValueError(f"Benchmark receipt not found inside benchmarks/: {rel_path!r}")
        if target.suffix.lower() != ".json":
            raise ValueError("Only JSON benchmark receipts can be loaded.")
        return {
            "path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "data": json.loads(target.read_text(encoding="utf-8")),
        }
