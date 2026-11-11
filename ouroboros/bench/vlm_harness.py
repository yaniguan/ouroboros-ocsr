"""Unified Frontier VLM (Claude / GPT / Gemini) & OCSR evaluation harness for ChemVeritas-Bench.

Builds structured multimodal prompts for T1-T5, parses free-form or JSON CoT responses
into Veritas-RPF 4-step reasoning checkpoints, imports MolScribe/DECIMER CSV/JSONL outputs,
and provides zero-SDK HTTPS adapters for Anthropic, OpenAI, and Google Gemini APIs.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import urllib.request
from dataclasses import replace
from pathlib import Path

from ouroboros.phys.epimers import expand_superatoms
from ouroboros.phys.hrms import score_hrms
from ouroboros.phys.posterior import ABLATION_PRESETS, rerank_candidates
from ouroboros.si.parser import SICharacterization, parse_si_block

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.S)


def build_vlm_prompt(item: dict) -> str:
    """Construct deterministic JSON-output prompt for a ChemVeritas task record."""
    task = item["task"]
    if task == "T1_visual_ocsr":
        return (
            "You are an expert organic chemist. Transcribe the depicted chemical structure into "
            "an exact isomeric SMILES string preserving all wedge/dash (@/@@) and E/Z stereo "
            "and expanding any condensed protecting-group abbreviations. Return ONLY JSON:\n"
            '{"smiles": "<isomeric_SMILES>", "confidence": 0.0_to_1.0}'
        )
    if task == "T2_multimodal_si_disambig":
        cand_lines = "\n".join(f"  [{i}] {s}" for i, s in enumerate(item["candidates"]))
        return (
            "You are given a Supporting Information (SI) molecular depiction, its experimental "
            "analytical block (HRMS + 1H/13C NMR), and candidate SMILES structures.\n"
            f"SI Data:\n{item['si_text']}\n"
            f"Candidates:\n{cand_lines}\n"
            "Verify: (1) molecular formula & HRMS adduct (<5 ppm), (2) 13C symmetry-equivalent "
            "carbon signal count, (3) 3D ring/diastereomer strain & CIP preservation, and "
            "(4) exact stereoisomer match. Return ONLY JSON:\n"
            '{"step1_formula": "<HillFormula>", "step1_hrms_adduct": "<adduct>", '
            '"step2_c13_sym_signals": <int>, "step3_cip_preserved": true, '
            '"selected_index": <int>, "smiles": "<SMILES>", "confidence": 0.0_to_1.0}'
        )
    if task == "T3_silent_error_audit":
        return (
            "Audit whether the proposed SMILES matches the depicted molecule and SI spectra, or "
            "contains a silent stereochemical/constitutional error.\n"
            f"Proposed SMILES: {item['proposed_smiles']}\n"
            f"SI Data:\n{item['si_text']}\n"
            "Classify category in {correct, enantiomer, diastereomer, constitutional, invalid} "
            "and set accept=true iff exact match. Return ONLY JSON:\n"
            '{"accept": <bool>, "category": "<category>", "confidence": 0.0_to_1.0}'
        )
    if task == "T4_thermodynamic_ranking":
        cand_lines = "\n".join(f"  [{i}] {s}" for i, s in enumerate(item["isomer_smiles"]))
        return (
            "Rank the following stereoisomers from LOWEST 3D internal conformational energy "
            "(most thermodynamically stable) to HIGHEST energy (most strained):\n"
            f"{cand_lines}\n"
            'Return ONLY JSON: {"ranked_smiles": ["<lowest>", "..."], "lowest_smiles": "<SMILES>"}'
        )
    if task == "T5_epistemic_parity_abstention":
        pair = item["enantiomer_pair"]
        return (
            "The 2D structure image has obliterated wedge/dash contrast. Only achiral solution "
            "NMR and HRMS are available:\n"
            f"{item['si_text']}\n"
            f"Candidate A: {pair[0]}\nCandidate B: {pair[1]}\n"
            "If achiral physics/NMR uniquely identifies one candidate, output its SMILES; if "
            "fundamental symmetry makes (R)/(S) enantiomers indistinguishable without chiroptical "
            'data, output action="ABSTAIN_PARITY". Return ONLY JSON:\n'
            '{"action": "<SELECT_A|SELECT_B|ABSTAIN_PARITY>", "smiles": "<SMILES_or_empty>"}'
        )
    raise ValueError(f"unsupported task {task!r}")


def parse_vlm_response(raw_text: str) -> dict:
    """Extract structured JSON fields from a VLM response (handles markdown code fences)."""
    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I)
        if parts:
            cleaned = parts[-1].strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    matches = _JSON_BLOCK_RE.findall(cleaned)
    for block in reversed(matches):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {"smiles": cleaned.splitlines()[0].strip() if cleaned else "", "parse_fallback": True}


def run_physics_posterior_baseline(
    item: dict, *, ablation: str = "A6_full_posterior", n_conf: int = 4, seed: int = 0
) -> dict:
    """Execute our deterministic Bayesian physics+spectroscopy engine on T2, T3, T4, or T5."""
    if ablation not in ABLATION_PRESETS:
        raise ValueError(f"unknown ablation {ablation!r}")
    task = item["task"]

    if task == "T2_multimodal_si_disambig":
        si = SICharacterization.from_dict(item["si"])
        cands = [(s, -0.02 * i) for i, s in enumerate(item["candidates"])]
        closed_w = replace(ABLATION_PRESETS[ablation], augment_epimers=False)
        dec = rerank_candidates(
            cands,
            si.to_observables(),
            ablation=ablation,
            weights=closed_w,
            n_conf=n_conf,
            seed=seed,
        )
        top = dec.candidates[0]
        h_info = score_hrms(top.canonical_smiles, si.hrms_obs_mz, adduct=si.hrms_adduct)
        sel_idx = (
            item["candidates"].index(top.raw_smiles)
            if top.raw_smiles in item["candidates"]
            else 0
        )
        return {
            "task_id": item["task_id"],
            "smiles": dec.selected_smiles,
            "selected_index": sel_idx,
            "confidence": dec.posterior_prob,
            "epistemic_status": dec.epistemic_status,
            "step1_formula": h_info.formula,
            "step1_hrms_adduct": top.hrms_adduct or si.hrms_adduct,
            "step2_c13_sym_signals": top.c13_sym_count,
            "step3_cip_preserved": top.cip_preserved,
        }

    if task == "T5_epistemic_parity_abstention":
        si = SICharacterization.from_dict(item["si"])
        cands = [(s, -1.0) for s in item["enantiomer_pair"]]
        closed_w = replace(ABLATION_PRESETS[ablation], augment_epimers=False)
        dec = rerank_candidates(
            cands,
            si.to_observables(),
            ablation=ablation,
            weights=closed_w,
            n_conf=n_conf,
            seed=seed,
        )
        abstain = dec.epistemic_status == "abstain_enantiomer_underdetermined"
        return {
            "task_id": item["task_id"],
            "action": "ABSTAIN_PARITY" if abstain else "SELECT_A",
            "smiles": "" if abstain else dec.selected_smiles,
            "confidence": dec.posterior_prob,
        }

    if task == "T3_silent_error_audit":
        from ouroboros.eval.standardize import standardize
        from ouroboros.phys.nmr import score_nmr
        from ouroboros.phys.strain import evaluate_3d_strain

        si = (
            SICharacterization.from_dict(item["si"])
            if "si" in item
            else parse_si_block(item["si_text"], compound_id=item.get("mol_id", "1"))
        )
        prop = expand_superatoms(item["proposed_smiles"])
        st = standardize(prop)
        if not st.valid or st.smiles is None:
            return {
                "task_id": item["task_id"],
                "accept": False,
                "category": "invalid",
                "confidence": 0.99,
            }
        obs = si.to_observables()
        hr = score_hrms(
            st.smiles, obs.hrms_obs_mz, adduct=obs.hrms_adduct, expected_formula=obs.formula
        )
        sr = evaluate_3d_strain(st.smiles, n_conf=n_conf, seed=seed, relax_steps=80)
        nr = score_nmr(
            st.smiles,
            obs_c13_ppm=obs.c13_peaks_ppm,
            obs_dept_counts=obs.dept_counts,
            obs_h1_multiplets=obs.h1_multiplets,
            strain=sr,
        )
        if not hr.within_window or nr.signal_count_penalty >= 4.0:
            return {
                "task_id": item["task_id"],
                "accept": False,
                "category": "constitutional",
                "confidence": 0.95,
            }
        dec = rerank_candidates(
            [(st.smiles, 0.0)], obs, ablation="A6_full_posterior", n_conf=n_conf, seed=seed
        )
        if dec.canonical_smiles != st.smiles or (nr.c13_mae_ppm or 0.0) > 1.5:
            return {
                "task_id": item["task_id"],
                "accept": False,
                "category": "diastereomer",
                "confidence": 0.85,
            }
        return {
            "task_id": item["task_id"],
            "accept": True,
            "category": "correct",
            "confidence": dec.posterior_prob,
        }

    if task == "T4_thermodynamic_ranking":
        from ouroboros.phys.strain import evaluate_3d_strain

        scored = []
        for s in item["isomer_smiles"]:
            sr = evaluate_3d_strain(s, n_conf=n_conf, seed=seed, relax_steps=80)
            scored.append((sr.min_energy_kcal if sr.min_energy_kcal is not None else 999.0, s))
        scored.sort(key=lambda t: t[0])
        ranked = [s for _, s in scored]
        return {
            "task_id": item["task_id"],
            "ranked_smiles": ranked,
            "lowest_smiles": ranked[0],
        }

    raise ValueError(f"physics baseline does not directly handle {task!r}")


def import_external_ocsr_predictions(path: Path) -> dict[str, dict]:
    """Load MolScribe / DECIMER / SwinOCSR predictions from CSV or JSONL keyed by task_id."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                tid = str(row.get("task_id") or row.get("id"))
                out[tid] = {
                    "task_id": tid,
                    "smiles": expand_superatoms(row.get("smiles") or row.get("pred") or "") or "",
                    "confidence": float(row.get("confidence", 1.0)),
                }
        return out

    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = str(row.get("task_id") or row.get("id") or "")
            smi = row.get("smiles") or row.get("pred") or row.get("prediction") or ""
            conf = float(row["confidence"]) if row.get("confidence") else 1.0
            out[tid] = {"task_id": tid, "smiles": expand_superatoms(smi) or "", "confidence": conf}
    return out


def call_frontier_vlm(
    provider: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict:
    """Zero-dependency HTTPS client for Claude (anthropic), GPT (openai), and Gemini (google)."""
    img_b64 = (
        base64.b64encode(Path(image_path).read_bytes()).decode("ascii") if image_path else None
    )
    prov = provider.lower()

    if prov == "anthropic":
        key = os.environ["ANTHROPIC_API_KEY"]
        content: list[dict] = []
        if img_b64:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                }
            )
        content.append({"type": "text", "text": prompt})
        body = json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return parse_vlm_response(text)

    if prov == "openai":
        key = os.environ["OPENAI_API_KEY"]
        content = [{"type": "text", "text": prompt}]
        if img_b64:
            content.insert(
                0,
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            )
        body = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return parse_vlm_response(text)

    if prov in ("gemini", "google"):
        key = os.environ["GEMINI_API_KEY"]
        parts: list[dict] = []
        if img_b64:
            parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        parts.append({"text": prompt})
        body = json.dumps(
            {
                "contents": [{"parts": parts}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
        ).encode("utf-8")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={key}"
        )
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_vlm_response(text)

    raise ValueError(f"unsupported VLM provider {provider!r}")
