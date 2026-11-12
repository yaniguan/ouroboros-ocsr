"""Tests for the Interactive Physics & Equivariance Workbench UI (ouroboros.ui)."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
import torch

from ouroboros.ui import WorkbenchEngine, audit_3d_physical_compliance
from ouroboros.ui.__main__ import WorkbenchHandler


@pytest.fixture(scope="module")
def engine():
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    eng = WorkbenchEngine(device="cpu")
    yield eng
    torch.set_num_threads(prev_threads)


def test_3d_compliance_all_10_gates_pass_chiral_molecule():
    res = audit_3d_physical_compliance(
        "N[C@@H](C)C(=O)O",
        potential="mmff94s",
        n_conf=2,
        do_relax=True,
        steps=80,
    )
    assert res["ok"] is True
    assert res["overall_status"] == "pass"
    assert res["summary_counts"]["total"] == 10
    assert res["summary_counts"]["fail"] == 0
    assert res["summary_counts"]["warn"] == 0
    assert len(res["gates"]) == 10
    assert all(g["status"] == "pass" for g in res["gates"])

    # Verify live SE(3)/O(3) equivariance, Noether conservation & conservative force field
    eq = res["equivariance_metrics"]
    assert eq["se3_energy_abs_err_ev"] < 1e-4
    assert eq["so3_force_max_err_ev_a"] < 1e-4
    assert eq["net_force_norm_ev_a"] < 1e-3
    assert eq["net_torque_norm_ev"] < 1e-3
    assert eq["parity_energy_abs_err_ev"] < 1e-4
    assert eq["parity_force_max_err_ev_a"] < 1e-4
    assert eq["conservative_grad_fd_err_ev_a"] < 2e-3

    # Mirroring L-Alanine in 3D yields its enantiomer D-Alanine
    assert res["mirror_relation"] == "enantiomer"
    assert res["structure_3d"]["n_atoms"] == 13
    assert res["standardized_smiles"] in res["exports"]["xyz"]
    assert "mace_forces" in res["exports"]["extxyz"]
    assert len(res["conformers"]) >= 1
    assert "coords" in res["conformers"][0]
    assert "forces_vec" in res["conformers"][0]


def test_3d_compliance_catches_pauli_steric_clash():
    res = audit_3d_physical_compliance(
        "N[C@@H](C)C(=O)O",
        potential="mmff94s",
        n_conf=1,
        do_relax=True,
        inject_steric_clash=True,
    )
    assert res["ok"] is False
    assert res["overall_status"] == "fail"
    gate_by_id = {g["id"]: g for g in res["gates"]}
    assert gate_by_id["G5_PAULI_STERIC_CLEARANCE"]["status"] == "fail"
    assert res["structure_3d"]["d_min_a"] < 0.75
    assert len(res["structure_3d"]["clashes"]) >= 1


@pytest.mark.parametrize(
    ("smiles", "potential", "expected_failed_gate"),
    [
        ("not_a_smiles", "mmff94s", "G1_VALENCE_TOPOLOGY"),
        ("CC(=O)[O-].[Na+]", "mmff94s", "G1_VALENCE_TOPOLOGY"),
        ("[CH2]c1ccccc1", "mmff94s", "G2_DOMAIN_ELIGIBILITY"),
        ("CC1(C)OB(c2ccc(F)cc2)OC1(C)C", "mace-small", "G2_DOMAIN_ELIGIBILITY"),
    ],
)
def test_3d_compliance_rejects_invalid_salt_radical_and_unsupported_element(
    smiles, potential, expected_failed_gate
):
    res = audit_3d_physical_compliance(smiles, potential=potential, n_conf=1)
    assert res["ok"] is False
    assert res["overall_status"] == "fail"
    gate_by_id = {g["id"]: g for g in res["gates"]}
    assert gate_by_id[expected_failed_gate]["status"] == "fail"


def test_3d_compliance_unassigned_stereo_warns_and_alias_expands():
    warn_res = audit_3d_physical_compliance("CC(O)C(=O)O", potential="mmff94s", n_conf=1)
    assert warn_res["ok"] is True
    assert warn_res["overall_status"] == "warn"
    gate_by_id = {g["id"]: g for g in warn_res["gates"]}
    assert gate_by_id["G3_STEREO_SPECIFICATION"]["status"] == "warn"

    alias_res = audit_3d_physical_compliance(
        "[Boc]NCc1ccc([Ph])cc1", potential="mmff94s", n_conf=1, expand_abbrev=True
    )
    assert alias_res["ok"] is True
    assert "*" not in alias_res["standardized_smiles"]


def test_2d_equivariance_steerable_vs_baseline_and_mirror_probe(engine):
    c8 = engine.run_2d_equivariance(
        {
            "mode": "arch",
            "arch": "steerable",
            "order": 8,
            "smiles": "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
            "angle_deg": 90.0,
        }
    )
    assert c8["focus_feature_error"] is not None and c8["focus_feature_error"] < 1e-4
    assert c8["symmetry_verdict"]["exact_c4_grid_equivariant"] is True
    assert c8["mirror_probe"]["reflection_non_invariant_ok"] is True
    assert c8["feature_maps"] is not None

    eq_attn = engine.run_2d_equivariance(
        {"mode": "arch", "arch": "equiv_attention", "order": 4, "smiles": "CCO", "angle_deg": 90.0}
    )
    assert eq_attn["symmetry_verdict"]["exact_c4_grid_equivariant"] is True

    base = engine.run_2d_equivariance(
        {"mode": "arch", "arch": "baseline", "order": 8, "smiles": "CCO", "angle_deg": 90.0}
    )
    assert base["focus_feature_error"] is None
    assert base["focus_logit_error"] > 1e-4


def test_error_propagation_enantiomer_zero_strain_vs_diastereomer(engine):
    prop = engine.run_error_propagation(
        {
            "truth_smiles": "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1",
            "potential": "mmff94s",
            "n_conf": 8,
            "relax": True,
        }
    )
    by_cat = {r["category"]: r for r in prop["candidates"]}
    assert "correct" in by_cat
    assert "enantiomer" in by_cat
    assert "diastereomer" in by_cat
    # Enantiomer must have exact 0 energy difference up to numerical tolerance
    assert abs(by_cat["enantiomer"]["delta_e_kcal"]) < 0.05
    # Diastereomer has distinct internal distances and non-zero energy gap
    assert abs(by_cat["diastereomer"]["delta_e_kcal"]) > 0.2


def test_ocsr_forcefield_and_batch_eval_workflows(engine):
    ocsr_res = engine.run_ocsr(
        {
            "model": "demo_c8",
            "beam": 2,
            "smiles": "N[C@@H](C)C(=O)O",
            "reference_smiles": "N[C@@H](C)C(=O)O",
            "angle_deg": 90.0,
            "augment_mode": "scan",
        }
    )
    assert ocsr_res["source_image"].startswith("data:image/png;base64,")
    assert ocsr_res["processed_image"].startswith("data:image/png;base64,")
    assert "confidence_gate" in ocsr_res

    ff_res = engine.run_forcefield_builder(
        {
            "smiles_list": ["N[C@@H](C)C(=O)O", "CNC(=O)C12CC(C1)C2", "CC(=O)[O-].[Na+]"],
            "potential": "mmff94s",
            "n_conf": 1,
            "displaced_samples": 1,
        }
    )
    assert ff_res["total_inputs"] == 3
    assert ff_res["accepted_molecules"] == 2
    assert ff_res["total_frames_written"] >= 2
    assert "mace_forces" in ff_res["extxyz_preview"]

    eval_res = engine.run_batch_evaluation(
        {
            "mode": "custom_pairs",
            "pairs": [
                {"gold": "N[C@@H](C)C(=O)O", "pred": "N[C@@H](C)C(=O)O", "margin": 0.9},
                {"gold": "N[C@@H](C)C(=O)O", "pred": "N[C@H](C)C(=O)O", "margin": 0.1},
            ],
        }
    )
    assert eval_res["n"] == 2
    assert eval_res["aggregate"]["exact"] == 0.5
    assert eval_res["aggregate"]["exact_flat"] == 1.0
    assert len(eval_res["selective_curve"]) >= 1

    st = engine.status()
    assert st["ready"] is True
    if st["repo_benchmarks"]:
        bench = engine.load_repo_benchmark(st["repo_benchmarks"][0])
        assert bench["path"].startswith("benchmarks/")


def test_workbench_http_handler_in_memory(engine):
    server = SimpleNamespace(prefix="/wb-test/", engine=engine)

    class MemConn:
        def __init__(self, raw_req: bytes) -> None:
            self.incoming = io.BytesIO(raw_req)
            self.outgoing = io.BytesIO()

        def makefile(self, *_args):
            return self.incoming

        def sendall(self, chunk: bytes) -> None:
            self.outgoing.write(chunk)

    def send(path: str, payload: dict | None = None, content_type: str = "application/json"):
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        method = "POST" if payload is not None else "GET"
        hdr = (
            f"{method} {path} HTTP/1.0\r\nHost: localhost\r\n"
            f"Content-Type: {content_type}\r\nContent-Length: {len(body)}\r\n\r\n"
        )
        conn = MemConn(hdr.encode("utf-8") + body)
        WorkbenchHandler(conn, ("127.0.0.1", 8787), server)
        raw_hdr, raw_body = conn.outgoing.getvalue().split(b"\r\n\r\n", 1)
        return int(raw_hdr.split()[1]), raw_body

    code, html_body = send("/wb-test/")
    assert code == 200
    assert b"Ouroboros Physics" in html_body

    code, health_body = send("/wb-test/api/health")
    assert code == 200 and json.loads(health_body)["ready"] is True

    code, status_body = send("/wb-test/api/status")
    assert code == 200 and json.loads(status_body)["ready"] is True

    code, c3d_body = send(
        "/wb-test/api/compliance3d",
        {"smiles": "CCO", "potential": "mmff94s", "n_conf": 1, "relax": False},
    )
    assert code == 200 and json.loads(c3d_body)["ok"] is True

    assert send("/wb-test/api/compliance3d", {"smiles": ""})[0] == 400
    assert send("/wb-test/api/compliance3d", {"smiles": "CCO"}, "text/plain")[0] == 415
    assert send("/wrong-prefix/api/health")[0] == 404
    assert send("/wb-test/api/unknown", {"x": 1})[0] == 404
