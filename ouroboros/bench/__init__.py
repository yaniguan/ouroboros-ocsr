"""ChemVeritas-Bench: Multimodal Physics & Spectroscopy Chemistry Benchmark."""

from ouroboros.bench.builder import VIEWS, HardNegative, build_chemveritas_benchmark
from ouroboros.bench.catalog import STRATA, CatalogEntry, build_catalog
from ouroboros.bench.scoring import (
    bootstrap_ci,
    evaluate_t1_visual_predictions,
    evaluate_t2_multimodal_predictions,
    evaluate_t3_audit_predictions,
    evaluate_t4_thermo_predictions,
    evaluate_t5_parity_abstention,
    score_veritas_rpf,
)
from ouroboros.bench.vlm_harness import (
    build_vlm_prompt,
    call_frontier_vlm,
    import_external_ocsr_predictions,
    parse_vlm_response,
    run_physics_posterior_baseline,
)

__all__ = [
    "CatalogEntry",
    "HardNegative",
    "STRATA",
    "VIEWS",
    "bootstrap_ci",
    "build_catalog",
    "build_chemveritas_benchmark",
    "build_vlm_prompt",
    "call_frontier_vlm",
    "evaluate_t1_visual_predictions",
    "evaluate_t2_multimodal_predictions",
    "evaluate_t3_audit_predictions",
    "evaluate_t4_thermo_predictions",
    "evaluate_t5_parity_abstention",
    "import_external_ocsr_predictions",
    "parse_vlm_response",
    "run_physics_posterior_baseline",
    "score_veritas_rpf",
]
