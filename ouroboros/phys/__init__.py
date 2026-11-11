"""Physics- and spectroscopy-informed Bayesian OCSR disambiguation package."""

from ouroboros.phys.epimers import (
    SUPERATOM_FRAGMENTS,
    condense_superatom_for_depiction,
    enumerate_targeted_epimers,
    expand_superatoms,
    stereo_hamming_distance,
)
from ouroboros.phys.hrms import STANDARD_ADDUCTS, HRMSMatch, exact_adduct_mz, score_hrms
from ouroboros.phys.nmr import (
    Boltzmann3DNMRPredictor,
    H1Multiplet,
    NMRMatchResult,
    NMRPrediction,
    optimal_1d_match_mae,
    optimal_1d_match_stats,
    predict_nmr,
    score_nmr,
)
from ouroboros.phys.posterior import (
    ABLATION_PRESETS,
    CandidateScore,
    PosteriorWeights,
    RerankDecision,
    SIObservables,
    area_under_risk_coverage,
    expected_calibration_error,
    rerank_candidates,
)
from ouroboros.phys.strain import StrainResult, embed_parity_invariant, evaluate_3d_strain

__all__ = [
    "ABLATION_PRESETS",
    "Boltzmann3DNMRPredictor",
    "CandidateScore",
    "H1Multiplet",
    "HRMSMatch",
    "NMRMatchResult",
    "NMRPrediction",
    "PosteriorWeights",
    "RerankDecision",
    "SIObservables",
    "STANDARD_ADDUCTS",
    "SUPERATOM_FRAGMENTS",
    "StrainResult",
    "area_under_risk_coverage",
    "condense_superatom_for_depiction",
    "embed_parity_invariant",
    "enumerate_targeted_epimers",
    "evaluate_3d_strain",
    "exact_adduct_mz",
    "expand_superatoms",
    "expected_calibration_error",
    "optimal_1d_match_mae",
    "optimal_1d_match_stats",
    "predict_nmr",
    "rerank_candidates",
    "score_hrms",
    "score_nmr",
    "stereo_hamming_distance",
]
