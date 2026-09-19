import pytest
import numpy as np
from backend.confidence.scorer import get_confidence_scorer, ConfidenceScorer
from backend.confidence.calibration import compute_ece, evaluate_calibration

def test_confidence_scorer_formula_and_breakdown():
    scorer = get_confidence_scorer()
    
    # 1. Nominal case with perfect cross-tool agreement and no quality penalties
    res = scorer.compute_confidence(
        raw_model_confidence=0.90,
        tool_name="optical_sar_fusion",
        cross_tool_agreement=1.0,
        actions_taken=[],
        computed_metrics={"cross_tool_ndwi_agreement": 1.0}
    )
    score = res["confidence_score"]
    bd = res["confidence_breakdown"]
    
    # Check that score is high (>0.85)
    assert score >= 0.85
    assert bd["token_probability"] == 0.90
    assert bd["cross_tool_agreement"] == 1.0
    assert bd["input_quality"] == 1.0
    assert "resampling_penalty" not in bd

def test_confidence_scorer_penalties():
    scorer = get_confidence_scorer()
    
    # Case with resampling and missing spectral bands
    res = scorer.compute_confidence(
        raw_model_confidence=0.80,
        tool_name="optical_sar_fusion",
        cross_tool_agreement=0.40,
        actions_taken=[
            "Auto-resampled SAR to Optical grid via bilinear",
            "Spectral NDWI cross-check bypassed: Required bands missing"
        ],
        computed_metrics={"degradation_mode": "single_band_degraded"}
    )
    score = res["confidence_score"]
    bd = res["confidence_breakdown"]
    
    # Quality score should reflect resampling and missing band deductions
    assert bd["input_quality"] < 1.0
    assert "resampling_penalty" in bd
    assert "missing_bands_penalty" in bd
    # Overall score should be appropriately penalized (< 0.75)
    assert score < 0.75

def test_ece_reliability_calibration():
    # Perfectly calibrated case: confidences = [0.8, 0.8], accuracies = [1, 1] => gap = 0.2
    confs = np.array([0.9, 0.9, 0.1, 0.1])
    accs = np.array([1.0, 1.0, 0.0, 0.0])
    ece_res = compute_ece(confs, accs, num_bins=5)
    
    assert "ece" in ece_res
    assert "mce" in ece_res
    assert "brier_score" in ece_res
    assert len(ece_res["bins"]) > 0
    # Perfect predictions with 0.9 conf and 1.0 acc have ECE = 0.1
    assert ece_res["ece"] == 0.10
