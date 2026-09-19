import os
import pytest
from pathlib import Path

from models.evaluation.eval_fusion import evaluate_fusion

def test_evaluate_fusion_pipeline():
    """Verify Optical+SAR fusion evaluation benchmark runs end-to-end and creates valid JSON report."""
    ann_path = "data/datasets/sample_optical_sar.json"
    out_path = "data/outputs/test_eval_fusion_results.json"

    assert os.path.exists(ann_path)

    report = evaluate_fusion(
        annotation_path=ann_path,
        output_report_path=out_path,
        tile_size=128,
        overlap=32
    )

    assert os.path.exists(out_path)
    assert report["benchmark"] == "Optical-SAR-Fusion"
    assert report["total_samples"] == 3
    assert 0.0 <= report["consistency_rate"] <= 1.0
    assert 0.0 <= report["mean_cross_tool_ndwi_agreement"] <= 1.0
    assert len(report["sample_evaluations"]) == 3

    for item in report["sample_evaluations"]:
        assert "water_percentage" in item
        assert "built_up_percentage" in item
        assert "cross_tool_ndwi_agreement" in item
        assert 0.0 <= item["cross_tool_ndwi_agreement"] <= 1.0
