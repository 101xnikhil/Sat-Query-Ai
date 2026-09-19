import os
import pytest
from models.evaluation.eval_cdvqa import evaluate_cdvqa, check_cdvqa_match

def test_cdvqa_match_logic():
    assert check_cdvqa_match("Yes, significant change detected", "yes") is True
    assert check_cdvqa_match("No change occurred", "no") is True
    assert check_cdvqa_match("New construction and urban expansion observed", "construction") is True
    assert check_cdvqa_match("Vegetation canopy loss identified", "vegetation") is True
    assert check_cdvqa_match("No change", "yes") is False

def test_cdvqa_evaluation_pipeline(test_data_dir):
    out_report = os.path.join(test_data_dir, "test_cdvqa_report.json")
    report = evaluate_cdvqa(
        annotation_path="data/datasets/sample_cdvqa.json",
        output_report_path=out_report,
        images_dir="data/samples"
    )
    assert os.path.exists(out_report)
    assert report["benchmark"] == "CDVQA"
    assert report["total_samples"] == 5
    assert 0.0 <= report["vqa_accuracy"] <= 1.0
    assert 0.0 <= report["change_detection_accuracy"] <= 1.0
    assert 0.0 <= report["mean_confidence"] <= 1.0
    assert "presence" in report["category_accuracy"]
