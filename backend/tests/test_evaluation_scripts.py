import os
import pytest
from models.evaluation.eval_rsvqa import evaluate_rsvqa
from models.evaluation.eval_vrsbench import evaluate_vrsbench, compute_box_iou
from models.training.train_lora import run_lora_training

def test_box_iou_computation():
    # Identical boxes -> IoU = 1.0
    b1 = [0.1, 0.1, 0.5, 0.5]
    b2 = [0.1, 0.1, 0.5, 0.5]
    assert compute_box_iou(b1, b2) == pytest.approx(1.0)

    # Disjoint boxes -> IoU = 0.0
    b3 = [0.6, 0.6, 0.9, 0.9]
    assert compute_box_iou(b1, b3) == 0.0

    # Half overlap
    # b1 area = 0.4 * 0.4 = 0.16
    # b4 from 0.1 to 0.5 (y), 0.3 to 0.7 (x): area = 0.4 * 0.4 = 0.16
    # intersection: y [0.1, 0.5], x [0.3, 0.5]: area = 0.4 * 0.2 = 0.08
    # union = 0.16 + 0.16 - 0.08 = 0.24
    # iou = 0.08 / 0.24 = 1/3 ~ 0.333
    b4 = [0.1, 0.3, 0.5, 0.7]
    assert compute_box_iou(b1, b4) == pytest.approx(0.3333, rel=1e-2)

def test_rsvqa_evaluation_pipeline(test_data_dir):
    out_report = os.path.join(test_data_dir, "test_rsvqa_report.json")
    report = evaluate_rsvqa(
        annotation_path="data/datasets/sample_rsvqa.json",
        output_report_path=out_report,
        images_dir="data/samples"
    )
    assert os.path.exists(out_report)
    assert report["benchmark"] == "RSVQA"
    assert report["total_samples"] == 5
    assert 0.0 <= report["overall_accuracy"] <= 1.0
    assert 0.0 <= report["mean_confidence"] <= 1.0
    assert "presence" in report["category_accuracy"]

def test_vrsbench_evaluation_pipeline(test_data_dir):
    out_report = os.path.join(test_data_dir, "test_vrsbench_report.json")
    report = evaluate_vrsbench(
        annotation_path="data/datasets/sample_vrsbench.json",
        output_report_path=out_report,
        images_dir="data/samples"
    )
    assert os.path.exists(out_report)
    assert report["benchmark"] == "VRSBench"
    assert report["total_samples"] == 4
    assert 0.0 <= report["mean_iou"] <= 1.0
    assert 0.0 <= report["acc_at_50"] <= 1.0

def test_lora_training_dry_run(test_data_dir):
    out_dir = os.path.join(test_data_dir, "test_lora_ckpt")
    summary = run_lora_training(
        config_path="models/configs/base_vlm.yaml",
        dataset_name="rsvqa",
        dataset_path="data/datasets/sample_rsvqa.json",
        output_dir=out_dir,
        epochs=1,
        dry_run=True
    )
    assert summary["status"] == "completed"
    assert os.path.exists(os.path.join(out_dir, "adapter_config.json"))
